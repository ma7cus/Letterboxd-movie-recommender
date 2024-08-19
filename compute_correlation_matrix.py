import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import logging

def make_training_data_gpu_tensor(training_data, device):
    """
    Encodes user and movie IDs, converts ratings into a dense tensor, and transfers it to the GPU.

    This function encodes user and movie IDs using label encoders, converts the training data into a 
    sparse tensor format, and then into a dense tensor, which is moved to the specified GPU device.

    Parameters
    ----------
    training_data : DataFrame
        A DataFrame containing the training data with columns 'user_id', 'movie_id', and 'rating_val'.
    device : torch.device
        The GPU device on which the tensor operations will be performed.

    Returns
    -------
    training_tensor : torch.Tensor
        The training data reshaped into a dense tensor and transferred to the GPU.
    user_encoder : LabelEncoder
        The label encoder used for encoding user IDs.
    movie_encoder : LabelEncoder
        The label encoder used for encoding movie IDs.
    """

    print("Encoding user and movie IDs...")
    logging.info("Encoding user and movie IDs...")
    
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    training_data = training_data.copy()

    training_data['user_id'] = user_encoder.fit_transform(training_data['user_id'].astype(str)).astype(int)
    training_data['movie_id'] = movie_encoder.fit_transform(training_data['movie_id'].astype(str)).astype(int)

    if training_data.isnull().values.any():
        logging.error("Training data contains missing values.")
        raise ValueError("Training data contains missing values. Please clean the data before proceeding.")
    

    row_indices = torch.tensor(training_data['user_id'].values, dtype=torch.long, device=device)
    col_indices = torch.tensor(training_data['movie_id'].values, dtype=torch.long, device=device)
    ratings = torch.tensor(training_data['rating_val'].values, dtype=torch.float32, device=device)

    num_users = row_indices.max().item() + 1
    num_movies = col_indices.max().item() + 1

    training_sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=ratings,
        size=(num_users, num_movies)
    )

    print("Converting sparse tensor to dense tensor...")
    logging.info("Converting sparse tensor to dense tensor...")

    training_tensor = training_sparse_tensor.to_dense()
    
    return training_tensor, user_encoder, movie_encoder

def compute_column_means(training_tensor, batch_size, device):
    """
    Computes the mean of each column (movie) in the training tensor, processing the data in batches to manage memory usage.

    This function calculates the mean ratings for each movie by summing the ratings column-wise 
    across all users, processing the data in batches to avoid memory overflow on the GPU.

    Parameters
    ----------
    training_tensor : torch.Tensor
        The dense tensor containing user-movie ratings.
    batch_size : int
        The number of rows to process in each batch to manage memory usage.
    device : torch.device
        The GPU device on which the operations will be performed.

    Returns
    -------
    column_means : torch.Tensor
        A tensor containing the mean rating for each movie, used for mean-centering the data.
    """

    logging.info("Computing column means...")
    
    num_rows, num_columns = training_tensor.shape

    column_sums = torch.zeros((1, num_columns), dtype=torch.float32, device=device) #Initialises a set for column sums

    with tqdm(total=(num_rows // batch_size) + 1, desc="Computing column means") as pbar: #Progress bar in steps of each batch up to total batches
        for start in range(0, num_rows, batch_size): #Iterates down the columns in batches of size 'batch_size'
            end = min(start + batch_size, num_rows)  #Ends at either the end of the batch or the end of the matrix
            batch_sum = training_tensor[start:end, :].sum(dim=0, keepdim=True) #Calculates the sum of the current batch for all columns
            column_sums += batch_sum #Adds this sum to the total
            
            torch.cuda.empty_cache()
            pbar.update(1)

    column_means = column_sums / num_rows #Calculates the mean from the total sum of each column
    
    return column_means

def mean_center_data(training_tensor, column_means, batch_size, device):
    """
    Mean-centers the data by subtracting the column means from each element in the training tensor.

    This function adjusts each rating in the training tensor by subtracting the mean rating 
    for the corresponding movie. The operation is performed in batches to manage GPU memory usage.

    Parameters
    ----------
    training_tensor : torch.Tensor
        The dense tensor containing user-movie ratings.
    column_means : torch.Tensor
        The precomputed mean ratings for each movie.
    batch_size : int
        The number of rows to process in each batch to manage memory usage.
    device : torch.device
        The GPU device on which the operations will be performed.

    Returns
    -------
    centered_data : torch.Tensor
        The mean-centered data tensor.
    """

    logging.info("Mean centering the data...")
    
    num_rows, num_columns = training_tensor.shape
    centered_data = torch.zeros_like(training_tensor, device=device) #Initialises a matrix of zeros the same size as the training_tensor

    with tqdm(total=(num_rows // batch_size) + 1, desc="Mean centering data") as pbar: #Progress bar which steps through each batch of rows in steps of 'batch_size'
        for start in range(0, num_rows, batch_size): #Iterates batches of 'batch_size' number of rows
            end = min(start + batch_size, num_rows) #Ends at the end of the batch or matrix
            centered_data[start:end, :] = training_tensor[start:end, :] - column_means #Centres the matrix by subtracting column means from the column they're in
                                                                                       #Note that in pytorch, the column means object is automatically applied to all rows even though it's just one row.
            pbar.update(1)

    return centered_data


def compute_partial_covariances_upper(centered_data, block_size, device):
    """
    Computes the upper triangle of the covariance matrix using block matrix multiplication on the GPU.

    This function calculates the covariance matrix's upper triangle by multiplying blocks of the 
    mean-centered data matrix with its transpose. The result is stored in a matrix on the CPU.

    Parameters
    ----------
    centered_data : torch.Tensor
        The mean-centered data tensor with shape (n, m), where n is the number of users and m is the number of movies.
    block_size : int
        The block size for dividing the matrices during multiplication.
    device : torch.device
        The GPU device on which the operations will be performed.

    Returns
    -------
    upper_covariance_matrix : torch.Tensor
        The upper triangle of the covariance matrix with shape (m, m), stored on the CPU.
    """

    logging.info("Computing covariance matrix")

    m, n = centered_data.shape  
    
    # Calculate the number of blocks required to cover the matrix vertically
    n_blocks_m = (n + block_size - 1) // block_size 
    
    # Calculate the total number of blocks to be processed in the upper triangle of the matrix.
    total_blocks = n_blocks_m * (n_blocks_m + 1) // 2
    
    # Initialize the covariance matrix on the CPU to store the results.
    upper_covariance_matrix = torch.zeros((n, n), device='cpu') 
    
    # Create a progress bar to track the processing of blocks.
    with tqdm(total=total_blocks, desc="Computing covariance matrix", leave=True) as pbar:
        # Iterate over the blocks in the row dimension of the matrix.
        for i in range(0, n, block_size): 

            i_end = min(i + block_size, n) 
            
            # For each block in the row dimension, iterate over the blocks in the column dimension.
            # Start j from i to only compute the upper triangle (since the matrix is symmetric).
            for j in range(i, n, block_size):  
                
                j_end = min(j + block_size, n) 
            
                # For the current block (i, j), iterate over the other blocks needed for block multiplication.
                for p in range(0, m, block_size): 
                    
                    p_end = min(p + block_size, m)  
                
                    # Extract the current block from the centered data matrix.
                    A_block = centered_data[p:p_end, i:i_end].to(device)
                    
                    # Extract the corresponding block from the centered data matrix.
                    B_block = centered_data[p:p_end, j:j_end].to(device)
                
                    # Perform the matrix multiplication of A_block.T and B_block on the GPU.
                    upper_covariance_matrix[i:i_end, j:j_end] += (torch.matmul(A_block.T, B_block) / (n-1)).cpu()
                
                    # Clear the GPU memory cache to manage memory efficiently and avoid overflow.
                    del A_block, B_block
                    torch.cuda.empty_cache()
                pbar.update(1)
    
    # Return the upper triangle of the covariance matrix.
    return upper_covariance_matrix


def normalise_covariance_matrix(upper_covariance_matrix, device, block_size):
    """
    Normalises the upper triangle of the covariance matrix to produce the correlation matrix.

    This function normalises each block of the upper triangle of the covariance matrix by dividing 
    by the product of the standard deviations of the corresponding movies. After normalisation, 
    the upper triangle is mirrored to the lower triangle to form the complete correlation matrix.

    Parameters
    ----------
    upper_covariance_matrix : torch.Tensor
        The upper triangle of the covariance matrix with shape (m, m), stored on the CPU.
    device : torch.device
        The GPU device on which the operations will be performed.
    block_size : int
        The size of the blocks to use for normalisation.

    Returns
    -------
    correlation_matrix : torch.Tensor
        The complete correlation matrix with shape (m, m).
    """

    logging.info("Normalising the covariance matrix...")
    
    m = upper_covariance_matrix.shape[0]
    
    # Compute the standard deviations (square root of the diagonal elements of the covariance matrix)
    norms = torch.sqrt(torch.diag(upper_covariance_matrix))

    # Calculate the number of blocks needed along one dimension of the matrix
    n_blocks_m = (m // block_size) + 1  # This is the number of blocks required to fill the matrix vertically

    # Calculate total number of blocks in the upper triangle
    total_blocks = n_blocks_m * (n_blocks_m + 1) // 2  # This is the sum of the first 'n_blocks_m' numbers, representing the number of blocks in the triangle.

    # Use tqdm to track progress of the normalisation process
    with tqdm(total=total_blocks, desc="Normalising covariance matrix: ") as pbar:
        # Iterate through the matrix in blocks, focusing on the upper triangle.
        for start_i in range(0, m, block_size):
            
            end_i = min(start_i + block_size, m)
            
            # For each block starting at 'start_i', iterate over the blocks in the column dimension,
            # ensuring that we only process the upper triangle (start_j >= start_i).
            for start_j in range(start_i, m, block_size):
                
                end_j = min(start_j + block_size, m)
                
                # Select the appropriate norms for the current block from the diagonal of the covariance matrix.
                norm_i = norms[start_i:end_i].to(device)
                norm_j = norms[start_j:end_j].to(device)
                
                # Extract the current block of the covariance matrix and move it to the GPU.
                block_covariance = upper_covariance_matrix[start_i:end_i, start_j:end_j].to(device)
                
                # Normalise the current block by dividing by the product of the corresponding norms.
                block_normalised = block_covariance / (norm_i.unsqueeze(1) * norm_j.unsqueeze(0))
                
                # Store the normalised block back in the covariance matrix on the CPU.
                upper_covariance_matrix[start_i:end_i, start_j:end_j] = block_normalised.cpu()
                
                # If the block is not on the diagonal (start_i != start_j), mirror the result to the lower triangle.
                if start_i != start_j:
                    upper_covariance_matrix[start_j:end_j, start_i:end_i] = block_normalised.T.cpu()

                # Clear the GPU memory cache to ensure efficient memory usage.
                torch.cuda.empty_cache()
                
                # Update the progress bar to reflect that one more block has been processed.
                pbar.update(1)

    return upper_covariance_matrix


def compute_correlation_matrix(training_data, device, row_batch_size=2000, col_block_size=2000):
    """
    Carries out the entire process of calculating the correlation matrix from raw training data.

    This function encodes user and movie IDs, constructs a dense tensor from the training data, 
    mean-centers the data, computes the covariance matrix, and normalizes it to produce the final 
    correlation matrix. The correlation matrix is calculated using GPU acceleration.

    Parameters
    ----------
    training_data : DataFrame
        A DataFrame containing the training data with columns 'user_id', 'movie_id', and 'rating_val'.
    device : torch.device
        The GPU device on which the operations will be performed.
    row_batch_size : int, optional
        The batch size for processing rows. Default is 2000.
    col_block_size : int, optional
        The block size for processing columns. Default is 2000.

    Returns
    -------
    correlation_matrix : torch.Tensor
        The final correlation matrix with shape (m, m).
    movie_encoder : LabelEncoder
        The label encoder used for encoding movie IDs.
    """

    print("Using device:", device)
    
    #Converting the training data to a tensor for the gpu
    training_tensor, user_encoder, movie_encoder = make_training_data_gpu_tensor(training_data, device)
    torch.cuda.empty_cache()

    #Calculating column means to mean centre the matrix
    column_means = compute_column_means(training_tensor, row_batch_size, device)
    torch.cuda.empty_cache()

    #Mean centring the data
    centered_data = mean_center_data(training_tensor, column_means, row_batch_size, device)
    torch.cuda.empty_cache()

    #Calculating the upper triangle of the covariance matrix
    partial_covariance_matrix_upper = compute_partial_covariances_upper(centered_data, col_block_size, device)
    torch.cuda.empty_cache()

    #Normalising the partial covariance matrix and mirroring the upper triangle to the lower triangle
    correlation_matrix = normalise_covariance_matrix(partial_covariance_matrix_upper, device, col_block_size)
    torch.cuda.empty_cache()

    print("")
    return correlation_matrix, movie_encoder
