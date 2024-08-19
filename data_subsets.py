import numpy as np
import pandas as pd
import time
import logging

def partition_and_sample(their_names, my_names, percentage, random_seed=None):
    """
    Partitions the movie names by excluding user-rated movies and then samples a percentage of the remaining movies.

    This function first removes the movies rated by the user from the larger set of movie names. 
    It then takes a random sample from the remaining movies up to the specified percentage of the 
    total set, combining this sample with the user-rated movies.

    Parameters
    ----------
    their_names : Series or list
        The names of movies in the larger set (e.g., all movies in the dataset).
    my_names : Series or list
        The names of movies rated by the user.
    percentage : float
        The percentage of the total 'their_names' to include in the sample. If set to 100, no sampling is done.
    random_seed : int, optional
        The random seed for reproducibility of the sampling process. Default is None.

    Returns
    -------
    Series
        A Series containing the combined set of user-rated movies and the sampled movies from the larger set.
    """

    # Ensure objects are pandas Series
    their_names = pd.Series(their_names)
    my_names = pd.Series(my_names)

    if percentage == 100:
        combined_names = their_names
    else:
        if random_seed is not None:
            np.random.seed(random_seed)

        # Partition 'their_names' by excluding 'my_names'
        partitioned_names = their_names[~their_names.isin(my_names)]
        logging.info(f"Partitioned names count: {len(partitioned_names)}")

        # Determine the number of movies to sample
        sample_size = int((percentage / 100) * len(their_names)) - len(my_names)
        if sample_size < len(my_names):
            sample_size = len(my_names)
        
        # Sample from the partitioned names
        sampled_names = partitioned_names.sample(sample_size, random_state=random_seed)
        logging.info(f"Sampled names count: {len(sampled_names)}")

        # Combine the sampled names with 'my_names'
        combined_names = pd.concat([sampled_names, my_names]).drop_duplicates()
        logging.info(f"Combined names count: {len(combined_names)}")

    return combined_names

def extract_training_data_subset(training_data, combined_names):
    """
    Extracts a subset of the training data that includes only the specified movie titles.

    This function filters the training data to retain only the rows corresponding to the 
    movies in the provided list of combined movie names.

    Parameters
    ----------
    training_data : DataFrame
        The full training data containing user ratings, with columns such as 'user_id', 'movie_id', and 'rating_val'.
    combined_names : Series
        A Series containing the combined set of user-rated movies and sampled movies.

    Returns
    -------
    DataFrame
        A DataFrame containing the subset of the training data that includes only the specified movie titles.
    """

    logging.info('Extracting training data subset...')

    # Filter the training data to include only the rows corresponding to the combined movie IDs.
    training_data_subset = training_data[training_data['movie_id'].isin(combined_names)]
    logging.info(f'Filtered training data to include {len(training_data_subset)} rows corresponding to combined movie IDs.')

    # Calculate the number of unique movies and unique users in the subset.
    unique_movies_in_subset = training_data_subset['movie_id'].nunique()
    unique_users_in_subset = training_data_subset['user_id'].nunique()
    
    # Logging/print information
    total_num_users_training = training_data['user_id'].nunique()
    total_num_movies_training = training_data['movie_id'].nunique()
    
    logging.info(f"Total number of unique movies in the training data subset: {unique_movies_in_subset}")
    logging.info(f"Total number of unique users in the training data subset: {unique_users_in_subset}")
    print(f"Total number of unique movies in the training data subset: {unique_movies_in_subset}/{total_num_movies_training}")
    print(f"Total number of unique users in the training data subset: {unique_users_in_subset}/{total_num_users_training}")
    print("")

    return training_data_subset

def generate_training_data_subset(training_data, my_data_names, their_names, percentage, iteration, count):
    """
    Generates a subset of the training data using random sampling, iteratively adjusting the sample size.

    This function partitions and samples the larger set of movie names, excluding those rated by the user, 
    and then extracts the corresponding subset of training data. It is designed to be used in an iterative 
    process to generate multiple training subsets with different random samples.

    Parameters
    ----------
    training_data : DataFrame
        The full training data containing user ratings, with columns such as 'user_id', 'movie_id', and 'rating_val'.
    my_data_names : Series
        A Series containing the names of movies rated by the user.
    their_names : Series
        A Series containing the names of movies in the larger set (e.g., all movies in the dataset).
    percentage : float
        The percentage of the total 'their_names' to include in the sample. This determines the size of the sample.
    iteration : int
        The current iteration number, used to generate a unique random seed for reproducibility.
    count : int
        The total number of iterations to run. This is used for logging purposes to indicate progress.

    Returns
    -------
    DataFrame
        A DataFrame containing the subset of the training data corresponding to the sampled and user-rated movies.
    """

    logging.info(f"Processing {percentage}% subset of the training data (Iteration {iteration + 1} of {count})")
    print(f"Processing {percentage}% subset of the training data (Iteration {iteration + 1} of {count})")
    
    # Generate a random seed for reproducibility
    random_seed = int(time.time() + iteration + percentage)
    
    # Partition and sample film names
    combined_names = partition_and_sample(their_names, my_data_names, percentage, random_seed)
    
    # Extract training data subset
    training_data_subset = extract_training_data_subset(training_data, combined_names)
    
    return training_data_subset
