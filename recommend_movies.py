import pandas as pd
import torch
from scipy.stats import norm
import numpy as np
import time
import logging

def sigmoid_scaling_lookup(rating_frequencies, total_ratings, beta=0.1):
    """
    Precomputes sigmoid scaling factors for ratings (1 to 10) and adjusts them to enforce monotonicity.

    The sigmoid function gives more weight to rarer ratings, but adjustments are made to ensure 
    that the scaling factors increase monotonically with the rating values.

    Parameters
    ----------
    rating_frequencies : pd.Series
        A Series with ratings (1 to 10) as the index and their corresponding frequencies as values.
    total_ratings : int
        The total number of ratings in the dataset, used to calculate the rarity of each rating.
    beta : float, optional
        A parameter controlling the steepness of the sigmoid function. Default is 0.1.

    Returns
    -------
    dict
        A dictionary mapping each rating (1 to 10) to its corresponding scaled value, adjusted for monotonicity.
    """

    
    scaling_factors = {}
    midpoint = 5  
    
    # Step 1: Generate the full set of scaling values
    for rating in range(1, 11):
        if rating in rating_frequencies:
            rarity_factor = total_ratings / rating_frequencies[rating] #Gives a value for rarity with rarer ratings getting a higher score
            scaling_factor = 1 / (1 + np.exp(-beta * rarity_factor)) #Gives a scaling factor based on this score (0-1)
            
            if rating > midpoint: #Positive ratings
                scaling_factors[rating] = (rating - midpoint) * scaling_factor
            elif rating < midpoint: #Negative ratings
                scaling_factors[rating] = -(midpoint - rating) * scaling_factor
            else: #Neutral ratings
                scaling_factors[rating] = 0  
        else:
            scaling_factors[rating] = 0  # Handle missing ratings
    
    # Step 2: Identify and adjust monotonicity violations
    #(Note that testing suggests the scaling rarely violates this but I have left it in as a precaution given it is computationaly simple)
    scaled_values = list(scaling_factors.values())
    for i in range(1, len(scaled_values)):
        if scaled_values[i] < scaled_values[i - 1]:
            # If there's a violation, find the next correct value
            j = i + 1
            while j < len(scaled_values) and scaled_values[j] < scaled_values[j - 1]:
                j += 1

            # Calculate the step size to evenly distribute values between correct points
            correct_start = scaled_values[i - 1]
            correct_end = scaled_values[j] if j < len(scaled_values) else scaled_values[i - 1] + 1
            step = (correct_end - correct_start) / (j - i + 1)
            
            # Adjust all values between i and j to enforce monotonicity
            for k in range(i, j):
                scaled_values[k] = correct_start + step * (k - i + 1)
    
    # Step 3: Update the dictionary with the adjusted values
    for i, rating in enumerate(range(1, 11)):
        scaling_factors[rating] = scaled_values[i]
    
    return scaling_factors


def recommend_movies(user_ratings, corr_matrix, movie_encoder, scaling_factors):
    """
    Generates movie recommendations based on user ratings and the correlation matrix, using sigmoid scaling.

    This function uses a precomputed correlation matrix and the user's past ratings, adjusted by 
    sigmoid scaling, to generate a list of recommended movies. Movies that the user has already 
    rated are excluded from the recommendations.

    Parameters
    ----------
    user_ratings : pd.Series
        A Series with movie titles as the index and the corresponding ratings as the values.
    corr_matrix : torch.Tensor
        The correlation matrix containing similarity scores between movies.
    movie_encoder : LabelEncoder
        An encoder that maps movie titles to numeric indices.
    scaling_factors : dict
        A dictionary mapping each rating to its precomputed scaled value.

    Returns
    -------
    recommended_movies : pd.Series
        A Series with movie titles as the index and their corresponding similarity scores as the values, 
        sorted in descending order.
    """

    start_time = time.time()
    print("Starting recommendation generation...")

    # Initialize lists to store the encoded movie indices and the adjusted ratings based on scaling factors
    user_rated_movie_indices = []
    adjusted_ratings = []

    # Iterate through the user's ratings
    for movie, rating in user_ratings.items():
        # Check if the movie is in the movie encoder's known classes
        if movie in movie_encoder.classes_:
            # Convert the movie title to its encoded numeric index
            encoded_index = movie_encoder.transform([movie])[0]
            user_rated_movie_indices.append(encoded_index)
            # Apply the precomputed scaling factor to the rating and store the adjusted rating
            adjusted_ratings.append(scaling_factors[rating])

    # Convert the adjusted ratings and movie indices into PyTorch tensors
    user_ratings_tensor = torch.tensor(adjusted_ratings, dtype=torch.float32, device=corr_matrix.device)
    user_rated_movie_indices_tensor = torch.tensor(user_rated_movie_indices, dtype=torch.long, device=corr_matrix.device)

    #Extracting only the columns of movies the user has rated.
    #(Note that all of the similarity scores for unrated movies are preserved here in the rows)
    corr_subset = corr_matrix[:, user_rated_movie_indices_tensor]

    print("Computing similarity scores...")
    logging.info("Computing similarity scores...")
    # Compute the similarity scores by performing a matrix multiplication between the correlation subset
    # and the user's adjusted ratings tensor
    # This gives a vector of all movies in the training set with a similarity score calculated as
    # the sum of all the scaled ratings of films each is correlated with multiplied by how correlated they are to each other.
    sim_scores = torch.matmul(corr_subset, user_ratings_tensor)  

    # Convert the computed similarity scores back to a pandas Series with movie titles as the index
    all_movie_indices = range(len(movie_encoder.classes_))  # Generate a range of indices for all movies
    all_movie_titles = movie_encoder.inverse_transform(all_movie_indices)  # Decode the indices back to movie titles
    sim_scores = pd.Series(sim_scores.cpu().numpy(), index=all_movie_titles)  # Convert to a pandas Series
    sim_scores = sim_scores.drop(user_ratings.index, errors='ignore')  # Exclude movies already rated by the user
    recommended_movies = sim_scores.sort_values(ascending=False)  # Sort the similarity scores in descending order

    end_time = time.time()  # Record the end time for tracking performance
    print(f"Recommendation process completed in {(end_time - start_time)/60:.2f} minutes")
    logging.info(f"Recommendation process completed in {(end_time - start_time)/60:.2f} minutes")
    print("")

    return recommended_movies  # Return the sorted Series of recommended movies

def normal_dist_normalise(filtered_sims, user_ratings):
    """
    Normalises similarity scores using the cumulative distribution function (CDF) of the normal distribution.

    This function applies a normal distribution-based normalisation to the similarity scores, 
    centering them around the user's mean rating.

    Parameters
    ----------
    filtered_sims : pd.Series
        A Series containing movie similarity scores.
    user_ratings : pd.Series
        A Series containing the user's ratings used to determine the normalisation parameters.

    Returns
    -------
    normalised_sims : pd.Series
        A Series containing the normalised similarity scores.
    """

    mean_rating = user_ratings.mean()
    std_rating = 2 * user_ratings.std()
    normalised_sims = norm.cdf((filtered_sims - filtered_sims.mean()) / filtered_sims.std()) * std_rating + mean_rating
    return pd.Series(normalised_sims, index=filtered_sims.index)

def min_max_normalise(series, min_value=1, max_value=10):
    """
    Scales a series of scores to a specified range using min-max normalisation.

    This function normalises the values in a Series to a defined range, typically from 1 to 10.

    Parameters
    ----------
    series : pd.Series
        A Series containing the scores to be normalised.
    min_value : int, optional
        The minimum value of the normalised range. Default is 1.
    max_value : int, optional
        The maximum value of the normalised range. Default is 10.

    Returns
    -------
    normalised_series : pd.Series
        A Series containing the scores normalised to the specified range.
    """

    normalised_series = (series - series.min()) / (series.max() - series.min()) * (max_value - min_value) + min_value
    return normalised_series

def generate_recommendations(my_ratings, corr_matrix, movie_encoder):
    """
    Generates movie recommendations by applying sigmoid scaling to the correlation matrix based on user ratings.

    This function processes the user's ratings to generate a list of movie recommendations. It first 
    calculates scaling factors for the ratings, uses these to adjust the similarity scores from the 
    correlation matrix, and then normalises the final recommendation scores.

    Parameters
    ----------
    my_ratings : pd.Series
        A Series with the user's ratings, where the index is the movie titles and the values are the ratings.
    corr_matrix : torch.Tensor
        The correlation matrix containing similarity scores between movies.
    movie_encoder : LabelEncoder
        An encoder that maps movie titles to numeric indices.

    Returns
    -------
    recommendations_df_sorted : pd.DataFrame
        A DataFrame containing the recommended movies with their raw and normalised similarity scores, 
        sorted by the average normalised score.
    normalised_recommendations : pd.Series
        A Series containing the normalised similarity scores for the recommended movies.
    """

    # Calculate the frequency of each rating and total ratings
    rating_frequencies = my_ratings.value_counts().sort_index()
    total_ratings = len(my_ratings)

    # Precompute the scaling factors for each rating
    scaling_factors = sigmoid_scaling_lookup(rating_frequencies, total_ratings)

    # Use the precomputed scaling factors in the recommendation process
    raw_sim_candidates = recommend_movies(my_ratings, corr_matrix, movie_encoder, scaling_factors)
    torch.cuda.empty_cache()

    #Normalise the sim candidates to the same range as ratings to act as predicted scores.
    #Note that the min_max generates a good distribution of scores and the normal distribution is used in an average 
    #to centre these scores a bit to the actual distribution seen in the user data
    normalised_recommendations = normal_dist_normalise(raw_sim_candidates, my_ratings) / 2 
    min_max_normalised_recommendations = min_max_normalise(raw_sim_candidates) / 2
    average_score = (normalised_recommendations + min_max_normalised_recommendations) / 2
    torch.cuda.empty_cache()

    recommendations_df = pd.DataFrame({
        'Movie Title': raw_sim_candidates.index,  
        'Raw sim score': raw_sim_candidates,
        'Normalised Score': normalised_recommendations,
        'Min-Max Normalised Score': min_max_normalised_recommendations,
        'Average Score': average_score
    })

    #Sort recommendations by average normalised scores
    recommendations_df_sorted = recommendations_df.sort_values(by='Average Score', ascending=False)

    return recommendations_df_sorted, normalised_recommendations
