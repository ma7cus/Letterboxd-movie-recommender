import time
import torch

from setup import setup
from preprocessing_training_data import clean_training_data
from preprocessing_and_merging_user_data import preprocess_data
from compute_correlation_matrix import compute_correlation_matrix
from recommend_movies import generate_recommendations
from data_subsets import generate_training_data_subset
from postprocessing import postprocess_data
from user_variables import training_data_path, user_data_path, my_user_id, subset_sizes, min_reviews, upper_percentile

def main():
    total_time_start = time.time()

    ##############################################################################################################
    #Reading in training data and user data
    ##############################################################################################################
    device, training_data, my_data = setup(training_data_path, user_data_path)

    ##############################################################################################################
    #Preprocessing if not already completed.
    ##############################################################################################################

    #Preprocessing training data by correcting known labelling issues in this set:
    training_data,their_names = clean_training_data(training_data)

    # Preprocessing my movie titles and adding them to the dataset
    training_data,my_data = preprocess_data(my_data, training_data, their_names, my_user_id)

    # Get useful sets and data
    my_names = my_data['Name']
    my_ratings = my_data.set_index('Name')['Rating']
    

    ##############################################################################################################
    # Iterate over each subset size and iteration count
    ##############################################################################################################
    all_recommendations = []

    # Iterate over each subset size and the specified number of iterations for that size
    for size, count in subset_sizes.items():
        torch.cuda.empty_cache()
        for iteration in range(count):
            torch.cuda.empty_cache()

            ##############################################################################################################
            #Generating the current subset
            ##############################################################################################################
            training_data_subset = generate_training_data_subset(training_data, my_names, their_names, size, iteration, count)
            
            ##############################################################################################################
            # Computing correlation matrix
            ##############################################################################################################
            correlation_matrix, movie_encoder = compute_correlation_matrix(training_data_subset, device)

            ##############################################################################################################
            # Generating and normalising recommendations
            ##############################################################################################################
            recommendations_df_sorted, normalised_recommendations = generate_recommendations(my_ratings, correlation_matrix, movie_encoder)

            # Add the normalised recommendations to the list
            all_recommendations.append(normalised_recommendations)
            torch.cuda.empty_cache()

            ##############################################################################################################
            # Postprocessing/printing to csv's
            ##############################################################################################################
            df = postprocess_data(recommendations_df_sorted, training_data, size, iteration, min_reviews, upper_percentile,1)

    total_time_end = time.time()
    print(f"Full script (including data loading, preprocessing, and postprocessing) ran in: {(total_time_end - total_time_start)/60:.2f} minutes")

if __name__ == "__main__":
    main()
