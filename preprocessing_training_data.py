import pandas as pd
import os
import logging

def handle_suffix_problem(training_data):
    """
    Handles the '-1' suffix problem in movie titles within the training data.

    This function identifies movie titles that incorrectly contain the '-1' suffix, removes the suffix 
    unless it's part of an exception, and merges the cleaned titles back into the original data. 
    It then removes any duplicates created by the suffix correction and ensures the rating values are numeric.

    Parameters
    ----------
    training_data : DataFrame
        A DataFrame containing user ratings with columns 'user_id', 'movie_id', and 'rating_val'.

    Returns
    -------
    DataFrame
        A DataFrame with cleaned movie titles and numeric ratings, retaining only necessary columns ('user_id', 'movie_id', 'rating_val').

    """
    logging.info('Handling suffix problem...')

    # Define exceptions where a '-1' suffix is actually part of the title
    exceptions = [
        'part-1', 'vol-1', 'pt-1', 'chapter-1', 'volume-1', 'episode-1', 'no-1', 'parte-1', 'day-1', 'film-1', 
        'night-1', 'number-1', 'periode-1', 'enemy-1'
    ]
    
    # Identify entries with '-1' suffix, excluding specified exceptions
    suffix_mask = training_data['movie_id'].str.endswith('-1') # This is a boolean mask of rows where the suffix is '-1'
    exception_mask = training_data['movie_id'].apply(lambda x: any(exc in x for exc in exceptions)) # This is a boolean mask of rows where each row is checked to see if it contains any of the exceptions in the set
    training_data_with_suffixes = training_data[suffix_mask & ~exception_mask].copy() # This extracts the rows in the dataframe which contain the suffix and not the exceptions

    logging.info(f"Identified {len(training_data_with_suffixes)} entries with '-1' suffix not in exceptions.")
    
    # Remove the '-1' suffix from relevant rows and adds it to a new column for comparison
    training_data_with_suffixes['cleaned_movie_id'] = training_data_with_suffixes['movie_id'].str.replace(r'-1$', '', regex=True)

    # Set had_suffix flag so we can see which ones were corrected clearly
    training_data_with_suffixes['had_suffix'] = True

    # Merge cleaned IDs back to the original dataframe (i.e. match user_id and movie_id rows with their corresponding correction in the 'cleaned_movie_id' column)
    training_data_merged = pd.merge(training_data, training_data_with_suffixes[['user_id', 'movie_id', 'cleaned_movie_id','had_suffix']], 
                         on=['user_id', 'movie_id'], how='left')

    # If cleaned_movie_id is NaN (i.e. row didn't need correcting), set it to the same as movie_id column 
    training_data_merged['cleaned_movie_id'] = training_data_merged['cleaned_movie_id'].fillna(training_data_merged['movie_id'])
    training_data_merged['had_suffix'] = training_data_merged['had_suffix'].fillna(False)

    # Dropping resultant duplicates where a user had entries for 'movie' and 'movie-1'
    training_data_merged.sort_values(by=['user_id', 'cleaned_movie_id', 'had_suffix'], ascending=[True, True, False], inplace=True) # Sort to prioritize entries with the '-1' suffix corrected   
    training_data_suffixes_corrected = training_data_merged.drop_duplicates(subset=['user_id', 'cleaned_movie_id'], keep='first') # Drop duplicates while keeping the first occurrence in the sorted DataFrame

    # Number of changes made
    num_changes = len(training_data_with_suffixes)
    print(f"Number of '-1' suffixes handled: {num_changes}")
    logging.info(f"Number of '-1' suffixes handled: {num_changes}")

    # Ensure 'rating_val' is numeric
    training_data_suffixes_corrected.loc[:, 'rating_val'] = pd.to_numeric(training_data_suffixes_corrected['rating_val'], errors='coerce')

    # Return the modified dataframe with only the necessary columns, correctly labeled
    return training_data_suffixes_corrected[['user_id', 'cleaned_movie_id', 'rating_val']].rename(columns={'cleaned_movie_id': 'movie_id'})

def correct_mislabeled_films(training_data):
    """
    Corrects known mislabeled films in the training data based on a predefined list.

    This function replaces incorrect movie titles with the correct ones according to a dictionary 
    of known corrections, ensuring consistency in the dataset.

    Parameters
    ----------
    training_data : DataFrame
        A DataFrame containing user ratings with columns 'user_id', 'movie_id', and 'rating_val'.

    Returns
    -------
    DataFrame
        A DataFrame with corrected movie titles.    
    """

    logging.info('Correcting mislabeled films...')

    corrections = {
        'the-meaning-of-life-1983': 'monty-pythons-the-meaning-of-life',
        'sherlock-a-scandal-in-belgravia-2012':'sherlock-a-scandal-in-belgravia',
        'sherlock-a-study-in-pink-2010':'sherlock-a-study-in-pink',
        'glass-onion-a-knives-out-mystery':'glass-onion'
    }

    for incorrect_title, correct_title in corrections.items():
        training_data.loc[training_data['movie_id'] == incorrect_title, 'movie_id'] = correct_title
    
    print("Mislabeled films have been corrected.")
    logging.info("Mislabeled films have been corrected.")

    return training_data

def clean_training_data(training_data):
    """
    Cleans the training data by handling the '-1' suffix problem and correcting mislabeled films.

    This function either reads in pre-cleaned data from a CSV file or performs data cleaning by 
    handling incorrect '-1' suffixes in movie titles and correcting known mislabeled films. 
    It then ensures all rating values are numeric and saves the cleaned data to a CSV file.

    Parameters
    ----------
    training_data : DataFrame
        A DataFrame containing user ratings with columns 'user_id', 'movie_id', and 'rating_val'.

    Returns
    -------
    DataFrame
        The cleaned training data as a DataFrame.
    ndarray
        An array of unique movie titles from the cleaned data.
    """    
    output_file = 'cleaned_training_data.csv'
    
    if os.path.exists(output_file): # Checks if you've already generated a cleaned version of the training data
        print(f"Reading in existing cleaned data from {output_file}")
        logging.info(f"Reading in existing cleaned data from {output_file}")

        corrected_training_data = pd.read_csv(output_file, dtype=str)

        # Ensure 'rating_val' is numeric after reading the existing cleaned data
        corrected_training_data['rating_val'] = pd.to_numeric(corrected_training_data['rating_val'], errors='coerce')
    else:
        print("Cleaning training data of known issues:")
        logging.info("Cleaning training data of known issues:")
                
        # Handling suffix issue
        training_data_with_suffix_handled = handle_suffix_problem(training_data)
        
        # Correcting known mislabellings
        corrected_training_data = correct_mislabeled_films(training_data_with_suffix_handled)

        # Ensure 'rating_val' is numeric before writing to CSV
        corrected_training_data['rating_val'] = pd.to_numeric(corrected_training_data['rating_val'], errors='coerce')

        # Writing to csv    
        corrected_training_data[['user_id', 'movie_id', 'rating_val']].to_csv(output_file, index=False, quoting=1)
        print(f"Resolved training data has been saved to {output_file}.")
        logging.info(f"Resolved training data has been saved to {output_file}.")

    #Pulling correct names
    their_names = corrected_training_data['movie_id'].unique()

    return corrected_training_data,their_names


if __name__ == "__main__":
    clean_training_data()
