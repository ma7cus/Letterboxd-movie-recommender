import re
import unicodedata
import pandas as pd
import time
import logging
import os

def replace_exceptions(text):
    """
    Applies specific replacements to handle known special cases in movie titles.

    This function replaces certain problematic characters or sequences in movie titles with more 
    standard equivalents to ensure consistent formatting.

    Parameters
    ----------
    text : str
        The original movie title text.

    Returns
    -------
    str
        The text with specific replacements applied.
    """

    # Handling specific known exceptions
    text = text.replace('½', ' ')
    text = text.replace('⅓', ' ')
    text = text.replace('...', ' ')  # As in tick, tick...BOOM!
    text = text.replace('/', ' ')  # As in Face/Off
    text = text.replace('³', '-3')  # As in Alien³
    
    return text

def reformat_name(name):
    """
    Reformats a movie name by applying special case replacements, converting to lower case, 
    removing punctuation, and replacing spaces with hyphens.

    Parameters
    ----------
    name : str
        The original movie name.

    Returns
    -------
    str
        The reformatted movie name.
    """

    name = replace_exceptions(name)
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = name.lower()  # Lower case all words
    name = re.sub(r'[^a-z0-9\s-]', '', name)  # Remove anything that isn't latin letters or numbers or hyphens
    name = re.sub(r'\s+', ' ', name)  # Remove any multiple spaces
    name = name.replace(" ", "-")  # Replace spaces with hyphens
    return name

def add_dates_to_name(my_names_and_years, name_with_year, current_new_names_vector):
    """
    Updates movie names by appending the year if a match is found between the provided name and year 
    in the user's data and the existing dataset.

    This function checks if a movie name ends with a year (e.g., "Title 1999") and, if the year matches 
    an entry in the user's dataset, updates the name in the provided vector.

    Parameters
    ----------
    my_names_and_years : DataFrame
        A DataFrame containing movie titles and their corresponding years with columns 'Name' and 'Year'.
    name_with_year : str
        A movie title string that potentially has a year appended to it.
    current_new_names_vector : Series
        Series containing the current state of the updated movie names.

    Returns
    -------
    Series
        The Series with updated movie names where applicable.
    """

    # Check if the title has 4 numbers at the end (i.e., a date)
    if len(name_with_year) > 4 and name_with_year[-4:].isdigit():
        name_name = name_with_year[:-5].strip()  # Extracts the title without the date
        name_year = int(name_with_year[-4:])  # Extracts the date

        # Find all matches for the extracted name
        matching_rows = my_names_and_years.loc[my_names_and_years['Name'] == name_name]

        # Iterate through all matching rows
        for idx, row in matching_rows.iterrows():
            corresponding_year = int(row['Year'])  # Extract the year in the row where the titles match

            # Check if the title year matches the year in the data
            if abs(name_year - corresponding_year) == 0:
                current_new_names_vector.loc[idx] = name_with_year  # Update the name in the cumulative vector

    return current_new_names_vector

def compare_names(my_names, their_names):
    """
    Compares the user's movie names with the training data's movie names and prints any differences.

    This function identifies and lists any discrepancies between the movies a user has rated and 
    those present in the training dataset, helping to ensure data consistency.

    Parameters
    ----------
    my_names : Series
        The Series containing the names of movies rated by the user.
    their_names : ndarray
        The array containing the names of movies in the training data.

    Returns
    -------
    None
        This function prints the differences and does not return a value.
    """

    # Convert my_names and their_names to sets
    set_my_new_names = set(my_names)
    set_their_names = set(their_names)

    # Find the difference
    films_not_in_their_names = set_my_new_names - set_their_names

    # Print the list of films
    if len(films_not_in_their_names) >= 1:
        print("The following films are in your set but not the large data set:")
        for film in films_not_in_their_names:
            print(film)

def apply_standard_formatting(my_data, their_names):
    """
    Applies standard formatting to the movie names in the user's data and compares them with the training data's names.

    This function reformats movie names by removing punctuation, normalizing case, adding dates where 
    appropriate, and saving the progress to a CSV file. It also compares the reformatted names with the 
    training data and rescales user ratings.

    Parameters
    ----------
    my_data : DataFrame
        The DataFrame containing the user's movie ratings.
    their_names : ndarray
        The array containing the names of movies in the training data.

    Returns
    -------
    DataFrame
        The user data with formatted movie names and rescaled ratings.
    """

    # Step 1: Reformat Names
    print("Reformatting names...")
    logging.info("Reformatting names...")
    
    my_names = my_data['Name']
    my_new_names_1 = my_names.apply(reformat_name)
    
    my_data['Name'] = my_new_names_1

    # Step 2: Add Dates
    print("Adding dates to names...")
    logging.info("Adding dates to names...")
    my_new_names_2 = my_new_names_1.copy()  # Ensure this is a separate copy
    start_time = time.time()
    for name in their_names:
        my_new_names_2 = add_dates_to_name(my_data[['Name', 'Year']], name, my_new_names_2)
    print(f"Added dates to names in {time.time() - start_time:.2f} seconds.")
    logging.info(f"Added dates to names in {time.time() - start_time:.2f} seconds.")

    # Assign the updated names back to the DataFrame
    my_data['Name'] = my_new_names_2

    # Step 3: Save Progress
    print("Saving progress to CSV...")
    logging.info("Saving progress to CSV...")
    my_progress = pd.DataFrame({'Original_name': my_names, 'Name_no_punctuation': my_new_names_1, 'Name_with_date': my_new_names_2})
    my_progress.to_csv('preprocessing_names.csv', index=False)

    # Step 4: Compare Names
    print("")
    print("Comparing names...")
    logging.info("Comparing names...")
    compare_names(my_new_names_2, their_names)

    # Step 5: Rescale Ratings
    print("")
    print("Rescaling ratings...")
    logging.info("Rescaling ratings...")
    my_data['Rating'] = 2 * my_data['Rating']
    print("")

    return my_data

def add_user_data_to_training_data(my_data, training_data, user_id):
    """
    Adds the user's formatted data to the training data table and saves the combined data to a CSV file.

    This function combines the user's formatted movie data with the existing training data and 
    saves the result to 'complete_merged_training_data.csv'.

    Parameters
    ----------
    my_data : DataFrame
        The DataFrame containing the user's formatted movie names and ratings.
    training_data : DataFrame
        The DataFrame containing the existing training data with user ratings.
    user_id : str
        The user ID to be added to the user data.

    Returns
    -------
    DataFrame
        The combined DataFrame with user data added to the training data.
    """

    logging.info("Adding user data to training data.")
    my_reformatted_data = pd.DataFrame({'user_id': user_id, 'movie_id': my_data['Name'], 'rating_val': my_data['Rating']})
    combined_data = pd.concat([training_data, my_reformatted_data], ignore_index=True)
    combined_data.to_csv('complete_merged_training_data.csv', index=False)
    logging.info("User data added to training data and saved to updated_training_data.csv.")
    return combined_data

def drop_duplicates(training_data):
    """
    Removes duplicate rows from the training data based on user_id and movie_id.

    This function identifies and removes duplicate entries in the training data where the 
    combination of 'user_id' and 'movie_id' is repeated, and logs the number of duplicates removed.

    Parameters
    ----------
    training_data : DataFrame
        The DataFrame containing the training data with user ratings.

    Returns
    -------
    DataFrame
        The DataFrame with duplicates removed.
    """

    duplicate_count = training_data.duplicated(subset=['user_id', 'movie_id']).sum()
    logging.info(f"Number of duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        training_data = training_data.drop_duplicates(subset=['user_id', 'movie_id'])
        logging.info(f"{duplicate_count} duplicate rows removed.")
    return training_data

def preprocess_data(my_data, training_data, their_names, user_id):
    """
    Preprocesses the user data, integrates it with the training data, and handles duplicates and rating rescaling.

    This function applies standard formatting to the user's movie data, adds the user data to the 
    training data, rescales the ratings, and removes duplicates. It also checks if a preprocessed 
    file exists and uses it if available.

    Parameters
    ----------
    my_data : DataFrame
        The DataFrame containing the user's movie ratings.
    training_data : DataFrame
        The DataFrame containing the existing training data with user ratings.
    their_names : ndarray
        The array containing the names of movies in the training data.
    user_id : str
        The user ID to be added to the user data.

    Returns
    -------
    DataFrame
        The combined DataFrame with user data added to the training data.
    DataFrame
        The preprocessed user data.
    """

    # Check if a combined data file already exists
    output_file = 'complete_merged_training_data.csv'
    if os.path.exists(output_file):
        """
        logging.info(f"Reading in existing combined data from {output_file}")
        combined_data = pd.read_csv(output_file)
        """
        my_data = pd.read_csv('cleaned_ratings.csv')
        combined_data = pd.read_csv('complete_merged_training_data.csv')
    else:
        logging.info("Starting apply_standard_formatting:")
        my_data = apply_standard_formatting(my_data, their_names)
        my_data.to_csv("cleaned_ratings.csv", index=False)
        logging.info("Completed apply_standard_formatting.")
        
        logging.info("Starting adding user data to training data:")
        combined_data = add_user_data_to_training_data(my_data, training_data, user_id)
        logging.info("Completed adding user data to training data.")
        
        combined_data = drop_duplicates(combined_data)
    
    # Ensure combined data includes the training data properly
    total_entries = combined_data.shape[0]
    user_entries = combined_data[combined_data['user_id'] == user_id].shape[0]
    logging.info(f"Number of entries in training data: {total_entries}")
    logging.info(f"Number of entries for user_id '{user_id}': {user_entries}")

    return combined_data,my_data

