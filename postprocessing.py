import logging
from datetime import datetime

def find_stats(training_data):
    """
    Calculates the average rating and number of reviews for each movie in the training data.

    This function groups the training data by movie IDs and computes the average rating and 
    the number of reviews for each movie. The average rating is then scaled down by dividing it by 2.

    Parameters
    ----------
    training_data : DataFrame
        A DataFrame containing user ratings with columns such as 'user_id', 'movie_id', and 'rating_val'.

    Returns
    -------
    DataFrame
        A DataFrame containing columns for 'movie_id', 'average_rating', and 'num_reviews' 
    """

    
    # Group the data by movie_id and calculate the average rating and number of reviews
    film_stats = training_data.groupby('movie_id').agg(
        average_rating=('rating_val', 'mean'),
        num_reviews=('user_id', 'count')
    )
    
    # Reset the index to make 'movie_id' a column again
    film_stats = film_stats.reset_index()

    # Scale down the average rating by dividing it by 2
    film_stats['average_rating'] = film_stats['average_rating'] / 2
    
    # Return the resulting DataFrame
    return film_stats


def merge_recommendations_with_stats(recommendations_df, training_data):
    """
    Merges movie recommendations with their corresponding statistics from the training data.

    This function merges a DataFrame of movie recommendations with a DataFrame of movie 
    statistics (average ratings and number of reviews) calculated from the training data.

    Parameters
    ----------
    recommendations_df : DataFrame
        A DataFrame containing movie recommendations with at least a 'Movie Title' column.
    training_data : DataFrame
        A DataFrame containing user ratings with columns such as 'user_id', 'movie_id', and 'rating_val'.

    Returns
    -------
    DataFrame
        A merged DataFrame containing movie recommendations alongside their average ratings 
        and the number of reviews from the training data.
    """

    training_film_stats = find_stats(training_data)
    merged_data = recommendations_df.merge(training_film_stats, left_on='Movie Title', right_on='movie_id', how='left')
    merged_data.drop(columns=['movie_id'], inplace=True)
    return merged_data

def format_columns(df):
    """
    Formats and renames columns in the DataFrame to the desired structure.

    This function rounds specific columns to two decimal places, renames columns 
    for clarity, and reorders the columns to a specific sequence.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing movie recommendations and their associated statistics.

    Returns
    -------
    DataFrame
        A DataFrame with formatted and renamed columns, reordered according to the specified structure.
    """

    
    # List of columns to format (round to 2 decimal places)
    columns_to_format = [
        'Normalised Score', 
        'Average Score', 
        'average_rating', 
        'Min-Max Normalised Score'
    ]
    df[columns_to_format] = df[columns_to_format].astype(float).round(2)

    # Dictionary for renaming columns
    renamed_columns = {
        'Normalised Score': 'Recommender rating (normalised)',
        'Average Score': 'Recommender rating',
        'average_rating': 'Training data average rating',
        'Min-Max Normalised Score': 'Recommender rating (min/max)',
        'num_reviews': 'Training data number of ratings'
    }
    df.rename(columns=renamed_columns, inplace=True)

    # Reorder the columns
    reordered_columns = [
        'Movie Title',
        'Recommender rating',
        'Training data average rating',
        'Raw sim score',
        'Training data number of ratings'
    ]
    
    # Return the DataFrame with columns reordered
    return df[reordered_columns]


def filter_upper_percentile(df, min_reviews, upper_percentile):
    """
    Filters the DataFrame to include only movies with a minimum number of reviews 
    and within the specified upper percentile of review counts.

    This function filters the DataFrame to retain only the rows where the number of reviews 
    meets or exceeds the minimum threshold and falls within the specified upper percentile.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing movie recommendations and their associated statistics.
    min_reviews : int
        The minimum number of reviews a movie must have to be included in the filtered DataFrame.
    upper_percentile : int
        The upper percentile of the number of reviews used as a threshold for filtering.

    Returns
    -------
    DataFrame
        A filtered DataFrame containing only the movies that meet the review count criteria.
    """



    df = df[df['Training data number of ratings'] >= min_reviews]

    upper_value = df['Training data number of ratings'].quantile(upper_percentile/100)

    df_filtered = df[(df['Training data number of ratings'] <= upper_value)]

    return df_filtered.reset_index(drop=True)

def save_to_csv(df, size, iteration,filtered):
    """
    Saves the DataFrame to a CSV file with a timestamped filename.

    This function saves the DataFrame to a CSV file. The filename includes the subset size, 
    iteration number, and a timestamp to ensure uniqueness. If the DataFrame is filtered, 
    this is reflected in the filename.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to save.
    size : float
        The subset size used in the recommendation process.
    iteration : int
        The iteration number in the recommendation process.
    filtered : bool
        A flag indicating whether the DataFrame has been filtered (affects the filename).

    Returns
    -------
    str
        The file path of the saved CSV file.
    """

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not filtered:
        file_path = f'recommendations_subset_{size}_iter_{iteration + 1}_{timestamp}.csv'
    else:
        file_path = f'filtered_recommendations_subset_{size}_iter_{iteration + 1}_{timestamp}.csv'

    df.to_csv(file_path, index=False)
    return file_path

def postprocess_data(recommendations_df, training_data, size, iteration, min_reviews=10, upper_percentile=80,to_csv=1):
    """
    Postprocesses the recommendation data to generate and optionally save the final recommendations.

    This function merges the recommendation data with training data statistics, formats 
    the resulting DataFrame, and optionally filters the recommendations based on the number 
    of reviews. The results are then saved to CSV files if specified.

    Parameters
    ----------
    recommendations_df : DataFrame
        A DataFrame containing movie recommendations.
    training_data : DataFrame
        The training data containing user ratings, with columns such as 'user_id', 'movie_id', and 'rating_val'.
    size : float
        The subset size used in the recommendation process.
    iteration : int
        The iteration number in the recommendation process.
    min_reviews : int, optional
        The minimum number of reviews required to include a movie in the final recommendations. Default is 10.
    upper_percentile : int, optional
        The upper percentile of the number of reviews used as a threshold for filtering. Default is 80.
    to_csv : bool, optional
        Whether to save the results to CSV files. Default is True.

    Returns
    -------
    DataFrame
        The final full recommendations DataFrame, which may have been filtered based on review counts.
    """


    #####################################################################################
    #Formatting full recommendations
    #####################################################################################

    # Merging recommendations with training data statistics
    merged_data = merge_recommendations_with_stats(recommendations_df, training_data)

    # Formatting columns
    full_recommendations = format_columns(merged_data)

    # Printing full recommendations to csv
    if to_csv:
        save_to_csv(full_recommendations,size,iteration,0)

    #####################################################################################
    #Formatting filtered recommendations
    #####################################################################################

    # Filtering for central percentile
    filtered_recommendations = filter_upper_percentile(full_recommendations, min_reviews, upper_percentile)

    # Printing filtered recommendations to csv
    if to_csv:
        save_to_csv(filtered_recommendations,size,iteration,1)

    #####################################################################################
    #Printing results
    #####################################################################################
    # Creating a dictionary for the top 5 recommendations
    top_5_recommendations = {row['Movie Title']: row['Recommender rating'] for index, row in full_recommendations.head(5).iterrows()}

    logging.info(f"Top 5 recommendations for subset size {size}% (Iteration {iteration + 1}): {top_5_recommendations}")
    print(f"Top 5 recommendations for subset size {size}% (Iteration {iteration + 1}): {top_5_recommendations}")
    print("")
    
    return full_recommendations
