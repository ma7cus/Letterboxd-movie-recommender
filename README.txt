########################################################################################
Movie Recommendation System
########################################################################################

########################################################################################
Project Description
########################################################################################
This project is a movie recommendation system that suggests movies based on user ratings. The system processes large datasets, calculates correlation matrices, and generates personalized movie recommendations using a subset of training data. The workflow involves multiple steps, including data setup, preprocessing, correlation computation, recommendation generation, and postprocessing, all aimed at delivering accurate and relevant movie suggestions.

########################################################################################
Features
########################################################################################
- Comprehensive data setup with logging and device configuration
- Detailed preprocessing of both user and movie data, including handling of known data issues
- Generation of training data subsets for efficient processing
- Calculation of correlation matrices using GPU-accelerated tensor operations
- Personalized recommendation generation with advanced scaling and normalization techniques
- Postprocessing of recommendations to merge with additional statistics, filter, and format the results for presentation

########################################################################################
Installation Instructions
########################################################################################
1) Clone the repository:
	git clone https://github.com/ma7cus/movierecommender.git

2) Navigate to the project directory:
	cd movierecommender

3) Install the required dependencies:
	pip install -r requirements.txt

########################################################################################
Usage Instructions
########################################################################################
1) Ensure your data files (training_data.csv and ratings.csv) are in the project directory.
2) Run the main script:
	python main.py

########################################################################################
Input Data
########################################################################################
- training_data.csv: Contains historical movie ratings data used for training. Extracted from the data gathered from the project: "https://github.com/sdl60660/letterboxd_recommendations/tree/main/data_processing/data"
- ratings.csv: Contains user-specific Letterboxd movie ratings data for generating recommendations. This is the ratings.csv file that you can download from your Letterboxd profile at: "https://letterboxd.com/settings/data/"

########################################################################################
Output Data
########################################################################################
- CSV files with recommended movies for different subsets and iterations, saved in the format recommendations_subset_<size>iter<iteration>_<timestamp>.csv.
- cleaned_ratings.csv: A CSV showing the user’s ratings after applying standard formatting.
- complete_merged_training_data.csv: A CSV of the training data updated with the user’s own ratings.
- preprocessing_names.csv: A CSV showing the pre-processing steps to normalize the movie titles to the same form as the training data.
- updated_training_data.csv: A CSV of the training data provided, updated with the user's own ratings.

########################################################################################
Directory Structure
########################################################################################
.
├── compute_correlation_matrix.py
├── data_subsets.py
├── main.py
├── postprocessing.py
├── preprocessing_and_merging_user_data.py
├── preprocessing_training_data.py
├── recommend_movies.py
├── setup.py
├── user_variables.py
└── requirements.txt

########################################################################################
Script Descriptions
########################################################################################
- setup.py: Initializes the environment, sets up logging, loads data, and configures the computation device (GPU or CPU).

- preprocessing_training_data.py: Cleans the training data by handling known issues such as suffixes, duplicates, and missing values.

- preprocessing_and_merging_user_data.py: Preprocesses the user’s movie ratings, aligns them with the training data, and merges them for use in recommendations.

- compute_correlation_matrix.py: Converts the data into GPU tensors, mean-centers the data, and computes the correlation matrix for the dataset.

- data_subsets.py: Creates subsets of data based on configuration settings for efficient processing.

- recommend_movies.py: Generates movie recommendations based on user preferences, applying advanced scaling and normalisation techniques to refine the results
.
- postprocessing.py: Merges recommendations with additional statistics, filters, formats, and saves the final output.

- user_variables.py: Stores user-specific variables and configuration settings.

- main.py: The main script that orchestrates the entire workflow, coordinating all stages from data setup to final recommendations.

########################################################################################
Algorithm Description
########################################################################################
This movie recommendation system employs a collaborative filtering approach, specifically using Pearson's correlation coefficient to generate personalized movie recommendations. The workflow is as follows:

- Data Preparation:
	A training dataset is created by merging the historical movie ratings data (training_data.csv) with the user's specific ratings (ratings.csv). Each entry in this dataset represents a user's rating for a specific movie.

- Correlation Matrix Calculation:
	The system calculates a correlation matrix for the movies using Pearson’s correlation coefficient. This matrix quantifies the relationship between any two movies based on how similarly users have rated them in the past.

-Similarity Score Computation:
	The correlation scores are then used to compute a similarity score for each movie relative to the user's preferences. This is done by multiplying the correlation scores by the user's own ratings, which effectively predicts how 	much the user is likely to enjoy a given movie. Higher similarity scores indicate a stronger recommendation.

-Score Scaling and Normalization:
	The similarity scores are scaled to highlight extreme values, emphasizing movies that are strongly similar to those the user rated highly or poorly. These scores are then normalized to match the range of typical user ratings, 	providing an approximate estimate of the rating the user might give to a movie they haven't seen before.

-Recommendation Filtering:
	The final set of recommendations is saved to a CSV file, along with a filtered version that removes a specified percentage of the most reviewed films. This filtering step aims to exclude obvious choices that the user is likely 	already aware of, allowing for more unique or less mainstream recommendations to be highlighted.

########################################################################################
Contributing
########################################################################################
If you would like to contribute to this project, please fork the repository and submit a pull request.

########################################################################################
License
########################################################################################
This project is licensed under the MIT License.

########################################################################################
Contact Information
########################################################################################
For questions or suggestions, please contact Marcus at marcusb196@outlook.com.