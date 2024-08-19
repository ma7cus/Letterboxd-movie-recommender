#Files to read in:
training_data_path = 'personalised_training_data.csv'
user_data_path = 'ratings.csv'

#User ID to append ratings into the training data
my_user_id = 'ma7cus'

# Define subset sizes and number of iterations
subset_sizes = {
    100: 1,  
    50: 0,   
    2.5: 0   
}

# Minimum number of reviews for a movie to be considered
min_reviews = 10

# Desired top n-th percentile for filtering
upper_percentile = 80
