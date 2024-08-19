import torch
import logging
import pandas as pd


def setup_logging():
    """
    Configures the logging system for the script.

    This function sets up logging to a file named 'logging_information.log' with a specific format 
    that includes the timestamp, log level, and message. It also logs the start of the script.

    """
    logging.basicConfig(filename='logging_information.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info('Script started')
    print('Script started:')

def setup_device():
    """
    Sets up the device for computation, preferring GPU if available.

    This function checks if a CUDA-compatible GPU is available and sets the computation device 
    accordingly. If not, it defaults to the CPU.

    Returns
    -------
    torch.device
        The device to use for computation.
    
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    print(f"Using device: {device}")
    torch.cuda.empty_cache()
    return device


def load_data(training_data_path, user_data_path):
    """
    Loads the training and user data from CSV files.

    This function reads data from two specified CSV files: one containing training data 
    and the other containing user data, and returns them as Pandas DataFrames.

    Parameters
    ----------
    training_data_path : str
        The file path to the CSV file containing the training data.
        
    user_data_path : str
        The file path to the CSV file containing the user data.

    Returns
    -------
    DataFrame, DataFrame
        The training data and user data as Pandas DataFrames.
    """
    logging.info('Loading training data from CSV file')
    training_data = pd.read_csv(training_data_path)
    
    logging.info('Loading user data from CSV file')
    my_data = pd.read_csv(user_data_path)

    return training_data, my_data


def setup(training_data_path, user_data_path):
    """
    Runs the setup steps for the script.

    This function initializes logging, sets up the computation device, and loads the 
    training and user data from the specified CSV file paths.

    Parameters
    ----------
    training_data_path : str
        The file path to the CSV file containing the training data.
        
    user_data_path : str
        The file path to the CSV file containing the user data.

    Returns
    -------
    torch.device, DataFrame, DataFrame
        The device for computation, the training data, and the user data as Pandas DataFrames.
    """
    setup_logging()
    device = setup_device()
    training_data, my_data = load_data(training_data_path, user_data_path)
    print("")
    logging.info("Setup completed successfully")
    return device, training_data, my_data


if __name__ == '__main__':
    setup()
