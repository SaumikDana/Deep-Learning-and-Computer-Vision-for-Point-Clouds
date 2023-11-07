"""
Example script for training a tracker using PointNet-LK without noise.

This script demonstrates how to train a model for point cloud registration using the PointNet-LK algorithm.
"""
__author__ = "Saumik"
__date__ = "11/03/2023"

import os
import sys
import logging

# Add the parent directory to the system path so that we can import modules from there.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# Import custom modules required for the script.
from Settings_pointnetlk import Settings
from Action_pointnetlk import Action
from Trainer_pointnetlk import Trainer
from Utils_pointnetlk import *

# Set up a logger for logging information during execution.
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def main(args):
    """
    Main function to set up datasets, action, and start the training process.
    
    :param args: Command line arguments.
    """
    # Prepare the training and testing datasets.
    trainset, testset = get_datasets(args)

    # Initialize the Action object with the command line arguments.
    action = Action(args)

    """
    Set up the training environment and start the training and evaluation process.
    
    :param args: Command line arguments.
    :param trainset: The training dataset.
    :param testset: The testing dataset.
    :param action: The Action object defining training and evaluation procedures.
    """
    # Initialize the Trainer object with the provided datasets, action, and command line arguments.
    trainer = Trainer(args, trainset, testset, action)
    
    # Set up the device (CPU or GPU) for training.
    trainer.setup_device()
    
    # Initialize the model and optimizer.
    trainer.setup_model_and_optimizer()
        
    # Start the training and evaluation process.
    trainer.train_and_evaluate()


if __name__ == '__main__':
    # Parse command line arguments.
    settings = Settings()
    ARGS = settings.parse_arguments()
    
    # Set up logging for debugging and tracking.
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)
    
    LOGGER = logging.getLogger(__name__)
    
    # Log the start of the training process.
    LOGGER.debug('Training (PID=%d), %s', os.getpid(), ARGS)
    
    # Call the main function to start the training.
    main(ARGS)
            
    LOGGER.debug('Training process ended (PID=%d)', os.getpid())