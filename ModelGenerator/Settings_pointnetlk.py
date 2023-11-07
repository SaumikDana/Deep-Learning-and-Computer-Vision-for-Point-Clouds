__author__ = "Saumik"
__date__ = "11/03/2023"

import argparse
import os

class Settings:
    """
    A class for parsing command line arguments and storing settings for the PointNet-LK program.
    """
    def __init__(self):
        """
        Initializes the Settings object.
        """
        pass

    def parse_arguments(self, argv=None):
        """
        Parses command line arguments.

        :param argv: A list of command line arguments. If None, the arguments are taken from sys.argv.
        :return: An argparse.Namespace object containing the parsed arguments.
        """
        # Create an ArgumentParser object for handling command line arguments.
        parser = argparse.ArgumentParser(description='PointNet-LK')

        # Define the command line arguments that the program accepts.
        parser.add_argument('-o', '--outfile', required=True, type=str,
                            metavar='BASENAME', help='output filename (prefix)')
        parser.add_argument('-i', '--h5-file', required=True, type=str,
                            metavar='PATH', help='path to the input h5 file')
        parser.add_argument('--num-points', default=1024, type=int,
                            metavar='N', help='points in point-cloud (default: 1024)')
        parser.add_argument('--mag', default=0.8, type=float,
                            metavar='T', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')
        parser.add_argument('--dim-k', default=1024, type=int,
                            metavar='K', help='dim. of the feature vector (default: 1024)')
        parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                            help='symmetric function (default: max)')
        parser.add_argument('--max-iter', default=10, type=int,
                            metavar='N', help='max-iter on LK. (default: 10)')
        parser.add_argument('--delta', default=1.0e-2, type=float,
                            metavar='D', help='step size for approx. Jacobian (default: 1.0e-2)')
        parser.add_argument('--learn-delta', dest='learn_delta', action='store_true',
                            help='flag for training step size delta')
        parser.add_argument('-j', '--workers', default=4, type=int,
                            metavar='N', help='number of data loading workers (default: 4)')
        parser.add_argument('-b', '--batch-size', default=32, type=int,
                            metavar='N', help='mini-batch size (default: 32)')
        parser.add_argument('--epochs', default=200, type=int,
                            metavar='N', help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int,
                            metavar='N', help='manual epoch number (useful on restarts)')
        parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                            metavar='METHOD', help='name of an optimizer (default: Adam)')
        parser.add_argument('--resume', default='', type=str,
                            metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
        parser.add_argument('--pretrained', default='', type=str,
                            metavar='PATH', help='path to pretrained model file (default: null (no-use))')
        parser.add_argument('--device', default='cuda:0', type=str,
                            metavar='DEVICE', help='use CUDA if available')
        parser.add_argument('--logfile', default='logfile.txt', type=str,
                            metavar='PATH', help='path to the log file (default: logfile.txt)')

        # Parse the command line arguments.
        args = parser.parse_args(argv)
        
        # Return the parsed arguments.
        return args
