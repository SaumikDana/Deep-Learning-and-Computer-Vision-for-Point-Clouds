__author__ = "Saumik"
__date__ = "11/03/2023"

import torch
import os
import logging
from torch.optim.lr_scheduler import StepLR

# Configure the logger for this module
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

def custom_collate(batch):
    # Find the maximum number of points in the current batch for both source and target
    max_points_source = max(item['source_points'].size(0) for item in batch)
    max_points_target = max(item['target_points'].size(0) for item in batch)
    max_points = max(max_points_source, max_points_target)
    
    # Pad the point clouds and collect the transformation matrices
    padded_source_points = []
    padded_target_points = []
    transformation_matrices = []
    for item in batch:
        source_points = item['source_points']
        target_points = item['target_points']
        trans_matrix = item['transformation_matrix']
        
        # Calculate the padding size for the current point cloud
        padding_size_source = max_points - source_points.size(0)
        padding_size_target = max_points - target_points.size(0)
        
        # Pad the source and target point clouds
        padded_source = torch.nn.functional.pad(source_points, (0, 0, 0, padding_size_source))
        padded_target = torch.nn.functional.pad(target_points, (0, 0, 0, padding_size_target))
        
        # Add the padded point clouds and transformation matrix to the batch
        padded_source_points.append(padded_source)
        padded_target_points.append(padded_target)
        transformation_matrices.append(trans_matrix)
    
    # Stack the padded point clouds and transformation matrices
    batched_source_points = torch.stack(padded_source_points)
    batched_target_points = torch.stack(padded_target_points)
    batched_trans_matrices = torch.stack(transformation_matrices)
    
    return {
        'source_points': batched_source_points,
        'target_points': batched_target_points,
        'transformation_matrix': batched_trans_matrices
    }

class Trainer:
    """
    The Trainer class handles the training and evaluation of a machine learning model.
    """
    def __init__(self, args, trainset, testset, action):
        """
        Initializes the Trainer object.

        :param args: Command line arguments or other configuration settings.
        :param trainset: The dataset used for training the model.
        :param testset: The dataset used for evaluating the model.
        :param action: An object that defines the create_model, train_1, and eval_1 methods.
        """
        self.args = args
        self.trainset = trainset
        self.testset = testset
        self.action = action
        self.model = None
        self.optimizer = None
        self.min_loss = float('inf')
        self.checkpoint = None

    def setup_device(self):
        """
        Sets up the device for training (CPU or GPU).
        """
        if not torch.cuda.is_available():
            self.args.device = 'cpu'
        else:
            self.args.device = 'cuda'
            
        self.args.device = torch.device(self.args.device)
        print(f"Running on: {self.args.device}")

    def load_pretrained_model(self):
        """
        Loads a pretrained model if a path is provided.
        """
        if self.args.pretrained:
            assert os.path.isfile(self.args.pretrained), "The provided pretrained model file does not exist."
            pretrained_state_dict = torch.load(self.args.pretrained, map_location='cpu')
            self.model.load_state_dict(pretrained_state_dict)

    def load_checkpoint(self):
        """
        Loads model and optimizer states from a checkpoint file if provided.
        """
        if self.args.resume:
            assert os.path.isfile(self.args.resume), "Checkpoint file not found."
            self.checkpoint = torch.load(self.args.resume)
            self.args.start_epoch = self.checkpoint['epoch']
            self.model.load_state_dict(self.checkpoint['model'])

    def setup_model_and_optimizer(self):
        """
        Initializes the model and optimizer.
        """
        self.model = self.action.create_model()
        self.model.to(self.args.device)
        learnable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(learnable_params)
        else:
            self.optimizer = torch.optim.SGD(learnable_params, lr=0.1)
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        if self.checkpoint is not None:
            self.min_loss = self.checkpoint['min_loss']
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])

    def train_and_evaluate(self):
        """
        Trains and evaluates the model.
        """
        print("Initializing data loaders...")
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers, collate_fn=custom_collate)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.workers, collate_fn=custom_collate)
        
        print("Starting training and evaluation...")
        for epoch in range(self.args.start_epoch, self.args.epochs):
            print(f'====================================')
            print(f"Epoch {epoch + 1}/{self.args.epochs}")
            print(f'====================================')
            
            print("======================= Training ============================")
            running_loss, running_info = self.action.train(self.model, trainloader, self.optimizer, self.args.device)
            print(f"Training Loss: {running_loss}, Training Info: {running_info}")
            
            print("======================= Evaluating ==========================")
            val_loss, val_info = self.action.eval(self.model, testloader, self.args.device)
            print(f"Validation Loss: {val_loss}, Validation Info: {val_info}")
            
            is_best = val_loss < self.min_loss
            if is_best:
                print("New best model found!")
            
            self.min_loss = min(val_loss, self.min_loss)
            
            LOGGER.info('epoch, %04d, %f, %f, %f, %f', epoch + 1, running_loss, val_loss, running_info, val_info)
            
            print("Saving checkpoint...")
            self.save_checkpoint(epoch, is_best)
            print("Checkpoint saved.\n")
            
            # Step the learning rate scheduler
            self.scheduler.step()

    def save_checkpoint(self, epoch, is_best):
        """
        Saves the current state of the model and optimizer.

        :param epoch: The current epoch number.
        :param is_best: Boolean indicating whether the current model is the best one so far.
        """
        snap = {
            'epoch': epoch + 1,
            'model': self.model.state_dict(),
            'min_loss': self.min_loss,
            'optimizer': self.optimizer.state_dict(),
        }
        if is_best:
            self._save_checkpoint(snap, 'snap_best')
            self._save_checkpoint(self.model.state_dict(), 'model_best')
        self._save_checkpoint(snap, 'snap_last')
        self._save_checkpoint(self.model.state_dict(), 'model_last')

    def _save_checkpoint(self, state, suffix):
        """
        Helper function to save a checkpoint.

        :param state: The state to be saved.
        :param suffix: Suffix to append to the filename.
        """
        torch.save(state, '{}_{}.pth'.format(self.args.outfile, suffix))
