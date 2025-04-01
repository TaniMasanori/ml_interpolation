"""
Training utilities for seismic interpolation models.

This module provides training loops, loss functions, and utilities for
training seismic interpolation models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import logging
import mlflow
from pathlib import Path

logger = logging.getLogger(__name__)

class SeismicTrainer:
    """Trainer class for seismic interpolation models."""
    
    def __init__(self, model, train_dataset, val_dataset, 
                 batch_size=32, learning_rate=1e-4, weight_decay=1e-5,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 experiment_name='seismic_interpolation'):
        """
        Initialize trainer.
        
        Args:
            model (torch.nn.Module): PyTorch model to train
            train_dataset (torch.utils.data.Dataset): Training dataset
            val_dataset (torch.utils.data.Dataset): Validation dataset
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay (L2 regularization)
            device (str): Device to use ('cuda' or 'cpu')
            experiment_name (str): MLflow experiment name
        """
        self.model = model.to(device)
        self.device = device
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        # Set up optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        
        # Set up learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # MLflow tracking
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Forward pass
            batch = self._prepare_batch(batch)
            outputs = self._forward_pass(batch)
            loss = self._compute_loss(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f'Train Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.6f}')
                
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """
        Validate the model.
        
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Forward pass
                batch = self._prepare_batch(batch)
                outputs = self._forward_pass(batch)
                loss = self._compute_loss(outputs, batch)
                
                # Update statistics
                total_loss += loss.item()
                
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _prepare_batch(self, batch):
        """
        Prepare batch for model input.
        
        This method should be overridden by subclasses to match their specific model input.
        
        Args:
            batch (tuple): Batch from data loader
            
        Returns:
            dict: Batch data ready for model input
        """
        # Default implementation for SeismicDataset
        masked_geophone, das, mask, target_geophone = batch
        
        return {
            'masked_geophone': masked_geophone.to(self.device),
            'das': das.to(self.device),
            'mask': mask.to(self.device),
            'target_geophone': target_geophone.to(self.device)
        }
    
    def _forward_pass(self, batch):
        """
        Perform forward pass through the model.
        
        This method should be overridden by subclasses to match their specific model.
        
        Args:
            batch (dict): Batch data
            
        Returns:
            torch.Tensor: Model output
        """
        # Default implementation, should be overridden
        output = self.model(batch['das'], batch['masked_geophone'], batch['mask'])
        return output
    
    def _compute_loss(self, outputs, batch):
        """
        Compute loss for the current batch.
        
        This method should be overridden by subclasses to match their specific loss computation.
        
        Args:
            outputs (torch.Tensor): Model outputs
            batch (dict): Batch data
            
        Returns:
            torch.Tensor: Loss value
        """
        # Default implementation, should be overridden
        # Compute MSE loss only on masked channels
        mask = batch['mask']
        target = batch['target_geophone']
        
        loss = self.criterion(outputs[mask], target[mask])
        return loss
    
    def train(self, num_epochs, save_dir='./models', save_freq=5):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs (int): Number of epochs to train
            save_dir (str): Directory to save model checkpoints
            save_freq (int): Frequency of saving checkpoints (in epochs)
            
        Returns:
            dict: Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Start MLflow run
        with mlflow.start_run():
            # Log model parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    mlflow.log_param(f"param_{name}", param.numel())
            
            mlflow.log_param('batch_size', self.train_loader.batch_size)
            mlflow.log_param('initial_lr', self.optimizer.param_groups[0]['lr'])
            mlflow.log_param('model_type', self.model.__class__.__name__)
            
            # Train for the specified number of epochs
            for epoch in range(num_epochs):
                start_time = time.time()
                
                # Train and validate
                train_loss = self.train_epoch()
                val_loss = self.validate()
                
                # Update learning rate
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['learning_rate'].append(current_lr)
                
                # Log to MLflow
                mlflow.log_metric('train_loss', train_loss, step=epoch)
                mlflow.log_metric('val_loss', val_loss, step=epoch)
                mlflow.log_metric('lr', current_lr, step=epoch)
                
                # Print epoch summary
                elapsed = time.time() - start_time
                logger.info(f'Epoch {epoch+1}/{num_epochs}, '
                           f'Train Loss: {train_loss:.6f}, '
                           f'Val Loss: {val_loss:.6f}, '
                           f'LR: {current_lr:.8f}, '
                           f'Time: {elapsed:.2f}s')
                
                # Save checkpoint
                if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
                    checkpoint_path = save_path / f'checkpoint_epoch_{epoch+1}.pt'
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    logger.info(f'Saved checkpoint to {checkpoint_path}')
                    
                    # Log model to MLflow
                    mlflow.pytorch.log_model(self.model, f'model_epoch_{epoch+1}')
            
            # Save final model
            final_model_path = save_path / 'final_model.pt'
            torch.save(self.model.state_dict(), final_model_path)
            logger.info(f'Saved final model to {final_model_path}')
            
            # Log final model to MLflow
            mlflow.pytorch.log_model(self.model, 'final_model')
        
        return history

class TransformerSeismicTrainer(SeismicTrainer):
    """Trainer class for transformer-based seismic interpolation models."""
    
    def _prepare_batch(self, batch):
        """
        Prepare batch for transformer model input.
        
        Args:
            batch (tuple): Batch from TransformerSeismicDataset
            
        Returns:
            dict: Batch data ready for transformer model input
        """
        # From TransformerSeismicDataset
        input_data, attention_mask, positions, target = batch
        
        return {
            'input_data': input_data.to(self.device),
            'attention_mask': attention_mask.to(self.device),
            'positions': positions.to(self.device) if positions is not None else None,
            'target': target.to(self.device)
        }
    
    def _forward_pass(self, batch):
        """
        Perform forward pass through the transformer model.
        
        Args:
            batch (dict): Batch data
            
        Returns:
            torch.Tensor: Model output
        """
        # For StorSeismicBERTModel
        output = self.model(
            batch['input_data'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['positions']
        )
        return output
    
    def _compute_loss(self, outputs, batch):
        """
        Compute loss for transformer model output.
        
        Args:
            outputs (torch.Tensor): Model outputs
            batch (dict): Batch data
            
        Returns:
            torch.Tensor: Loss value
        """
        # Compute loss only on geophone channels (which are the target)
        # This assumes output has the same sequence as input_data
        target = batch['target']
        
        # Extract only the geophone portion of the outputs
        # Assuming das_channels are followed by geo_channels in the output
        n_das_channels = outputs.shape[1] - target.shape[1]
        predicted_geophone = outputs[:, n_das_channels:, :]
        
        # Compute MSE loss
        loss = self.criterion(predicted_geophone, target)
        return loss

class MultimodalTrainer(SeismicTrainer):
    """Trainer class for multimodal seismic interpolation models."""
    
    def _prepare_batch(self, batch):
        """
        Prepare batch for multimodal model input.
        
        Args:
            batch (tuple): Batch from SeismicDataset
            
        Returns:
            dict: Batch data ready for multimodal model input
        """
        # From SeismicDataset
        masked_geophone, das, mask, target_geophone = batch
        
        return {
            'masked_geophone': masked_geophone.to(self.device),
            'das': das.to(self.device),
            'mask': mask.to(self.device),
            'target_geophone': target_geophone.to(self.device)
        }
    
    def _forward_pass(self, batch):
        """
        Perform forward pass through the multimodal model.
        
        Args:
            batch (dict): Batch data
            
        Returns:
            torch.Tensor: Model output
        """
        # For MultimodalSeismicTransformer
        output = self.model(batch['das'], batch['masked_geophone'], batch['mask'])
        return output
    
    def _compute_loss(self, outputs, batch):
        """
        Compute loss for multimodal model output.
        
        Args:
            outputs (torch.Tensor): Model outputs
            batch (dict): Batch data
            
        Returns:
            torch.Tensor: Loss value
        """
        target = batch['target_geophone']
        mask = batch['mask']
        
        # Compute loss on the masked (missing) channels
        # First, create binary mask to select only the masked channels
        batch_size, n_channels = mask.shape
        masked_indices = torch.where(mask)
        
        # Extract predictions and targets for masked channels
        predicted_values = outputs[masked_indices[0], masked_indices[1], :]
        target_values = target[masked_indices[0], masked_indices[1], :]
        
        # Compute MSE loss
        loss = self.criterion(predicted_values, target_values)
        return loss