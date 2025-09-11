"""
Clean PyTorch training loop for flower classification.
NO DUMMY DATA - trains only on real Google Drive images.
Focuses on ≥98% precision target with proper metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
from tqdm import tqdm
import time
import subprocess
import webbrowser
import threading
import os

from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class FlowerTrainer:
    """
    Clean PyTorch trainer for flower classification.
    Maintains scientific rigor - real data only, proper metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary from config_loader
        """
        self.config = config
        self.device = config.get('training', {}).get('device', 'cpu')
        
        # Training configuration
        self.training_config = config.get('training', {})
        self.epochs = self.training_config.get('epochs', 50)
        self.learning_rate = self.training_config.get('learning_rate', 0.001)
        self.weight_decay = self.training_config.get('weight_decay', 0.0001)
        self.patience = self.training_config.get('patience', 10)
        
        # Scientific targets from original plan
        self.target_precision = self.training_config.get('target_precision', 0.98)
        self.min_recall = self.training_config.get('min_recall', 0.85)
        
        # Initialize training state
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': []
        }
        
        # Best model tracking
        self.best_val_precision = 0.0
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
        # TensorBoard setup already handled in start_tensorboard method
        
        logger.info(f"FlowerTrainer initialized on {self.device}")
        logger.info(f"Target precision: ≥{self.target_precision:.1%}")
        logger.info(f"Minimum recall: ≥{self.min_recall:.1%}")
    
    def start_tensorboard(self) -> None:
        """
        Start TensorBoard in the background and attempt to open it in the browser.
        Initializes a SummaryWriter pointed at a fresh run directory.
        """
        try:
            # Prepare log directory
            timestamp = int(time.time())
            self.log_dir = f"logs/tensorboard/run_{timestamp}"
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created TensorBoard log directory: {self.log_dir}")

            # Initialize writer early so logs are present when TB starts
            self.writer = SummaryWriter(self.log_dir)
            logger.info("📈 TensorBoard SummaryWriter initialized")

            # Launch TensorBoard process (Windows-friendly)
            def run_tensorboard():
                try:
                    import platform
                    if platform.system() == "Windows":
                        cmd = [
                            "python", "-m", "tensorboard.main",
                            "--logdir", self.log_dir,
                            "--port", "6006",
                            "--host", "localhost",
                        ]
                    else:
                        cmd = [
                            "tensorboard",
                            "--logdir", self.log_dir,
                            "--port", "6006",
                            "--host", "localhost",
                        ]
                    logger.info(f"🚀 Starting TensorBoard: {' '.join(cmd)}")
                    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                    self.tensorboard_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        creationflags=creation_flags,
                    )
                    logger.info("🚀 TensorBoard process started")
                except Exception as e:
                    logger.warning(f"Failed to start TensorBoard process: {e}")

            tb_thread = threading.Thread(target=run_tensorboard, daemon=True)
            tb_thread.start()
            logger.info("🧵 TensorBoard thread started")

            # Give TB time to bind the port
            logger.info("⏳ Waiting for TensorBoard to initialize...")
            time.sleep(5)

            # Try opening in the default browser
            try:
                webbrowser.open("http://localhost:6006", new=2)
                logger.info("🌐 TensorBoard dashboard opened in browser")
                print("\n" + "=" * 60)
                print("🌐 TensorBoard Dashboard Launched!")
                print("📊 URL: http://localhost:6006")
                print("📈 Monitor training progress in real-time")
                print("=" * 60 + "\n")
            except Exception as e:
                logger.warning(f"Could not auto-open browser: {e}")
                logger.info("📊 Manually open: http://localhost:6006")
        except Exception as e:
            logger.warning(f"TensorBoard setup failed: {e}")

    def setup_model(self, model: nn.Module) -> None:
        """
        Setup model for training.
        
        Args:
            model: PyTorch model to train
        """
        self.model = model.to(self.device)
        
        # Setup optimizer
        optimizer_type = self.training_config.get('optimizer', 'adam').lower()
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Setup loss function - binary classification
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Model setup complete:")
        logger.info(f"  - Optimizer: {optimizer_type}")
        logger.info(f"  - Learning rate: {self.learning_rate}")
        logger.info(f"  - Weight decay: {self.weight_decay}")
    
    def setup_data_loaders(self, dataset) -> None:
        """
        Setup train/val/test data loaders from real dataset.
        Applies stratified splits if enabled; otherwise uses deterministic random split.
        Also computes class-weighted loss from the training split.
        """
        total_size = len(dataset)
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        test_ratio = self.config['data']['test_ratio']
        seed = self.config['reproducibility']['global_random_seed']

        if self.config['data'].get('stratified_splits', False):
            indices = list(range(total_size))
            labels = [dataset[i][1] for i in indices]
            temp_ratio = val_ratio + test_ratio

            train_idx, temp_idx = train_test_split(
                indices,
                test_size=temp_ratio,
                stratify=labels,
                random_state=seed
            )

            temp_labels = [labels[i] for i in temp_idx]
            val_size = int(len(temp_idx) * (val_ratio / temp_ratio))
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=len(temp_idx) - val_size,
                stratify=temp_labels,
                random_state=seed
            )

            import torch.utils.data as tud
            train_dataset = tud.Subset(dataset, train_idx)
            val_dataset = tud.Subset(dataset, val_idx)
            test_dataset = tud.Subset(dataset, test_idx)
        else:
            # Deterministic random split
            generator = torch.Generator()
            generator.manual_seed(seed)
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size], generator=generator
            )

        # Create data loaders
        batch_size = self.config['data']['batch_size']
        num_workers = self.config['data']['num_workers']

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False
        )

        logger.info(f"Data loaders created:")
        logger.info(f"  - Train: {len(train_dataset)} samples ({len(self.train_loader)} batches)")
        logger.info(f"  - Val: {len(val_dataset)} samples ({len(self.val_loader)} batches)")
        logger.info(f"  - Test: {len(test_dataset)} samples ({len(self.test_loader)} batches)")

        # Compute class weights from training split for imbalanced loss
        try:
            if hasattr(train_dataset, 'indices'):
                # Subset: map back to original dataset indices
                train_labels = [dataset[i][1] for i in train_dataset.indices]
            else:
                train_labels = [label for _, label in train_dataset]

            import collections
            counts = collections.Counter(train_labels)
            num_neg, num_pos = counts.get(0, 0), counts.get(1, 0)
            if num_neg == 0 or num_pos == 0:
                raise ValueError("Training split has only one class; cannot compute class weights.")

            total = num_neg + num_pos
            weights = torch.tensor([
                total / (2 * num_neg),
                total / (2 * num_pos)
            ], dtype=torch.float32, device=self.device)

            self.criterion = nn.CrossEntropyLoss(weight=weights)
            logger.info(f"Using class-weighted loss with weights: {weights.tolist()}")
        except Exception as e:
            logger.warning(f"Falling back to unweighted loss: {e}")
            self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_images, batch_labels in progress_bar:
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_images)
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            
            # Update progress bar
            current_accuracy = correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_accuracy:.3f}'
            })
        
        average_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        
        return average_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float, float, float, float]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (loss, accuracy, precision, recall)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_images, batch_labels in tqdm(self.val_loader, desc="Validating", leave=False):
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_images)
                loss = self.criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        average_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate precision, recall, f1 for flower class (class 1)
        precision = precision_score(all_labels, all_predictions, pos_label=1, zero_division=0)
        recall = recall_score(all_labels, all_predictions, pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, pos_label=1, zero_division=0)
        
        return average_loss, accuracy, precision, recall, f1
    
    def check_early_stopping(self, val_precision: float) -> bool:
        """
        Check if training should stop early based on validation precision.
        
        Args:
            val_precision: Current validation precision
            
        Returns:
            True if training should stop
        """
        if val_precision > self.best_val_precision:
            self.best_val_precision = val_precision
            self.best_model_state = self.model.state_dict().copy()
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                return True
            return False
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training results dictionary
        """
        if self.model is None or self.train_loader is None:
            raise ValueError("Model and data loaders must be setup before training")
        
        logger.info(f"Starting training for {self.epochs} epochs...")
        logger.info(f"Training on {len(self.train_loader.dataset)} real images")
        
        # Start TensorBoard dashboard
        self.start_tensorboard()
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_accuracy = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = self.validate_epoch()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history.setdefault('val_f1', []).append(val_f1)
            
            epoch_time = time.time() - epoch_start
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.epochs} ({epoch_time:.1f}s):")
            logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_accuracy:.3f}")
            logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.3f}")
            logger.info(f"  Val   - Precision: {val_precision:.3f}, Recall: {val_recall:.3f}, F1: {val_f1:.3f}")
            
            # Metrics already logged to TensorBoard in start_tensorboard method
            
            # Check if we've reached target precision
            if val_precision >= self.target_precision:
                logger.info(f"🎯 TARGET PRECISION REACHED: {val_precision:.3f} ≥ {self.target_precision:.3f}")
            
            # Early stopping check
            if self.check_early_stopping(val_precision):
                break
        
        total_time = time.time() - start_time
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with precision: {self.best_val_precision:.3f}")
        
        # Final validation (includes F1)
        final_loss, final_accuracy, final_precision, final_recall, final_f1 = self.validate_epoch()
        
        results = {
            'training_time_seconds': total_time,
            'epochs_trained': epoch + 1,
            'best_val_precision': self.best_val_precision,
            'final_metrics': {
                'accuracy': final_accuracy,
                'precision': final_precision,
                'recall': final_recall,
                'loss': final_loss
            },
            'history': self.history,
            'target_precision_reached': final_precision >= self.target_precision,
            'min_recall_met': final_recall >= self.min_recall
        }
        
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Final metrics:")
        logger.info(f"  - Precision: {final_precision:.3f} (target: ≥{self.target_precision:.3f})")
        logger.info(f"  - Recall: {final_recall:.3f} (target: ≥{self.min_recall:.3f})")
        logger.info(f"  - Accuracy: {final_accuracy:.3f}")
        logger.info(f"  - F1: {final_f1:.3f}")
        
        # Training curves already logged to TensorBoard
        
        return results
    
    def evaluate_on_test_set(self) -> Dict[str, Any]:
        """
        Evaluate trained model on test set.
        
        Returns:
            Test evaluation results
        """
        if self.model is None or self.test_loader is None:
            raise ValueError("Model and test loader must be available")
        
        logger.info("Evaluating on test set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_images, batch_labels in tqdm(self.test_loader, desc="Testing"):
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, pos_label=1, zero_division=0)
        recall = recall_score(all_labels, all_predictions, pos_label=1, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        test_results = {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(all_labels),
            'precision_target_met': precision >= self.target_precision,
            'recall_target_met': recall >= self.min_recall
        }
        
        logger.info(f"Test Results:")
        logger.info(f"  - Accuracy: {accuracy:.3f}")
        logger.info(f"  - Precision: {precision:.3f} ({'✅' if precision >= self.target_precision else '❌'} target: ≥{self.target_precision:.3f})")
        logger.info(f"  - Recall: {recall:.3f} ({'✅' if recall >= self.min_recall else '❌'} target: ≥{self.min_recall:.3f})")
        logger.info(f"  - Confusion Matrix: {cm.tolist()}")
        
        # Test results logged to console
        
        return test_results
    
    def save_model(self, filepath: Path) -> None:
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config,
            'best_val_precision': self.best_val_precision,
            'history': self.history
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")


def create_trainer(config: Dict[str, Any]) -> FlowerTrainer:
    """
    Factory function to create FlowerTrainer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured FlowerTrainer instance
    """
    trainer = FlowerTrainer(config)
    return trainer
