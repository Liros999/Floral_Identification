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
import torchvision


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
        self.scheduler = None
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
        
        # TensorBoard
        self.writer = None
        self.log_dir = None
        self._graph_logged = False
        
        logger.info(f"FlowerTrainer initialized on {self.device}")
        logger.info(f"Target precision: ≥{self.target_precision:.1%}")
        logger.info(f"Minimum recall: ≥{self.min_recall:.1%}")
    
    def _find_open_port(self, starting_port: int = 6006, max_tries: int = 20) -> int:
        """
        Find an available localhost TCP port starting from starting_port.
        """
        import socket
        port = starting_port
        for _ in range(max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    port += 1
        return starting_port

    def start_tensorboard(self) -> None:
        """
        Start TensorBoard in the background and attempt to open it in the browser.
        Initializes a SummaryWriter pointed at a fresh run directory.
        """
        try:
            # Prepare log directories (absolute paths anchored to project root)
            timestamp = int(time.time())
            # Use config.yaml log_dir anchored to project root, not CWD
            project_root = Path(__file__).resolve().parents[2]  # Project/FlowerDetector_Clean
            configured_log_dir = self.config.get('tensorboard', {}).get('log_dir', 'logs/tensorboard')
            log_root = (project_root / configured_log_dir).resolve()
            run_dir = log_root / f"run_{timestamp}"
            log_root.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir = str(run_dir)
            self._tb_log_root = str(log_root)
            logger.info(f"📁 TensorBoard log root: {self._tb_log_root}")
            logger.info(f"📁 Run directory: {self.log_dir}")

            # Find and store the chosen port to use consistently
            self._tb_port = self._find_open_port(self.config.get('tensorboard', {}).get('port', 6006))

            # Initialize writer early so logs are present when TB starts
            self.writer = SummaryWriter(self.log_dir)
            logger.info("📈 TensorBoard SummaryWriter initialized")
            # Write immediate summaries so dashboards activate instantly
            try:
                self.writer.add_text('run/meta', 'run started', 0)
                self.writer.add_scalar('run/started', 1, 0)
                self.writer.flush()
            except Exception:
                pass

            # Launch TensorBoard process (Windows-friendly)
            def run_tensorboard():
                try:
                    import platform
                    # Minimize TensorFlow verbosity inside TensorBoard while keeping performance
                    tb_env = os.environ.copy()
                    # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
                    tb_env["TF_CPP_MIN_LOG_LEVEL"] = tb_env.get("TF_CPP_MIN_LOG_LEVEL", "2")
                    if platform.system() == "Windows":
                        cmd = [
                            "python", "-m", "tensorboard.main",
                            "--logdir", self._tb_log_root,
                            "--port", str(self._tb_port),
                            "--host", "localhost",
                        ]
                    else:
                        cmd = [
                            "tensorboard",
                            "--logdir", self._tb_log_root,
                            "--port", str(self._tb_port),
                            "--host", "localhost",
                        ]
                    logger.info(f"🚀 Starting TensorBoard: {' '.join(cmd)}")
                    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                    self.tensorboard_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        creationflags=creation_flags,
                        env=tb_env,
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

            # Try opening in the default browser using the actual chosen port
            try:
                webbrowser.open(f"http://localhost:{self._tb_port}", new=2)
                logger.info("🌐 TensorBoard dashboard opened in browser")
                print("\n" + "=" * 60)
                print("🌐 TensorBoard Dashboard Launched!")
                print(f"📊 URL: http://localhost:{self._tb_port}")
                print("📈 Monitor training progress in real-time")
                print("=" * 60 + "\n")
            except Exception as e:
                logger.warning(f"Could not auto-open browser: {e}")
                logger.info(f"📊 Manually open: http://localhost:{self._tb_port}")
        except Exception as e:
            logger.warning(f"TensorBoard setup failed: {e}")

    def setup_model(self, model: nn.Module) -> None:
        """
        Setup model for training with Intel optimizations.
        
        Args:
            model: PyTorch model to train
        """
        self.model = model.to(self.device)
        
        # CPU optimizations applied via system settings
        logger.info("Using standard PyTorch CPU optimizations")
        
        # CPU threading optimizations are handled by system settings
        
        # Setup optimizer
        optimizer_type = self.training_config.get('optimizer', 'adam').lower()
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
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
        
        # Setup learning rate scheduler
        scheduler_type = self.training_config.get('scheduler', 'cosine_annealing').lower()
        if scheduler_type == 'cosine_annealing':
            min_lr = self.training_config.get('min_lr', self.learning_rate * 0.01)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.epochs,
                eta_min=min_lr
            )
        elif scheduler_type == 'step_lr':
            step_size = self.training_config.get('scheduler_step_size', self.epochs // 3)
            gamma = self.training_config.get('scheduler_gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, using no scheduler")
            self.scheduler = None
        
        # Setup loss function - binary classification
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info("=" * 60)
        logger.info("🎯 MODEL SETUP COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"🔧 Optimizer: {optimizer_type}")
        logger.info(f"📈 Scheduler: {scheduler_type}")
        logger.info(f"🎯 Learning rate: {self.learning_rate}")
        logger.info(f"⚖️ Weight decay: {self.weight_decay}")
        logger.info("=" * 60)
    
    
    def setup_data_loaders(self, dataset) -> None:
        """
        Setup train/val/test data loaders from real dataset.
        Applies stratified splits if enabled; otherwise uses deterministic random split.
        Also computes class-weighted loss from the training split.
        """
        logger.info("🔄 Setting up data loaders...")
        
        total_size = len(dataset)
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        test_ratio = self.config['data']['test_ratio']
        seed = self.config['reproducibility']['global_random_seed']
        
        logger.info(f"📊 Dataset size: {total_size} samples")
        logger.info(f"📈 Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")

        if self.config['data'].get('stratified_splits', False):
            logger.info("⚖️ Using stratified splits for balanced class distribution...")
            indices = list(range(total_size))
            # Use stored labels instead of loading all images - much faster!
            labels = [dataset.labels[i] for i in indices]
            temp_ratio = val_ratio + test_ratio

            logger.info("🔄 Creating stratified train/test split...")
            train_idx, temp_idx = train_test_split(
                indices,
                test_size=temp_ratio,
                stratify=labels,
                random_state=seed
            )

            logger.info("🔄 Creating stratified validation/test split...")
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
            logger.info("🎲 Using deterministic random split...")
            # Deterministic random split
            generator = torch.Generator()
            generator.manual_seed(seed)
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size
            
            logger.info(f"📊 Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size], generator=generator
            )

        # Create data loaders
        logger.info("🔧 Creating optimized data loaders...")
        batch_size = self.config['data']['batch_size']
        num_workers = self.config['data']['num_workers']
        
        logger.info(f"📦 Batch size: {batch_size}")
        logger.info(f"👥 Workers: {num_workers}")
        logger.info(f"💾 Pin memory: False (CPU training)")

        logger.info("🚂 Creating training data loader...")
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False, persistent_workers=True
        )
        
        logger.info("✅ Creating validation data loader...")
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False, persistent_workers=True
        )
        
        logger.info("🧪 Creating test data loader...")
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False, persistent_workers=True
        )

        logger.info("=" * 60)
        logger.info("📊 DATA LOADERS CREATED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"🚂 Train: {len(train_dataset)} samples ({len(self.train_loader)} batches)")
        logger.info(f"✅ Val: {len(val_dataset)} samples ({len(self.val_loader)} batches)")
        logger.info(f"🧪 Test: {len(test_dataset)} samples ({len(self.test_loader)} batches)")
        logger.info("=" * 60)

        # Compute class weights from training split for imbalanced loss
        logger.info("⚖️ Computing class weights for balanced training...")
        logger.info("🔄 Analyzing training data class distribution...")
        
        try:
            if hasattr(train_dataset, 'indices'):
                # Subset: map back to original dataset indices
                logger.info("📊 Extracting labels from stratified subset...")
                train_labels = [dataset.labels[i] for i in train_dataset.indices]
            else:
                logger.info("📊 Extracting labels from random split...")
                train_labels = [dataset.labels[i] for i in range(len(train_dataset))]

            logger.info("🔢 Counting class occurrences...")
            import collections
            counts = collections.Counter(train_labels)
            num_neg, num_pos = counts.get(0, 0), counts.get(1, 0)
            if num_neg == 0 or num_pos == 0:
                raise ValueError("Training split has only one class; cannot compute class weights.")

            logger.info("⚖️ Computing balanced class weights...")
            total = num_neg + num_pos
            weights = torch.tensor([
                total / (2 * num_neg),
                total / (2 * num_pos)
            ], dtype=torch.float32, device=self.device)

            self.criterion = nn.CrossEntropyLoss(weight=weights)
            logger.info(f"✅ Class-weighted loss enabled:")
            logger.info(f"  🌺 Positive weight: {weights[1]:.4f}")
            logger.info(f"  🌿 Negative weight: {weights[0]:.4f}")
            logger.info(f"  📊 Class distribution: {num_pos} positive, {num_neg} negative")
        except Exception as e:
            logger.warning(f"⚠️ Falling back to unweighted loss: {e}")
            self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch with gradient accumulation support.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Get gradient accumulation setting
        accumulate_grad_batches = self.config.get('data', {}).get('accumulate_grad_batches', 1)
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (batch_images, batch_labels) in enumerate(progress_bar):
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            outputs = self.model(batch_images)

            # Log the model graph once using a real batch
            if self.writer is not None and not self._graph_logged:
                try:
                    self.writer.add_graph(self.model, batch_images)
                    self._graph_logged = True
                except Exception:
                    # Attempt only once; skip on failure to avoid slowing training
                    self._graph_logged = True
            loss = self.criterion(outputs, batch_labels)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulate_grad_batches
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation: only step optimizer every N batches
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                # Gradient clipping for training stability
                clip_val = self.training_config.get('gradient_clip_val', 1.0)
                if clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_val)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Step scheduler if available
                if self.scheduler is not None:
                    self.scheduler.step()
            
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
    
    def check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop early based on configured monitor metric.
        
        Args:
            metrics: Dictionary containing validation metrics
            
        Returns:
            True if training should stop
        """
        # Get monitor metric from config (default to val_precision for backward compatibility)
        monitor_key = self.training_config.get('early_stopping', {}).get('monitor', 'val_precision')
        current_metric = metrics.get(monitor_key, 0.0)
        
        if current_metric > self.best_val_precision:
            self.best_val_precision = current_metric
            self.best_model_state = self.model.state_dict().copy()
            self.epochs_without_improvement = 0
            logger.info(f"New best {monitor_key}: {current_metric:.4f}")
            return False
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement in {monitor_key}")
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
        
        # Create progress bar for epochs
        from tqdm import tqdm
        epoch_pbar = tqdm(range(self.epochs), desc="Training Progress", unit="epoch", 
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        # Two-stage training setup
        two_stage = self.training_config.get('two_stage_training', False)
        stage1_epochs = self.training_config.get('stage1_epochs', 10)
        stage2_lr = self.training_config.get('stage2_learning_rate', 1e-5)
        
        for epoch in epoch_pbar:
            epoch_start = time.time()
            
            # Two-stage training logic
            if two_stage and epoch < stage1_epochs:
                # Stage 1: Freeze backbone, train only classifier
                if epoch == 0:
                    logger.info(f"STAGE 1: Freezing backbone for {stage1_epochs} epochs")
                    self._freeze_backbone()
                current_stage = "Stage 1 (Frozen Backbone)"
            else:
                # Stage 2: Unfreeze all, fine-tune
                if two_stage and epoch == stage1_epochs:
                    logger.info(f"STAGE 2: Unfreezing backbone, reducing LR to {stage2_lr}")
                    self._unfreeze_backbone()
                    self._adjust_learning_rate(stage2_lr)
                current_stage = "Stage 2 (Fine-tuning)"
            
            # Update progress bar description
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{self.epochs} ({current_stage})")
            
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
            
            # Update progress bar with metrics
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Val Acc': f'{val_accuracy:.3f}',
                'Val F1': f'{val_f1:.3f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Write TensorBoard scalars
            if self.writer is not None:
                global_step = epoch + 1
                self.writer.add_scalar('train/loss', train_loss, global_step)
                self.writer.add_scalar('train/accuracy', train_accuracy, global_step)
                self.writer.add_scalar('val/loss', val_loss, global_step)
                self.writer.add_scalar('val/accuracy', val_accuracy, global_step)
                self.writer.add_scalar('val/precision', val_precision, global_step)
                self.writer.add_scalar('val/recall', val_recall, global_step)
                self.writer.add_scalar('val/f1', val_f1, global_step)
                # Learning rate logging (supports schedulers or static lr)
                try:
                    current_lr = next(iter(self.optimizer.param_groups))['lr'] if self.optimizer else self.learning_rate
                    self.writer.add_scalar('training/learning_rate', current_lr, global_step)
                except Exception:
                    pass

                # Heavy logging only every 5 epochs to reduce overhead
                if epoch % 5 == 0 or epoch == self.epochs - 1:
                    # Parameter and gradient histograms (epoch-level)
                    try:
                        for name, param in self.model.named_parameters():
                            try:
                                self.writer.add_histogram(f'params/{name}', param.detach().cpu().numpy(), global_step)
                                if param.grad is not None:
                                    self.writer.add_histogram(f'grads/{name}', param.grad.detach().cpu().numpy(), global_step)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Sample image grids from train/val
                    try:
                        train_batch = next(iter(self.train_loader))
                        train_imgs = train_batch[0][:8].detach().cpu()
                        train_imgs = (train_imgs - train_imgs.min()) / (train_imgs.max() - train_imgs.min() + 1e-8)
                        train_grid = torchvision.utils.make_grid(train_imgs, nrow=4)
                        self.writer.add_image('train/sample_batch', train_grid, global_step)
                    except Exception:
                        pass
                    try:
                        val_batch = next(iter(self.val_loader))
                        val_imgs = val_batch[0][:8].detach().cpu()
                        val_imgs = (val_imgs - val_imgs.min()) / (val_imgs.max() - val_imgs.min() + 1e-8)
                        val_grid = torchvision.utils.make_grid(val_imgs, nrow=4)
                        self.writer.add_image('val/sample_batch', val_grid, global_step)
                    except Exception:
                        pass
                self.writer.flush()
            
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
            metrics_dict = {
                'val_precision': val_precision,
                'val_f1': val_f1,
                'val_recall': val_recall,
                'val_accuracy': val_accuracy
            }
            if self.check_early_stopping(metrics_dict):
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
        
        # Finalize TensorBoard writer
        try:
            if self.writer is not None:
                # Log final metrics as hparams summary for quick comparison between runs
                try:
                    hparams = {
                        'epochs': self.epochs,
                        'learning_rate': self.learning_rate,
                        'weight_decay': self.weight_decay,
                        'target_precision': self.target_precision,
                        'min_recall': self.min_recall,
                    }
                    metrics = {
                        'hparam/accuracy': final_accuracy,
                        'hparam/precision': final_precision,
                        'hparam/recall': final_recall,
                        'hparam/loss': final_loss,
                    }
                    self.writer.add_hparams(hparams, metrics)
                except Exception:
                    pass
                self.writer.flush()
                self.writer.close()
        except Exception:
            pass

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
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'best_val_precision': self.best_val_precision,
            'history': self.history
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def _freeze_backbone(self):
        """Freeze backbone parameters for Stage 1 training."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen - only classifier will be trained")
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters for Stage 2 fine-tuning."""
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen - full model will be fine-tuned")
    
    def _adjust_learning_rate(self, new_lr):
        """Adjust learning rate for Stage 2 fine-tuning."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        logger.info(f"Learning rate adjusted to {new_lr}")


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

