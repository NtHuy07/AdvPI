"""
ENTRep trainer for contrastive learning
"""

import os
import json
from typing import Dict, Optional, Any, Union, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from tqdm import tqdm
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from ..dataset.factory import DatasetFactory
from ..utils.logging_config import get_logger
from ..utils.helpers import setup_seed

logger = get_logger(__name__)


class VisionLanguageTrainer:
    """
    General trainer for Vision-Language Models with contrastive learning
    
    Supports:
    - ENTRep Model
    - MedCLIP Model  
    - BioMedCLIP Model
    
    Features:
    - Contrastive learning với CLIP loss
    - Multi-GPU training support
    - Mixed precision training
    - Checkpoint saving/loading
    - WandB logging support
    - Validation tracking
    - Flexible dataset support (MIMIC, COVID, RSNA, ENTREP)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        output_dir: str = './checkpoints',
        experiment_name: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = 'vlm-training',
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize Vision-Language Model trainer
        
        Args:
            model: Vision-Language Model instance (ENTRepModel, MedCLIPModel, BioMedCLIPModel)
            config: Training configuration dict containing:
                - dataset: dataset config (dataset_name, model_type, batch_size, etc.)
                - optimizer: optimizer config (type, lr, weight_decay, etc.)
                - scheduler: scheduler config (type, T_max, etc.)
                - training: training config (num_epochs, use_amp, etc.)
            device: Device to use (auto-detect if None)
            output_dir: Directory for checkpoints
            experiment_name: Name for experiment
            use_wandb: Whether to use WandB logging
            wandb_project: WandB project name
            seed: Random seed
        """
        self.model = model
        self.config = config
        self.seed = seed
        setup_seed(seed)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            self.is_parallel = True
        else:
            self.is_parallel = False
            
        # Output directory
        self.output_dir = Path(output_dir)
        if experiment_name is None:
            # Auto-generate experiment name from model and dataset
            model_name = config.get('model_type', 'vlm')
            dataset_name = config.get('dataset', {}).get('dataset_name', 'data')
            experiment_name = f"{model_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.checkpoint_dir = self.output_dir / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.checkpoint_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        # Initialize WandB if requested
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=config
            )
            wandb.watch(model)
            
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking for plotting
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.learning_rates = []
        
        logger.info(f"✅ Trainer initialized")
        logger.info(f"📁 Checkpoints will be saved to: {self.checkpoint_dir}")
        logger.info(f"🔧 Device: {self.device}")
        
    def create_optimizer(self) -> Optimizer:
        """Create optimizer from config"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adam').lower()
        lr = float(opt_config.get('lr', 1e-4))
        weight_decay = float(opt_config.get('weight_decay', 0.01))
        
        # Get model parameters
        if self.is_parallel:
            params = self.model.module.parameters()
        else:
            params = self.model.parameters()
            
        if opt_type == 'adam':
            optimizer = torch.optim.Adam(
                params, 
                lr=lr, 
                weight_decay=weight_decay,
                betas=(float(opt_config.get('betas', (0.9, 0.999))[0]), float(opt_config.get('betas', (0.9, 0.999))[1]))
            )
        elif opt_type == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(float(opt_config.get('betas', (0.9, 0.999))[0]), float(opt_config.get('betas', (0.9, 0.999))[1]))
            )
        elif opt_type == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=float(opt_config.get('momentum', 0.9)),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
            
        logger.info(f"Created {opt_type} optimizer with lr={lr}")
        return optimizer
        
    def create_scheduler(self, optimizer: Optimizer) -> Optional[_LRScheduler]:
        """Create learning rate scheduler from config"""
        sched_config = self.config.get('scheduler', None)
        if sched_config is None:
            return None
            
        sched_type = sched_config.get('type', 'cosine').lower()
        
        if sched_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(sched_config.get('T_max', 100)),
                eta_min=float(sched_config.get('eta_min', 1e-6))
            )
        elif sched_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(sched_config.get('step_size', 30)),
                gamma=float(sched_config.get('gamma', 0.1))
            )
        elif sched_type == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(map(int, sched_config.get('milestones', [30, 60, 90]))),
                gamma=float(sched_config.get('gamma', 0.1))
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")
            
        logger.info(f"Created {sched_type} scheduler")
        return scheduler
        
    def create_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Create train and validation dataloaders using DatasetFactory
        
        Supports multiple datasets: MIMIC, COVID, RSNA, ENTREP
        """
        dataset_config = self.config.get('dataset', {})
        
        # Common config
        dataset_name = dataset_config.get('dataset_name', 'entrep')
        dataset_type = dataset_config.get('dataset_type', 'contrastive')
        task_type = dataset_config.get('task_type', 'contrastive')
        data_root = dataset_config.get('data_root', 'local_data')
        model_type = dataset_config.get('model_type', 'entrep')
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 4)
        
        # Additional kwargs for specific datasets
        extra_kwargs = {}
        if 'tokenizer_name' in dataset_config:
            extra_kwargs['tokenizer_name'] = dataset_config['tokenizer_name']
        if 'transform' in dataset_config:
            extra_kwargs['transform'] = dataset_config['transform']
            
        logger.info(f"📊 Creating dataloaders for {dataset_name} dataset...")
        logger.info(f"   Model type: {model_type}")
        logger.info(f"   Dataset type: {dataset_type}")
        logger.info(f"   Task type: {task_type}")
        
        # Create train dataloader
        train_loader = DatasetFactory.create_dataloader(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            task_type=task_type,
            model_type=model_type,
            split='train',
            data_root=data_root,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            **extra_kwargs
        )
        
        # Create validation dataloader
        val_loader = DatasetFactory.create_dataloader(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            task_type=task_type,
            model_type=model_type,
            split='val',
            data_root=data_root,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            **extra_kwargs
        )
        
        logger.info(f"✅ Created dataloaders:")
        logger.info(f"   Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        logger.info(f"   Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
        
        return {
            'train': train_loader,
            'val': val_loader
        }
        
    def train_epoch(
        self, 
        dataloader: DataLoader, 
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> Dict[str, float]:
        """Train for one epoch - supports all Vision-Language Models"""
        self.model.train()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        # Use mixed precision if scaler provided
        use_amp = scaler is not None
        
        with tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # Prepare model inputs (handle different model interfaces)
                model_inputs = self._prepare_model_inputs(batch)
                
                # Forward pass with mixed precision
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**model_inputs)
                        loss = self._extract_loss(outputs)
                else:
                    outputs = self.model(**model_inputs)
                    loss = self._extract_loss(outputs)
                
                # Backward pass
                optimizer.zero_grad()
                
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Log to WandB
                if self.use_wandb and self.global_step % 10 == 0:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'global_step': self.global_step
                    })
                    
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
            
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def _prepare_model_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare inputs for different model types
        
        Args:
            batch: Batch dict from dataloader
            
        Returns:
            Model inputs dict
        """
        model_inputs = {
            'pixel_values': batch['pixel_values'],
            'return_loss': True
        }
        
        # Add text inputs if available
        # Handle different text input formats from different collators
        if 'input_ids' in batch:
            model_inputs['input_ids'] = batch['input_ids']
        if 'attention_mask' in batch:
            model_inputs['attention_mask'] = batch['attention_mask']
        if 'texts' in batch:
            model_inputs['texts'] = batch['texts']
        elif 'text' in batch:  
            model_inputs['texts'] = batch['text']
            
        return model_inputs
    
    def _extract_loss(self, outputs: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Extract loss from model outputs
        
        Different models return loss in different formats:
        - ENTRep: outputs['loss_value']
        - MedCLIP: outputs['loss_value'] or outputs['loss']
        - BioMedCLIP: outputs['loss_value']
        
        Args:
            outputs: Model outputs
            
        Returns:
            Loss tensor
        """
        if isinstance(outputs, torch.Tensor):
            return outputs
        elif isinstance(outputs, dict):
            # Try different loss keys
            loss = None
            if 'loss_value' in outputs:
                loss = outputs['loss_value']
            elif 'loss' in outputs:
                loss = outputs['loss']
            else:
                raise ValueError(f"Cannot find loss in model outputs. Available keys: {outputs.keys()}")
            
            # Validate loss
            if loss is None:
                raise ValueError(f"Loss is None. Model may be in eval mode or return_loss=False. Outputs: {outputs.keys()}")
            
            if not isinstance(loss, torch.Tensor):
                raise ValueError(f"Loss must be a Tensor, got {type(loss)}: {loss}")
                
            return loss
        else:
            raise ValueError(f"Unexpected output type: {type(outputs)}")
        
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model - supports all Vision-Language Models"""
        self.model.eval()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch in tqdm(dataloader, desc="Validation"):
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            
            # Prepare model inputs
            model_inputs = self._prepare_model_inputs(batch)
                    
            # Forward pass
            outputs = self.model(**model_inputs)
            loss = self._extract_loss(outputs)
            
            total_loss += loss.item()
            
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
        
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        # Get model state dict
        if self.is_parallel:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 Saved best checkpoint to {best_path}")
            
        # Save epoch checkpoint
        if self.current_epoch % self.config.get('save_every', 5) == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
            torch.save(checkpoint, epoch_path)
            
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.is_parallel:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")
        logger.info(f"   Epoch: {self.current_epoch}, Step: {self.global_step}")
        
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Vẽ biểu đồ train và validation loss
        
        Args:
            save_path: Đường dẫn lưu biểu đồ (mặc định lưu vào checkpoint_dir)
        """
        if not self.train_losses:
            logger.warning("Không có dữ liệu để vẽ biểu đồ")
            return
            
        # Tạo figure với 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Train và Val Loss
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        if self.val_losses:
            ax1.plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training và Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate
        if self.learning_rates:
            ax2.plot(self.epochs, self.learning_rates, 'g-', label='Learning Rate', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'Không có dữ liệu Learning Rate', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        if save_path is None:
            save_path = self.checkpoint_dir / 'training_curves.png'
        else:
            save_path = Path(save_path)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Đã lưu biểu đồ training curves tại: {save_path}")
        
    def plot_loss_comparison(self, save_path: Optional[str] = None):
        """
        Vẽ biểu đồ so sánh train và val loss chi tiết hơn
        
        Args:
            save_path: Đường dẫn lưu biểu đồ
        """
        if not self.train_losses:
            logger.warning("Không có dữ liệu để vẽ biểu đồ")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Vẽ train loss
        plt.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', 
                linewidth=2, alpha=0.8)
        
        # Vẽ val loss nếu có
        if self.val_losses:
            plt.plot(self.epochs, self.val_losses, 'r-', label='Val Loss', 
                    linewidth=2, alpha=0.8)
            
            # Tìm điểm best validation loss
            best_epoch = self.epochs[np.argmin(self.val_losses)]
            best_loss = min(self.val_losses)
            plt.scatter([best_epoch], [best_loss], color='red', s=100, 
                       zorder=5, label=f'Best Val Loss: {best_loss:.4f}')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Progress - Loss Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Thêm thông tin thống kê
        stats_text = f"Epochs: {len(self.epochs)}\n"
        stats_text += f"Final Train Loss: {self.train_losses[-1]:.4f}\n"
        if self.val_losses:
            stats_text += f"Final Val Loss: {self.val_losses[-1]:.4f}\n"
            stats_text += f"Best Val Loss: {min(self.val_losses):.4f}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        if save_path is None:
            save_path = self.checkpoint_dir / 'loss_comparison.png'
        else:
            save_path = Path(save_path)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Đã lưu biểu đồ loss comparison tại: {save_path}")
        
    def update_metrics(self, train_loss: float, val_loss: Optional[float] = None, 
                      learning_rate: Optional[float] = None):
        """
        Cập nhật metrics cho plotting
        
        Args:
            train_loss: Train loss của epoch hiện tại
            val_loss: Validation loss (optional)
            learning_rate: Learning rate hiện tại (optional)
        """
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        else:
            # Thêm None để giữ index đồng bộ
            self.val_losses.append(None)
            
        self.epochs.append(self.current_epoch + 1)
        
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
            
    def save_metrics(self, save_path: Optional[str] = None):
        """
        Lưu metrics ra file JSON
        
        Args:
            save_path: Đường dẫn lưu file metrics
        """
        if save_path is None:
            save_path = self.checkpoint_dir / 'training_metrics.json'
        else:
            save_path = Path(save_path)
            
        metrics_data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.epochs)
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
            
        logger.info(f"💾 Đã lưu metrics tại: {save_path}")
        
    def train(self):
        """Main training loop"""
        # Create optimizer and scheduler
        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(optimizer)
        
        # Create dataloaders
        dataloaders = self.create_dataloaders()
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        # Training config
        training_config = self.config.get('training', {})
        num_epochs = training_config.get('num_epochs', self.config.get('num_epochs', 100))
        val_every = training_config.get('val_every', self.config.get('val_every', 1))
        
        # Mixed precision training
        use_amp = training_config.get('use_amp', self.config.get('use_amp', False)) and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        logger.info(f"🎓 Starting training for {num_epochs} epochs")
        if use_amp:
            logger.info("⚡ Using mixed precision training")
            
        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(
                train_loader, 
                optimizer, 
                scheduler,
                scaler
            )
            logger.info(f"📈 Train loss: {train_metrics['loss']:.4f}")
            
            # Validate
            val_loss = None
            if (epoch + 1) % val_every == 0:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['loss']
                logger.info(f"📊 Val loss: {val_loss:.4f}")
                
                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    
                # Save checkpoint
                metrics = {
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_loss
                }
                self.save_checkpoint(metrics, is_best)
                
                # Log to WandB
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train/epoch_loss': train_metrics['loss'],
                        'val/epoch_loss': val_loss,
                        'val/best_loss': self.best_val_loss
                    })
            else:
                # Save checkpoint without validation
                metrics = {'train_loss': train_metrics['loss']}
                self.save_checkpoint(metrics, is_best=False)
                
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train/epoch_loss': train_metrics['loss']
                    })
            
            # Update metrics for plotting
            current_lr = optimizer.param_groups[0]['lr']
            self.update_metrics(
                train_loss=train_metrics['loss'],
                val_loss=val_loss,
                learning_rate=current_lr
            )
            
            # Vẽ biểu đồ mỗi plot_every epochs
            plot_every = training_config.get('plot_every', 5)
            if (epoch + 1) % plot_every == 0:
                self.plot_training_curves()
                self.plot_loss_comparison()
                self.save_metrics()
                    
        logger.info(f"\n🎉 Training completed!")
        logger.info(f"📊 Best validation loss: {self.best_val_loss:.4f}")
        
        # Vẽ biểu đồ cuối cùng
        logger.info("📊 Đang tạo biểu đồ cuối cùng...")
        self.plot_training_curves()
        self.plot_loss_comparison()
        self.save_metrics()
        
        if self.use_wandb:
            wandb.finish()


# Convenience functions for creating trainers
def create_trainer_for_entrep(model, config, **kwargs):
    """
    Convenience function to create trainer for ENTRep model
    
    Args:
        model: ENTRepModel instance
        config: Training config
        **kwargs: Additional trainer arguments
    """
    return VisionLanguageTrainer(model=model, config=config, **kwargs)


def create_trainer_for_medclip(model, config, **kwargs):
    """
    Convenience function to create trainer for MedCLIP model
    
    Args:
        model: MedCLIPModel instance
        config: Training config
        **kwargs: Additional trainer arguments
    """
    return VisionLanguageTrainer(model=model, config=config, **kwargs)


def create_trainer_for_biomedclip(model, config, **kwargs):
    """
    Convenience function to create trainer for BioMedCLIP model
    
    Args:
        model: BioMedCLIPModel instance
        config: Training config
        **kwargs: Additional trainer arguments
    """
    return VisionLanguageTrainer(model=model, config=config, **kwargs)
