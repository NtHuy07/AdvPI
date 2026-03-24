from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

from .base import BaseEvaluator
from modules.utils.logging_config import get_logger
from modules.utils import constants

logger = get_logger(__name__)


class ZeroShotEvaluator(BaseEvaluator):
    """
    Evaluator for zero-shot classification tasks
    """
    
    def __init__(
        self,
        model,
        class_names: List[str],
        templates: Optional[List[str]] = None,
        mode: str = 'binary',
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            model: Vision-language model (MedCLIP, BioMedCLIP, etc.)
            class_names: List of class names
            templates: Text templates for prompts (if None will use default)
            mode: 'binary' or 'multilabel'
            device: Device to run evaluation
        """
        super().__init__(model, device, **kwargs)
        
        self.class_names = class_names
        self.mode = mode
        
        # Setup default templates
        if templates is None:
            model_name = getattr(model, 'model_name', 'general')
            if model_name in constants.DEFAULT_TEMPLATES:
                base_template = constants.DEFAULT_TEMPLATES[model_name]
                self.templates = [base_template + '{}']
            else:
                self.templates = ['this is a photo of {}']
        else:
            self.templates = templates
        
        # Create text prompts for all classes
        self.text_prompts = self._create_text_prompts()
        
        # Pre-encode text prompts to speed up
        self.text_features = self._encode_text_prompts()
    
    def _create_text_prompts(self) -> List[str]:
        """Create text prompts from class names and templates"""
        prompts = []
        for class_name in self.class_names:
            for template in self.templates:
                prompts.append(template.format(class_name))
        return prompts
    
    def _encode_text_prompts(self) -> torch.Tensor:
        """Pre-encode text prompts"""
        with torch.no_grad():
            if hasattr(self.model, 'encode_text'):
                # For MedCLIP and BioMedCLIP
                text_features = self.model.encode_text(self.text_prompts, normalize=True)
            else:
                logger.error("Model does not have `encode_text` method")
            
            return text_features.to(self.device)
    
    def evaluate(
        self, 
        dataloader, 
        top_k: List[int] = [1, 5],
        return_predictions: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform zero-shot classification evaluation
        
        Args:
            dataloader: DataLoader test data
            top_k: List of k values for top-k accuracy
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary contains accuracy, precision, recall, f1, and optionally predictions
        """
        all_predictions = []
        all_labels = []
        all_logits = []
        
        logger.info(f"🔄 Evaluating zero-shot classification with {len(self.class_names)} classes...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Zero-shot evaluation"):
                batch = self._move_to_device(batch)
                
                # Encode images
                if hasattr(self.model, 'encode_image'):
                    image_features = self.model.encode_image(
                        batch['pixel_values'], normalize=True
                    )
                else:
                    logger.error("Model does not have `encode_image` method")
                
                # Compute similarities with text prompts
                similarities = torch.matmul(image_features, self.text_features.t())
                
                # Aggregate similarities per class if there are multiple templates
                if len(self.templates) > 1:
                    # Reshape: (batch_size, num_classes, num_templates)
                    similarities = similarities.view(
                        similarities.size(0), 
                        len(self.class_names), 
                        len(self.templates)
                    )
                    # Average over templates
                    logits = similarities.mean(dim=-1)
                else:
                    logits = similarities
                
                # Get predictions
                if self.mode == 'multiclass' or self.mode == 'binary':
                    predictions = torch.argmax(logits, dim=-1)
                else:  # multilabel
                    predictions = torch.sigmoid(logits) > 0.5
                
                all_logits.append(logits.cpu())
                all_predictions.append(predictions.cpu())
                all_labels.append(batch['labels'].cpu())
        
        # Concatenate all results
        all_logits = torch.cat(all_logits, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        metrics = self._compute_metrics(
            all_predictions, all_labels, all_logits, top_k
        )
        
        if return_predictions:
            metrics['predictions'] = all_predictions.numpy()
            metrics['labels'] = all_labels.numpy()
            metrics['logits'] = all_logits.numpy()
        
        return metrics
    
    def _compute_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        logits: torch.Tensor,
        top_k: List[int]
    ) -> Dict[str, Any]:
        """Compute metrics"""
        metrics = {}
        
        predictions_np = predictions.numpy()
        labels_np = labels.numpy()
        logits_np = logits.numpy()
        
        if self.mode == 'binary' or self.mode == 'multiclass':
            # Accuracy
            accuracy = (predictions_np == labels_np).mean()
            metrics['accuracy'] = accuracy
            
            # Top-k accuracy
            for k in top_k:
                if k <= len(self.class_names):
                    top_k_preds = torch.topk(logits, k, dim=-1)[1]
                    top_k_acc = (top_k_preds == labels.unsqueeze(-1)).any(dim=-1).float().mean()
                    metrics[f'top_{k}_accuracy'] = top_k_acc.item()
            
            # Classification report
            try:
                report = classification_report(
                    labels_np, predictions_np, 
                    target_names=self.class_names,
                    output_dict=True,
                    zero_division=0
                )
                metrics['precision'] = report['macro avg']['precision']
                metrics['recall'] = report['macro avg']['recall']
                metrics['f1'] = report['macro avg']['f1-score']
                
                # Per-class metrics
                for i, class_name in enumerate(self.class_names):
                    if str(i) in report:
                        metrics[f'{class_name}_precision'] = report[str(i)]['precision']
                        metrics[f'{class_name}_recall'] = report[str(i)]['recall']
                        metrics[f'{class_name}_f1'] = report[str(i)]['f1-score']
            except Exception as e:
                logger.info(f"⚠️ Cannot compute classification report: {e}")
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1'] = 0.0
        
        return metrics