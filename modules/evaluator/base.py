from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Any

import torch
from modules.utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators
    """
    
    def __init__(
        self,
        model,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            model: Vision-language model to evaluate
            device: Device to run evaluation ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Put model in eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
    
    @abstractmethod
    def evaluate(self, dataloader, **kwargs) -> Dict[str, Any]:
        """
        Perform evaluation
        
        Args:
            dataloader: DataLoader test data
            
        Returns:
            Dictionary contains metrics
        """
        pass
    
    def _move_to_device(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Move data to device"""
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device)
        return data
