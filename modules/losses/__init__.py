"""
Loss functions for medical vision-language models
"""

from .contrastive import ImageTextContrastiveLoss
from .supervised import ImageSuperviseLoss

__all__ = [
    'ImageTextContrastiveLoss',
    'ImageSuperviseLoss'
]
