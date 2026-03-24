"""
Trainer module for Vision-Language Model training

Supports:
- ENTRep Model
- MedCLIP Model
- BioMedCLIP Model
"""

from .vlm_trainer import (
    VisionLanguageTrainer,
    create_trainer_for_entrep,
    create_trainer_for_medclip,
    create_trainer_for_biomedclip,
)

__all__ = [
    'VisionLanguageTrainer',
    'create_trainer_for_entrep',
    'create_trainer_for_medclip',
    'create_trainer_for_biomedclip',
]
