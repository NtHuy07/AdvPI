from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .base import BaseEvaluator
from modules.utils.logging_config import get_logger

logger = get_logger(__name__)


class TextToImageRetrievalEvaluator(BaseEvaluator):
    """
    Evaluator for text-to-image retrieval tasks
    """
    
    def __init__(
        self,
        model,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            model: Vision-language model (MedCLIP, BioMedCLIP, etc.)
            device: Device to run evaluation
        """
        super().__init__(model, device, **kwargs)
    
    def evaluate(
        self,
        image_dataloader,
        text_queries: List[str],
        ground_truth_pairs: List[Tuple[int, int]],  # (text_idx, image_idx)
        top_k_list: List[int] = [1, 5, 10, 20, 50],
        batch_size: int = 64,
        return_rankings: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform text-to-image retrieval evaluation
        
        Args:
            image_dataloader: DataLoader test data
            text_queries: List of text queries
            ground_truth_pairs: List of (text_query_idx, correct_image_idx)
            top_k_list: List of k values for Recall@k
            batch_size: Batch size for text encoding
            return_rankings: Whether to return rankings
            
        Returns:
            Dictionary contains Recall@k, MRR, Mean Rank metrics
        """
        logger.info(f"🔄 Encoding {len(image_dataloader.dataset)} images...")
        
        # Encode all images
        image_embeddings = self._encode_images(image_dataloader)
        
        logger.info(f"🔄 Encoding {len(text_queries)} text queries...")
        
        # Encode all text queries
        text_embeddings = self._encode_texts(text_queries, batch_size)
        
        logger.info(f"🔄 Computing similarities and rankings...")
        
        # Compute similarities
        similarities = cosine_similarity(
            text_embeddings.cpu().numpy(),
            image_embeddings.cpu().numpy()
        )
        
        # Compute rankings for each query
        all_rankings = []
        
        for text_idx, correct_image_idx in tqdm(ground_truth_pairs, desc="Computing rankings"):
            query_similarities = similarities[text_idx]
            
            # Sort by similarity (descending)
            sorted_indices = np.argsort(query_similarities)[::-1]
            
            # Find rank of correct image
            rank_positions = np.where(sorted_indices == correct_image_idx)[0]
            if len(rank_positions) > 0:
                rank = rank_positions[0] + 1  # 1-indexed
            else:
                rank = len(sorted_indices) + 1  # Worst case
            
            all_rankings.append(rank)
        
        # Compute metrics
        metrics = self._compute_retrieval_metrics(all_rankings, top_k_list)
        
        if return_rankings:
            metrics['rankings'] = all_rankings
            metrics['similarities'] = similarities
        
        return metrics
    
    def _encode_images(self, image_dataloader) -> torch.Tensor:
        """Encode all images in dataloader"""
        image_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(image_dataloader, desc="Encoding images"):
                batch = self._move_to_device(batch)
                
                if hasattr(self.model, 'encode_image'):
                    embeddings = self.model.encode_image(
                        batch['pixel_values'], normalize=True
                    )
                else:
                    logger.error("Model does not have `encode_image` method")
                
                image_embeddings.append(embeddings.cpu())
        
        return torch.cat(image_embeddings, dim=0)
    
    def _encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """Encode list of texts"""
        text_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
                batch_texts = texts[i:i + batch_size]
                
                if hasattr(self.model, 'encode_text'):
                    embeddings = self.model.encode_text(batch_texts, normalize=True)
                else:
                    logger.error("Model does not have `encode_text` method")
                
                text_embeddings.append(embeddings.cpu())
        
        return torch.cat(text_embeddings, dim=0)
    
    def _compute_retrieval_metrics(
        self, 
        rankings: List[int], 
        top_k_list: List[int]
    ) -> Dict[str, Any]:
        """Compute retrieval metrics"""
        metrics = {}
        
        # Recall@k
        for k in top_k_list:
            hits = sum(1 for rank in rankings if 0 < rank <= k)
            recall_at_k = hits / len(rankings)
            metrics[f'Recall@{k}'] = recall_at_k
        
        # Mean Reciprocal Rank (MRR)
        reciprocal_ranks = [1.0 / rank if rank > 0 else 0.0 for rank in rankings]
        mrr = np.mean(reciprocal_ranks)
        metrics['MRR'] = mrr
        
        # Mean Rank
        valid_rankings = [rank for rank in rankings if rank > 0]
        if valid_rankings:
            metrics['Mean_Rank'] = np.mean(valid_rankings)
            metrics['Median_Rank'] = np.median(valid_rankings)
        else:
            metrics['Mean_Rank'] = float('inf')
            metrics['Median_Rank'] = float('inf')
        
        # Success rate (percentage of queries that found correct image)
        found_count = sum(1 for rank in rankings if rank > 0)
        metrics['Success_Rate'] = found_count / len(rankings)
        
        return metrics
    
    def evaluate_image_to_text_retrieval(
        self,
        image_dataloader,
        text_queries: List[str], 
        ground_truth_pairs: List[Tuple[int, int]],  # (image_idx, text_idx)
        top_k_list: List[int] = [1, 5, 10, 20, 50],
        batch_size: int = 64,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform image-to-text retrieval evaluation (opposite of text-to-image)
        
        Args:
            image_dataloader: DataLoader test data
            text_queries: List of text queries
            ground_truth_pairs: List of (image_idx, correct_text_idx)
            top_k_list: List of k values for Recall@k
            batch_size: Batch size for text encoding
            
        Returns:
            Dictionary contains Recall@k, MRR, Mean Rank metrics
        """
        logger.info(f"🔄 Image-to-text retrieval evaluation...")
        
        # Encode images and texts
        image_embeddings = self._encode_images(image_dataloader)
        text_embeddings = self._encode_texts(text_queries, batch_size)
        
        # Compute similarities (images x texts)
        similarities = cosine_similarity(
            image_embeddings.cpu().numpy(),
            text_embeddings.cpu().numpy()
        )
        
        # Compute rankings for each image
        all_rankings = []
        
        for image_idx, correct_text_idx in tqdm(ground_truth_pairs, desc="Computing rankings"):
            image_similarities = similarities[image_idx]
            
            # Sort by similarity (descending)
            sorted_indices = np.argsort(image_similarities)[::-1]
            
            # Find rank of correct text
            rank_positions = np.where(sorted_indices == correct_text_idx)[0]
            if len(rank_positions) > 0:
                rank = rank_positions[0] + 1  # 1-indexed
            else:
                rank = len(sorted_indices) + 1  # Worst case
            
            all_rankings.append(rank)
        
        # Compute metrics
        metrics = self._compute_retrieval_metrics(all_rankings, top_k_list)
        
        # Add prefix to distinguish from text-to-image
        prefixed_metrics = {}
        for key, value in metrics.items():
            prefixed_metrics[f'I2T_{key}'] = value
        
        return prefixed_metrics
