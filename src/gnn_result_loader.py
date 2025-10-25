#!/usr/bin/env python3
"""
GNN Result Loader for KGEAR Pipeline

Loads and parses pre-computed GNN retrieval results from SubgraphRAG.
These results contain top-K scored triplets for each question in the CWQ test set.
"""

import torch
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class GNNResultLoader:
    """
    Loader for pre-computed GNN retrieval results.

    The GNN has already performed:
    1. 2-hop neighbor extraction from topic entity
    2. GNN-based scoring and pruning to top-K triplets

    This class provides easy access to those pre-computed results.
    """

    def __init__(
        self,
        result_path: str = "/home/haoyang/private/KGRAG/SubgraphRAG/retrieve/cwq_Jun15-05:26:49/retrieval_result.pth"
    ):
        """
        Initialize the GNN result loader.

        Args:
            result_path: Path to the retrieval_result.pth file (81.6 MB)
        """
        logger.info(f"Loading GNN results from {result_path}...")

        try:
            self.results = torch.load(result_path, map_location='cpu')
            logger.info(f"Loaded {len(self.results)} samples from GNN results")
        except Exception as e:
            logger.error(f"Failed to load GNN results: {e}")
            self.results = {}

    def get_by_id(self, question_id: str) -> Optional[Dict]:
        """
        Get GNN results for a specific question ID.

        Args:
            question_id: Question identifier (e.g., "WebQTest-983_...")

        Returns:
            Dictionary containing:
            - question: str
            - scored_triplets: [(subj, pred, obj, score), ...]
            - q_entity: [topic_entity_name]
            - a_entity: [ground_truth_answer]
            - max_path_length: int
            - target_relevant_triples: [(s, p, o), ...] (reasoning path)
        """
        if question_id not in self.results:
            logger.warning(f"Question ID not found in GNN results: {question_id}")
            return None

        return self.results[question_id]

    def get_top_k_triplets(
        self,
        question_id: str,
        k: int = 50
    ) -> List[Tuple[str, str, str, float]]:
        """
        Get top-K GNN-scored triplets for a question.

        Args:
            question_id: Question identifier
            k: Number of top triplets to return (default: 50)

        Returns:
            List of (subject, predicate, object, score) tuples
            Sorted by GNN score in descending order
        """
        sample = self.get_by_id(question_id)

        if sample is None:
            logger.warning(f"No GNN results for {question_id}, returning empty list")
            return []

        # Note: GNN results use 'scored_triples' (not 'scored_triplets')
        scored_triplets = sample.get('scored_triples', sample.get('scored_triplets', []))

        # Return top-K
        top_k = scored_triplets[:k]
        logger.info(f"Retrieved {len(top_k)} top-K triplets for {question_id}")

        return top_k

    def get_ground_truth_path(self, question_id: str) -> List[Tuple[str, str, str]]:
        """
        Get the ground truth reasoning path for a question.

        This is used for computing path coverage metrics.

        Args:
            question_id: Question identifier

        Returns:
            List of (subject, predicate, object) tuples representing the reasoning path
        """
        sample = self.get_by_id(question_id)

        if sample is None:
            return []

        return sample.get('target_relevant_triples', [])

    def get_question_text(self, question_id: str) -> Optional[str]:
        """Get the question text for a question ID."""
        sample = self.get_by_id(question_id)
        return sample.get('question') if sample else None

    def get_topic_entity(self, question_id: str) -> Optional[str]:
        """Get the topic entity for a question."""
        sample = self.get_by_id(question_id)

        if sample is None:
            return None

        q_entity = sample.get('q_entity', [])
        return q_entity[0] if q_entity else None

    def get_ground_truth_answer(self, question_id: str) -> Optional[str]:
        """Get the ground truth answer for a question."""
        sample = self.get_by_id(question_id)

        if sample is None:
            return None

        a_entity = sample.get('a_entity', [])
        return a_entity[0] if a_entity else None

    def get_max_path_length(self, question_id: str) -> Optional[int]:
        """Get the maximum path length (hop count) for a question."""
        sample = self.get_by_id(question_id)
        return sample.get('max_path_length') if sample else None

    def get_all_question_ids(self) -> List[str]:
        """Get all available question IDs in the GNN results."""
        return list(self.results.keys())

    def has_question_id(self, question_id: str) -> bool:
        """Check if a question ID exists in the GNN results."""
        return question_id in self.results

    def get_stats(self) -> Dict:
        """Get statistics about the loaded GNN results."""
        if not self.results:
            return {'total_samples': 0}

        total_samples = len(self.results)

        # Compute average triplet count
        triplet_counts = []
        path_lengths = []

        for sample in self.results.values():
            triplet_counts.append(len(sample.get('scored_triplets', [])))
            path_lengths.append(sample.get('max_path_length', 0))

        return {
            'total_samples': total_samples,
            'avg_triplets_per_sample': sum(triplet_counts) / len(triplet_counts) if triplet_counts else 0,
            'avg_path_length': sum(path_lengths) / len(path_lengths) if path_lengths else 0,
            'max_triplets': max(triplet_counts) if triplet_counts else 0,
            'min_triplets': min(triplet_counts) if triplet_counts else 0
        }


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)

    loader = GNNResultLoader()
    stats = loader.get_stats()

    print("=== GNN Result Loader Stats ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Average triplets per sample: {stats['avg_triplets_per_sample']:.1f}")
    print(f"Average path length: {stats['avg_path_length']:.1f}")
    print(f"Max triplets: {stats['max_triplets']}")
    print(f"Min triplets: {stats['min_triplets']}")

    # Test with first question
    question_ids = loader.get_all_question_ids()
    if question_ids:
        test_id = question_ids[0]
        print(f"\n=== Sample Question: {test_id} ===")
        print(f"Question: {loader.get_question_text(test_id)}")
        print(f"Topic Entity: {loader.get_topic_entity(test_id)}")
        print(f"Ground Truth: {loader.get_ground_truth_answer(test_id)}")
        print(f"Path Length: {loader.get_max_path_length(test_id)}")

        top_k = loader.get_top_k_triplets(test_id, k=5)
        print(f"\nTop-5 Triplets:")
        for i, (s, p, o, score) in enumerate(top_k, 1):
            print(f"  [{i}] {s} --{p}--> {o} (score: {score:.4f})")

        path = loader.get_ground_truth_path(test_id)
        print(f"\nReasoning Path ({len(path)} triplets):")
        for s, p, o in path:
            print(f"  {s} --{p}--> {o}")
