#!/usr/bin/env python3
"""
Metrics Evaluator for KGEAR Pipeline

Implements all 4 metrics specified in task.md:
1. After GNN pruning: ground truth hit + reasoning path coverage
2. After LLM pregeneration: token count + sufficiency judgment
3. After path extension: triplet count + ground truth hit + path coverage
4. Final answer: exact match with ground truth
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MetricsEvaluator:
    """
    Evaluator for KGEAR pipeline metrics.

    Tracks performance at each stage:
    - Stage 1 (GNN): Coverage of ground truth and reasoning path
    - Stage 2 (LLM): Token usage and sufficiency decision
    - Stage 3 (Extension): Post-expansion coverage
    - Stage 4 (Final): Answer accuracy
    """

    def __init__(self):
        """Initialize the metrics evaluator."""
        self.results = []

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        return str(text).lower().strip()

    def calculate_ground_truth_hit(
        self,
        triplets: List[Tuple],
        answer: str
    ) -> bool:
        """
        Check if ground truth answer appears in any triplet.

        Args:
            triplets: List of (subj, pred, obj) or (subj, pred, obj, score) tuples
            answer: Ground truth answer string

        Returns:
            True if answer found in any subject or object, False otherwise
        """
        if not answer or not triplets:
            return False

        answer_norm = self.normalize_text(answer)

        for triplet in triplets:
            # Handle both 3-tuple and 4-tuple (with score)
            subj = self.normalize_text(triplet[0])
            obj = self.normalize_text(triplet[2])

            # Check for partial match
            if answer_norm in subj or subj in answer_norm:
                logger.debug(f"Ground truth '{answer}' found in subject: {triplet[0]}")
                return True

            if answer_norm in obj or obj in answer_norm:
                logger.debug(f"Ground truth '{answer}' found in object: {triplet[2]}")
                return True

        return False

    def calculate_path_coverage(
        self,
        retrieved_triplets: List[Tuple],
        reasoning_path: List[Tuple[str, str, str]]
    ) -> float:
        """
        Calculate reasoning path coverage: (matched triplets) / (total in path).

        Args:
            retrieved_triplets: Triplets retrieved by system
            reasoning_path: Ground truth reasoning path

        Returns:
            Coverage ratio between 0.0 and 1.0
        """
        if not reasoning_path:
            logger.warning("No reasoning path provided, returning 0.0 coverage")
            return 0.0

        if not retrieved_triplets:
            logger.warning("No retrieved triplets, returning 0.0 coverage")
            return 0.0

        # Normalize retrieved triplets to (s, p, o) tuples
        retrieved_set = set()
        for triplet in retrieved_triplets:
            s = self.normalize_text(triplet[0])
            p = self.normalize_text(triplet[1])
            o = self.normalize_text(triplet[2])
            retrieved_set.add((s, p, o))

        # Check which reasoning path triplets are in retrieved set
        matched = 0
        for s, p, o in reasoning_path:
            s_norm = self.normalize_text(s)
            p_norm = self.normalize_text(p)
            o_norm = self.normalize_text(o)

            if (s_norm, p_norm, o_norm) in retrieved_set:
                matched += 1
                logger.debug(f"Matched reasoning triplet: {s} --{p}--> {o}")

        coverage = matched / len(reasoning_path)
        logger.info(f"Path coverage: {matched}/{len(reasoning_path)} = {coverage:.2%}")

        return coverage

    def evaluate_gnn_stage(
        self,
        gnn_triplets: List[Tuple],
        ground_truth_answer: str,
        reasoning_path: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        """
        Metric 1: Evaluate GNN pruning stage.

        Tests:
        - Whether ground truth answer appears in top-K GNN triplets
        - How many reasoning path triplets are covered

        Args:
            gnn_triplets: Top-K triplets from GNN
            ground_truth_answer: Expected answer
            reasoning_path: Ground truth reasoning triplets

        Returns:
            Dictionary with:
            - ground_truth_hit: bool
            - path_coverage: float (0.0-1.0)
            - num_triplets: int
            - num_reasoning_triplets: int
            - matched_reasoning_triplets: int
        """
        ground_truth_hit = self.calculate_ground_truth_hit(
            gnn_triplets, ground_truth_answer
        )

        path_coverage = self.calculate_path_coverage(
            gnn_triplets, reasoning_path
        )

        matched_count = int(path_coverage * len(reasoning_path)) if reasoning_path else 0

        return {
            'ground_truth_hit': ground_truth_hit,
            'path_coverage': path_coverage,
            'num_triplets': len(gnn_triplets),
            'num_reasoning_triplets': len(reasoning_path),
            'matched_reasoning_triplets': matched_count
        }

    def evaluate_llm_stage(
        self,
        input_tokens: int,
        output_tokens: int,
        is_sufficient: bool
    ) -> Dict[str, Any]:
        """
        Metric 2: Evaluate LLM pregeneration stage.

        Tracks:
        - Input token count
        - Output token count
        - Sufficiency judgment (can answer directly or needs extension)

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            is_sufficient: Whether LLM judged evidence sufficient

        Returns:
            Dictionary with token counts and sufficiency flag
        """
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'is_sufficient': is_sufficient,
            'decision': 'direct_answer' if is_sufficient else 'needs_extension'
        }

    def evaluate_extension_stage(
        self,
        extended_triplets: List[Tuple],
        ground_truth_answer: str,
        reasoning_path: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        """
        Metric 3: Evaluate path extension stage.

        Tests after bridge entity expansion:
        - Total triplet count
        - Whether ground truth now appears
        - Updated reasoning path coverage

        Args:
            extended_triplets: All triplets after extension
            ground_truth_answer: Expected answer
            reasoning_path: Ground truth reasoning triplets

        Returns:
            Dictionary with coverage metrics after extension
        """
        ground_truth_hit = self.calculate_ground_truth_hit(
            extended_triplets, ground_truth_answer
        )

        path_coverage = self.calculate_path_coverage(
            extended_triplets, reasoning_path
        )

        matched_count = int(path_coverage * len(reasoning_path)) if reasoning_path else 0

        return {
            'num_triplets_after_extension': len(extended_triplets),
            'ground_truth_hit': ground_truth_hit,
            'path_coverage': path_coverage,
            'num_reasoning_triplets': len(reasoning_path),
            'matched_reasoning_triplets': matched_count
        }

    def evaluate_final_answer(
        self,
        predicted_answer: str,
        ground_truth_answer: str
    ) -> Dict[str, Any]:
        """
        Metric 4: Evaluate final answer.

        Checks exact match between predicted and ground truth.

        Args:
            predicted_answer: Answer generated by LLM
            ground_truth_answer: Expected answer

        Returns:
            Dictionary with exact match flag and normalized strings
        """
        pred_norm = self.normalize_text(predicted_answer)
        truth_norm = self.normalize_text(ground_truth_answer)

        # Exact match
        exact_match = (pred_norm == truth_norm)

        # Partial match (for softer evaluation)
        partial_match = (
            (pred_norm in truth_norm) or
            (truth_norm in pred_norm)
        )

        return {
            'exact_match': exact_match,
            'partial_match': partial_match,
            'predicted_answer': predicted_answer,
            'ground_truth_answer': ground_truth_answer,
            'predicted_normalized': pred_norm,
            'ground_truth_normalized': truth_norm
        }

    def compute_aggregate_metrics(
        self,
        all_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute aggregate metrics across all questions.

        Args:
            all_results: List of per-question result dictionaries

        Returns:
            Aggregated statistics
        """
        if not all_results:
            return {}

        total = len(all_results)

        # GNN stage metrics
        gnn_hits = sum(1 for r in all_results if r.get('after_gnn', {}).get('ground_truth_hit', False))
        avg_gnn_coverage = sum(r.get('after_gnn', {}).get('path_coverage', 0.0) for r in all_results) / total

        # LLM stage metrics
        avg_input_tokens = sum(r.get('after_llm', {}).get('input_tokens', 0) for r in all_results) / total
        avg_output_tokens = sum(r.get('after_llm', {}).get('output_tokens', 0) for r in all_results) / total
        sufficient_count = sum(1 for r in all_results if r.get('after_llm', {}).get('is_sufficient', False))

        # Extension stage metrics (only for questions that needed extension)
        extended = [r for r in all_results if 'after_extension' in r]
        if extended:
            ext_hits = sum(1 for r in extended if r.get('after_extension', {}).get('ground_truth_hit', False))
            avg_ext_coverage = sum(r.get('after_extension', {}).get('path_coverage', 0.0) for r in extended) / len(extended)
        else:
            ext_hits = 0
            avg_ext_coverage = 0.0

        # Final answer metrics
        exact_matches = sum(1 for r in all_results if r.get('final_answer_eval', {}).get('exact_match', False))
        partial_matches = sum(1 for r in all_results if r.get('final_answer_eval', {}).get('partial_match', False))

        return {
            'total_questions': total,
            'gnn_stage': {
                'ground_truth_hit_rate': gnn_hits / total,
                'avg_path_coverage': avg_gnn_coverage,
                'hit_count': gnn_hits
            },
            'llm_stage': {
                'avg_input_tokens': avg_input_tokens,
                'avg_output_tokens': avg_output_tokens,
                'sufficient_rate': sufficient_count / total,
                'sufficient_count': sufficient_count,
                'extension_needed_count': total - sufficient_count
            },
            'extension_stage': {
                'questions_extended': len(extended),
                'ground_truth_hit_rate': ext_hits / len(extended) if extended else 0.0,
                'avg_path_coverage': avg_ext_coverage,
                'hit_count': ext_hits
            },
            'final_answer': {
                'exact_match_rate': exact_matches / total,
                'partial_match_rate': partial_matches / total,
                'exact_match_count': exact_matches,
                'partial_match_count': partial_matches
            }
        }


if __name__ == "__main__":
    # Test the evaluator
    logging.basicConfig(level=logging.INFO)

    evaluator = MetricsEvaluator()

    # Test triplets
    triplets = [
        ('Brad Paisley', 'people.person.profession', 'Singer', 0.95),
        ('Brad Paisley', 'people.person.education', 'Belmont University', 0.90),
        ('University of Missouri', 'education.institution.students_graduates', 'Brad Paisley', 0.85)
    ]

    reasoning_path = [
        ('Brad Paisley', 'people.person.education', 'Belmont University'),
        ('Belmont University', 'location.location.state', 'Tennessee')
    ]

    # Test GNN stage
    gnn_metrics = evaluator.evaluate_gnn_stage(
        triplets, 'Belmont University', reasoning_path
    )
    print("=== GNN Stage Metrics ===")
    print(f"Ground truth hit: {gnn_metrics['ground_truth_hit']}")
    print(f"Path coverage: {gnn_metrics['path_coverage']:.2%}")
    print(f"Matched: {gnn_metrics['matched_reasoning_triplets']}/{gnn_metrics['num_reasoning_triplets']}")

    # Test LLM stage
    llm_metrics = evaluator.evaluate_llm_stage(2500, 150, True)
    print("\n=== LLM Stage Metrics ===")
    print(f"Input tokens: {llm_metrics['input_tokens']}")
    print(f"Output tokens: {llm_metrics['output_tokens']}")
    print(f"Decision: {llm_metrics['decision']}")

    # Test final answer
    final_metrics = evaluator.evaluate_final_answer(
        'Belmont University', 'Belmont University'
    )
    print("\n=== Final Answer Metrics ===")
    print(f"Exact match: {final_metrics['exact_match']}")
    print(f"Partial match: {final_metrics['partial_match']}")
