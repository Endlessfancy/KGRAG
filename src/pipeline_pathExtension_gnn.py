#!/usr/bin/env python3
"""
KGEAR Pipeline with Real GNN-Based Path Extension - Task2

Two-stage filtering architecture:
1. PPR ranking: 2000 triplets per bridge entity (coarse filter)
2. Real GNN scoring: 50 triplets globally (fine filter)

Main improvements over simplified pipeline:
1. Real GNN model from SubgraphRAG (not heuristics)
2. PPR pre-filtering at 2000 per entity (higher than simplified's 100)
3. GNN fine-tuning on combined graph for global top-50
4. Better quality filtering with learned parameters

Pipeline stages:
1. Offline: GNN training + Freebase import
2a. Offline: 2-hop neighbor extraction → Pre-computed
2b. Offline: GNN pruning to top-K → Pre-computed (50 triplets)
2c. Online: LLM Pregeneration
2d. Online: **2-hop Path Extension with PPR + Real GNN** (NEW)
    - PPR per entity → 2000
    - GNN on combined → 50
2e. Online: Final Answer Generation (with 50 + 50 = 100 total triplets)

Uses vLLM API for LLM inference
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any

# Core components
from gnn_result_loader import GNNResultLoader
from metrics_evaluator import MetricsEvaluator
from prompt_templates import (
    build_pregeneration_prompt,
    build_final_answer_prompt,
    parse_llm_response
)
# Use vLLM version for LLM
from llm_bridge_extractor_vllm import load_model, call_llm_with_prompt
# Task2: Real GNN components
from bridge_extension_2hop_gnn import TwoHopBridgeExtensionGNN
from gnn_scorer_real import RealGNNScorer
from answer_validator import validate_llm_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGEARPipelineWithGNN:
    """
    KGEAR pipeline with real GNN-based path extension.

    Two-stage filtering:
    Stage 1 (PPR): Coarse filter per entity → 2000 triplets
    Stage 2 (GNN): Fine filter globally → 50 triplets

    Pipeline:
    1. Load pre-computed GNN triplets (50)
    2. LLM pregeneration
    3. If insufficient:
       a. PPR pre-filter: 2000 per bridge entity
       b. GNN fine-tune: 50 globally from combined
    4. Final generation with 100 total triplets (50 initial + 50 extension)

    Metrics (task.md):
    1. After initial GNN: ground_truth_hit, path_coverage
    2. After pregeneration: tokens, sufficiency, bridge_entities
    3. After extension: triplet_count, ground_truth_hit, path_coverage
    4. Final: exact_match
    """

    def __init__(
        self,
        gnn_result_path: str = "/home/haoyang/private/KGRAG/SubgraphRAG/retrieve/cwq_Jun15-05:26:49/retrieval_result.pth",
        gnn_model_path: str = None,  # Will use default from RealGNNScorer
        model_name: str = "Qwen/Qwen2.5-32B-Instruct",
        device_map: str = "auto",
        gnn_device: str = "cuda:0",
        top_k: int = 50,
        max_bridge_entities: int = 3,
        extension_fetch_limit: int = 2000,
        ppr_limit_per_entity: int = 2000,  # PPR coarse filter
        gnn_return_limit: int = 50,        # GNN fine filter
        use_graph_structure: bool = False
    ):
        """
        Initialize KGEAR pipeline with real GNN.

        Args:
            gnn_result_path: Path to pre-computed GNN results
            gnn_model_path: Path to GNN model checkpoint (default: SubgraphRAG trained model)
            model_name: HuggingFace model name for LLM
            device_map: Device mapping for LLM (ignored in vLLM mode)
            gnn_device: Device for GNN model (e.g., 'cuda:0')
            top_k: Top-K triplets from initial GNN (default: 50)
            max_bridge_entities: Max bridge entities to extract (default: 3)
            extension_fetch_limit: SPARQL fetch limit per entity (default: 2000)
            ppr_limit_per_entity: PPR filter per entity (default: 2000)
            gnn_return_limit: GNN global selection (default: 50)
            use_graph_structure: Use graph structure format in prompts
        """
        logger.info("Initializing KGEAR Pipeline with Real GNN...")

        # Load components
        self.gnn_loader = GNNResultLoader(gnn_result_path)
        self.evaluator = MetricsEvaluator()

        # Initialize PPR pre-filtering
        self.extender = TwoHopBridgeExtensionGNN()
        logger.info("✓ PPR pre-filtering initialized")

        # Initialize real GNN scorer
        logger.info(f"Loading real GNN model on {gnn_device}...")
        self.gnn_scorer = RealGNNScorer(model_path=gnn_model_path, device=gnn_device)
        logger.info("✓ Real GNN scorer initialized")

        self.top_k = top_k
        self.max_bridge_entities = max_bridge_entities
        self.extension_fetch_limit = extension_fetch_limit
        self.ppr_limit_per_entity = ppr_limit_per_entity
        self.gnn_return_limit = gnn_return_limit
        self.use_graph_structure = use_graph_structure

        # Connect to vLLM server
        logger.info(f"Connecting to vLLM server for model: {model_name}...")
        self.model, self.tokenizer = load_model(model_name=model_name, device_map=device_map)

        logger.info(f"✅ Real GNN Pipeline ready!")
        logger.info(f"  - 2-hop exploration: ENABLED")
        logger.info(f"  - PPR pre-filtering: {ppr_limit_per_entity} per entity")
        logger.info(f"  - GNN scoring: {gnn_return_limit} globally")
        logger.info(f"  - Total triplets: {top_k} (initial) + {gnn_return_limit} (extension) = {top_k + gnn_return_limit}")

    def process_question(
        self,
        question_id: str,
        question: str,
        ground_truth_answer: str = None,
        reasoning_path: List = None
    ) -> Dict[str, Any]:
        """
        Process a single question through the KGEAR pipeline with real GNN.

        Args:
            question_id: Question identifier
            question: Question text
            ground_truth_answer: Expected answer (for evaluation)
            reasoning_path: Expected reasoning triplets (for evaluation)

        Returns:
            Complete results dictionary with all metrics
        """
        start_time = time.time()

        result = {
            'question_id': question_id,
            'question': question,
            'ground_truth_answer': ground_truth_answer,
            'success': False,
            'final_answer': None,
            'method': None,
            'stages': {},
            'total_time': 0
        }

        # =====================================================================
        # Stage 1 & 2: Load Pre-computed GNN Results
        # =====================================================================

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing Question: {question_id}")
        logger.info(f"Question: {question}")
        logger.info(f"{'='*80}")

        # Load GNN results
        gnn_sample = self.gnn_loader.get_by_id(question_id)

        if not gnn_sample:
            logger.error(f"Question ID not found in GNN results: {question_id}")
            result['error'] = "Question ID not in GNN results"
            return result

        # Get top-K triplets from pre-computed GNN
        gnn_triplets = self.gnn_loader.get_top_k_triplets(question_id, k=self.top_k)

        # Use provided reasoning_path if available
        if reasoning_path is None:
            reasoning_path = self.gnn_loader.get_ground_truth_path(question_id)

        logger.info(f"Loaded {len(gnn_triplets)} top-K GNN triplets")
        if reasoning_path:
            logger.info(f"Reasoning path has {len(reasoning_path)} triplets")

        # Metric 1: Evaluate initial GNN stage
        gnn_metrics = self.evaluator.evaluate_gnn_stage(
            gnn_triplets, ground_truth_answer or "", reasoning_path or []
        )

        result['stages']['gnn_pruning'] = gnn_metrics

        logger.info(f"[Metric 1 - After Initial GNN]")
        logger.info(f"  Ground truth hit: {gnn_metrics['ground_truth_hit']}")
        logger.info(f"  Path coverage: {gnn_metrics['path_coverage']:.2%}")
        logger.info(f"  Matched triplets: {gnn_metrics['matched_reasoning_triplets']}/{gnn_metrics['num_reasoning_triplets']}")

        # =====================================================================
        # Stage 2c: LLM Pregeneration
        # =====================================================================

        logger.info(f"\n--- Stage 2c: LLM Pregeneration ---")

        # Build pregeneration prompt
        pregen_prompt = build_pregeneration_prompt(
            question, gnn_triplets, max_triplets=self.top_k, max_bridge_entities=self.max_bridge_entities,
            use_graph_structure=self.use_graph_structure
        )

        # Call LLM
        pregen_response, input_tokens, output_tokens = call_llm_with_prompt(
            pregen_prompt, self.model, self.tokenizer, max_new_tokens=800
        )

        # Parse response
        pregen_parsed = parse_llm_response(pregen_response)

        # Validate response
        validation_result = validate_llm_response(pregen_parsed, gnn_triplets, question)

        if validation_result['should_retry']:
            logger.warning(f"Validation failed for question {question_id}")
            logger.warning(f"  Validation errors: {validation_result['errors']}")
            logger.warning(f"  Forcing re-judgment as INSUFFICIENT")

            pregen_parsed['can_answer'] = False
            pregen_parsed['answer'] = 'unknown'

            if not pregen_parsed.get('bridge_entities'):
                pregen_parsed['bridge_entities'] = []

            result['validation_failed'] = True
            result['validation_errors'] = validation_result['errors']

        # Metric 2: Evaluate LLM stage
        llm_metrics = self.evaluator.evaluate_llm_stage(
            input_tokens, output_tokens, pregen_parsed['can_answer']
        )

        result['stages']['llm_pregeneration'] = llm_metrics
        result['stages']['llm_pregeneration']['raw_response'] = pregen_response[:500]

        # Record bridge entities if insufficient
        if not pregen_parsed['can_answer']:
            bridge_entities = pregen_parsed.get('bridge_entities', [])
            result['stages']['llm_pregeneration']['bridge_entities'] = bridge_entities[:self.max_bridge_entities]

        logger.info(f"[Metric 2 - After LLM Pregeneration]")
        logger.info(f"  Input tokens: {input_tokens}")
        logger.info(f"  Output tokens: {output_tokens}")
        logger.info(f"  Can answer: {pregen_parsed['can_answer']}")

        # Log bridge entities
        if not pregen_parsed['can_answer'] and 'bridge_entities' in result['stages']['llm_pregeneration']:
            logger.info(f"  Bridge entities (N={len(result['stages']['llm_pregeneration']['bridge_entities'])}): {[be.get('entity_name', 'unknown') for be in result['stages']['llm_pregeneration']['bridge_entities']]}")

        # Check if LLM can answer directly
        if pregen_parsed['can_answer']:
            logger.info(f"  Direct answer: {pregen_parsed['answer']}")

            result['success'] = True
            result['final_answer'] = pregen_parsed['answer']
            result['method'] = 'direct_pregeneration'

            # Evaluate final answer
            if ground_truth_answer:
                final_metrics = self.evaluator.evaluate_final_answer(
                    pregen_parsed['answer'], ground_truth_answer
                )
                result['stages']['final_answer'] = final_metrics

                logger.info(f"\n[Metric 4 - Final Answer]")
                logger.info(f"  Predicted: {pregen_parsed['answer']}")
                logger.info(f"  Ground truth: {ground_truth_answer}")
                logger.info(f"  Exact match: {final_metrics['exact_match']}")

            result['total_time'] = time.time() - start_time
            return result

        # =====================================================================
        # Stage 2d: Path Extension with PPR + Real GNN
        # =====================================================================

        logger.info(f"\n--- Stage 2d: Path Extension (PPR + Real GNN) ---")

        # Extract bridge entities
        bridge_entities = pregen_parsed.get('bridge_entities', [])

        if not bridge_entities:
            logger.warning("LLM said insufficient but provided no bridge entities!")
            result['error'] = "No bridge entities from LLM"
            result['total_time'] = time.time() - start_time
            return result

        logger.info(f"LLM identified {len(bridge_entities)} bridge entities:")
        for i, be in enumerate(bridge_entities, 1):
            logger.info(f"  [{i}] {be.get('entity_name', 'unknown')}")

        # Two-stage filtering: PPR → GNN
        try:
            # Step 1: PPR pre-filtering (2000 per entity)
            logger.info(f"\nStep 1: PPR pre-filtering (up to {self.ppr_limit_per_entity} per entity)...")
            ppr_filtered_triplets = self.extender.extend_from_bridge_entities(
                bridge_entities, question,
                fetch_limit=self.extension_fetch_limit,
                ppr_limit_per_entity=self.ppr_limit_per_entity
            )

            logger.info(f"PPR pre-filtering: collected {len(ppr_filtered_triplets)} triplets")
            logger.info(f"  Expected max: {self.ppr_limit_per_entity} × {len(bridge_entities)} = {self.ppr_limit_per_entity * len(bridge_entities)}")

            if not ppr_filtered_triplets:
                logger.warning("No triplets from PPR pre-filtering!")
                result['error'] = "PPR pre-filtering returned no triplets"
                result['total_time'] = time.time() - start_time
                return result

            # Step 2: GNN fine-tuning (top 50 globally)
            logger.info(f"\nStep 2: GNN fine-tuning (selecting top {self.gnn_return_limit} globally)...")
            extended_triplets = self.gnn_scorer.score_and_rank_triplets(
                ppr_filtered_triplets,
                question,
                top_k=self.gnn_return_limit
            )

            logger.info(f"GNN fine-tuning: selected {len(extended_triplets)} triplets")
            logger.info(f"  Reduction: {len(ppr_filtered_triplets)} → {len(extended_triplets)} ({len(extended_triplets)/len(ppr_filtered_triplets)*100:.1f}%)")

        except Exception as e:
            logger.error(f"Path extension failed: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = f"Path extension error: {str(e)}"
            result['total_time'] = time.time() - start_time
            return result

        # Merge with initial GNN triplets
        all_triplets = list(gnn_triplets) + extended_triplets

        logger.info(f"\nTotal triplets after merging: {len(all_triplets)} (initial {len(gnn_triplets)} + extension {len(extended_triplets)})")

        # Metric 3: Evaluate extension stage
        if ground_truth_answer:
            ext_metrics = self.evaluator.evaluate_extension_stage(
                all_triplets, ground_truth_answer, reasoning_path or []
            )

            result['stages']['path_extension'] = ext_metrics

            logger.info(f"[Metric 3 - After Path Extension]")
            logger.info(f"  Total triplets: {ext_metrics['num_triplets_after_extension']}")
            logger.info(f"  Ground truth hit: {ext_metrics['ground_truth_hit']}")
            logger.info(f"  Path coverage: {ext_metrics['path_coverage']:.2%}")

        # =====================================================================
        # Stage 2e: Final Answer Generation
        # =====================================================================

        logger.info(f"\n--- Stage 2e: Final Answer Generation ---")

        # Build final prompt
        final_prompt = build_final_answer_prompt(
            question, all_triplets, max_triplets=100, use_graph_structure=self.use_graph_structure
        )

        # Call LLM
        final_response, final_input_tokens, final_output_tokens = call_llm_with_prompt(
            final_prompt, self.model, self.tokenizer, max_new_tokens=200
        )

        # Parse final answer
        final_parsed = parse_llm_response(final_response)

        # Validate final answer
        final_validation = validate_llm_response(final_parsed, all_triplets, question)

        if final_validation['should_retry']:
            logger.warning(f"Final answer validation failed for question {question_id}")
            logger.warning(f"  Validation errors: {final_validation['errors']}")

            final_parsed['can_answer'] = False
            final_parsed['answer'] = 'unknown'

            result['final_validation_failed'] = True
            result['final_validation_errors'] = final_validation['errors']

        result['success'] = final_parsed['can_answer']
        result['final_answer'] = final_parsed.get('answer', 'unknown')
        result['method'] = 'extension_then_generation'

        # Record final generation metrics
        result['stages']['final_generation'] = {
            'input_tokens': final_input_tokens,
            'output_tokens': final_output_tokens,
            'is_sufficient': final_parsed['can_answer'],
            'raw_response': final_response[:200]
        }

        logger.info(f"  Final answer: {result['final_answer']}")
        logger.info(f"  Final sufficiency: {final_parsed['can_answer']}")
        logger.info(f"  Final generation tokens: {final_input_tokens} in + {final_output_tokens} out")

        # Metric 4: Evaluate final answer
        if ground_truth_answer:
            final_metrics = self.evaluator.evaluate_final_answer(
                result['final_answer'], ground_truth_answer
            )

            result['stages']['final_answer'] = final_metrics

            logger.info(f"\n[Metric 4 - Final Answer]")
            logger.info(f"  Predicted: {result['final_answer']}")
            logger.info(f"  Ground truth: {ground_truth_answer}")
            logger.info(f"  Exact match: {final_metrics['exact_match']}")
            logger.info(f"  Partial match: {final_metrics['partial_match']}")

        result['total_time'] = time.time() - start_time
        logger.info(f"\nTotal processing time: {result['total_time']:.2f}s")

        return result


if __name__ == "__main__":
    # Quick test
    import json

    pipeline = KGEARPipelineWithGNN(top_k=50, gnn_device='cuda:0')

    # Test with a sample question ID from GNN results
    question_ids = pipeline.gnn_loader.get_all_question_ids()

    if question_ids:
        test_id = question_ids[0]

        # Get question info
        question_text = pipeline.gnn_loader.get_question_text(test_id)
        ground_truth = pipeline.gnn_loader.get_ground_truth_answer(test_id)

        print(f"\n{'='*80}")
        print(f"Testing KGEAR Pipeline with Real GNN")
        print(f"{'='*80}")
        print(f"Question ID: {test_id}")
        print(f"Question: {question_text}")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'='*80}\n")

        # Process
        result = pipeline.process_question(test_id, question_text, ground_truth)

        # Print summary
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Final Answer: {result['final_answer']}")
        print(f"Method: {result['method']}")
        print(f"Success: {result['success']}")
        print(f"Total Time: {result['total_time']:.2f}s")

        if 'final_answer' in result.get('stages', {}):
            print(f"Exact Match: {result['stages']['final_answer']['exact_match']}")

        print(f"{'='*80}\n")

        # Save result
        with open('kgear_gnn_test_result.json', 'w') as f:
            json.dump(result, f, indent=2)

        print("Result saved to kgear_gnn_test_result.json")
    else:
        print("No questions found in GNN results!")
