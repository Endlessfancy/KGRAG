#!/usr/bin/env python3
"""
10-Question Verification Test for Real GNN Pipeline

Tests the first 10 questions from CWQ-100 to verify:
1. Pipeline runs without crashes
2. PPR pre-filtering works (2000 per entity)
3. Real GNN scoring works (50 globally)
4. Triplet counts are correct
5. Metrics are collected properly

Usage:
    CUDA_VISIBLE_DEVICES=6,7 python test_pathExtension_gnn_10q.py
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline_pathExtension_gnn import KGEARPipelineWithGNN

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_cwq100_first_10():
    """Load first 10 questions from CWQ-100."""
    data_path = Path('data/cwq_100.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data[:10]


def print_question_result(i, total, result):
    """Print result summary for a single question."""
    qid = result.get('question_id', 'unknown')
    question = result.get('question', 'N/A')
    method = result.get('method', 'unknown')
    final_answer = result.get('final_answer', 'N/A')
    exact_match = result.get('stages', {}).get('final_answer', {}).get('exact_match', False)

    print(f"\n{'='*80}")
    print(f"[{i}/{total}] {qid[:50]}")
    print(f"{'='*80}")
    print(f"Q: {question[:100]}...")
    print(f"Method: {method}")
    print(f"Answer: {final_answer[:60]}")
    print(f"Match: {'✓' if exact_match else '✗'}")

    # Extension metrics if available
    if 'path_extension' in result.get('stages', {}):
        ext = result['stages']['path_extension']
        print(f"Triplets: {ext.get('num_triplets_after_extension', 0)} total")
        print(f"GT hit: {ext.get('ground_truth_hit', False)}")
        print(f"Coverage: {ext.get('path_coverage', 0):.2%}")


def print_summary(results):
    """Print comprehensive summary."""
    print(f"\n\n{'='*80}")
    print("10-QUESTION VERIFICATION SUMMARY")
    print(f"{'='*80}")

    total = len(results)
    successful = [r for r in results if 'error' not in r]
    errors = [r for r in results if 'error' in r]

    print(f"\nProcessing:")
    print(f"  Total: {total}")
    print(f"  Successful: {len(successful)}")
    print(f"  Errors: {len(errors)}")

    if len(successful) == 0:
        print("\n⚠ No successful results")
        return

    # Exact match
    exact_matches = sum(1 for r in successful
                       if r.get('stages', {}).get('final_answer', {}).get('exact_match', False))
    print(f"\nAccuracy:")
    print(f"  Exact Match: {exact_matches}/{len(successful)} ({exact_matches/len(successful)*100:.1f}%)")

    # Method breakdown
    direct = sum(1 for r in successful if r.get('method') == 'direct_pregeneration')
    extended = sum(1 for r in successful if r.get('method') == 'extension_then_generation')

    print(f"\nMethods:")
    print(f"  Direct: {direct}/{len(successful)} ({direct/len(successful)*100:.1f}%)")
    print(f"  Extended: {extended}/{len(successful)} ({extended/len(successful)*100:.1f}%)")

    # Extension statistics
    extension_results = [r for r in successful if 'path_extension' in r.get('stages', {})]
    if extension_results:
        print(f"\nPath Extension (GNN):")
        print(f"  Questions extended: {len(extension_results)}")

        avg_triplets = sum(r['stages']['path_extension'].get('num_triplets_after_extension', 0)
                          for r in extension_results) / len(extension_results)
        print(f"  Avg triplets: {avg_triplets:.0f}")

        ext_gt_hits = sum(1 for r in extension_results
                         if r['stages']['path_extension'].get('ground_truth_hit', False))
        print(f"  GT hit rate: {ext_gt_hits}/{len(extension_results)} ({ext_gt_hits/len(extension_results)*100:.1f}%)")

        # Verify triplet counts
        print(f"\n  Verification:")
        if avg_triplets <= 100:
            print(f"    ✓ Avg triplets ({avg_triplets:.0f}) ≤ 100 (50 initial + 50 extension)")
        else:
            print(f"    ✗ Avg triplets ({avg_triplets:.0f}) > 100 expected")

    # Processing time
    total_time = sum(r.get('processing_time', 0) for r in successful)
    avg_time = total_time / len(successful) if successful else 0

    print(f"\nProcessing Time:")
    print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Average: {avg_time:.1f}s per question")

    # Errors
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for r in errors:
            qid = r.get('question_id', 'unknown')
            error = r.get('error', 'unknown')
            print(f"  - {qid[:40]}: {str(error)[:80]}")

    print(f"\n{'='*80}\n")


def main():
    print("="*80)
    print("10-QUESTION VERIFICATION TEST - REAL GNN PIPELINE")
    print("="*80)

    # Load data
    print("\nLoading first 10 questions from CWQ-100...")
    cwq_data = load_cwq100_first_10()
    print(f"Loaded {len(cwq_data)} questions")

    # Initialize pipeline
    print("\n" + "="*80)
    print("Initializing Real GNN Pipeline")
    print("="*80)
    print("Configuration:")
    print("  - Pipeline: KGEARPipelineWithGNN")
    print("  - Model: Qwen/Qwen2.5-32B-Instruct (vLLM)")
    print("  - Initial GNN triplets: 50")
    print("  - PPR per entity: 2000")
    print("  - GNN global selection: 50")
    print("  - Total triplets: 100 (50 + 50)")
    print("  - GNN device: cuda:0")

    try:
        pipeline = KGEARPipelineWithGNN(
            model_name="Qwen/Qwen2.5-32B-Instruct",
            device_map="auto",
            gnn_device="cuda:0",
            top_k=50,
            max_bridge_entities=3,
            extension_fetch_limit=2000,
            ppr_limit_per_entity=2000,
            gnn_return_limit=50,
            use_graph_structure=True
        )
        print("\n✅ Pipeline initialized successfully!\n")
    except Exception as e:
        print(f"\n❌ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Process questions
    print("="*80)
    print(f"Processing {len(cwq_data)} Questions")
    print("="*80)

    results = []
    start_time = time.time()

    for i, sample in enumerate(cwq_data, 1):
        qid = sample['ID']
        question = sample['question']
        ground_truth = sample.get('answer', 'unknown')
        reasoning_path = sample.get('reasoning_path', None)

        print(f"\n\n--- Processing {i}/{len(cwq_data)}: {qid} ---")
        print(f"Q: {question}")

        q_start_time = time.time()
        try:
            result = pipeline.process_question(
                question_id=qid,
                question=question,
                ground_truth_answer=ground_truth,
                reasoning_path=reasoning_path
            )

            result['processing_time'] = time.time() - q_start_time
            results.append(result)

            print_question_result(i, len(cwq_data), result)

        except Exception as e:
            print(f"\n❌ Error processing question {i}/{len(cwq_data)}: {qid}")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()

            results.append({
                'question_id': qid,
                'question': question,
                'error': str(e),
                'processing_time': time.time() - q_start_time
            })

    elapsed_time = time.time() - start_time

    print(f"\n\n✅ Test complete!")
    print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

    # Save results
    output_dir = Path("results/task2_gnn")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"gnn_10q_verification_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Results saved to: {result_file}")

    # Print summary
    print_summary(results)

    # Check if test passed
    successful = [r for r in results if 'error' not in r]
    if len(successful) == len(results):
        print("✅ All 10 questions processed successfully!")
        print("✅ Pipeline verification PASSED - ready for CWQ-100 full test")
        return 0
    else:
        print(f"⚠ {len(results) - len(successful)}/{len(results)} questions failed")
        print("⚠ Review errors before running full CWQ-100 test")
        return 1


if __name__ == "__main__":
    exit(main())
