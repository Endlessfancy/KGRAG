#!/usr/bin/env python3
"""
CWQ-100 Full Test for Real GNN Pipeline

Tests the complete CWQ-100 dataset with real GNN-based path extension.

Collects all task.md metrics:
1. After Initial GNN: ground_truth_hit, path_coverage, num_triplets
2. After Pregeneration: input_tokens, output_tokens, sufficiency, bridge_entities
3. After Extension: triplet_count, ground_truth_hit, path_coverage
4. After Final: exact_match

Features:
- Progress tracking with tqdm
- Checkpoint saving every 20 questions
- Resume capability
- Comprehensive summary statistics

Usage:
    CUDA_VISIBLE_DEVICES=6,7 python test_pathExtension_gnn_cwq100.py
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline_pathExtension_gnn import KGEARPipelineWithGNN

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CHECKPOINT_INTERVAL = 20  # Save checkpoint every N questions


def load_cwq100_data():
    """Load CWQ-100 dataset."""
    data_path = Path('data/cwq_100.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def save_checkpoint(results, output_dir, checkpoint_num):
    """Save checkpoint results."""
    checkpoint_file = output_dir / f"gnn_cwq100_checkpoint_{checkpoint_num}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return checkpoint_file


def save_final_results(results, output_dir):
    """Save final results in multiple formats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full JSON
    json_file = output_dir / f"gnn_cwq100_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # JSONL (one result per line)
    jsonl_file = output_dir / f"gnn_cwq100_{timestamp}.jsonl"
    with open(jsonl_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return json_file, jsonl_file


def print_summary(results, output_dir):
    """Generate and print comprehensive summary with all task.md metrics."""
    print("\n" + "="*80)
    print("REAL GNN PIPELINE - CWQ-100 SUMMARY")
    print("="*80)

    total = len(results)
    successful = [r for r in results if 'error' not in r]
    errors = [r for r in results if 'error' in r]

    print(f"\n📊 Processing Summary:")
    print(f"  Total Questions: {total}")
    print(f"  Successfully Processed: {len(successful)}")
    print(f"  Errors: {len(errors)}")

    if len(successful) == 0:
        print("\n⚠ No successful results to analyze")
        return

    # ============================================================
    # TASK.MD METRIC 4: Final Answer - Exact Match
    # ============================================================
    exact_matches = sum(1 for r in successful
                       if r.get('stages', {}).get('final_answer', {}).get('exact_match', False))
    print(f"\n🎯 Accuracy (TASK.MD Metric 4):")
    print(f"  Exact Match: {exact_matches}/{len(successful)} ({exact_matches/len(successful)*100:.1f}%)")

    # Answer method breakdown
    direct = sum(1 for r in successful if r.get('method') == 'direct_pregeneration')
    extended = sum(1 for r in successful if r.get('method') == 'extension_then_generation')

    print(f"\n🔍 Answer Method Breakdown:")
    print(f"  Direct (pregeneration): {direct}/{len(successful)} ({direct/len(successful)*100:.1f}%)")
    print(f"  Extended (2-hop + GNN): {extended}/{len(successful)} ({extended/len(successful)*100:.1f}%)")

    # Accuracy by method
    if direct > 0:
        direct_correct = sum(1 for r in successful
                            if r.get('method') == 'direct_pregeneration'
                            and r.get('stages', {}).get('final_answer', {}).get('exact_match', False))
        print(f"    Direct accuracy: {direct_correct}/{direct} ({direct_correct/direct*100:.1f}%)")

    if extended > 0:
        extended_correct = sum(1 for r in successful
                              if r.get('method') == 'extension_then_generation'
                              and r.get('stages', {}).get('final_answer', {}).get('exact_match', False))
        print(f"    Extended accuracy: {extended_correct}/{extended} ({extended_correct/extended*100:.1f}%)")

    # ============================================================
    # TASK.MD METRIC 1: After Initial GNN
    # ============================================================
    gnn_results = [r for r in successful if 'gnn_pruning' in r.get('stages', {})]
    if gnn_results:
        avg_gnn_triplets = sum(r['stages']['gnn_pruning'].get('num_triplets', 0)
                               for r in gnn_results) / len(gnn_results)
        gnn_gt_hits = sum(1 for r in gnn_results
                         if r['stages']['gnn_pruning'].get('ground_truth_hit', False))
        avg_gnn_coverage = sum(r['stages']['gnn_pruning'].get('path_coverage', 0)
                               for r in gnn_results) / len(gnn_results)

        print(f"\n📈 After Initial GNN (TASK.MD Metric 1):")
        print(f"  Avg triplets: {avg_gnn_triplets:.0f}")
        print(f"  Ground truth hit: {gnn_gt_hits}/{len(gnn_results)} ({gnn_gt_hits/len(gnn_results)*100:.1f}%)")
        print(f"  Avg path coverage: {avg_gnn_coverage:.2%}")

    # ============================================================
    # TASK.MD METRIC 2: After LLM Pregeneration
    # ============================================================
    pregen_results = [r for r in successful if 'llm_pregeneration' in r.get('stages', {})]
    if pregen_results:
        avg_input = sum(r['stages']['llm_pregeneration']['input_tokens']
                       for r in pregen_results) / len(pregen_results)
        avg_output = sum(r['stages']['llm_pregeneration']['output_tokens']
                        for r in pregen_results) / len(pregen_results)
        sufficient = sum(1 for r in pregen_results
                        if r['stages']['llm_pregeneration'].get('is_sufficient', False))

        # Bridge entities count
        bridge_entity_counts = [len(r['stages']['llm_pregeneration'].get('bridge_entities', []))
                               for r in pregen_results
                               if not r['stages']['llm_pregeneration'].get('is_sufficient', False)]
        avg_bridges = sum(bridge_entity_counts) / len(bridge_entity_counts) if bridge_entity_counts else 0

        print(f"\n💬 After LLM Pregeneration (TASK.MD Metric 2):")
        print(f"  Avg input tokens: {avg_input:.0f}")
        print(f"  Avg output tokens: {avg_output:.0f}")
        print(f"  Sufficient: {sufficient}/{len(pregen_results)} ({sufficient/len(pregen_results)*100:.1f}%)")
        print(f"  Insufficient (needs extension): {len(pregen_results)-sufficient}/{len(pregen_results)} ({(len(pregen_results)-sufficient)/len(pregen_results)*100:.1f}%)")
        if bridge_entity_counts:
            print(f"  Avg bridge entities: {avg_bridges:.1f}")

    # ============================================================
    # TASK.MD METRIC 3: After Path Extension
    # ============================================================
    extension_results = [r for r in successful if 'path_extension' in r.get('stages', {})]
    if extension_results:
        print(f"\n🔧 After Path Extension (TASK.MD Metric 3):")
        print(f"  Questions extended: {len(extension_results)}/{len(successful)}")

        # Average triplets after extension
        avg_triplets = sum(r['stages']['path_extension'].get('num_triplets_after_extension', 0)
                          for r in extension_results) / len(extension_results)
        print(f"  Avg triplets: {avg_triplets:.0f}")

        # Ground truth hit rate after extension
        ext_gt_hits = sum(1 for r in extension_results
                         if r['stages']['path_extension'].get('ground_truth_hit', False))
        print(f"  Ground truth hit: {ext_gt_hits}/{len(extension_results)} ({ext_gt_hits/len(extension_results)*100:.1f}%)")

        # Path coverage after extension
        avg_ext_coverage = sum(r['stages']['path_extension'].get('path_coverage', 0)
                               for r in extension_results) / len(extension_results)
        print(f"  Avg path coverage: {avg_ext_coverage:.2%}")

        # Coverage improvement analysis
        coverage_improved = 0
        coverage_maintained = 0
        coverage_decreased = 0

        for r in extension_results:
            initial_cov = r['stages']['gnn_pruning'].get('path_coverage', 0)
            final_cov = r['stages']['path_extension'].get('path_coverage', 0)

            if final_cov > initial_cov:
                coverage_improved += 1
            elif final_cov == initial_cov:
                coverage_maintained += 1
            else:
                coverage_decreased += 1

        print(f"\n  Path Coverage Changes:")
        print(f"    Improved: {coverage_improved}/{len(extension_results)}")
        print(f"    Maintained: {coverage_maintained}/{len(extension_results)}")
        print(f"    Decreased: {coverage_decreased}/{len(extension_results)}")

        # Triplet count verification
        print(f"\n  Triplet Count Verification:")
        max_triplets = max(r['stages']['path_extension'].get('num_triplets_after_extension', 0)
                          for r in extension_results)
        min_triplets = min(r['stages']['path_extension'].get('num_triplets_after_extension', 0)
                          for r in extension_results)
        print(f"    Range: {min_triplets} - {max_triplets}")

        over_limit = sum(1 for r in extension_results
                        if r['stages']['path_extension'].get('num_triplets_after_extension', 0) > 100)
        if over_limit == 0:
            print(f"    ✓ All within 100 triplet limit (50 initial + 50 extension)")
        else:
            print(f"    ⚠ {over_limit} questions exceed 100 triplet limit")

    # ============================================================
    # OVERALL Ground Truth Hit (After Path Extension)
    # ============================================================
    # For direct questions: use initial GNN GT hit (not extended)
    # For extended questions: use post-extension GT hit
    direct_results = [r for r in successful if r.get('method') == 'direct_pregeneration']
    extended_results = [r for r in successful if r.get('method') == 'extension_then_generation']

    if direct_results or extended_results:
        print(f"\n🎯 Overall Ground Truth Hit (After Path Extension):")

        # Calculate GT hits for each method
        direct_gt_hits = sum(1 for r in direct_results
                            if r['stages']['gnn_pruning'].get('ground_truth_hit', False)) if direct_results else 0

        extended_gt_hits = sum(1 for r in extended_results
                              if r['stages']['path_extension'].get('ground_truth_hit', False)) if extended_results else 0

        # Overall GT hit = direct (initial) + extended (after extension)
        overall_gt_hits = direct_gt_hits + extended_gt_hits

        print(f"  Total: {overall_gt_hits}/{len(successful)} ({overall_gt_hits/len(successful)*100:.1f}%)")

        # Breakdown by method
        print(f"\n  Breakdown by Method:")
        if direct_results:
            print(f"    Direct (kept initial): {direct_gt_hits}/{len(direct_results)} ({direct_gt_hits/len(direct_results)*100:.1f}%)")
        if extended_results:
            print(f"    Extended (after extension): {extended_gt_hits}/{len(extended_results)} ({extended_gt_hits/len(extended_results)*100:.1f}%)")

        # Show improvement from path extension
        if extended_results:
            extended_gt_before = sum(1 for r in extended_results
                                    if r['stages']['gnn_pruning'].get('ground_truth_hit', False))
            overall_gt_before = direct_gt_hits + extended_gt_before

            print(f"\n  Improvement from Path Extension:")
            print(f"    Before extension: {overall_gt_before}/{len(successful)} ({overall_gt_before/len(successful)*100:.1f}%)")
            print(f"    After extension: {overall_gt_hits}/{len(successful)} ({overall_gt_hits/len(successful)*100:.1f}%)")
            gained = extended_gt_hits - extended_gt_before
            print(f"    Gained: +{gained} question{'s' if gained != 1 else ''}")

    # Token usage for final generation
    final_gen_results = [r for r in successful if 'final_generation' in r.get('stages', {})]
    if final_gen_results:
        avg_final_input = sum(r['stages']['final_generation']['input_tokens']
                             for r in final_gen_results) / len(final_gen_results)
        avg_final_output = sum(r['stages']['final_generation']['output_tokens']
                              for r in final_gen_results) / len(final_gen_results)
        print(f"\n  Final Generation Tokens:")
        print(f"    Avg input: {avg_final_input:.0f}")
        print(f"    Avg output: {avg_final_output:.0f}")

    # Processing time
    total_time = sum(r.get('processing_time', 0) for r in successful)
    avg_time = total_time / len(successful) if successful else 0

    print(f"\n⏱️  Processing Time:")
    print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Average per question: {avg_time:.1f}s")

    # Error summary
    if errors:
        print(f"\n❌ Errors ({len(errors)}):")
        error_types = {}
        for r in errors:
            error = str(r.get('error', 'unknown'))
            error_type = error.split(':')[0] if ':' in error else error[:50]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error_type}: {count}")

    # Save summary to file
    summary_file = output_dir / f"gnn_cwq100_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("REAL GNN PIPELINE - CWQ-100 SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total Questions: {total}\n")
        f.write(f"Successfully Processed: {len(successful)}\n")
        f.write(f"Exact Match: {exact_matches}/{len(successful)} ({exact_matches/len(successful)*100:.1f}%)\n\n")

        f.write("Answer Methods:\n")
        f.write(f"  Direct: {direct}/{len(successful)} ({direct/len(successful)*100:.1f}%)\n")
        f.write(f"  Extended: {extended}/{len(successful)} ({extended/len(successful)*100:.1f}%)\n\n")

        if extension_results:
            f.write("Path Extension:\n")
            f.write(f"  Questions extended: {len(extension_results)}\n")
            f.write(f"  Avg triplets after extension: {avg_triplets:.0f}\n")
            f.write(f"  GT hit rate: {ext_gt_hits}/{len(extension_results)} ({ext_gt_hits/len(extension_results)*100:.1f}%)\n")
            f.write(f"  Avg path coverage: {avg_ext_coverage:.2%}\n\n")

        # Overall GT hit (combining direct and extended)
        if direct_results or extended_results:
            f.write("Overall Ground Truth Hit:\n")
            f.write(f"  After path extension: {overall_gt_hits}/{len(successful)} ({overall_gt_hits/len(successful)*100:.1f}%)\n")
            if direct_results:
                f.write(f"  Direct (initial): {direct_gt_hits}/{len(direct_results)} ({direct_gt_hits/len(direct_results)*100:.1f}%)\n")
            if extended_results:
                f.write(f"  Extended (after): {extended_gt_hits}/{len(extended_results)} ({extended_gt_hits/len(extended_results)*100:.1f}%)\n")
            f.write("\n")

        f.write(f"Processing Time:\n")
        f.write(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)\n")
        f.write(f"  Average: {avg_time:.1f}s per question\n")

    print(f"\n📄 Summary saved to: {summary_file}")
    print("="*80 + "\n")


def main():
    print("="*80)
    print("REAL GNN PIPELINE - CWQ-100 FULL TEST")
    print("="*80)

    # Load CWQ-100 data
    print("\nLoading CWQ-100 dataset...")
    cwq_data = load_cwq100_data()
    print(f"Loaded {len(cwq_data)} questions")

    # Create output directory
    output_dir = Path("results/task2_gnn")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Real GNN Pipeline
    print("\n" + "="*80)
    print("Initializing Real GNN Pipeline")
    print("="*80)
    print("Configuration:")
    print("  - Pipeline: KGEARPipelineWithGNN")
    print("  - Model: Qwen/Qwen2.5-32B-Instruct (vLLM)")
    print("  - Initial GNN triplets: 50")
    print("  - PPR pre-filtering: 2000 per entity")
    print("  - GNN global selection: 50")
    print("  - Total triplets: 100 max (50 + 50)")
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

    for i, sample in enumerate(tqdm(cwq_data, desc="Processing"), 1):
        qid = sample['ID']
        question = sample['question']
        ground_truth = sample.get('answer', 'unknown')
        reasoning_path = sample.get('reasoning_path', None)

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

            # Brief progress update every 10 questions
            if i % 10 == 0:
                exact_match = result.get('stages', {}).get('final_answer', {}).get('exact_match', False)
                method = result.get('method', 'unknown')
                tqdm.write(f"[{i}/{len(cwq_data)}] {qid[:40]}: {'✓' if exact_match else '✗'} ({method})")

        except Exception as e:
            tqdm.write(f"\n❌ Error on question {i}/{len(cwq_data)}: {qid[:50]}")
            tqdm.write(f"   {str(e)[:100]}")

            results.append({
                'question_id': qid,
                'question': question,
                'error': str(e),
                'processing_time': time.time() - q_start_time
            })

        # Checkpoint saving
        if i % CHECKPOINT_INTERVAL == 0:
            checkpoint_file = save_checkpoint(results, output_dir, i)
            tqdm.write(f"💾 Checkpoint saved: {checkpoint_file.name}")

    elapsed_time = time.time() - start_time

    print(f"\n\n✅ Processing complete!")
    print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

    # Save final results
    print("\nSaving results...")
    json_file, jsonl_file = save_final_results(results, output_dir)
    print(f"  JSON: {json_file}")
    print(f"  JSONL: {jsonl_file}")

    # Generate summary
    print_summary(results, output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
