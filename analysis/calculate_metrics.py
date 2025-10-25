#!/usr/bin/env python3
"""
Calculate Average Metrics from CWQ Evaluation Results

Computes metrics per task.md requirements:
1. After GNN (2b): Ground truth hit + path coverage
2. After LLM (2c): Token counts + sufficiency judgment + bridge entities
3. After Extension (2d): Triplet count + ground truth hit + path coverage
4. Final (2e): Exact match
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def calculate_metrics(results_file):
    """Calculate average metrics from full_results.jsonl"""

    # Initialize counters
    total = 0
    metrics = defaultdict(lambda: defaultdict(float))

    # Read results
    with open(results_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            result = json.loads(line)
            total += 1

            # Skip if error
            if 'error' in result:
                continue

            # Stage 1: GNN metrics
            if 'stage1_gnn' in result:
                gnn = result['stage1_gnn']
                if gnn.get('ground_truth_hit'):
                    metrics['stage1']['gnn_gt_hit'] += 1
                metrics['stage1']['path_coverage'] += gnn.get('path_coverage', 0)
                metrics['stage1']['triplet_count'] += gnn.get('triplet_count', 0)

            # Stage 2: LLM Pregeneration
            if 'stage2_pregeneration' in result:
                pregen = result['stage2_pregeneration']
                metrics['stage2']['input_tokens'] += pregen.get('input_tokens', 0)
                metrics['stage2']['output_tokens'] += pregen.get('output_tokens', 0)

                if pregen.get('can_answer'):
                    metrics['stage2']['sufficient_count'] += 1

                # Count bridge entities
                bridges = pregen.get('bridge_entities', [])
                if bridges:
                    metrics['stage2']['bridge_entity_count'] += len(bridges)
                    metrics['stage2']['questions_with_bridges'] += 1

            # Stage 3: Path Extension
            if 'stage3_extension' in result:
                ext = result['stage3_extension']
                if ext.get('triggered'):
                    metrics['stage3']['extension_triggered'] += 1
                    metrics['stage3']['extended_triplet_count'] += ext.get('triplet_count', 0)

                    if ext.get('ground_truth_hit'):
                        metrics['stage3']['gt_hit_after_extension'] += 1

                    metrics['stage3']['path_coverage_after_extension'] += ext.get('path_coverage', 0)

            # Stage 4: Final Answer
            if 'stage4_final' in result:
                final = result['stage4_final']
                if final.get('exact_match'):
                    metrics['stage4']['exact_match'] += 1
                if final.get('partial_match'):
                    metrics['stage4']['partial_match'] += 1

                metrics['stage4']['processing_time'] += final.get('processing_time_seconds', 0)

    # Calculate averages and rates
    summary = {
        'total_questions': total,
        'stage1_gnn': {
            'ground_truth_hit_rate': metrics['stage1']['gnn_gt_hit'] / total if total > 0 else 0,
            'avg_path_coverage': metrics['stage1']['path_coverage'] / total if total > 0 else 0,
            'avg_triplet_count': metrics['stage1']['triplet_count'] / total if total > 0 else 0,
        },
        'stage2_pregeneration': {
            'avg_input_tokens': metrics['stage2']['input_tokens'] / total if total > 0 else 0,
            'avg_output_tokens': metrics['stage2']['output_tokens'] / total if total > 0 else 0,
            'sufficiency_rate': metrics['stage2']['sufficient_count'] / total if total > 0 else 0,
            'avg_bridge_entities_per_question': (
                metrics['stage2']['bridge_entity_count'] / metrics['stage2']['questions_with_bridges']
                if metrics['stage2']['questions_with_bridges'] > 0 else 0
            ),
            'questions_with_bridge_entities': int(metrics['stage2']['questions_with_bridges']),
        },
        'stage3_extension': {
            'extension_trigger_rate': metrics['stage3']['extension_triggered'] / total if total > 0 else 0,
            'avg_extended_triplets': (
                metrics['stage3']['extended_triplet_count'] / metrics['stage3']['extension_triggered']
                if metrics['stage3']['extension_triggered'] > 0 else 0
            ),
            'gt_hit_after_extension_rate': (
                metrics['stage3']['gt_hit_after_extension'] / metrics['stage3']['extension_triggered']
                if metrics['stage3']['extension_triggered'] > 0 else 0
            ),
            'avg_path_coverage_after_extension': (
                metrics['stage3']['path_coverage_after_extension'] / metrics['stage3']['extension_triggered']
                if metrics['stage3']['extension_triggered'] > 0 else 0
            ),
            'extension_triggered_count': int(metrics['stage3']['extension_triggered']),
        },
        'stage4_final': {
            'exact_match_rate': metrics['stage4']['exact_match'] / total if total > 0 else 0,
            'partial_match_rate': metrics['stage4']['partial_match'] / total if total > 0 else 0,
            'avg_processing_time_seconds': metrics['stage4']['processing_time'] / total if total > 0 else 0,
        },

        # Raw counts for reference
        'raw_counts': {
            'exact_match': int(metrics['stage4']['exact_match']),
            'partial_match': int(metrics['stage4']['partial_match']),
            'gnn_gt_hit': int(metrics['stage1']['gnn_gt_hit']),
            'sufficient_judgments': int(metrics['stage2']['sufficient_count']),
            'extension_triggered': int(metrics['stage3']['extension_triggered']),
        }
    }

    return summary

def print_metrics(summary):
    """Print metrics in task.md format"""
    print("="*80)
    print("KGEAR PIPELINE METRICS (per task.md)")
    print("="*80)
    print(f"Total Questions Evaluated: {summary['total_questions']}")
    print()

    print("STAGE 1: GNN-based Pruning (task.md 2b)")
    print("-"*80)
    gnn = summary['stage1_gnn']
    print(f"  [GNN pruning GT Hit]:               {gnn['ground_truth_hit_rate']:.2%}")
    print(f"  [GNN pruning Reasoning Path Coverage]: {gnn['avg_path_coverage']:.2%}")
    print(f"  Average Triplet Count:              {gnn['avg_triplet_count']:.1f}")
    print()

    print("STAGE 2: LLM Pregeneration (task.md 2c)")
    print("-"*80)
    pregen = summary['stage2_pregeneration']
    print(f"  [Pre Generation input token]:       {pregen['avg_input_tokens']:.0f} tokens")
    print(f"  [Pre Generation output token]:      {pregen['avg_output_tokens']:.0f} tokens")
    print(f"  [Sufficiency judgement]:            {pregen['sufficiency_rate']:.2%} (sufficient)")
    print(f"  [Bridge Entities]:                  {pregen['questions_with_bridge_entities']} questions with bridges")
    print(f"  Average Bridge Entities (when used): {pregen['avg_bridge_entities_per_question']:.2f}")
    print()

    print("STAGE 3: Path Extension (task.md 2d)")
    print("-"*80)
    ext = summary['stage3_extension']
    print(f"  Extension Trigger Rate:             {ext['extension_trigger_rate']:.2%}")
    print(f"  Extension Triggered Count:          {ext['extension_triggered_count']} questions")
    print(f"  [Path extension triplets]:          {ext['avg_extended_triplets']:.0f} avg")
    print(f"  [Path extension GT Hit]:            {ext['gt_hit_after_extension_rate']:.2%}")
    print(f"  [Path extension Coverage]:          {ext['avg_path_coverage_after_extension']:.2%}")
    print()

    print("STAGE 4: Final Answer (task.md 2e)")
    print("-"*80)
    final = summary['stage4_final']
    print(f"  [Generation Exact Match]:           {final['exact_match_rate']:.2%}")
    print(f"  Partial Match Rate:                 {final['partial_match_rate']:.2%}")
    print(f"  Average Processing Time:            {final['avg_processing_time_seconds']:.2f}s")
    print()

    print("RAW COUNTS")
    print("-"*80)
    counts = summary['raw_counts']
    print(f"  Exact Match:                        {counts['exact_match']}")
    print(f"  Partial Match:                      {counts['partial_match']}")
    print(f"  GNN GT Hit:                         {counts['gnn_gt_hit']}")
    print(f"  Sufficient Judgments:               {counts['sufficient_judgments']}")
    print(f"  Extension Triggered:                {counts['extension_triggered']}")
    print("="*80)

def main():
    results_file = Path('results/test_v1/full_results.jsonl')

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return 1

    print("Calculating metrics from evaluation results...")
    print()

    summary = calculate_metrics(results_file)
    print_metrics(summary)

    # Save to JSON
    output_file = Path('results/test_v1/average_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Detailed metrics saved to: {output_file}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
