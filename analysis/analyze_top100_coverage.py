#!/usr/bin/env python3
"""
Analyze Ground Truth Coverage: Top-100 vs Top-50 GNN Results

Compares GT coverage and path coverage between:
- Current: top-50 GNN triplets
- Proposed: top-100 GNN triplets

This helps decide if doubling the triplet count justifies the token cost.
"""

import sys
import json
import torch
from pathlib import Path
from typing import List, Tuple, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from gnn_result_loader import GNNResultLoader


def normalize_triplet(triplet: Tuple) -> Tuple[str, str, str]:
    """Normalize triplet format for comparison."""
    if len(triplet) == 3:
        return tuple(str(x) for x in triplet)
    elif len(triplet) == 4:  # (s, p, o, score)
        return tuple(str(x) for x in triplet[:3])
    return triplet


def calculate_coverage(
    top_k_triplets: List[Tuple],
    reasoning_path: List[Tuple]
) -> Tuple[bool, float, int, int]:
    """
    Calculate GT coverage metrics.

    Args:
        top_k_triplets: List of (s, p, o, score) or (s, p, o)
        reasoning_path: List of (s, p, o) ground truth triplets

    Returns:
        (gt_hit, coverage_ratio, matched_count, total_count)
    """
    if not reasoning_path:
        return False, 0.0, 0, 0

    # Normalize triplets for comparison
    top_k_set = {normalize_triplet(t) for t in top_k_triplets}
    reasoning_set = {normalize_triplet(t) for t in reasoning_path}

    # Count matches
    matched = reasoning_set & top_k_set
    matched_count = len(matched)
    total_count = len(reasoning_set)

    # GT hit = all reasoning triplets found
    gt_hit = matched_count == total_count

    # Coverage ratio
    coverage = matched_count / total_count if total_count > 0 else 0.0

    return gt_hit, coverage, matched_count, total_count


def main():
    print("=" * 80)
    print("TOP-100 vs TOP-50 GNN COVERAGE ANALYSIS")
    print("=" * 80)

    # Load GNN results
    print("\nLoading GNN results...")
    gnn_loader = GNNResultLoader()

    # Load CWQ-100 data
    print("Loading CWQ-100 dataset...")
    with open('data/cwq_100.json', 'r') as f:
        cwq_data = json.load(f)

    print(f"Loaded {len(cwq_data)} questions\n")

    # Load existing top-50 results for comparison
    print("Loading existing top-50 results...")
    with open('results/task2_gnn/gnn_cwq100_20251009_100945.json', 'r') as f:
        existing_results = json.load(f)

    # Create lookup for existing results
    existing_lookup = {r['question_id']: r for r in existing_results}

    # Analyze both top-50 and top-100
    top50_results = []
    top100_results = []
    questions_processed = 0

    print("\nAnalyzing coverage...")
    print("-" * 80)

    for sample in cwq_data:
        qid = sample['ID']

        # Get reasoning path
        reasoning_path = sample.get('reasoning_path', [])

        if not reasoning_path:
            continue

        # Get top-50 triplets
        top50 = gnn_loader.get_top_k_triplets(qid, k=50)
        if not top50:
            continue

        # Get top-100 triplets
        top100 = gnn_loader.get_top_k_triplets(qid, k=100)
        if not top100:
            continue

        # Calculate coverage for both
        gt_hit_50, cov_50, matched_50, total_50 = calculate_coverage(top50, reasoning_path)
        gt_hit_100, cov_100, matched_100, total_100 = calculate_coverage(top100, reasoning_path)

        top50_results.append({
            'qid': qid,
            'gt_hit': gt_hit_50,
            'coverage': cov_50,
            'matched': matched_50,
            'total': total_50
        })

        top100_results.append({
            'qid': qid,
            'gt_hit': gt_hit_100,
            'coverage': cov_100,
            'matched': matched_100,
            'total': total_100
        })

        questions_processed += 1

    print(f"Processed {questions_processed} questions with reasoning paths\n")

    # Calculate aggregate statistics
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Top-50 stats
    gt_hits_50 = sum(1 for r in top50_results if r['gt_hit'])
    avg_cov_50 = sum(r['coverage'] for r in top50_results) / len(top50_results)

    print(f"\n📊 Top-50 GNN Results (Current):")
    print(f"  Questions analyzed: {len(top50_results)}")
    print(f"  GT hit: {gt_hits_50}/{len(top50_results)} ({gt_hits_50/len(top50_results)*100:.1f}%)")
    print(f"  Avg path coverage: {avg_cov_50:.2%}")

    # Top-100 stats
    gt_hits_100 = sum(1 for r in top100_results if r['gt_hit'])
    avg_cov_100 = sum(r['coverage'] for r in top100_results) / len(top100_results)

    print(f"\n📊 Top-100 GNN Results (Proposed):")
    print(f"  Questions analyzed: {len(top100_results)}")
    print(f"  GT hit: {gt_hits_100}/{len(top100_results)} ({gt_hits_100/len(top100_results)*100:.1f}%)")
    print(f"  Avg path coverage: {avg_cov_100:.2%}")

    # Improvement
    print(f"\n📈 Improvement:")
    additional_hits = gt_hits_100 - gt_hits_50
    coverage_gain = avg_cov_100 - avg_cov_50
    print(f"  Additional GT hits: +{additional_hits} questions")
    print(f"  Avg coverage gain: +{coverage_gain:.2%}")
    print(f"  Relative improvement: {(additional_hits/gt_hits_50*100) if gt_hits_50 > 0 else 0:.1f}%")

    # Questions that gain GT coverage
    gained_questions = []
    for r50, r100 in zip(top50_results, top100_results):
        if not r50['gt_hit'] and r100['gt_hit']:
            gained_questions.append({
                'qid': r50['qid'],
                'cov_50': r50['coverage'],
                'cov_100': r100['coverage'],
                'matched_50': r50['matched'],
                'matched_100': r100['matched'],
                'total': r50['total']
            })

    if gained_questions:
        print(f"\n📋 Questions that GAIN GT coverage with top-100:")
        print(f"  Total: {len(gained_questions)} questions")
        print(f"\n  Details:")
        for i, q in enumerate(gained_questions[:10], 1):  # Show first 10
            print(f"    [{i}] {q['qid'][:50]}")
            print(f"        Top-50: {q['matched_50']}/{q['total']} triplets ({q['cov_50']:.1%})")
            print(f"        Top-100: {q['matched_100']}/{q['total']} triplets ({q['cov_100']:.1%})")

        if len(gained_questions) > 10:
            print(f"    ... and {len(gained_questions) - 10} more")

    # Coverage improvement (not full GT hit, but higher coverage)
    improved_coverage = []
    for r50, r100 in zip(top50_results, top100_results):
        if r100['coverage'] > r50['coverage']:
            improved_coverage.append({
                'qid': r50['qid'],
                'gain': r100['coverage'] - r50['coverage'],
                'cov_50': r50['coverage'],
                'cov_100': r100['coverage']
            })

    print(f"\n📋 Questions with IMPROVED coverage (including partial gains):")
    print(f"  Total: {len(improved_coverage)} questions")

    if improved_coverage:
        # Sort by gain
        improved_coverage.sort(key=lambda x: x['gain'], reverse=True)
        print(f"\n  Top 10 largest improvements:")
        for i, q in enumerate(improved_coverage[:10], 1):
            print(f"    [{i}] {q['qid'][:50]}")
            print(f"        Coverage: {q['cov_50']:.1%} → {q['cov_100']:.1%} (gain: +{q['gain']:.1%})")

    # Token cost analysis
    print(f"\n💰 Token Cost Analysis:")
    avg_triplet_tokens = 25  # Rough estimate: 25 tokens per triplet
    token_increase = 50 * avg_triplet_tokens
    print(f"  Estimated token increase: ~{token_increase} tokens/question")
    print(f"  For 100 questions: ~{token_increase * 100:,} additional tokens")
    print(f"  GT hit improvement: {additional_hits} questions")
    if additional_hits > 0:
        print(f"  Tokens per gained GT hit: ~{(token_increase * 100) / additional_hits:,.0f} tokens")

    # Recommendation
    print(f"\n💡 Recommendation:")
    improvement_rate = (additional_hits / len(top50_results) * 100) if top50_results else 0
    if improvement_rate >= 5:
        print(f"  ✅ CONSIDER increasing to top-100")
        print(f"     - Gains {additional_hits} GT hits (+{improvement_rate:.1f}%)")
        print(f"     - Coverage improves from {avg_cov_50:.1%} to {avg_cov_100:.1%}")
    elif improvement_rate >= 2:
        print(f"  ⚠️  MARGINAL benefit from top-100")
        print(f"     - Only gains {additional_hits} GT hits (+{improvement_rate:.1f}%)")
        print(f"     - 2x tokens for modest improvement")
    else:
        print(f"  ❌ NOT recommended to increase to top-100")
        print(f"     - Minimal gain: {additional_hits} GT hits (+{improvement_rate:.1f}%)")
        print(f"     - 2x tokens not justified")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
