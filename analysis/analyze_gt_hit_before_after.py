#!/usr/bin/env python3
"""
Analyze Ground Truth Hit Rate: Before vs After Path Extension

Shows the impact of path extension on GT coverage by comparing:
- GT hit with initial 50 GNN triplets (before extension)
- GT hit after path extension is applied
"""

import json

def main():
    print("=" * 80)
    print("GROUND TRUTH HIT RATE: BEFORE vs AFTER PATH EXTENSION")
    print("=" * 80)

    # Load results
    print("\nLoading results from: results/task2_gnn/gnn_cwq100_20251009_100945.json")
    with open('results/task2_gnn/gnn_cwq100_20251009_100945.json', 'r') as f:
        results = json.load(f)

    # Filter successful results
    successful = [r for r in results if r.get('success', False)]
    print(f"Successfully processed: {len(successful)}/100 questions\n")

    # Split by method
    direct_results = [r for r in successful if r.get('method') == 'direct_pregeneration']
    extended_results = [r for r in successful if r.get('method') == 'extension_then_generation']

    print(f"Direct questions (no extension): {len(direct_results)}")
    print(f"Extended questions (with extension): {len(extended_results)}")

    # ================================================================
    # BEFORE PATH EXTENSION - Initial 50 GNN triplets
    # ================================================================
    print("\n" + "=" * 80)
    print("📊 BEFORE PATH EXTENSION (Initial 50 GNN Triplets)")
    print("=" * 80)

    # Direct questions: use initial GNN GT hit
    direct_gt_before = sum(1 for r in direct_results
                          if r['stages']['gnn_pruning'].get('ground_truth_hit', False))

    # Extended questions: use initial GNN GT hit (BEFORE extension)
    extended_gt_before = sum(1 for r in extended_results
                            if r['stages']['gnn_pruning'].get('ground_truth_hit', False))

    overall_gt_before = direct_gt_before + extended_gt_before

    print(f"\nOverall GT Hit: {overall_gt_before}/{len(successful)} ({overall_gt_before/len(successful)*100:.1f}%)")
    print(f"\nBreakdown:")
    print(f"  Direct questions: {direct_gt_before}/{len(direct_results)} ({direct_gt_before/len(direct_results)*100:.1f}%)")
    print(f"  Extended questions: {extended_gt_before}/{len(extended_results)} ({extended_gt_before/len(extended_results)*100:.1f}%)")

    # Average coverage
    direct_cov_before = sum(r['stages']['gnn_pruning'].get('path_coverage', 0) for r in direct_results) / len(direct_results) if direct_results else 0
    extended_cov_before = sum(r['stages']['gnn_pruning'].get('path_coverage', 0) for r in extended_results) / len(extended_results) if extended_results else 0
    overall_cov_before = sum(r['stages']['gnn_pruning'].get('path_coverage', 0) for r in successful) / len(successful)

    print(f"\nAverage Path Coverage:")
    print(f"  Overall: {overall_cov_before:.1%}")
    print(f"  Direct: {direct_cov_before:.1%}")
    print(f"  Extended: {extended_cov_before:.1%}")

    # ================================================================
    # AFTER PATH EXTENSION
    # ================================================================
    print("\n" + "=" * 80)
    print("📊 AFTER PATH EXTENSION")
    print("=" * 80)

    # Direct questions: still use initial GNN (no extension applied)
    direct_gt_after = direct_gt_before  # Same as before

    # Extended questions: use post-extension GT hit
    extended_gt_after = sum(1 for r in extended_results
                           if r['stages']['path_extension'].get('ground_truth_hit', False))

    overall_gt_after = direct_gt_after + extended_gt_after

    print(f"\nOverall GT Hit: {overall_gt_after}/{len(successful)} ({overall_gt_after/len(successful)*100:.1f}%)")
    print(f"\nBreakdown:")
    print(f"  Direct questions: {direct_gt_after}/{len(direct_results)} ({direct_gt_after/len(direct_results)*100:.1f}%)")
    print(f"    (no extension applied)")
    print(f"  Extended questions: {extended_gt_after}/{len(extended_results)} ({extended_gt_after/len(extended_results)*100:.1f}%)")
    print(f"    (after path extension)")

    # Average coverage
    direct_cov_after = direct_cov_before  # Same as before
    extended_cov_after = sum(r['stages']['path_extension'].get('path_coverage', 0) for r in extended_results) / len(extended_results) if extended_results else 0

    # Overall coverage after (weighted by method)
    overall_cov_after = (direct_cov_before * len(direct_results) + extended_cov_after * len(extended_results)) / len(successful)

    print(f"\nAverage Path Coverage:")
    print(f"  Overall: {overall_cov_after:.1%}")
    print(f"  Direct: {direct_cov_after:.1%}")
    print(f"  Extended: {extended_cov_after:.1%}")

    # ================================================================
    # IMPROVEMENT FROM PATH EXTENSION
    # ================================================================
    print("\n" + "=" * 80)
    print("📈 IMPROVEMENT FROM PATH EXTENSION")
    print("=" * 80)

    gt_gained = overall_gt_after - overall_gt_before
    gt_gained_extended = extended_gt_after - extended_gt_before
    cov_gain = overall_cov_after - overall_cov_before
    cov_gain_extended = extended_cov_after - extended_cov_before

    print(f"\nOverall (all {len(successful)} questions):")
    print(f"  GT Hit: {overall_gt_before}/{len(successful)} → {overall_gt_after}/{len(successful)}")
    print(f"  Gained: +{gt_gained} questions ({gt_gained/len(successful)*100:+.1f} percentage points)")
    print(f"  Path Coverage: {overall_cov_before:.1%} → {overall_cov_after:.1%} ({cov_gain:+.1%})")

    print(f"\nExtended questions only ({len(extended_results)} questions):")
    print(f"  GT Hit: {extended_gt_before}/{len(extended_results)} → {extended_gt_after}/{len(extended_results)}")
    print(f"  Gained: +{gt_gained_extended} questions ({gt_gained_extended/len(extended_results)*100:+.1f} percentage points)")
    print(f"  Path Coverage: {extended_cov_before:.1%} → {extended_cov_after:.1%} ({cov_gain_extended:+.1%})")

    # ================================================================
    # DETAILED ANALYSIS OF GAINED QUESTIONS
    # ================================================================
    print("\n" + "=" * 80)
    print("🔍 DETAILED ANALYSIS: Questions That Gained GT Coverage")
    print("=" * 80)

    gained_questions = []
    for r in extended_results:
        before_hit = r['stages']['gnn_pruning'].get('ground_truth_hit', False)
        after_hit = r['stages']['path_extension'].get('ground_truth_hit', False)

        if not before_hit and after_hit:
            # Get triplet counts
            num_before = r['stages']['gnn_pruning'].get('num_triplets', 0)
            num_after = r['stages']['path_extension'].get('num_triplets_after_extension', 0)

            # Get bridge entities if available
            bridge_entities = 0
            if 'llm_pregeneration' in r['stages']:
                pregen = r['stages']['llm_pregeneration']
                if isinstance(pregen.get('bridge_entities'), list):
                    bridge_entities = len(pregen['bridge_entities'])

            gained_questions.append({
                'qid': r['question_id'],
                'question': r['question'],
                'cov_before': r['stages']['gnn_pruning'].get('path_coverage', 0),
                'cov_after': r['stages']['path_extension'].get('path_coverage', 0),
                'bridge_entities': bridge_entities,
                'triplets_before': num_before,
                'triplets_after': num_after,
                'triplets_added': num_after - num_before
            })

    print(f"\nTotal questions that gained GT hit: {len(gained_questions)}")

    if gained_questions:
        print(f"\nExamples:")
        for i, q in enumerate(gained_questions[:5], 1):
            print(f"\n  [{i}] Question: {q['question'][:80]}...")
            print(f"      QID: {q['qid']}")
            print(f"      Coverage: {q['cov_before']:.1%} → {q['cov_after']:.1%}")
            print(f"      Triplets: {q['triplets_before']} → {q['triplets_after']} (+{q['triplets_added']})")
            print(f"      Bridge entities: {q['bridge_entities']}")

        if len(gained_questions) > 5:
            print(f"\n  ... and {len(gained_questions) - 5} more questions")

    # ================================================================
    # QUESTIONS THAT STILL MISSED GT
    # ================================================================
    missed_questions = []
    for r in extended_results:
        before_hit = r['stages']['gnn_pruning'].get('ground_truth_hit', False)
        after_hit = r['stages']['path_extension'].get('ground_truth_hit', False)

        if not before_hit and not after_hit:
            missed_questions.append({
                'qid': r['question_id'],
                'question': r['question'],
                'cov_before': r['stages']['gnn_pruning'].get('path_coverage', 0),
                'cov_after': r['stages']['path_extension'].get('path_coverage', 0),
                'improvement': r['stages']['path_extension'].get('path_coverage', 0) - r['stages']['gnn_pruning'].get('path_coverage', 0)
            })

    print("\n" + "=" * 80)
    print("❌ Questions That Still Missed GT Hit (Even After Extension)")
    print("=" * 80)

    print(f"\nTotal: {len(missed_questions)} questions")

    if missed_questions:
        # Sort by improvement (even if didn't reach 100%)
        missed_questions.sort(key=lambda x: x['improvement'], reverse=True)

        print(f"\nTop 5 with largest coverage improvement:")
        for i, q in enumerate(missed_questions[:5], 1):
            print(f"\n  [{i}] Question: {q['question'][:80]}...")
            print(f"      QID: {q['qid']}")
            print(f"      Coverage: {q['cov_before']:.1%} → {q['cov_after']:.1%} ({q['improvement']:+.1%})")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✅ Path extension improved GT hit from {overall_gt_before} to {overall_gt_after} (+{gt_gained} questions)")
    print(f"✅ For extended questions specifically: {extended_gt_before}/{len(extended_results)} → {extended_gt_after}/{len(extended_results)} (+{gt_gained_extended})")
    print(f"✅ Overall path coverage: {overall_cov_before:.1%} → {overall_cov_after:.1%}")
    print(f"\n💡 Path extension helps {len(gained_questions)} questions achieve full GT coverage")
    print(f"⚠️  But {len(missed_questions)} extended questions still miss GT even after extension")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
