#!/usr/bin/env python3
"""
2-Hop Bridge Extension with PPR + Real GNN Scoring

Two-stage filtering architecture:
1. PPR ranking: 2000 triplets per bridge entity (coarse filter)
2. GNN scoring: 50 triplets globally (fine filter, done in pipeline)

This module handles Stage 1 (PPR filtering):
- Fetches 1-hop neighbors from bridge entity
- Extracts intermediate entities
- Fetches 2-hop neighbors from intermediate entities
- Applies PPR ranking to all collected triplets
- Returns top 2000 per entity for GNN scoring

Key differences from simplified version:
1. Higher PPR limit: 2000 per entity (vs 100)
2. No global limit here (GNN will apply global limit of 50)
3. Returns raw PPR-filtered triplets for GNN to process
"""

import re
import logging
from typing import List, Tuple, Dict, Set
from collections import defaultdict

from bridge_extension_improved import (
    ImprovedBridgeExtension,
    DEFAULT_FETCH_LIMIT,
    MID_PATTERN
)

logger = logging.getLogger(__name__)

# PPR limits for GNN pipeline
PPR_RETURN_LIMIT_PER_ENTITY = 2000  # PPR coarse filter: 2000 per entity
INTERMEDIATE_ENTITY_LIMIT = 20  # Max intermediate entities to explore per bridge


class TwoHopBridgeExtensionGNN(ImprovedBridgeExtension):
    """
    2-hop bridge extension with PPR ranking for GNN pipeline.

    Two-stage filtering:
    1. PPR pre-filtering (this module): 2000 triplets per entity
    2. GNN fine-tuning (in pipeline): 50 triplets globally

    Exploration strategy:
    1. Fetch 1-hop neighbors from bridge entity
    2. Extract intermediate entities (filter by relevance)
    3. Fetch 2-hop neighbors from intermediate entities
    4. Combine 1-hop and 2-hop triplets
    5. Apply PPR ranking → top 2000 per entity
    6. Return all for GNN scoring (up to 2000 × N entities)
    """

    def __init__(self):
        super().__init__()
        logger.info("Initialized TwoHopBridgeExtensionGNN for PPR + GNN pipeline")

    def extract_intermediate_entities(
        self,
        one_hop_neighbors: List[Tuple[str, str, str, str]],
        question: str,
        limit: int = INTERMEDIATE_ENTITY_LIMIT
    ) -> List[Tuple[str, float]]:
        """
        Extract promising intermediate entities from 1-hop neighbors.

        Select entities that:
        1. Are proper Freebase MIDs (not literals)
        2. Have relevant relations
        3. Match question keywords

        Args:
            one_hop_neighbors: List of (subject, relation, object, direction) tuples
            question: Question text for relevance matching
            limit: Max intermediate entities to return

        Returns:
            List of (entity_mid, relevance_score) tuples, sorted by relevance
        """
        entity_scores = defaultdict(float)
        question_lower = question.lower()
        question_words = set(question_lower.split())

        for subj, rel, obj, direction in one_hop_neighbors:
            # Only consider objects from outgoing edges as intermediate entities
            if direction != "out":
                continue

            # Must be a valid Freebase MID
            if not MID_PATTERN.match(obj):
                continue

            # Skip CVT nodes (we handle them separately)
            if re.match(r'^[mg]\.[0-9a-z_]+$', obj) and len(obj) > 12:
                continue

            # Compute relevance score
            score = 0.0

            # 1. Relation importance (from PPR weights)
            rel_lower = rel.lower()
            if any(keyword in rel_lower for keyword in ['education', 'employment', 'profession', 'nationality']):
                score += 0.3
            if any(keyword in rel_lower for keyword in ['location', 'country', 'city']):
                score += 0.25
            if any(keyword in rel_lower for keyword in ['film', 'tv', 'actor', 'director']):
                score += 0.25

            # 2. Object name relevance (if it has a readable name)
            obj_words = set(obj.lower().replace('_', ' ').split('.'))
            common_words = question_words & obj_words
            if common_words:
                score += len(common_words) * 0.2

            # 3. Prefer certain relation patterns
            if direction == "out":
                score += 0.1

            entity_scores[obj] = max(entity_scores[obj], score)

        # Sort by score and return top-K
        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        top_entities = sorted_entities[:limit]

        logger.info(f"Selected {len(top_entities)} intermediate entities from {len(entity_scores)} candidates")
        if top_entities:
            logger.debug(f"Top intermediate entity: {top_entities[0][0]} (score: {top_entities[0][1]:.3f})")

        return top_entities

    def get_2hop_neighbors(
        self,
        entity_mid: str,
        question: str,
        fetch_limit_1hop: int = 500,
        fetch_limit_2hop_per_intermediate: int = 100
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Get 2-hop neighbors by:
        1. Fetching 1-hop neighbors
        2. Extracting intermediate entities
        3. Fetching 2-hop from intermediates

        Args:
            entity_mid: Starting entity MID
            question: Question for relevance filtering
            fetch_limit_1hop: Limit for 1-hop queries
            fetch_limit_2hop_per_intermediate: Limit for 2-hop queries per intermediate

        Returns:
            Tuple of (one_hop_triplets, two_hop_triplets)
        """
        logger.info(f"Starting 2-hop exploration from {entity_mid}")

        # Step 1: Get 1-hop neighbors
        one_hop_neighbors = self.get_1hop_neighbors(
            entity_mid,
            direction="both",
            limit=fetch_limit_1hop
        )
        logger.info(f"Collected {len(one_hop_neighbors)} 1-hop neighbors")

        # Step 2: Extract intermediate entities
        intermediate_entities = self.extract_intermediate_entities(
            one_hop_neighbors,
            question,
            limit=INTERMEDIATE_ENTITY_LIMIT
        )

        if not intermediate_entities:
            logger.warning("No intermediate entities found for 2-hop exploration")
            return one_hop_neighbors, []

        # Step 3: Fetch 2-hop neighbors from intermediate entities
        two_hop_neighbors = []
        seen_triplets = set()  # Deduplication

        for intermediate_mid, relevance_score in intermediate_entities:
            try:
                # Fetch neighbors of intermediate entity
                intermediate_neighbors = self.get_1hop_neighbors(
                    intermediate_mid,
                    direction="both",
                    limit=fetch_limit_2hop_per_intermediate
                )

                # Add to 2-hop results with deduplication
                for triplet in intermediate_neighbors:
                    triplet_key = (triplet[0], triplet[1], triplet[2])
                    if triplet_key not in seen_triplets:
                        two_hop_neighbors.append(triplet)
                        seen_triplets.add(triplet_key)

            except Exception as e:
                logger.warning(f"Failed to fetch 2-hop from {intermediate_mid}: {e}")
                continue

        logger.info(f"Collected {len(two_hop_neighbors)} unique 2-hop neighbors from {len(intermediate_entities)} intermediate entities")

        return one_hop_neighbors, two_hop_neighbors

    def extend_from_entity_with_ppr(
        self,
        entity_mid: str,
        entity_name: str,
        question: str,
        fetch_limit: int = DEFAULT_FETCH_LIMIT,
        ppr_limit: int = PPR_RETURN_LIMIT_PER_ENTITY
    ) -> List[Tuple]:
        """
        Extend from entity with 2-hop exploration and PPR ranking.

        Process:
        1. Get CVT relations and expand (pseudo-2-hop)
        2. Get 1-hop and 2-hop neighbors
        3. Combine all triplets
        4. Rank using PPR
        5. Return top 2000 for GNN scoring

        Args:
            entity_mid: Entity MID to extend from
            entity_name: Entity name (for logging and scoring)
            question: Question for relevance scoring
            fetch_limit: Total fetch limit (distributed between 1-hop and 2-hop)
            ppr_limit: Number of triplets to return after PPR ranking (2000)

        Returns:
            Top 2000 PPR-ranked triplets for GNN scoring
        """
        if not entity_mid:
            return []

        logger.info(f"PPR pre-filtering from {entity_name} ({entity_mid})")

        # Detect question type for better ranking
        question_types = self.detect_question_type(question)
        logger.info(f"Question types: {question_types}")

        # Step 1: Get CVT relations and expand
        cvt_relations = self.get_cvt_relations(entity_mid, limit=50)
        cvt_expanded = self.expand_cvt_nodes(cvt_relations, limit_per_cvt=50)
        logger.info(f"CVT expansion: {len(cvt_expanded)} triplets")

        # Step 2: Get 1-hop and 2-hop neighbors
        one_hop_limit = min(500, fetch_limit // 2)  # Reserve half for 1-hop
        two_hop_limit_per_intermediate = 100

        one_hop_neighbors, two_hop_neighbors = self.get_2hop_neighbors(
            entity_mid,
            question,
            fetch_limit_1hop=one_hop_limit,
            fetch_limit_2hop_per_intermediate=two_hop_limit_per_intermediate
        )

        # Step 3: Combine all triplets
        all_triplets = cvt_expanded + one_hop_neighbors + two_hop_neighbors
        logger.info(f"Total triplets before PPR: {len(all_triplets)} (CVT: {len(cvt_expanded)}, 1-hop: {len(one_hop_neighbors)}, 2-hop: {len(two_hop_neighbors)})")

        # Step 4: Compute PPR scores and rank
        scored_triplets = []
        for triplet in all_triplets:
            score = self.compute_ppr_score(triplet, question, question_types, entity_name)
            scored_triplets.append((score, triplet))

        # Sort by score (descending)
        scored_triplets.sort(key=lambda x: x[0], reverse=True)

        # Step 5: Return top 2000 for GNN scoring
        top_triplets = [triplet for score, triplet in scored_triplets[:ppr_limit]]

        logger.info(f"PPR pre-filtering: returning top {len(top_triplets)} triplets (limit: {ppr_limit})")

        # Log top-ranked triplets for debugging
        if scored_triplets:
            logger.debug("Top 3 PPR-ranked triplets:")
            for i, (score, triplet) in enumerate(scored_triplets[:3]):
                _, rel, obj, _ = triplet
                logger.debug(f"  {i+1}. Score: {score:.3f} | {rel} -> {obj[:50]}")

        return top_triplets

    def extend_from_bridge_entities(
        self,
        bridge_entities: List[Dict],
        question: str,
        fetch_limit: int = None,
        ppr_limit_per_entity: int = None
    ) -> List[Tuple[str, str, str]]:
        """
        Extend from multiple bridge entities with PPR pre-filtering.

        Two-stage filtering:
        1. PPR per entity → 2000 triplets (this method)
        2. GNN on combined → 50 triplets (done in pipeline)

        Args:
            bridge_entities: List of bridge entity dicts from LLM
            question: Question text
            fetch_limit: Per-entity fetch limit (default: 2000)
            ppr_limit_per_entity: PPR limit per entity (default: 2000)

        Returns:
            Combined list of PPR-filtered triplets (up to 2000 × N entities)
            These will be fed to GNN for final top-50 selection
        """
        if fetch_limit is None:
            fetch_limit = DEFAULT_FETCH_LIMIT
        if ppr_limit_per_entity is None:
            ppr_limit_per_entity = PPR_RETURN_LIMIT_PER_ENTITY

        all_triplets = []

        logger.info(f"PPR pre-filtering from {len(bridge_entities)} bridge entities")
        logger.info(f"Limits: fetch={fetch_limit}, PPR_per_entity={ppr_limit_per_entity}")
        logger.info(f"Expected max triplets before GNN: {ppr_limit_per_entity * len(bridge_entities)}")

        for bridge_dict in bridge_entities:
            entity_name = bridge_dict.get('entity_name', '')
            if not entity_name:
                logger.warning(f"Bridge entity missing name: {bridge_dict}")
                continue

            entity_mid = bridge_dict.get('entity_mid')
            if not entity_mid:
                if MID_PATTERN.match(entity_name):
                    entity_mid = entity_name
                    logger.info(f"  Detected MID from LLM: {entity_mid}")
                else:
                    # entity_name is text, look up the MID from Virtuoso
                    from entity_verifier_improved import verify_bridge_entity_mid
                    result = verify_bridge_entity_mid(name=entity_name)
                    entity_mid = result.get('mid')
                    if entity_mid:
                        logger.info(f"  Resolved '{entity_name}' to MID: {entity_mid}")

            if not entity_mid:
                logger.warning(f"Could not find MID for entity: {entity_name}")
                continue

            # PPR pre-filtering for this entity
            try:
                triplets = self.extend_from_entity_with_ppr(
                    entity_mid=entity_mid,
                    entity_name=entity_name,
                    question=question,
                    fetch_limit=fetch_limit,
                    ppr_limit=ppr_limit_per_entity
                )

                all_triplets.extend(triplets)
                logger.info(f"PPR filtered {len(triplets)} triplets from '{entity_name}' (total so far: {len(all_triplets)})")

            except Exception as e:
                logger.error(f"Failed to extend from '{entity_name}': {e}")
                continue

        logger.info(f"PPR pre-filtering complete: {len(all_triplets)} triplets from {len(bridge_entities)} entities")
        logger.info(f"These triplets will be fed to GNN for final top-50 selection")

        return all_triplets


if __name__ == "__main__":
    # Test PPR pre-filtering for GNN pipeline
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    extender = TwoHopBridgeExtensionGNN()

    # Test on Brad Paisley (education example)
    test_mid = "m.03gr7w"  # Brad Paisley
    test_question = "Where did Brad Paisley go to college?"

    print(f"Testing PPR pre-filtering for GNN pipeline")
    print(f"Question: {test_question}")
    print("="*80)

    triplets = extender.extend_from_entity_with_ppr(
        entity_mid=test_mid,
        entity_name="Brad Paisley",
        question=test_question,
        fetch_limit=2000,
        ppr_limit=2000
    )

    print(f"\nPPR pre-filtering: collected {len(triplets)} triplets (limit: 2000)")
    print("These will be fed to GNN for final top-50 selection")
    print("\nTop 10 PPR-ranked triplets:")
    for i, (s, r, o, d) in enumerate(triplets[:10], 1):
        print(f"{i}. {s} --{r}--> {o} [{d}]")

    # Check if Belmont University is found
    found_belmont = False
    for s, r, o, d in triplets:
        if "belmont" in str(o).lower():
            print(f"\n✓ Found Belmont University: {r} -> {o}")
            found_belmont = True
            break

    if not found_belmont:
        print("\n✗ Belmont University not found in PPR results")
