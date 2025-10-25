#!/usr/bin/env python3
"""
Real GNN Scorer using SubgraphRAG's Trained Model

Loads the trained GNN model from SubgraphRAG and uses it for scoring triplets.

Architecture:
- Text Encoder: Alibaba-NLP/gte-large-en-v1.5 (1024-dim embeddings)
- GNN Model: Retriever from SubgraphRAG (DDE with message passing)
- Input: Triplets + Question → Output: GNN scores

Usage:
    scorer = RealGNNScorer(model_path, device='cuda:0')
    top_triplets = scorer.score_and_rank_triplets(
        triplets, question, top_k=50
    )
"""

import sys
import os
import torch
import torch.nn.functional as F
import logging
from typing import List, Tuple, Dict
from pathlib import Path

# Add SubgraphRAG to path
SUBGRAPH_RAG_PATH = Path(__file__).parent.parent.parent / 'SubgraphRAG' / 'retrieve'
sys.path.insert(0, str(SUBGRAPH_RAG_PATH))

from src.model.retriever import Retriever
from src.model.text_encoders.gte_large_en import GTELargeEN

logger = logging.getLogger(__name__)


class RealGNNScorer:
    """
    Real GNN scorer using SubgraphRAG's trained model.

    Two-stage scoring:
    1. PPR pre-filtering (done externally)
    2. GNN fine-tuning with learned model
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = 'cuda:0'
    ):
        """
        Initialize GNN scorer.

        Args:
            model_path: Path to trained GNN checkpoint (cpt.pth)
                       Default: SubgraphRAG/retrieve/cwq_*/cpt.pth
            device: Device to run model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Default model path
        if model_path is None:
            model_path = str(SUBGRAPH_RAG_PATH / 'cwq_Jun15-05:26:49' / 'cpt.pth')

        logger.info(f"Loading GNN model from: {model_path}")
        logger.info(f"Using device: {self.device}")

        # Load checkpoint
        cpt = torch.load(model_path, map_location='cpu')
        self.config = cpt['config']

        # Initialize text encoder
        logger.info("Loading text encoder: Alibaba-NLP/gte-large-en-v1.5...")
        self.text_encoder = GTELargeEN(device=self.device, normalize=True)

        # Initialize GNN model
        emb_size = 1024  # GTE-large-en-v1.5 embedding size
        self.model = Retriever(emb_size, **self.config['retriever']).to(self.device)
        self.model.load_state_dict(cpt['model_state_dict'])
        self.model.eval()

        logger.info("✓ GNN model loaded successfully")

    def build_graph_from_triplets(
        self,
        triplets: List[Tuple[str, str, str, str]],
        question: str
    ) -> Dict:
        """
        Build graph structure from triplets for GNN input.

        Args:
            triplets: List of (subject, relation, object, direction) tuples
            question: Question text

        Returns:
            Dictionary with:
                - h_id_tensor, r_id_tensor, t_id_tensor: Triplet indices
                - q_emb: Question embedding
                - entity_embs: Entity embeddings
                - relation_embs: Relation embeddings
                - num_non_text_entities: Count
                - topic_entity_one_hot: One-hot for topic entities
        """
        # Extract unique entities and relations
        text_entities = set()
        relations = set()

        for subj, rel, obj, direction in triplets:
            text_entities.add(subj)
            text_entities.add(obj)
            relations.add(rel)

        # Convert to lists (order matters for indexing)
        text_entity_list = sorted(list(text_entities))
        relation_list = sorted(list(relations))

        # Build entity and relation to ID mappings
        entity_to_id = {entity: idx for idx, entity in enumerate(text_entity_list)}
        relation_to_id = {rel: idx for idx, rel in enumerate(relation_list)}

        # Build triplet tensors
        h_id_list = []
        r_id_list = []
        t_id_list = []

        for subj, rel, obj, direction in triplets:
            h_id = entity_to_id[subj]
            r_id = relation_to_id[rel]
            t_id = entity_to_id[obj]

            h_id_list.append(h_id)
            r_id_list.append(r_id)
            t_id_list.append(t_id)

        # Create embeddings
        logger.debug(f"Computing embeddings for {len(text_entity_list)} entities, {len(relation_list)} relations")

        q_emb, entity_embs, relation_embs = self.text_encoder(
            question,
            text_entity_list,
            relation_list
        )

        # Topic entity one-hot (we don't have topic entities in extension, so all zeros)
        num_entities = len(text_entity_list)
        topic_entity_mask = torch.zeros(num_entities)
        topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2).float()

        return {
            'h_id_tensor': torch.tensor(h_id_list, dtype=torch.long),
            'r_id_tensor': torch.tensor(r_id_list, dtype=torch.long),
            't_id_tensor': torch.tensor(t_id_list, dtype=torch.long),
            'q_emb': q_emb,
            'entity_embs': entity_embs,
            'relation_embs': relation_embs,
            'num_non_text_entities': 0,  # All entities are text entities
            'topic_entity_one_hot': topic_entity_one_hot
        }

    @torch.no_grad()
    def score_triplets_with_gnn(
        self,
        graph_data: Dict
    ) -> torch.Tensor:
        """
        Score triplets using GNN forward pass.

        Args:
            graph_data: Graph structure from build_graph_from_triplets

        Returns:
            Tensor of scores for each triplet
        """
        # Move data to device
        h_id_tensor = graph_data['h_id_tensor'].to(self.device)
        r_id_tensor = graph_data['r_id_tensor'].to(self.device)
        t_id_tensor = graph_data['t_id_tensor'].to(self.device)
        q_emb = graph_data['q_emb'].to(self.device)
        entity_embs = graph_data['entity_embs'].to(self.device)
        relation_embs = graph_data['relation_embs'].to(self.device)
        topic_entity_one_hot = graph_data['topic_entity_one_hot'].to(self.device)
        num_non_text_entities = graph_data['num_non_text_entities']

        # GNN forward pass
        logits = self.model(
            h_id_tensor,
            r_id_tensor,
            t_id_tensor,
            q_emb,
            entity_embs,
            num_non_text_entities,
            relation_embs,
            topic_entity_one_hot
        )

        # Convert logits to scores (sigmoid)
        scores = torch.sigmoid(logits).reshape(-1)

        return scores

    def score_and_rank_triplets(
        self,
        triplets: List[Tuple[str, str, str, str]],
        question: str,
        top_k: int = 50
    ) -> List[Tuple[str, str, str, str]]:
        """
        Score all triplets with GNN and return top-K.

        Args:
            triplets: List of (subject, relation, object, direction) tuples
                     (Already PPR-filtered, up to 2000 × N entities)
            question: Question text
            top_k: Number of top triplets to return

        Returns:
            Top-K triplets by GNN score
        """
        if len(triplets) == 0:
            logger.warning("No triplets to score")
            return []

        logger.info(f"Scoring {len(triplets)} triplets with GNN...")

        # Build graph structure
        graph_data = self.build_graph_from_triplets(triplets, question)

        # GNN forward pass
        scores = self.score_triplets_with_gnn(graph_data)

        # Rank and select top-K
        scores_cpu = scores.cpu().numpy()
        scored_triplets = list(zip(scores_cpu, triplets))
        scored_triplets.sort(key=lambda x: x[0], reverse=True)

        top_triplets = [triplet for score, triplet in scored_triplets[:top_k]]

        logger.info(f"GNN selected top-{len(top_triplets)} triplets")
        if scored_triplets:
            logger.info(f"Score range: {scored_triplets[-1][0]:.4f} to {scored_triplets[0][0]:.4f}")

        # Log top 3 for debugging
        if len(scored_triplets) >= 3:
            logger.debug("Top 3 GNN-scored triplets:")
            for i, (score, triplet) in enumerate(scored_triplets[:3], 1):
                subj, rel, obj, direction = triplet
                logger.debug(f"  {i}. Score: {score:.4f} | {subj} --{rel}--> {obj}")

        return top_triplets


if __name__ == "__main__":
    # Test the GNN scorer
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("Testing RealGNNScorer")
    print("="*80)

    # Initialize scorer (device will be cuda:0 with CUDA_VISIBLE_DEVICES)
    scorer = RealGNNScorer(device='cuda:0')

    # Test triplets (Brad Paisley education example)
    test_triplets = [
        ("Brad Paisley", "people.person.education", "Education CVT", "out"),
        ("Education CVT", "education.education.institution", "Belmont University", "cvt"),
        ("Brad Paisley", "people.person.profession", "Singer", "out"),
        ("Brad Paisley", "people.person.nationality", "United States", "out"),
        ("Belmont University", "education.educational_institution.students_graduates", "Brad Paisley", "in"),
        ("Brad Paisley", "music.artist.genre", "Country Music", "out"),
    ]

    test_question = "Where did Brad Paisley go to college?"

    print(f"\nQuestion: {test_question}")
    print(f"Scoring {len(test_triplets)} test triplets\n")

    # Score and rank
    top_triplets = scorer.score_and_rank_triplets(
        test_triplets,
        test_question,
        top_k=3
    )

    print("\nTop-3 Results:")
    print("-"*80)
    for i, (subj, rel, obj, direction) in enumerate(top_triplets, 1):
        print(f"{i}. {subj} --{rel}--> {obj} [{direction}]")

    print("\n✓ Test completed successfully")
