#!/usr/bin/env python3
"""
Prompt Templates for KGEAR Pipeline

Implements the 3-section prompt structure:
1. System Prompt: Instructions + one-shot example
2. Retrieval Section: Top-K GNN-scored triplets
3. Query Section: The question to answer

This follows the task.md requirement for structured prompts.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# SYSTEM PROMPT with One-Shot Example
# =============================================================================

def get_system_prompt(max_bridge_entities: int = 3) -> str:
    """Generate simplified system prompt with stricter rules to prevent errors."""
    return f"""You are a Knowledge Graph Question Answering assistant.
Analyze the provided triplets to either answer the question OR identify bridge entities for expansion.

STRICT RULES:
- Ignore metadata predicates: type.*, common.*, freebase.*
- NEVER return CVT node IDs (m.xxxxx format) as answers
- Only return real entity names, dates, or numerical values
- Keep all notes under 20 words

CRITICAL ANSWER REQUIREMENTS:
- Return EXACTLY ONE entity/value, NEVER a comma-separated list
- If multiple valid answers exist, choose the MOST SPECIFIC one
- ONLY use information from provided triplets - DO NOT use world knowledge
- If unsure about answer granularity (e.g., city vs country), mark as INSUFFICIENT
- You MUST cite at least one SUPPORT triplet for sufficient answers
- If no triplets directly support the answer, mark as INSUFFICIENT

OUTPUT FORMAT (choose one):

If SUFFICIENT evidence:
ANSWER: <exactly one entity/value>
SUPPORT: [<comma-separated triplet indices>]
NOTE: <brief explanation, ≤20 words>

If INSUFFICIENT evidence:
ANSWER: unknown
BRIDGES: <entity1>, <entity2>, <entity3>
NOTE: <what information is missing, ≤20 words>"""


# Default system prompt for backwards compatibility
SYSTEM_PROMPT = get_system_prompt(3) + """

ONE-SHOT EXAMPLES:

Example 1 (Sufficient - Single Specific Answer):
Question: Where did Brad Paisley go to college?
Triplets:
[1] Brad Paisley --people.person.profession--> Singer
[2] Belmont University --education.institution.students_graduates--> Brad Paisley
[3] Brad Paisley --music.artist.genre--> Country music

Your response:
ANSWER: Belmont University
SUPPORT: [2]
NOTE: Direct education connection in triplet 2

❌ WRONG (Multiple Answers): "Belmont University, Vanderbilt University"
✅ CORRECT: Return only the ONE answer directly supported by triplets

Example 2 (Insufficient):
Question: When did the San Francisco Giants last win the World Series?
Triplets:
[1] San Francisco Giants --sports.sports_team.sport--> Baseball
[2] San Francisco Giants --sports.sports_team.location--> San Francisco
[3] Lou Seal --sports.mascot.team--> San Francisco Giants

Your response:
ANSWER: unknown
BRIDGES: San Francisco Giants
NOTE: Missing championship information

❌ WRONG: Guessing "2014" based on world knowledge
✅ CORRECT: Mark as insufficient when no triplet contains the answer
"""

# =============================================================================
# Prompt Formatting Functions
# =============================================================================

def format_retrieval_section(
    triplets: List[Tuple],
    max_triplets: int = 50
) -> str:
    """
    Format the retrieval section with GNN-scored triplets.

    Args:
        triplets: List of (subj, pred, obj, score) tuples from GNN
        max_triplets: Maximum number of triplets to include

    Returns:
        Formatted retrieval section string
    """
    lines = []

    for i, triplet in enumerate(triplets[:max_triplets], 1):
        if len(triplet) == 4:
            subj, pred, obj, score = triplet
            # Handle both float scores (from GNN) and string labels (from path extension)
            if isinstance(score, (int, float)):
                lines.append(f"[{i}] {subj} --{pred}--> {obj} (score: {score:.3f})")
            else:
                # String label like "cvt", "out", "in" from path extension
                lines.append(f"[{i}] {subj} --{pred}--> {obj}")
        elif len(triplet) == 3:
            subj, pred, obj = triplet
            lines.append(f"[{i}] {subj} --{pred}--> {obj}")
        else:
            logger.warning(f"Unexpected triplet format: {triplet}")
            continue

    retrieval_text = "\n".join(lines)

    return f"""# Retrieved Evidence (Top-{len(lines)} by GNN):
{retrieval_text}"""


def format_retrieval_section_graphstructure(
    triplets: List[Tuple],
    max_triplets: int = 100
) -> str:
    """
    Format triplets as graph structure (grouped by subject).

    Better for LLM understanding of entity relationships by showing
    all information about each entity together.

    Args:
        triplets: List of (subj, pred, obj, score) tuples from GNN
        max_triplets: Maximum number of triplets to include

    Returns:
        Formatted graph structure string
    """
    from collections import defaultdict

    # Group by subject
    subject_map = defaultdict(list)
    triplet_count = 0

    for triplet in triplets:
        if triplet_count >= max_triplets:
            break

        if len(triplet) >= 3:
            subj, pred, obj = triplet[0], triplet[1], triplet[2]
            subject_map[subj].append((pred, obj))
            triplet_count += 1
        else:
            logger.warning(f"Unexpected triplet format: {triplet}")

    # Sort subjects for deterministic output
    lines = []
    for subject in sorted(subject_map.keys()):
        lines.append(f"{subject}:")

        # Sort predicates for each subject
        predicates = sorted(subject_map[subject], key=lambda x: (x[0], x[1]))
        for pred, obj in predicates:
            lines.append(f"  - {pred} → {obj}")

        lines.append("")  # Blank line between entities

    graph_text = "\n".join(lines)

    return f"""# Retrieved Evidence (Graph Structure, {triplet_count} triplets):
{graph_text}"""


def build_pregeneration_prompt(
    question: str,
    gnn_triplets: List[Tuple],
    max_triplets: int = 50,
    max_bridge_entities: int = 3,
    use_graph_structure: bool = False
) -> str:
    """
    Build the complete pregeneration prompt with simplified format.

    This is used for the initial LLM call (Stage 2c in task.md).

    Args:
        question: The question to answer
        gnn_triplets: Top-K triplets from GNN
        max_triplets: Maximum triplets to include (default: 50)
        max_bridge_entities: Maximum bridge entities to extract (default: 3)
        use_graph_structure: Use graph structure format (grouped by subject) instead of flat list

    Returns:
        Complete prompt string with simplified format
    """
    # Section 1: System prompt with examples
    section_1 = get_system_prompt(max_bridge_entities) + """

ONE-SHOT EXAMPLES:

Example 1 (Sufficient):
Question: Where did Brad Paisley go to college?
Triplets:
[1] Brad Paisley --people.person.profession--> Singer
[2] Belmont University --education.institution.students_graduates--> Brad Paisley
[3] Brad Paisley --music.artist.genre--> Country music

Your response:
ANSWER: Belmont University
SUPPORT: [2]
NOTE: Direct education connection in triplet 2

Example 2 (Insufficient):
Question: When did the San Francisco Giants last win the World Series?
Triplets:
[1] San Francisco Giants --sports.sports_team.sport--> Baseball
[2] San Francisco Giants --sports.sports_team.location--> San Francisco
[3] Lou Seal --sports.mascot.team--> San Francisco Giants

Your response:
ANSWER: unknown
BRIDGES: San Francisco Giants
NOTE: Missing championship information
"""

    # Section 2: Retrieval section (use graph structure or flat list)
    if use_graph_structure:
        section_2 = format_retrieval_section_graphstructure(gnn_triplets, max_triplets)
    else:
        section_2 = format_retrieval_section(gnn_triplets, max_triplets)

    # Section 3: Query section
    section_3 = f"""
================================================================================
QUESTION: {question}

Provide your response following the format above:"""

    # Combine all sections
    full_prompt = f"""{section_1}

================================================================================
RETRIEVED EVIDENCE (Top-{len(gnn_triplets[:max_triplets])} by GNN):
{section_2}
{section_3}"""

    return full_prompt


def build_final_answer_prompt(
    question: str,
    extended_triplets: List[Tuple],
    max_triplets: int = 100,
    use_graph_structure: bool = False
) -> str:
    """
    Build the final answer prompt after path extension.

    This is used after bridge entity expansion (Stage 2e in task.md).

    Args:
        question: The question to answer
        extended_triplets: Combined GNN + extended triplets
        max_triplets: Maximum triplets to include (default: 100)
        use_graph_structure: Use graph structure format (grouped by subject) instead of flat list

    Returns:
        Complete prompt string
    """
    # Simplified system prompt for final answer (no bridge entity option)
    final_system = """You are a Knowledge Graph Question Answering assistant.
Review the extended evidence and generate the final answer.

STRICT RULES:
- Ignore metadata predicates: type.*, common.*, freebase.*
- NEVER return CVT node IDs (m.xxxxx format) as answers
- Only return real entity names, dates, or numerical values
- Keep notes under 20 words

OUTPUT FORMAT:
ANSWER: <your final answer>
SUPPORT: [<comma-separated triplet indices>]
NOTE: <brief explanation, ≤20 words>
"""

    # Retrieval section (use graph structure or flat list)
    if use_graph_structure:
        retrieval = format_retrieval_section_graphstructure(extended_triplets, max_triplets)
    else:
        retrieval = format_retrieval_section(extended_triplets, max_triplets)

    # Query section
    query = f"""
================================================================================
QUESTION: {question}

Analyze the evidence and provide your answer in the format above:"""

    return f"""{final_system}

================================================================================
EXTENDED EVIDENCE (Top-{len(extended_triplets[:max_triplets])}):
{retrieval}
{query}"""


def extract_cvt_emphasis_note(attempt: int) -> str:
    """
    Generate CVT emphasis text for retry attempts.

    Args:
        attempt: Retry attempt number (0-indexed)

    Returns:
        Emphasis text or empty string
    """
    if attempt == 0:
        return ""

    return """
⚠️ IMPORTANT REMINDER:
- Do NOT return CVT node IDs (format: m.0xxxxx) as answers
- CVT nodes are intermediate connection points, not final answers
- If you see a CVT node, look for related triplets that contain the actual answer
"""


# =============================================================================
# Response Parsing
# =============================================================================

def parse_llm_response(response: str) -> dict:
    """
    Parse the simplified LLM response into structured format.

    Expected format (plain text, no JSON):
        ANSWER: <answer>
        SUPPORT: [1, 3, 5]
        NOTE: <explanation>
        BRIDGES: <entity1>, <entity2>, <entity3>  (if insufficient)

    Args:
        response: Raw LLM response text

    Returns:
        Dictionary with:
        - can_answer: bool
        - answer: str or None
        - support: List[int]
        - note: str or None
        - bridge_entities: List[dict]
    """
    result = {
        'can_answer': False,
        'answer': None,
        'support': [],
        'note': None,
        'bridge_entities': []
    }

    lines = response.strip().split('\n')

    # Parse main fields line by line
    for line in lines:
        line = line.strip()

        if line.startswith('ANSWER:'):
            answer = line.replace('ANSWER:', '').strip()
            result['answer'] = answer
            result['can_answer'] = (answer.lower() != 'unknown')

        elif line.startswith('SUPPORT:'):
            support_str = line.replace('SUPPORT:', '').strip()
            if support_str and support_str != '[]':
                # Extract numbers from [1, 2, 3] format
                support_str = support_str.strip('[]')
                if support_str:
                    try:
                        result['support'] = [
                            int(x.strip())
                            for x in support_str.split(',')
                            if x.strip().isdigit()
                        ]
                    except:
                        logger.warning(f"Failed to parse SUPPORT: {support_str}")

        elif line.startswith('NOTE:'):
            result['note'] = line.replace('NOTE:', '').strip()

        elif line.startswith('BRIDGES:'):
            # Parse comma-separated bridge entities (plain text, no JSON!)
            bridges_str = line.replace('BRIDGES:', '').strip()
            if bridges_str:
                # Split by comma and clean up
                bridge_names = [
                    name.strip()
                    for name in bridges_str.split(',')
                    if name.strip()
                ]
                # Convert to list of dicts for compatibility with existing code
                result['bridge_entities'] = [
                    {'entity_name': name, 'entity_mid': None, 'why': ''}
                    for name in bridge_names[:3]  # Max 3 as per task.md
                ]
                logger.info(f"Extracted {len(result['bridge_entities'])} bridge entities from BRIDGES field")

    return result


if __name__ == "__main__":
    # Test the prompt templates
    logging.basicConfig(level=logging.INFO)

    # Test triplets
    test_triplets = [
        ('Lou Seal', 'sports.mascot.team', 'San Francisco Giants', 0.999),
        ('San Francisco Giants', 'sports.sports_team.team_mascot', 'Lou Seal', 0.997),
        ('2014 World Series', 'sports.sports_championship_event.champion', 'San Francisco Giants', 0.449),
        ('San Francisco Giants', 'sports.sports_team.sport', 'Baseball', 0.420)
    ]

    test_question = "Lou Seal is the mascot for the team that last won the World Series when?"

    # Build pregeneration prompt
    prompt = build_pregeneration_prompt(test_question, test_triplets, max_triplets=10)

    print("=== PREGENERATION PROMPT ===")
    print(prompt)
    print("\n" + "="*80 + "\n")

    # Test response parsing
    test_response = """ANSWER: 2014 World Series
SUPPORT: [3]
NOTE: Triplet 3 shows that 2014 World Series champion was San Francisco Giants, and triplet 1 connects Lou Seal to the Giants.
[]"""

    parsed = parse_llm_response(test_response)
    print("=== PARSED RESPONSE ===")
    print(f"Can answer: {parsed['can_answer']}")
    print(f"Answer: {parsed['answer']}")
    print(f"Support: {parsed['support']}")
    print(f"Note: {parsed['note']}")
    print(f"Bridge entities: {parsed['bridge_entities']}")

    # Test bridge entity response
    test_response_2 = """ANSWER: unknown
SUPPORT: []
NOTE: No championship information available. Need to explore championships relation.
[{"entity_name": "San Francisco Giants", "entity_mid": "m.0713r", "why": "Need championship history"}]"""

    parsed_2 = parse_llm_response(test_response_2)
    print("\n=== PARSED BRIDGE RESPONSE ===")
    print(f"Can answer: {parsed_2['can_answer']}")
    print(f"Answer: {parsed_2['answer']}")
    print(f"Bridge entities: {parsed_2['bridge_entities']}")
