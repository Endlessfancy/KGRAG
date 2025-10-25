#!/usr/bin/env python3
"""
Answer Validation Module for KGEAR Pipeline

Validates LLM responses to catch common error patterns:
1. Multiple answers (comma-separated lists)
2. Missing SUPPORT triplet citations
3. CVT node IDs returned as answers
4. Hallucination indicators

This helps reduce "GNN Hit + LLM Wrong" errors by forcing re-judgment
when validation fails.
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def validate_answer_format(answer: str) -> Tuple[bool, str]:
    """
    Check if answer contains multiple values instead of single entity.

    Common error patterns:
    - "Author, Scientist, Philosopher" (comma-separated)
    - "Belmont University and Vanderbilt" (and/or connectors)
    - "University of X, Y College" (multiple institutions)

    Args:
        answer: The answer string from LLM

    Returns:
        (is_valid, reason) - True if valid single answer, False with reason if not
    """
    if not answer or answer.lower() == 'unknown':
        return True, ""

    # Check for comma-separated multiple answers
    if ',' in answer:
        # Allow commas in dates/numbers: "1,000" or "January 1, 2020"
        # But reject: "Author, Scientist" or "City A, City B"
        parts = [p.strip() for p in answer.split(',')]

        # If we have multiple non-numeric parts, it's likely multiple answers
        non_numeric_parts = [p for p in parts if not p.replace(',', '').replace('.', '').isdigit()]

        if len(non_numeric_parts) > 2:  # More than "Name, Detail" format
            return False, f"Multiple answers detected: {answer}"

        # Check if parts look like separate entities (capitalized words)
        if len(parts) >= 2:
            # Both parts start with capital and are multi-word
            if all(p[0].isupper() and len(p.split()) >= 1 for p in parts[:2] if p):
                # Allow "City, State" or "Name, Title" patterns
                # But reject if both are long entity names
                if all(len(p.split()) >= 2 for p in parts[:2]):
                    return False, f"Multiple entity answers: {answer}"

    # Check for "and" or "or" connectors (except in proper names)
    and_or_pattern = r'\b(and|or)\b'
    matches = re.findall(and_or_pattern, answer, re.IGNORECASE)

    # Allow "and" in proper names like "Trinidad and Tobago"
    # but reject "Author and Scientist"
    if matches and len(answer.split()) > 3:
        # If we have capitalized words on both sides of "and/or", likely multiple answers
        split_by_connector = re.split(r'\b(and|or)\b', answer, flags=re.IGNORECASE)
        if len(split_by_connector) >= 3:
            left = split_by_connector[0].strip()
            right = split_by_connector[2].strip()

            # Both sides are capitalized multi-word phrases
            if (left and right and
                left[0].isupper() and right[0].isupper() and
                ' ' in left and ' ' in right):
                return False, f"Multiple answers with '{matches[0]}': {answer}"

    # Check for list indicators
    list_patterns = [
        r'^\d+\.',  # "1. Answer"
        r'^[-•]',   # "- Answer" or "• Answer"
        r'\bincluding\b',
        r'\bsuch as\b',
    ]

    for pattern in list_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            return False, f"List format detected: {answer}"

    return True, ""


def validate_cvt_answer(answer: str) -> Tuple[bool, str]:
    """
    Check if answer is a CVT node ID (should never be returned as answer).

    CVT nodes have format: m.0xxxxx (Freebase MID)

    Args:
        answer: The answer string from LLM

    Returns:
        (is_valid, reason) - False if CVT node detected, True otherwise
    """
    if not answer or answer.lower() == 'unknown':
        return True, ""

    # Check for Freebase MID pattern
    cvt_pattern = r'\bm\.[0-9a-z_]+\b'

    if re.search(cvt_pattern, answer, re.IGNORECASE):
        return False, f"CVT node ID detected: {answer}"

    return True, ""


def validate_support_triplets(
    support_indices: List[int],
    can_answer: bool,
    total_triplets: int
) -> Tuple[bool, str]:
    """
    Validate that SUPPORT triplets are provided when answer is sufficient.

    Args:
        support_indices: List of triplet indices cited as support
        can_answer: Whether LLM marked answer as sufficient
        total_triplets: Total number of triplets in context

    Returns:
        (is_valid, reason) - True if valid, False with reason if not
    """
    # If answer is "unknown", no support needed
    if not can_answer:
        return True, ""

    # If answer is sufficient, MUST have at least one support triplet
    if not support_indices or len(support_indices) == 0:
        return False, "Sufficient answer requires at least one SUPPORT triplet"

    # Check for out-of-bounds indices
    invalid_indices = [idx for idx in support_indices if idx < 1 or idx > total_triplets]
    if invalid_indices:
        return False, f"Invalid SUPPORT indices: {invalid_indices} (total triplets: {total_triplets})"

    return True, ""


def check_hallucination_indicators(
    answer: str,
    note: str,
    triplets: List[Tuple]
) -> Tuple[bool, str]:
    """
    Check for common hallucination patterns in answer and note.

    Indicators:
    - Answer contains entities not present in any triplet
    - Note mentions information not in triplets
    - Vague language: "likely", "probably", "seems"

    Args:
        answer: The answer string
        note: The explanation note
        triplets: List of (subj, pred, obj, score) tuples

    Returns:
        (is_valid, reason) - False if hallucination suspected, True otherwise
    """
    if not answer or answer.lower() == 'unknown':
        return True, ""

    # Extract all entity names from triplets
    triplet_entities = set()
    for triplet in triplets:
        if len(triplet) >= 3:
            subj, pred, obj = triplet[0], triplet[1], triplet[2]
            triplet_entities.add(subj.lower())
            triplet_entities.add(obj.lower())

    # Check if answer appears in any triplet (case-insensitive substring match)
    answer_lower = answer.lower()
    answer_found = False

    for entity in triplet_entities:
        if answer_lower in entity or entity in answer_lower:
            answer_found = True
            break

    # Also check for exact match or partial match
    if not answer_found:
        # Split answer into words and check if significant portion appears
        answer_words = set(answer_lower.split())
        for entity in triplet_entities:
            entity_words = set(entity.split())
            # If at least 50% of answer words appear in entity, consider it found
            overlap = len(answer_words & entity_words)
            if overlap >= len(answer_words) * 0.5:
                answer_found = True
                break

    if not answer_found:
        # Could be a date/number not directly in triplets but derived from them
        # Allow if answer is purely numeric or date-like
        if re.match(r'^[\d\s,.-]+$', answer) or re.match(r'^\d{4}', answer):
            answer_found = True  # Allow numeric answers

    # Check note for uncertainty language
    uncertainty_patterns = [
        r'\blikely\b',
        r'\bprobably\b',
        r'\bseems\b',
        r'\bmight be\b',
        r'\bcould be\b',
        r'\bpossibly\b',
        r'\bmaybe\b',
    ]

    if note:
        for pattern in uncertainty_patterns:
            if re.search(pattern, note, re.IGNORECASE):
                return False, f"Uncertainty in note: '{pattern}' detected"

    # If answer not found in triplets and not numeric, flag as potential hallucination
    if not answer_found:
        return False, f"Answer '{answer}' not found in provided triplets"

    return True, ""


def validate_llm_response(
    parsed_response: Dict,
    triplets: List[Tuple],
    question: str = ""
) -> Dict:
    """
    Comprehensive validation of LLM response.

    Runs all validation checks and returns detailed results.

    Args:
        parsed_response: Dict from parse_llm_response() with keys:
            - can_answer: bool
            - answer: str
            - support: List[int]
            - note: str
            - bridge_entities: List[dict]
        triplets: List of triplets provided to LLM
        question: Original question (optional, for context)

    Returns:
        Dict with:
        - is_valid: bool (overall validity)
        - errors: List[str] (all validation errors found)
        - warnings: List[str] (potential issues)
        - should_retry: bool (whether to force re-judgment as insufficient)
    """
    errors = []
    warnings = []

    answer = parsed_response.get('answer', '')
    can_answer = parsed_response.get('can_answer', False)
    support = parsed_response.get('support', [])
    note = parsed_response.get('note', '')

    # Only validate if LLM claims answer is sufficient
    if not can_answer:
        return {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'should_retry': False
        }

    # Validation 1: Answer format (no multiple answers)
    valid_format, format_reason = validate_answer_format(answer)
    if not valid_format:
        errors.append(f"FORMAT: {format_reason}")

    # Validation 2: CVT node check
    valid_cvt, cvt_reason = validate_cvt_answer(answer)
    if not valid_cvt:
        errors.append(f"CVT: {cvt_reason}")

    # Validation 3: Support triplets
    valid_support, support_reason = validate_support_triplets(support, can_answer, len(triplets))
    if not valid_support:
        errors.append(f"SUPPORT: {support_reason}")

    # Validation 4: Hallucination check
    valid_hallucination, hallucination_reason = check_hallucination_indicators(
        answer, note, triplets
    )
    if not valid_hallucination:
        warnings.append(f"HALLUCINATION: {hallucination_reason}")

    # Determine if we should retry (force insufficient judgment)
    # Retry on hard errors (format, CVT, support), but just warn on hallucination
    should_retry = len(errors) > 0

    is_valid = len(errors) == 0

    result = {
        'is_valid': is_valid,
        'errors': errors,
        'warnings': warnings,
        'should_retry': should_retry
    }

    if errors or warnings:
        logger.warning(f"Validation issues for answer '{answer}': {errors + warnings}")

    return result


if __name__ == "__main__":
    # Unit tests
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Answer Validator Unit Tests")
    print("="*80)

    # Test 1: Multiple answers detection
    print("\nTest 1: Multiple Answers Detection")
    test_cases = [
        ("Belmont University", True, "Single answer"),
        ("Author, Scientist, Philosopher", False, "Comma-separated list"),
        ("University of A and University of B", False, "And connector"),
        ("January 1, 2020", True, "Date with comma"),
        ("1,000", True, "Number with comma"),
        ("Trinidad and Tobago", True, "Country name with 'and'"),
    ]

    for answer, expected_valid, description in test_cases:
        valid, reason = validate_answer_format(answer)
        status = "✅" if valid == expected_valid else "❌"
        print(f"  {status} {description}: '{answer}' -> valid={valid}")
        if reason:
            print(f"      Reason: {reason}")

    # Test 2: CVT detection
    print("\nTest 2: CVT Node Detection")
    cvt_cases = [
        ("Belmont University", True, "Normal answer"),
        ("m.0123abc", False, "CVT node ID"),
        ("The answer is m.0xyz123", False, "CVT in text"),
    ]

    for answer, expected_valid, description in cvt_cases:
        valid, reason = validate_cvt_answer(answer)
        status = "✅" if valid == expected_valid else "❌"
        print(f"  {status} {description}: '{answer}' -> valid={valid}")

    # Test 3: Support validation
    print("\nTest 3: SUPPORT Triplet Validation")
    support_cases = [
        ([1, 2], True, True, 50, "Valid support indices"),
        ([], True, True, 50, "Missing support for sufficient answer"),
        ([1, 100], True, True, 50, "Out-of-bounds index"),
        ([], False, False, 50, "No support needed for insufficient"),
    ]

    for support, can_answer, expected_valid, total, description in support_cases:
        valid, reason = validate_support_triplets(support, can_answer, total)
        status = "✅" if valid == expected_valid else "❌"
        print(f"  {status} {description}: support={support}, can_answer={can_answer} -> valid={valid}")

    # Test 4: Full validation
    print("\nTest 4: Full Validation")
    test_triplets = [
        ('Brad Paisley', 'education', 'Belmont University', 0.9),
        ('Belmont University', 'location', 'Nashville', 0.8),
    ]

    good_response = {
        'can_answer': True,
        'answer': 'Belmont University',
        'support': [1],
        'note': 'Direct education connection in triplet 1',
        'bridge_entities': []
    }

    bad_response_multiple = {
        'can_answer': True,
        'answer': 'Belmont University, Vanderbilt University',
        'support': [1],
        'note': 'Multiple schools found',
        'bridge_entities': []
    }

    bad_response_no_support = {
        'can_answer': True,
        'answer': 'Belmont University',
        'support': [],
        'note': 'From triplet',
        'bridge_entities': []
    }

    for name, response in [
        ("Good response", good_response),
        ("Multiple answers", bad_response_multiple),
        ("No support", bad_response_no_support)
    ]:
        result = validate_llm_response(response, test_triplets)
        status = "✅" if result['is_valid'] else "❌"
        print(f"  {status} {name}: valid={result['is_valid']}, errors={result['errors']}")

    print("\n" + "="*80)
    print("All tests completed!")
