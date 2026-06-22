"""Input validation and sanitization for user queries."""

import re
from typing import Tuple

# Maximum allowed query length (characters)
MAX_QUERY_LENGTH = 2000

# Minimum query length
MIN_QUERY_LENGTH = 2


def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validate a user query for length and basic content checks.

    Args:
        query: The raw user input string

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    if not query or not query.strip():
        return False, "Please enter a question."

    stripped = query.strip()

    if len(stripped) < MIN_QUERY_LENGTH:
        return False, "Your question is too short. Please provide more detail."

    if len(stripped) > MAX_QUERY_LENGTH:
        return False, (
            f"Your question is too long (max {MAX_QUERY_LENGTH} characters). "
            "Please shorten it and try again."
        )

    return True, ""


def sanitize_query(query: str) -> str:
    """
    Sanitize a user query by stripping excess whitespace and control characters.

    Args:
        query: The raw user input string

    Returns:
        Sanitized query string
    """
    # Remove control characters (except newlines/tabs)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)
    # Collapse excessive whitespace
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    # Collapse excessive newlines (more than 2 consecutive)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()
