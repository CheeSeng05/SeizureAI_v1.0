"""Dataset-specific raw annotation readers producing one canonical schema."""

from .base import CANONICAL_COLUMNS, REQUIRED_CANONICAL_COLUMNS, AnnotationAdapter
from .factory import create_adapter

__all__ = [
    "AnnotationAdapter",
    "CANONICAL_COLUMNS",
    "REQUIRED_CANONICAL_COLUMNS",
    "create_adapter",
]
