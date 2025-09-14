"""
Utility functions for LLM Oracle
"""

from .response_handlers import response_failure, response_failure_embed
from .clients import (
    openai_chat_query,
    openai_embed_query,
    google_chat_query,
    google_embed_query,
    azure_chat_query
)

__all__ = [
    'response_failure',
    'response_failure_embed',
    'openai_chat_query',
    'openai_embed_query',
    'google_chat_query',
    'google_embed_query',
    'azure_chat_query'
]
