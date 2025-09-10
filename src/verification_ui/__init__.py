"""
Human verification UI for hard negative mining.

Provides Streamlit-based interface for:
- False positive confirmation
- Efficient human-in-the-loop workflow
- Asynchronous verification process
- Progress tracking and logging
"""

from .app import VerificationApp

__all__ = ["VerificationApp"]
