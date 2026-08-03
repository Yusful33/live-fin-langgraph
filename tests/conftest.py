import os
import pytest

# Set dummy credentials before any test imports graph.py (which instantiates ChatOpenAI at module level)
os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")
os.environ.setdefault("FMP_API_KEY", "test-key-placeholder")
