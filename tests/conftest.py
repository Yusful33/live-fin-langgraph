import os

# The graph module instantiates ChatOpenAI at import time, which requires
# OPENAI_API_KEY. Tests only exercise tool wiring and static prompts, so a
# placeholder is sufficient — no OpenAI calls actually happen.
os.environ.setdefault("OPENAI_API_KEY", "test-placeholder")
os.environ.setdefault("FMP_API_KEY", "test-placeholder")
