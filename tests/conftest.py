import sys
import os
import pytest

# Set HF_HOME *before* importing anything else that might use it
# This ensures tests use a local cache directory
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(TEST_DIR, ".cache")
os.environ["HF_HOME"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"\n[Global] HF_HOME set to: {CACHE_DIR}")

# Add the project root to sys.path so that src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(TEST_DIR, "..")))


