import sys
import os

# Ensure the backend src directory is in the Python path so it can import rag_persona
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/src")))

from rag_persona.ingestion.cli import main

if __name__ == "__main__":
    main()
