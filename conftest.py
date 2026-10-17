import sys
from pathlib import Path

# Make "import src..." work when pytest is run from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
