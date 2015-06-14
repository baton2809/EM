"""Main entry point"""

import sys
from .faireanalysis import faireanalysis

if sys.argv[0].endswith("__main__.py"):
    sys.argv[0] = "python -m faireanalysis"

faireanalysis()