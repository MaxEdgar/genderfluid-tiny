#!/usr/bin/env python3
"""Backward-compatible CLI entry point. Delegates to genderfluid.cli."""

import sys

# Insert the project root so genderfluid is importable
import os
sys.path.insert(0, os.path.dirname(__file__))

from genderfluid.cli import main

if __name__ == "__main__":
    main()
