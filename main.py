#!/usr/bin/env python3
"""
Entry point for the Genetic Algorithm for Data Compression Parameter Optimization.
This file serves as the main entry point and delegates to the actual implementation in src/.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main function from src
from main import main

if __name__ == "__main__":
    main()