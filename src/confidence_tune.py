#!/usr/bin/env python3
# Wrapper to run the top-level confidence_tune.py from src/
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'confidence_tune.py'), run_name='__main__')
