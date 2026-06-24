#!/usr/bin/env python3
# Wrapper to run the top-level ensemble_final.py from src/
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'ensemble_final.py'), run_name='__main__')
