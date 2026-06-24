#!/usr/bin/env python3
# Wrapper to run the top-level metal_confidence.py from src/
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'metal_confidence.py'), run_name='__main__')
