#!/usr/bin/env python3
# Wrapper to run the top-level make_figures.py from src/
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'make_figures.py'), run_name='__main__')
