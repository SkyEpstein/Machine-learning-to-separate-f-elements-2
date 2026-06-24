#!/usr/bin/env python3
# Wrapper to run the top-level tabpfn_in_stack.py from src/
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'tabpfn_in_stack.py'), run_name='__main__')
