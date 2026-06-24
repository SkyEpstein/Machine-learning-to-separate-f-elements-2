#!/usr/bin/env python3
# Wrapper to run the top-level zhang_his_split.py from src/
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'zhang_his_split.py'), run_name='__main__')
