#!/usr/bin/env python3
# Wrapper to run the top-level zhang_data_model.py from src/
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'zhang_data_model.py'), run_name='__main__')
