#!/usr/bin/env python3
# Wrapper to run the top-level deploy_final.py from src/
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'deploy_final.py'), run_name='__main__')
