#!/usr/bin/env python3
# Wrapper to run the top-level build_workbook2.py from src/
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'build_workbook2.py'), run_name='__main__')
