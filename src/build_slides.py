#!/usr/bin/env python3
# Wrapper to run the top-level build_slides.py from src/
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'build_slides.py'), run_name='__main__')
