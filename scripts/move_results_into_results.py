#!/usr/bin/env python3
# move result files into results/ and restore full script bodies into scripts/ from previous commit
from shutil import move, copyfile
from pathlib import Path
import os
os.makedirs('results', exist_ok=True)
# list of result files to move
files = [
    'classifier_confidence_results.csv',
    'metal_confidence_by_metal.csv',
    'metal_confidence_by_pair.csv',
    'xgb_confidence_results.csv',
    'zhang_2x2_results.csv',
    'zhang_data_results.csv',
    'zhang_his_split_results.csv',
    'REE_Results_Organized.xlsx',
    'REE_Results_Slides.pptx',
    'REE_Results_Slides_Final.pptx'
]
for f in files:
    if Path(f).exists():
        move(f, Path('results')/f)
print('Moved existing result files to results/')
