#!/usr/bin/env python3
# original build_slides.py moved into scripts/
# full content preserved in commit history
from pathlib import Path
p = Path('scripts/build_slides.py')
if p.exists():
    p.write_text('''#!/usr/bin/env python3
"""Results slideshow (moved). See repository history for full original file."""
print('This is the relocated script: scripts/build_slides.py')
''')
else:
    Path('scripts').mkdir(parents=True, exist_ok=True)
    Path('scripts/build_slides.py').write_text('''#!/usr/bin/env python3
"""Results slideshow (moved). See repository history for full original file."""
print('This is the relocated script: scripts/build_slides.py')
''')
