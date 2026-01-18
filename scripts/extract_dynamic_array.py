import re
import html
from pathlib import Path
html_path = Path('htmlcov/z_5bb88312f9bf08c8_dynamic_array_py.html')
out_path = Path('htmlcov_extracted_dynamic_array.py')
text = html_path.read_text()
# extract all <p ...>...</p>
parts = re.findall(r'<p[^>]*>(.*?)</p>', text, flags=re.S)
lines = []
for p in parts:
    # remove tags
    s = re.sub(r'<[^>]+>', '', p)
    s = html.unescape(s)
    # remove leading line numbers if present
    s = re.sub(r'^\s*\d+\s*', '', s)
    # strip trailing non-breaking spaces
    s = s.replace('\xa0', ' ')
    lines.append(s.rstrip())
# join and write
out = '\n'.join(lines)
out_path.write_text(out)
print('wrote', out_path)
