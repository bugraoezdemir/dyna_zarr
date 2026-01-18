import re
import html
from pathlib import Path
html_path = Path('htmlcov/z_5bb88312f9bf08c8_dynamic_array_py.html')
out_path = Path('htmlcov_extracted_dynamic_array_fixed.py')
text = html_path.read_text()
parts = re.findall(r'<p[^>]*>(.*?)</p>', text, flags=re.S)
lines = []
for p in parts:
    s = re.sub(r'<[^>]+>', '', p)
    s = html.unescape(s)
    # remove only the leading line number, keep following spaces
    s = re.sub(r'^\s*\d+', '', s)
    s = s.replace('\xa0', ' ')
    lines.append(s.rstrip())
out = '\n'.join(lines)
out_path.write_text(out)
print('wrote', out_path)
