from pathlib import Path
p = Path('htmlcov_extracted_dynamic_array_fixed.py')
text = p.read_text()
lines = text.splitlines()
# find first line that starts with 'import zarr'
start = 0
for i,l in enumerate(lines):
    if l.strip().startswith('import zarr'):
        start = i
        break
# find last occurrence of the write return
end = 0
for i in range(len(lines)-1, -1, -1):
    if 'return result._with_transform(transform)' in lines[i] or 'return array._with_transform(transform)' in lines[i]:
        end = i
        break
clean = '\n'.join(lines[start:end+1])
out = Path('htmlcov_extracted_dynamic_array_clean.py')
out.write_text(clean)
print('wrote', out)
