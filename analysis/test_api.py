import urllib.request
import json

# Test Problem 1
r1 = urllib.request.urlopen('http://127.0.0.1:5000/api/problem1')
d1 = json.load(r1)
print("=== Problem 1 ===")
print("metrics keys:", list(d1['metrics'].keys()))
print("charts:", {k: 'OK' if v else 'MISSING' for k, v in d1['charts'].items()})
print("has crisp_dm:", 'crisp_dm_phases' in d1)
print("test_r2:", d1['metrics']['test']['r2'])

# Test Problem 2
print()
print("=== Problem 2 ===")
r2 = urllib.request.urlopen('http://127.0.0.1:5000/api/problem2')
d2 = json.load(r2)
print("methods:", list(d2['methods'].keys()))
for k, v in d2['methods'].items():
    if 'error' in v:
        print(f"  {k}: ERROR = {v['error']}")
    else:
        print(f"  {k}: test_r2={v.get('test_r2')}, features={v.get('selected_features')}")
print("DONE")
