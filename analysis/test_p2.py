import sys, os
sys.path.insert(0, os.path.abspath('.'))
from analysis.problem2 import run_all_feature_selection
r = run_all_feature_selection('data/50_Startups.csv')
print("baseline test_r2:", r['baseline']['test_r2'])
for k, v in r['methods'].items():
    if 'error' in v:
        print(k, 'ERROR:', v['error'])
    else:
        print(k, '| features:', v.get('selected_features'), '| test_r2:', v.get('test_r2'))
print("DONE")
