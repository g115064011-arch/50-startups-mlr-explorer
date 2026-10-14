import urllib.request
import json

r1 = urllib.request.urlopen('http://127.0.0.1:5000/api/problem1')
d1 = json.load(r1)
print("=== P1 Live API ===")
print("Train R2:", d1['metrics']['train']['r2'])
print("Test R2:", d1['metrics']['test']['r2'])
print("Test RMSE:", d1['metrics']['test']['rmse'])
print("Insights count:", len(d1.get('research_insights', [])))

r2 = urllib.request.urlopen('http://127.0.0.1:5000/api/problem2')
d2 = json.load(r2)
print("\n=== P2 Live API ===")
print("Baseline -> Train CV R2:", d2['baseline'].get('train_cv_r2'), "| Test R2:", d2['baseline'].get('test_r2'))

for k, m in d2['methods'].items():
    print(f"{k:15} -> Train CV R2: {m.get('train_cv_r2')} | Test R2: {m.get('test_r2')} | Features: {m.get('selected_features')}")

sfs_steps = d2['methods']['forward_sfs']['steps']
print("\n=== SFS Steps CV R2 ===")
for s in sfs_steps:
    print(f"k={s['n_features']}: Train CV R2 = {s.get('train_cv_r2')} | Test R2 = {s.get('test_r2')} | Features = {s['features']}")

exh_top5 = d2['methods']['exhaustive']['all_combinations_top10'][:5]
print("\n=== Exhaustive Top 5 (Sorted by Train CV R2) ===")
for i, item in enumerate(exh_top5):
    print(f"#{i+1}: Train CV R2 = {item.get('train_cv_r2')} | Test R2 = {item.get('test_r2')} | Features = {item['features']}")
