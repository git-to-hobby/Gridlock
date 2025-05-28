training scripts - only if u wanna retrain from scratch

full_train_v2.py - catboost + lgbm baseline
full_train_v3_advanced.py - tail lgbm with te/lags
morning_to_daytime.py - day48 only model
apply_mega_jump.py - max() tail fix on csvs

run from repo root like: python scripts/full_train_v2.py
needs train test in dataset/

faster way: just python run_predict.py uses output/ csvs already
