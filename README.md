# gridlock

hackerearth traffic demand prediction. my best score **91.9389** (r2*100)

basically predict demand 0-1 for geohash areas on day 49 afternoon slots

## setup

```
pip install -r requirements.txt
```

download train.csv test.csv from hackerearth, put in `dataset/`

## run

```
python run_predict.py
```

makes `output/submission.csv` — same thing i submitted

notebook version: `notebooks/gridlock_solution.ipynb` if u like jupyter

## whats inside

- `output/` — saved preds from my models (final csv is here)
- `scripts/` — training code if u wanna retrain (takes forever tho)
- `src/advanced/` — target encoding, lag stuff, asymmetric loss for lgbm
- `APPROACH.txt` — what i did
- `NOTES.txt` — what failed lol

## stack

pandas numpy sklearn lightgbm catboost
