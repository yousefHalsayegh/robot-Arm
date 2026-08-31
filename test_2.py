import wandb
import pandas as pd

api = wandb.Api()
run = api.run("/models-imperial-college-london6785/RL for Games/runs/clsjttea")

# ── Run metadata — context for interpreting the metrics themselves ──────────
print("=== Run config (hyperparameters) ===")
for k, v in run.config.items():
    print(f"  {k}: {v}")

print("\n=== Run summary (final/last logged value per metric) ===")
for k, v in run.summary.items():
    print(f"  {k}: {v}")

# ── Full history — scan_history() pulls EVERY logged step, no downsampling ──
# run.history() silently samples/bins by default (500 pts) and can hide
# exactly the kind of intermittent or sparse-column issues you're trying to
# spot — scan_history() avoids that entirely.
print("\n=== Pulling full history (this can take a moment for a long run) ===")
rows = list(run.scan_history())
df = pd.DataFrame(rows)

print(f"\nshape: {df.shape}")
print(f"columns ({len(df.columns)}):")
for col in df.columns:
    print(f"  {col}")

# ── Data-quality overview — surfaces exactly the kind of "questionable"
# things worth flagging before even opening the CSV: columns that are mostly
# NaN (logged rarely/inconsistently), columns with suspicious constant
# values, or numeric columns with implausible ranges.
print("\n=== Per-column NaN fraction (high = logged sparsely/inconsistently) ===")
nan_frac = df.isna().mean().sort_values(ascending=False)
for col, frac in nan_frac.items():
    if frac > 0:
        print(f"  {col}: {frac*100:.1f}% missing")

print("\n=== Numeric column summary stats ===")
numeric_cols = df.select_dtypes(include="number").columns
print(df[numeric_cols].describe().T)

# ── Save full, unabridged CSV ────────────────────────────────────────────────
csv_path = "run_415nnqxf_full_history.csv"
df.to_csv(csv_path, index=False)
print(f"\nsaved full history to {csv_path}")