

import wandb
import pandas as pd
import os
import sys

api = wandb.Api()

sweep = api.sweep("models-imperial-college-london6785/RL for Games/2weqoipj")
runs = sweep.runs

print(f"Found {len(runs)} runs")
for run in runs:
    print(f"  {run.name} (id={run.id}) — state: {run.state}")
sys.stdout.flush()

# Only fully scan runs that have actually stopped logging — running/crashed
# runs get a lightweight snapshot instead, so the script never blocks on a
# moving target.
FINISHED_STATES = {"finished", "crashed", "failed"}

os.makedirs("sweep_export", exist_ok=True)
all_summaries = []
all_history_frames = []

for run in runs:
    group = run.group or "ungrouped"
    print(f"\n--- {run.name} (id={run.id}, group={group}, state={run.state}) ---")
    sys.stdout.flush()   # force it to appear immediately on Windows, don't wait on buffering

    summary_row = {"run_id": run.id, "run_name": run.name, "group": group, "state": run.state}
    summary_row.update(run.config)
    summary_row.update(run.summary._json_dict)
    all_summaries.append(summary_row)

    if run.state not in FINISHED_STATES:
        print(f"  state is '{run.state}' — skipping full scan_history() (would race a moving target).")
        print(f"  taking a quick snapshot instead via run.history(samples=200) ...")
        sys.stdout.flush()
        df = run.history(samples=200, pandas=True)   # bounded, single call, no pagination loop
        if df.empty:
            print("  (no data logged yet)")
            continue
    else:
        print(f"  scanning full history...")
        sys.stdout.flush()
        rows = []
        for i, row in enumerate(run.scan_history()):
            rows.append(row)
            if i % 500 == 0:
                print(f"    ...{i} rows so far")
                sys.stdout.flush()
        if not rows:
            print("  (no history logged — skipping)")
            continue
        df = pd.DataFrame(rows)

    df["run_id"] = run.id
    df["run_name"] = run.name
    df["group"] = group

    safe_name = run.name.replace("/", "_")
    df.to_csv(f"sweep_export/{safe_name}_{run.id}_history.csv", index=False)
    all_history_frames.append(df)

summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv("sweep_export/all_runs_summary.csv", index=False)
print(f"\nSaved combined summary ({summary_df.shape}) -> sweep_export/all_runs_summary.csv")

if all_history_frames:
    combined_history = pd.concat(all_history_frames, ignore_index=True, sort=False)
    combined_history.to_csv("sweep_export/all_runs_full_history.csv", index=False)
    print(f"Saved combined history ({combined_history.shape}) -> sweep_export/all_runs_full_history.csv")