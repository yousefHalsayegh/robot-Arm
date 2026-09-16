#!/usr/bin/env python3
"""
Extracts full wandb run histories for every run listed in the ablation
sweep's summary file, writing one CSV per run plus a combined CSV with a
`job_name`/`group` column for cross-run analysis.

Usage:
    python extract_run_csvs.py --summary ablation_summary.md --out csv_exports/

Requires: pip install wandb pandas --break-system-packages
Requires you to be logged in (`wandb login`) with access to the runs.
"""

import argparse
import os
import re
import sys

import pandas as pd
import wandb

RUN_URL_RE = re.compile(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([^\s/?#]+)")
JOB_NAME_RE = re.compile(r"^\-\s+\*\*(.+?)\*\*\s+—\s+(.+)$")
RUN_LINE_RE = re.compile(r"^\s*-\s+run:\s+(\S.*)$")


def parse_summary(summary_path: str):
    """
    Parses the sweep's markdown summary and returns a list of dicts:
        {"job_name": ..., "status": ..., "entity": ..., "project": ..., "run_id": ...}
    Entries whose run URL couldn't be found or parsed are skipped, with a
    warning printed — this is expected for FAILED runs that never reached
    wandb.init(), not a bug in this script.
    """
    entries = []
    current_job = None
    current_status = None

    with open(summary_path, "r") as f:
        for line in f:
            job_match = JOB_NAME_RE.match(line.strip())
            if job_match:
                current_job, current_status = job_match.group(1), job_match.group(2)
                continue

            run_match = RUN_LINE_RE.match(line)
            if run_match and current_job is not None:
                url_line = run_match.group(1)
                url_match = RUN_URL_RE.search(url_line)
                if url_match:
                    entity, project, run_id = url_match.groups()
                    entries.append({
                        "job_name": current_job,
                        "status": current_status,
                        "entity": entity,
                        "project": project,
                        "run_id": run_id,
                    })
                else:
                    print(f"[skip] {current_job}: no parseable wandb URL "
                          f"('{url_line.strip()}') — likely a failed run, skipping.")
                current_job = None  # each job_name block is only used once

    return entries


def fetch_run_history(api: "wandb.Api", entity: str, project: str, run_id: str) -> pd.DataFrame:
    """
    Uses scan_history() rather than .history() — .history() silently
    subsamples to a default of 500 rows unless told otherwise, which is
    exactly the sparse-logging trap that caused unreliable conclusions
    earlier in this project. scan_history() returns every logged row,
    unsampled, via a paginated generator.
    """
    run = api.run(f"{entity}/{project}/{run_id}")
    rows = list(run.scan_history())
    df = pd.DataFrame(rows)
    df["run_id"] = run_id
    df["run_name"] = run.name
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="ablation_summary.md",
                         help="Path to the sweep's markdown summary (produced by run_ablation_sweep.sh)")
    parser.add_argument("--out", default="csv_exports",
                         help="Output directory for per-run and combined CSVs")
    parser.add_argument("--combined_name", default="all_runs_full_history.csv",
                         help="Filename for the combined, concatenated CSV")
    parser.add_argument("--skip_combined", action="store_true",
                         help="Only write per-run CSVs, skip the combined file "
                              "(useful if runs have very different/incompatible columns)")
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        print(f"error: summary file not found: {args.summary}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    entries = parse_summary(args.summary)
    if not entries:
        print("error: no run URLs found in summary file — nothing to extract.", file=sys.stderr)
        sys.exit(1)

    print(f"found {len(entries)} run(s) with a parseable wandb URL in {args.summary}")

    api = wandb.Api()
    combined_frames = []

    for entry in entries:
        job_name = entry["job_name"]
        print(f"\n>>> fetching: {job_name}  ({entry['entity']}/{entry['project']}/{entry['run_id']})")

        try:
            df = fetch_run_history(api, entry["entity"], entry["project"], entry["run_id"])
        except Exception as e:
            print(f"[error] failed to fetch {job_name}: {type(e).__name__}: {e}")
            continue

        df["job_name"] = job_name
        df["group"] = job_name  # kept for compatibility with earlier per-group CSV exports

        per_run_path = os.path.join(args.out, f"{job_name}.csv")
        df.to_csv(per_run_path, index=False)
        print(f"    wrote {len(df)} rows -> {per_run_path}")

        combined_frames.append(df)

    if not combined_frames:
        print("\nno runs were successfully fetched — nothing to combine.", file=sys.stderr)
        sys.exit(1)

    if not args.skip_combined:
        # Different configs (camera on/off, multi vs single task) log different
        # metric columns — outer-join concat keeps every column, filling gaps
        # with NaN rather than dropping columns that not every run shares.
        combined = pd.concat(combined_frames, axis=0, ignore_index=True, sort=False)
        combined_path = os.path.join(args.out, args.combined_name)
        combined.to_csv(combined_path, index=False)
        print(f"\nwrote combined CSV: {combined_path}  ({len(combined)} total rows, "
              f"{combined['job_name'].nunique()} runs, {len(combined.columns)} columns)")

    print("\ndone.")


if __name__ == "__main__":
    main()