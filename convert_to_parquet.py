"""
convert_to_parquet.py  — Run this ONCE before launching the app.
-----------------------------------------------------------------
Reads each CSV, collapses village rows → 1 mandal row per date,
and saves a fast Parquet file alongside the CSV.

FIXES in this version:
  - Uses mean() for rainfall_mm (correct aggregation across villages)
  - Normalises district/mandal to lowercase+strip (consistent with app)
  - Sorts by date for optimal Parquet predicate-pushdown performance
  - Validates output row counts and warns on anomalies

Run:  python convert_to_parquet.py
Time: ~1-2 min total (one-time only)
"""

import os
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RAINFALL_DIR = os.path.join(BASE_DIR, "IMD_AP_Historical Rain Fall")

CSV_FILES = {
    2021: "AP_Village_Daily_Rainfall_21_r.csv",
    2022: "AP_Village_Daily_Rainfall_22_r.csv",
    2023: "AP_Village_Daily_Rainfall_23_r.csv",
    2024: "AP_Village_Daily_Rainfall_24_r.csv",
    2025: "AP_Village_Daily_Rainfall_25_r.csv",
}

REQUIRED_COLS = ["district", "mandal", "date", "rainfall_mm", "centroid_lat", "centroid_lon"]


def _normalize_str(s: str) -> str:
    return str(s).strip().lower()


for year, fname in CSV_FILES.items():
    csv_path    = os.path.join(RAINFALL_DIR, fname)
    parquet_path = csv_path.replace(".csv", "_mandal.parquet")

    if not os.path.exists(csv_path):
        print(f"  SKIP  {fname} — file not found")
        continue

    if os.path.exists(parquet_path):
        print(f"  SKIP  {fname} — parquet already exists")
        continue

    print(f"  Converting {fname} ...", end=" ", flush=True)

    # ── Detect available columns (handle uppercase/mixed-case headers) ─────────
    header = pd.read_csv(csv_path, nrows=0)
    col_map = {c.strip().lower(): c for c in header.columns}

    use_cols = []
    rename   = {}
    for req in REQUIRED_COLS:
        if req in col_map:
            use_cols.append(col_map[req])
            if col_map[req] != req:
                rename[col_map[req]] = req
        else:
            print(f"\n  WARNING: column '{req}' not found in {fname} — skipping year {year}")
            break
    else:
        # All required columns found — proceed
        parts = []
        for chunk in pd.read_csv(
            csv_path,
            usecols=use_cols,
            dtype={
                col_map["district"]     : str,
                col_map["mandal"]       : str,
                col_map["rainfall_mm"]  : float,
                col_map["centroid_lat"] : float,
                col_map["centroid_lon"] : float,
            },
            parse_dates=[col_map["date"]],
            chunksize=500_000,
        ):
            if rename:
                chunk = chunk.rename(columns=rename)

            # ── Normalise text keys BEFORE groupby ────────────────────────────
            chunk["district"] = chunk["district"].apply(_normalize_str)
            chunk["mandal"]   = chunk["mandal"].apply(_normalize_str)

            # ── FIX: use mean() for rainfall, mean() for coordinates ──────────
            mandal_chunk = (
                chunk.groupby(["district", "mandal", "date"], as_index=False)
                .agg(
                    rainfall_mm  = ("rainfall_mm",  "mean"),   # average across villages
                    centroid_lat = ("centroid_lat", "mean"),
                    centroid_lon = ("centroid_lon", "mean"),
                )
            )
            parts.append(mandal_chunk)

        df = pd.concat(parts, ignore_index=True)

        # ── Deduplicate (in case chunk boundaries split a single date) ─────────
        df = (
            df.groupby(["district", "mandal", "date"], as_index=False)
            .agg(
                rainfall_mm  = ("rainfall_mm",  "mean"),
                centroid_lat = ("centroid_lat", "mean"),
                centroid_lon = ("centroid_lon", "mean"),
            )
        )

        # ── Ensure date column is proper datetime ──────────────────────────────
        df["date"] = pd.to_datetime(df["date"])

        # ── Sort for optimal Parquet predicate-pushdown on date ───────────────
        df = df.sort_values(["date", "district", "mandal"]).reset_index(drop=True)

        # ── Round numeric columns ──────────────────────────────────────────────
        df["rainfall_mm"]  = df["rainfall_mm"].round(2)
        df["centroid_lat"] = df["centroid_lat"].round(6)
        df["centroid_lon"] = df["centroid_lon"].round(6)

        # ── Validate ──────────────────────────────────────────────────────────
        n_districts = df["district"].nunique()
        n_mandals   = df["mandal"].nunique()
        n_dates     = df["date"].nunique()
        print(
            f"done → {os.path.basename(parquet_path)}  "
            f"({len(df):,} rows | {n_districts} districts | "
            f"{n_mandals} mandals | {n_dates} dates | "
            f"{os.path.getsize(parquet_path) // 1024 if os.path.exists(parquet_path) else '?'} KB)"
        )
        if n_districts < 10:
            print(f"  ⚠️  Only {n_districts} districts found — check CSV district column")

        df.to_parquet(
            parquet_path,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )

        # Print file size after writing
        kb = os.path.getsize(parquet_path) // 1024
        print(f"       → file size: {kb:,} KB")

print("\nAll done. Now run:  streamlit run streamlit_app.py")