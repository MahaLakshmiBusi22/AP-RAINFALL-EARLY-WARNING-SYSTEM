import pandas as pd
import numpy as np
import pickle
import os
from datetime import timedelta

# ─── Load Models ───────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RAINFALL_DIR = os.path.join(BASE_DIR, "IMD_AP_Historical Rain Fall")

CSV_YEAR_MAP = {
    2021: "AP_Village_Daily_Rainfall_21_r.csv",
    2022: "AP_Village_Daily_Rainfall_22_r.csv",
    2023: "AP_Village_Daily_Rainfall_23_r.csv",
    2024: "AP_Village_Daily_Rainfall_24_r.csv",
    2025: "AP_Village_Daily_Rainfall_25_r.csv",
}

PARQUET_YEAR_MAP = {
    yr: fname.replace(".csv", "_mandal.parquet")
    for yr, fname in CSV_YEAR_MAP.items()
}


def _parquet_path(year: int) -> str:
    return os.path.join(RAINFALL_DIR, PARQUET_YEAR_MAP[year])


def _parquet_exists(year: int) -> bool:
    return os.path.exists(_parquet_path(year))


def _norm(s) -> str:
    """Lowercase + strip. Safe for NaN/None."""
    if pd.isna(s):
        return ""
    return str(s).strip().lower()


print("Loading models...")

with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
    rf_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
    FEATURE_COLS = pickle.load(f)

village_lookup = pd.read_parquet(
    os.path.join(MODEL_DIR, "village_lookup.parquet")
)
# Normalise keys once at load time
village_lookup["district"] = village_lookup["district"].apply(_norm)
village_lookup["mandal"]   = village_lookup["mandal"].apply(_norm)

print(f"✅ Models loaded | Villages in lookup: {len(village_lookup):,}")


# ─── Constants ─────────────────────────────────────────────────────────────────
ALERT_META = {
    "GREEN" : {
        "color"   : "#2ecc71",
        "bg"      : "#eafaf1",
        "emoji"   : "🟢",
        "label"   : "No Risk",
        "message" : "No significant flood risk",
        "priority": 4,
    },
    "YELLOW": {
        "color"   : "#f39c12",
        "bg"      : "#fef9e7",
        "emoji"   : "🟡",
        "label"   : "Moderate Risk",
        "message" : "Monitor water levels — moderate risk",
        "priority": 3,
    },
    "ORANGE": {
        "color"   : "#e67e22",
        "bg"      : "#fdf2e9",
        "emoji"   : "🟠",
        "label"   : "High Risk",
        "message" : "Prepare evacuation — high flood risk",
        "priority": 2,
    },
    "RED"   : {
        "color"   : "#e74c3c",
        "bg"      : "#fdedec",
        "emoji"   : "🔴",
        "label"   : "Extreme Risk",
        "message" : "Immediate action required — extreme flood risk",
        "priority": 1,
    },
}

RISK_ORDER = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "GREEN": 3}


# ─────────────────────────────────────────────────────────────────────────────
# SEASONAL RAINFALL MODEL
# ─────────────────────────────────────────────────────────────────────────────
class SeasonalRainfallModel:

    def __init__(self):
        self._mandal_stats:   dict = {}
        self._district_stats: dict = {}
        self._ap_stats:       dict = {}
        self._grand_mean:     float = 0.0
        self.is_trained:      bool  = False

    def train(self) -> "SeasonalRainfallModel":
        all_parts: list[pd.DataFrame] = []

        for year, csv_name in CSV_YEAR_MAP.items():
            if _parquet_exists(year):
                try:
                    df = pd.read_parquet(_parquet_path(year), engine="pyarrow")
                    df["district"] = df["district"].apply(_norm)
                    df["mandal"]   = df["mandal"].apply(_norm)
                    df["date"]     = pd.to_datetime(df["date"])
                    all_parts.append(
                        df[["district", "mandal", "date", "rainfall_mm"]]
                        .dropna(subset=["rainfall_mm"])
                    )
                    continue
                except Exception as e:
                    print(f"[SeasonalModel] parquet read failed for {year}: {e}")

            csv_path = os.path.join(RAINFALL_DIR, csv_name)
            if not os.path.exists(csv_path):
                continue
            try:
                for chunk in pd.read_csv(
                    csv_path,
                    usecols=["district", "mandal", "date", "rainfall_mm"],
                    dtype={"district": str, "mandal": str, "rainfall_mm": float},
                    parse_dates=["date"],
                    chunksize=500_000,
                ):
                    chunk["district"] = chunk["district"].apply(_norm)
                    chunk["mandal"]   = chunk["mandal"].apply(_norm)
                    all_parts.append(
                        chunk[["district", "mandal", "date", "rainfall_mm"]]
                        .dropna(subset=["rainfall_mm"])
                    )
            except Exception as e:
                print(f"[SeasonalModel] CSV read failed for {year}: {e}")

        if not all_parts:
            print("[SeasonalModel] WARNING: no training data found.")
            self.is_trained = False
            return self

        hist = (
            pd.concat(all_parts, ignore_index=True)
            .sort_values(["district", "mandal", "date"])
        )
        hist["month"] = hist["date"].dt.month

        rolled_parts: list[pd.DataFrame] = []

        for (dist, mandal), grp in hist.groupby(["district", "mandal"], sort=False):
            grp   = grp.set_index("date").sort_index()
            daily = (
                grp["rainfall_mm"]
                .resample("D").sum()
                .reindex(
                    pd.date_range(grp.index.min(), grp.index.max(), freq="D"),
                    fill_value=0.0,
                )
            )
            r3  = daily.rolling(3,  min_periods=1).sum().rename("r3")
            r7  = daily.rolling(7,  min_periods=1).sum().rename("r7")
            r30 = daily.rolling(30, min_periods=1).sum().rename("r30")

            combined = pd.concat(
                [daily.rename("rainfall_mm"), r3, r7, r30], axis=1
            )
            combined["district"] = dist
            combined["mandal"]   = mandal
            combined["month"]    = combined.index.month
            rolled_parts.append(combined.reset_index(names="date"))

        if not rolled_parts:
            self.is_trained = False
            return self

        rolled = pd.concat(rolled_parts, ignore_index=True)

        mandal_agg = (
            rolled
            .groupby(["district", "mandal", "month"])[
                ["rainfall_mm", "r3", "r7", "r30"]
            ]
            .mean()
            .round(2)
            .reset_index()
        )
        for _, row in mandal_agg.iterrows():
            key = (_norm(row["district"]), _norm(row["mandal"]), int(row["month"]))
            self._mandal_stats[key] = {
                "mm" : float(row["rainfall_mm"]),
                "r3" : float(row["r3"]),
                "r7" : float(row["r7"]),
                "r30": float(row["r30"]),
            }

        district_agg = (
            rolled
            .groupby(["district", "month"])[["rainfall_mm", "r3", "r7", "r30"]]
            .mean()
            .round(2)
            .reset_index()
        )
        for _, row in district_agg.iterrows():
            key = (_norm(row["district"]), int(row["month"]))
            self._district_stats[key] = {
                "mm" : float(row["rainfall_mm"]),
                "r3" : float(row["r3"]),
                "r7" : float(row["r7"]),
                "r30": float(row["r30"]),
            }

        ap_agg = (
            rolled
            .groupby("month")[["rainfall_mm", "r3", "r7", "r30"]]
            .mean()
            .round(2)
            .reset_index()
        )
        for _, row in ap_agg.iterrows():
            self._ap_stats[int(row["month"])] = {
                "mm" : float(row["rainfall_mm"]),
                "r3" : float(row["r3"]),
                "r7" : float(row["r7"]),
                "r30": float(row["r30"]),
            }

        self._grand_mean = float(rolled["rainfall_mm"].mean())
        self.is_trained  = True

        print(
            f"[SeasonalModel] trained | "
            f"{len(self._mandal_stats):,} mandal×month entries | "
            f"grand mean = {self._grand_mean:.2f} mm"
        )
        return self

    def _lookup(self, district: str, mandal: str, month: int) -> dict:
        """Return {mm, r3, r7, r30} using fallback hierarchy."""
        d, m = _norm(district), _norm(mandal)

        v = self._mandal_stats.get((d, m, month))
        if v:
            return v

        v = self._district_stats.get((d, month))
        if v:
            return v

        v = self._ap_stats.get(month)
        if v:
            return v

        # ── FIX 1: grand-mean fallback was missing a return statement ─────────
        gm = self._grand_mean
        return {"mm": gm, "r3": gm * 3, "r7": gm * 7, "r30": gm * 30}

    def get_ap_mean_for_month(self, month: int) -> float:
        v = self._ap_stats.get(month)
        if v:
            return round(v["mm"], 1)
        return round(self._grand_mean, 1)

    def estimate(self, month: int, vl: pd.DataFrame) -> pd.DataFrame:
        mandal_df = (
            vl
            .groupby(["district", "mandal"], as_index=False)
            .agg(
                centroid_lat=("centroid_lat", "mean"),
                centroid_lon=("centroid_lon", "mean"),
            )
        )

        mm_vals, r3_vals, r7_vals, r30_vals = [], [], [], []
        for _, row in mandal_df.iterrows():
            s = self._lookup(row["district"], row["mandal"], month)
            mm_vals.append(round(s["mm"],  2))
            r3_vals.append(round(s["r3"],  1))
            r7_vals.append(round(s["r7"],  1))
            r30_vals.append(round(s["r30"], 1))

        mandal_df["rainfall_mm"]    = mm_vals
        mandal_df["rainfall_3day"]  = r3_vals
        mandal_df["rainfall_7day"]  = r7_vals
        mandal_df["rainfall_30day"] = r30_vals
        return mandal_df

    def estimate_for_village(
        self, district: str, mandal: str, month: int
    ) -> dict:
        return self._lookup(district, mandal, month)


# ─── Build the seasonal model once at module load ─────────────────────────────
seasonal_model = SeasonalRainfallModel().train()


# ─── Dropdowns ────────────────────────────────────────────────────────────────
def get_districts():
    return sorted(village_lookup["district"].unique().tolist())


def get_mandals(district):
    df = village_lookup[village_lookup["district"] == district]
    return sorted(df["mandal"].unique().tolist())


def get_villages(district, mandal):
    df = village_lookup[
        (village_lookup["district"] == district) &
        (village_lookup["mandal"]   == mandal)
    ]
    return sorted(df["village"].unique().tolist())


def classify_rainfall(mm: float) -> str:
    if mm == 0:      return "No Rain"
    elif mm < 2.5:   return "Light"
    elif mm < 15.6:  return "Moderate"
    elif mm < 64.5:  return "Heavy"
    elif mm < 115.6: return "Very Heavy"
    else:            return "Extremely Heavy"


# ─── Core Prediction ──────────────────────────────────────────────────────────
def predict_risk(district, date_str, rainfall_mm,
                 rainfall_3day=None, rainfall_7day=None,
                 rainfall_30day=None, mandal=None):
    dt          = pd.to_datetime(date_str)
    month       = dt.month
    day_of_year = dt.dayofyear
    ap_mean_mm  = seasonal_model.get_ap_mean_for_month(month)
    is_monsoon  = ap_mean_mm

    # Filter villages
    mask = village_lookup["district"] == _norm(district)
    if mandal:
        mask = mask & (village_lookup["mandal"] == _norm(mandal))

    villages = village_lookup[mask].copy().reset_index(drop=True)

    if villages.empty:
        return []

    # ── FIX 2: removed erroneous merge of mandal_rain (undefined variable) ────
    # Rolling accumulators: use seasonal model when not supplied by caller
    need_seasonal = (
        rainfall_3day  is None or
        rainfall_7day  is None or
        rainfall_30day is None
    )

    if need_seasonal:
        seasonal_df = seasonal_model.estimate(
            month,
            villages[["district", "mandal"]].drop_duplicates()
        )
        seasonal_map = {
            (_norm(r["district"]), _norm(r["mandal"])): r
            for _, r in seasonal_df.iterrows()
        }

    r3_vals, r7_vals, r30_vals = [], [], []

    for _, vrow in villages.iterrows():
        if need_seasonal:
            key = (_norm(vrow["district"]), _norm(vrow["mandal"]))
            s   = seasonal_map.get(
                key,
                seasonal_model.estimate_for_village(
                    vrow["district"], vrow["mandal"], month
                ),
            )
            r3  = rainfall_3day  if rainfall_3day  is not None else round(s["r3"],  1)
            r7  = rainfall_7day  if rainfall_7day  is not None else round(s["r7"],  1)
            r30 = rainfall_30day if rainfall_30day is not None else round(s["r30"], 1)
        else:
            r3  = rainfall_3day
            r7  = rainfall_7day
            r30 = rainfall_30day

        r3_vals.append(r3)
        r7_vals.append(r7)
        r30_vals.append(r30)

    # ── FIX 3: preserve the caller-supplied rainfall_mm instead of overwriting─
    villages["rainfall_mm"]    = float(rainfall_mm)
    villages["rainfall_3day"]  = r3_vals
    villages["rainfall_7day"]  = r7_vals
    villages["rainfall_30day"] = r30_vals
    # anomaly vs 30-day daily average
    villages["rainfall_anomaly"] = float(rainfall_mm) - (
        pd.Series(r30_vals, index=villages.index) / 30.0
    )
    villages["month"]       = month
    villages["day_of_year"] = day_of_year
    villages["is_monsoon"]  = is_monsoon

    X            = villages[FEATURE_COLS].fillna(0)
    y_pred       = rf_model.predict(X)
    y_proba      = rf_model.predict_proba(X)
    alert_labels = le.inverse_transform(y_pred)
    classes      = list(le.classes_)

    results = []
    for i, (_, row) in enumerate(villages.iterrows()):
        alert      = alert_labels[i]
        meta       = ALERT_META[alert]
        proba      = y_proba[i]
        confidence = round(float(proba[y_pred[i]]) * 100, 1)

        has_embankment = (
            float(row["dist_embankment_km"]) < 50.0
            if "dist_embankment_km" in row.index
            else False
        )

        results.append({
            # Location
            "district"                  : row["district"],
            "mandal"                    : row["mandal"],
            "village"                   : row["village"],
            "latitude"                  : round(float(row["centroid_lat"]), 6),
            "longitude"                 : round(float(row["centroid_lon"]), 6),

            # Rainfall
            "rainfall_mm"               : rainfall_mm,
            "rainfall_category"         : classify_rainfall(float(rainfall_mm)),
            "rainfall_3day"             : r3_vals[i],
            "rainfall_7day"             : r7_vals[i],
            "rainfall_30day"            : r30_vals[i],

            # Infrastructure
            "dist_canal_km"             : round(float(row["dist_canal_km"]), 2)
                                          if "dist_canal_km" in row.index else 0.0,
            "nearest_canal_name"        : str(row["nearest_canal_name"])
                                          if "nearest_canal_name" in row.index
                                          and pd.notna(row["nearest_canal_name"])
                                          else "N/A",
            "dist_embankment_km"        : round(float(row["dist_embankment_km"]), 2)
                                          if "dist_embankment_km" in row.index else 0.0,
            "nearest_embankment_name"   : str(row["nearest_embankment_name"])
                                          if "nearest_embankment_name" in row.index
                                          and pd.notna(row["nearest_embankment_name"])
                                          else "N/A",
            "canal_proximity_score"     : int(row["canal_proximity_score"])
                                          if "canal_proximity_score" in row.index else 0,
            "embankment_proximity_score": int(row["embankment_proximity_score"])
                                          if "embankment_proximity_score" in row.index else 0,
            "has_embankment_nearby"     : has_embankment,

            # Alert
            "alert_level"               : alert,
            "alert_color"               : meta["color"],
            "alert_bg"                  : meta["bg"],
            "alert_emoji"               : meta["emoji"],
            "alert_label"               : meta["label"],
            "alert_message"             : meta["message"],
            "confidence_pct"            : confidence,

            # Probabilities
            "prob_green"                : round(float(proba[classes.index("GREEN")])  * 100, 1),
            "prob_yellow"               : round(float(proba[classes.index("YELLOW")]) * 100, 1),
            "prob_orange"               : round(float(proba[classes.index("ORANGE")]) * 100, 1),
            "prob_red"                  : round(float(proba[classes.index("RED")])    * 100, 1),
        })

    results.sort(key=lambda x: RISK_ORDER[x["alert_level"]])
    return results


# ─── Summary ──────────────────────────────────────────────────────────────────
def get_summary(results):
    summary = {
        "total" : len(results),
        "RED"   : 0,
        "ORANGE": 0,
        "YELLOW": 0,
        "GREEN" : 0,
    }
    for r in results:
        summary[r["alert_level"]] += 1

    if   summary["RED"]    > 0: summary["overall_alert"] = "RED"
    elif summary["ORANGE"] > 0: summary["overall_alert"] = "ORANGE"
    elif summary["YELLOW"] > 0: summary["overall_alert"] = "YELLOW"
    else:                        summary["overall_alert"] = "GREEN"

    return summary


# ─── Historical Stats (for graphs) ───────────────────────────────────────────
def get_district_stats(district, featured_parquet_path):
    df = pd.read_parquet(
        featured_parquet_path,
        filters=[("district", "==", district)],
    )
    df["date"] = pd.to_datetime(df["date"])

    monthly = (
        df.groupby(["year", "month"])["rainfall_mm"]
        .mean()
        .reset_index()
        .rename(columns={"rainfall_mm": "avg_rainfall"})
    )
    monthly["avg_rainfall"] = monthly["avg_rainfall"].round(2)

    alert_yearly = (
        df.groupby(["year", "alert_level"])
        .size()
        .reset_index(name="count")
    )

    top_villages = (
        df.groupby(["mandal", "village"])["rainfall_mm"]
        .max()
        .reset_index()
        .sort_values("rainfall_mm", ascending=False)
        .head(10)
    )

    return {
        "monthly_avg" : monthly.to_dict(orient="records"),
        "alert_yearly": alert_yearly.to_dict(orient="records"),
        "top_villages": top_villages.to_dict(orient="records"),
    }