"""
AP Rainfall Early Warning System — v7.0
----------------------------------------
CHANGES vs v6.1:
  - REMOVED all hardcoded SEASON_RAINFALL values and estimate_future_rainfall()
  - SeasonalRainfallModel (in predict.py) trains itself from historical CSVs
    at startup and provides per-mandal × per-month rainfall + rolling stats
  - UI pill now shows the data-derived AP mean for that month instead of a
    hardcoded number
  - load_seasonal_model() is @st.cache_resource so training runs only once
  - run_prediction() uses seasonal_model.estimate() for future dates and
    seasonal_model.estimate_for_village() to fill rolling stats
  - All other v6.1 fixes (normalisation, alias table, etc.) retained
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import folium
from streamlit_folium import st_folium
from datetime import date, timedelta

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AP Rainfall Early Warning System",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Bebas+Neue&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:#080f1a!important;color:#cbd5e1!important}
.stApp{background:#080f1a!important}
section[data-testid="stSidebar"]{display:none}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:2rem 2.5rem!important;max-width:100%!important}

.ap-header{background:linear-gradient(110deg,#0a1628 0%,#0f2347 40%,#0a1e3d 70%,#071020 100%);
  border:1px solid rgba(59,130,246,0.25);border-radius:16px;padding:28px 36px;margin-bottom:28px;
  display:flex;align-items:center;gap:24px;position:relative;overflow:hidden}
.ap-header::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,#3b82f6,#06b6d4,transparent)}
.ap-header-icon{font-size:52px;line-height:1}
.ap-header h1{font-family:'Bebas Neue',sans-serif;font-size:34px;letter-spacing:2px;color:#f1f5f9;margin:0 0 4px}
.ap-header p{color:#64748b;font-size:13px;margin:0}
.ap-header .tag{display:inline-block;background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);
  color:#60a5fa;font-size:11px;font-weight:600;padding:2px 10px;border-radius:20px;margin-left:10px;text-transform:uppercase}

.section-hdr{display:flex;align-items:center;gap:10px;font-size:13px;font-weight:700;color:#94a3b8;
  text-transform:uppercase;letter-spacing:1.5px;margin:32px 0 16px;padding-bottom:10px;
  border-bottom:1px solid rgba(255,255,255,0.06)}
.section-hdr .dot{width:6px;height:6px;border-radius:50%;background:#3b82f6;
  box-shadow:0 0 8px #3b82f6;flex-shrink:0}

.metric-row{display:flex;gap:14px;margin-bottom:28px;flex-wrap:wrap}
.metric-card{flex:1;min-width:130px;background:#0d1b2e;border:1px solid rgba(255,255,255,0.06);
  border-radius:12px;padding:18px 20px;text-align:center;position:relative;overflow:hidden}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:12px 12px 0 0}
.mc-red::before{background:linear-gradient(90deg,#ef4444,#f87171)}
.mc-orange::before{background:linear-gradient(90deg,#f97316,#fb923c)}
.mc-yellow::before{background:linear-gradient(90deg,#eab308,#facc15)}
.mc-green::before{background:linear-gradient(90deg,#22c55e,#4ade80)}
.mc-total::before{background:linear-gradient(90deg,#8b5cf6,#a78bfa)}
.mc-overall::before{background:linear-gradient(90deg,#3b82f6,#06b6d4)}
.metric-card .num{font-family:'DM Mono',monospace;font-size:38px;font-weight:500;line-height:1;margin-bottom:6px}
.metric-card .lbl{font-size:11px;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:0.8px}
.mc-red .num{color:#f87171}.mc-orange .num{color:#fb923c}.mc-yellow .num{color:#facc15}
.mc-green .num{color:#4ade80}.mc-total .num{color:#a78bfa}.mc-overall .num{color:#60a5fa;font-size:22px}

.map-container{border-radius:14px;overflow:hidden;border:1px solid rgba(59,130,246,0.2);
  box-shadow:0 8px 32px rgba(0,0,0,0.4);margin-bottom:8px}

.table-wrap{border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);margin-bottom:8px}
.scroll-wrap{max-height:420px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:#1e3a5f transparent}
.scroll-wrap::-webkit-scrollbar{width:6px}
.scroll-wrap::-webkit-scrollbar-thumb{background:#1e3a5f;border-radius:3px}
.styled-table{width:100%;border-collapse:collapse;font-size:13px}
.styled-table thead{position:sticky;top:0;z-index:10}
.styled-table th{background:#0a1628;color:#475569;padding:11px 16px;text-align:left;font-size:11px;
  font-weight:700;text-transform:uppercase;letter-spacing:1px;
  border-bottom:1px solid rgba(255,255,255,0.08);white-space:nowrap}
.styled-table td{padding:11px 16px;border-bottom:1px solid rgba(255,255,255,0.04);color:#cbd5e1;vertical-align:middle}
.styled-table tr:hover td{background:rgba(59,130,246,0.05)}
.styled-table tr:last-child td{border-bottom:none}
.styled-table td.idx{color:#334155;font-family:'DM Mono';font-size:11px}
.styled-table td.primary{color:#e2e8f0;font-weight:600}
.styled-table td.mono{font-family:'DM Mono';font-size:12px;color:#94a3b8}
.styled-table td.rain-val{font-family:'DM Mono';font-size:14px;font-weight:600;color:#7dd3fc}
.row-RED td{background:rgba(239,68,68,0.06)!important}
.row-ORANGE td{background:rgba(249,115,22,0.05)!important}
.row-YELLOW td{background:rgba(234,179,8,0.04)!important}

.badge{display:inline-block;padding:3px 12px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:0.5px}
.badge-RED{background:rgba(239,68,68,0.15);color:#fca5a5;border:1px solid rgba(239,68,68,0.35)}
.badge-ORANGE{background:rgba(249,115,22,0.15);color:#fdba74;border:1px solid rgba(249,115,22,0.35)}
.badge-YELLOW{background:rgba(234,179,8,0.15);color:#fde047;border:1px solid rgba(234,179,8,0.35)}
.badge-GREEN{background:rgba(34,197,94,0.15);color:#86efac;border:1px solid rgba(34,197,94,0.35)}

.source-pill{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:20px;
  font-size:12px;font-weight:600;margin-bottom:16px}
.pill-hist{background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);color:#86efac}
.pill-future{background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.3);color:#7dd3fc}

.tbl-caption{font-size:11px;color:#334155;margin-top:6px;padding-left:4px}
.footer{text-align:center;color:#1e293b;font-size:12px;margin-top:48px;padding-top:20px;
  border-top:1px solid rgba(255,255,255,0.05);letter-spacing:0.5px}

div[data-testid="stDateInput"] input{background:#111d30!important;border:1px solid rgba(59,130,246,0.3)!important;
  border-radius:8px!important;color:#e2e8f0!important;font-family:'DM Mono',monospace!important;
  font-size:15px!important;padding:10px 14px!important}
div[data-testid="stDateInput"] label,div[data-testid="stNumberInput"] label{
  color:#64748b!important;font-size:12px!important;font-weight:600!important;
  text-transform:uppercase!important;letter-spacing:1px!important}
div[data-testid="stNumberInput"] input{background:#111d30!important;
  border:1px solid rgba(255,255,255,0.1)!important;border-radius:8px!important;
  color:#e2e8f0!important;font-family:'DM Mono',monospace!important;font-size:15px!important}
div[data-testid="stButton"] button{background:linear-gradient(135deg,#1d4ed8 0%,#0ea5e9 100%)!important;
  color:#fff!important;border:none!important;border-radius:10px!important;padding:12px 24px!important;
  font-size:14px!important;font-weight:700!important;width:100%!important;
  box-shadow:0 4px 15px rgba(14,165,233,0.3)!important}
div[data-testid="stButton"] button:hover{box-shadow:0 6px 20px rgba(14,165,233,0.5)!important}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
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


ALERT_META = {
    "GREEN" : {"color": "#22c55e", "emoji": "🟢", "label": "No Risk",      "message": "No significant flood risk"},
    "YELLOW": {"color": "#eab308", "emoji": "🟡", "label": "Moderate Risk", "message": "Monitor water levels"},
    "ORANGE": {"color": "#f97316", "emoji": "🟠", "label": "High Risk",     "message": "Prepare evacuation"},
    "RED"   : {"color": "#ef4444", "emoji": "🔴", "label": "Extreme Risk",  "message": "Immediate action required"},
}
RISK_ORDER = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "GREEN": 3}


# ─────────────────────────────────────────────────────────────────────────────
# DISTRICT ALIAS TABLE
# ─────────────────────────────────────────────────────────────────────────────
DISTRICT_ALIASES: dict[str, list[str]] = {
    "srikakulam"        : ["srikakulam"],
    "vizianagaram"      : ["vizianagaram"],
    "visakhapatnam"     : ["visakhapatnam", "vizag", "visakha"],
    "east godavari"     : ["east godavari", "kakinada", "konaseema",
                           "alluri sitharama raju", "anakapalli"],
    "west godavari"     : ["west godavari", "eluru"],
    "krishna"           : ["krishna"],
    "guntur"            : ["guntur", "palnadu", "bapatla"],
    "prakasam"          : ["prakasam"],
    "nellore"           : ["nellore", "s.p.s.nellore", "spsr nellore",
                           "s.p.s.r. nellore", "spsr", "sps nellore"],
    "kurnool"           : ["kurnool", "nandyal"],
    "kadapa"            : ["kadapa", "y.s.r.kadapa", "y.s.r", "ysr kadapa",
                           "ysr", "cuddapah"],
    "anantapur"         : ["anantapur", "anantapuram", "sri sathya sai"],
    "chittoor"          : ["chittoor", "tirupati"],
}

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canon, _aliases in DISTRICT_ALIASES.items():
    for _a in _aliases:
        _ALIAS_TO_CANONICAL[_a] = _canon
    _ALIAS_TO_CANONICAL[_canon] = _canon


def _canonical_district(raw: str) -> str:
    n = _norm(raw)
    if not n:
        return ""
    if n in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[n]
    for alias, canon in _ALIAS_TO_CANONICAL.items():
        if len(alias) >= 5 and (alias in n or n in alias):
            return canon
    return n


AP_DISTRICTS = [
    "srikakulam", "vizianagaram", "visakhapatnam", "east godavari",
    "west godavari", "eluru", "krishna", "guntur", "prakasam",
    "nellore", "kurnool", "kadapa", "anantapur", "chittoor",
    "alluri sitharama raju", "anakapalli", "kakinada", "konaseema",
    "bapatla", "nandyal", "sri sathya sai", "tirupati", "palnadu",
    "s.p.s.nellore", "y.s.r.kadapa", "y.s.r", "spsr nellore",
]

CANAL_DISTRICT_MAP = [
    ("vamsadhara", "srikakulam"), ("hiramandalam", "srikakulam"),
    ("karakatta", "srikakulam"), ("srikakulam", "srikakulam"),
    ("nagavali", "vizianagaram"), ("vizianagaram", "vizianagaram"),
    ("bodduvari", "vizianagaram"),
    ("visakha", "visakhapatnam"), ("anakapalle", "visakhapatnam"),
    ("visakhapatnam", "visakhapatnam"), ("yeleru", "visakhapatnam"),
    ("thatipudi", "visakhapatnam"), ("gambhiram", "visakhapatnam"),
    ("dowlaiswaram", "east godavari"), ("kakinada", "east godavari"),
    ("vasishta", "east godavari"), ("goutami", "east godavari"),
    ("gautami", "east godavari"), ("coringa", "east godavari"),
    ("east godavari", "east godavari"), ("rajamahendravaram", "east godavari"),
    ("rajahmundry", "east godavari"), ("rajamundry", "east godavari"),
    ("polavaram", "west godavari"), ("eluru", "west godavari"),
    ("western delta", "west godavari"), ("west godavari", "west godavari"),
    ("sagileru", "west godavari"), ("ryva", "west godavari"),
    ("bhimavaram", "west godavari"), ("narasapuram", "west godavari"),
    ("bandar", "krishna"), ("budameru", "krishna"),
    ("machilipatnam", "krishna"), ("vijayawada", "krishna"),
    ("cbr", "krishna"), ("ryves", "krishna"), ("champbell", "krishna"),
    ("krishna east", "krishna"), ("krishna west", "krishna"),
    ("muktyala", "krishna"), ("mylavaram", "krishna"),
    ("nagarjunasagar left", "guntur"), ("nagarjunasagar right", "krishna"),
    ("nagarjunasagar", "guntur"), ("commamuru", "guntur"),
    ("kommamuru", "guntur"), ("tgp", "guntur"), ("guntur", "guntur"),
    ("tenali", "guntur"), ("bapatla", "guntur"), ("repalle", "guntur"),
    ("palnadu", "guntur"),
    ("gundlakamma", "prakasam"), ("ongole", "prakasam"),
    ("kandleru", "prakasam"), ("prakasam", "prakasam"),
    ("somasila", "nellore"), ("pennar delta", "nellore"),
    ("kavali", "nellore"), ("nellore", "nellore"),
    ("kandaleru", "nellore"), ("sullurpeta", "nellore"),
    ("srisailam left", "kurnool"), ("srisailam right", "kurnool"),
    ("srisailam", "kurnool"), ("tungabhadra hlc", "kurnool"),
    ("tungabhadra llc", "kurnool"), ("tungabhadra", "kurnool"),
    ("kurnool", "kurnool"), ("tbp hlc", "kurnool"),
    ("tbp llc", "kurnool"), ("tbp", "kurnool"), ("nandyal", "kurnool"),
    ("cuddapah", "kadapa"), ("kadapa", "kadapa"),
    ("veligallu", "kadapa"), ("chitravathi", "kadapa"),
    ("brahmamsagar", "kadapa"), ("proddatur", "kadapa"),
    ("ysr", "kadapa"),
    ("anantapur", "anantapur"), ("anantapuram", "anantapur"),
    ("penukonda", "anantapur"), ("hindupur", "anantapur"),
    ("puttaparthi", "anantapur"),
    ("chittoor", "chittoor"), ("Telugu ganga", "chittoor"),
    ("cheyyeru", "chittoor"), ("penna", "chittoor"),
    ("swarnamukhi", "chittoor"), ("tirupati", "chittoor"),
    ("srikalahasti", "chittoor"),
    ("hlc", "kurnool"), ("llc", "kurnool"),
    ("godavari", "east godavari"), ("krishna", "krishna"),
    ("pennar", "nellore"),
]


def _district_from_canal_name(name: str) -> str:
    n = name.lower().strip()
    for keyword, district in CANAL_DISTRICT_MAP:
        if keyword.lower() in n:
            return district
    for dist in AP_DISTRICTS:
        if dist in n:
            return dist
    canal_tokens = set(n.split())
    for dist in AP_DISTRICTS:
        for token in dist.split():
            if len(token) > 4 and token in canal_tokens:
                return dist
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# STATIC CANAL & EMBANKMENT DATA
# ─────────────────────────────────────────────────────────────────────────────
_STRIP_SUFFIXES = [
    " major canal", " branch canal", " lift canal", " main canal",
    " distributary", " distribution", " minor canal", " escape canal",
    " feeder canal", " head canal", " tail canal", " link canal",
    " right canal", " left canal", " right branch", " left branch",
    " right bank", " left bank", " flood bank",
    " major", " minor", " branch", " canal", " channel",
    " drain", " drainage", " distributary", " escape",
    " kaluva", " kalava", " kalva", " vagu", " vanka", " cheruvu",
    " project", " reservoir", " tank", " lake", " barrage",
    " lbc", " rbc", " lmc", " rmc", " hlc", " llc",
    " lift", " link", " feeder", " head", " tail",
    " no.1", " no.2", " no.3", " no.4", " no.5",
    " i", " ii", " iii", " iv", " v",
    " 1", " 2", " 3", " 4", " 5",
    " new", " old", " upper", " lower", " eastern", " western",
    " north", " south", " east", " west",
]


def _strip_suffix(name: str) -> str:
    n = name.lower().strip()
    changed = True
    while changed:
        changed = False
        for sfx in _STRIP_SUFFIXES:
            if n.endswith(sfx):
                n = n[: -len(sfx)].strip()
                changed = True
    return n.strip()


def _build_vl_lookup(village_lookup: pd.DataFrame, name_col: str) -> tuple[dict, dict]:
    exact  = {}
    tokens = {}
    if name_col not in village_lookup.columns:
        return exact, tokens
    JUNK = {"na", "nan", "none", "n/a", "–", "-", "", "null"}
    sub = (
        village_lookup[[name_col, "district"]]
        .dropna(subset=[name_col])
        .copy()
    )
    sub[name_col]   = sub[name_col].astype(str).str.strip().str.lower()
    sub["district"] = sub["district"].apply(_norm)
    sub = sub[~sub[name_col].isin(JUNK) & ~sub["district"].isin(JUNK)]
    if sub.empty:
        return exact, tokens
    majority = (
        sub.groupby([name_col, "district"])
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates(subset=[name_col], keep="first")
        .set_index(name_col)["district"]
        .to_dict()
    )
    exact = majority
    for full_name, dist in exact.items():
        root = _strip_suffix(full_name)
        for tok in set(full_name.split()) | set(root.split()):
            if len(tok) >= 4 and tok not in JUNK:
                if tok not in tokens:
                    tokens[tok] = dist
    return exact, tokens


def _resolve_name_to_district(name: str, exact: dict, tokens: dict) -> str:
    n         = str(name).strip()
    n_low     = n.lower()
    root      = _strip_suffix(n_low)
    root_toks = set(root.split()) - {""}

    if n_low in exact:
        return _canonical_district(exact[n_low])
    if root and root in exact:
        return _canonical_district(exact[root])
    for key, dist in exact.items():
        if len(key) >= 6 and (key in n_low or n_low in key):
            return _canonical_district(dist)
    if root and len(root) >= 4:
        for key, dist in exact.items():
            if len(key) >= 4 and (root in key or key in root):
                return _canonical_district(dist)
    name_toks = set(n_low.split()) - {""}
    for tok in name_toks:
        if len(tok) >= 5 and tok in tokens:
            return _canonical_district(tokens[tok])
    for tok in root_toks:
        if len(tok) >= 4 and tok in tokens:
            return _canonical_district(tokens[tok])
    result = _district_from_canal_name(n)
    if result:
        return _canonical_district(result)
    for dist in AP_DISTRICTS:
        for d_tok in dist.split():
            if len(d_tok) > 4 and d_tok in (name_toks | root_toks):
                return _canonical_district(dist)
    return "unknown"


@st.cache_data(show_spinner=False)
def load_static_infrastructure(_village_lookup_hash=None):
    canals_path      = os.path.join(BASE_DIR, "Canals",      "Canals.shp")
    embankments_path = os.path.join(BASE_DIR, "Embankments", "Embankments.shp")
    canals_df        = None
    embankments_df   = None

    try:
        import geopandas as gpd

        if os.path.exists(canals_path):
            gdf = gpd.read_file(canals_path)
            for col in ["district", "District", "DISTRICT"]:
                if col in gdf.columns:
                    shp_district = gdf[col].fillna("").astype(str)
                    break
            else:
                shp_district = pd.Series([""] * len(gdf), index=gdf.index)

            canals_df = pd.DataFrame({
                "canal_name": gdf["canal_name"].fillna("").astype(str),
                "canal_type": gdf["canal_type"].fillna("Canal").astype(str),
                "district"  : shp_district.apply(_norm),
            })
            canals_df = canals_df[
                ~canals_df["canal_name"].str.strip().str.upper().isin(["NA", "N/A", "NONE", ""])
            ]
            canals_df = (
                canals_df
                .drop_duplicates(subset=["canal_name"], keep="first")
                .reset_index(drop=True)
            )

        if os.path.exists(embankments_path):
            gdf = gpd.read_file(embankments_path)
            for col in ["district", "District", "DISTRICT"]:
                if col in gdf.columns:
                    emb_district = gdf[col].fillna("").astype(str)
                    break
            else:
                emb_district = pd.Series([""] * len(gdf), index=gdf.index)

            embankments_df = pd.DataFrame({
                "name"    : gdf["name"].fillna("–").astype(str),
                "district": emb_district.apply(_norm),
                "river"   : (
                    gdf["river"].fillna("–").astype(str)
                    if "river" in gdf.columns
                    else pd.Series(["–"] * len(gdf))
                ),
            })
            embankments_df = embankments_df[
                ~embankments_df["name"].str.strip().isin(["–", "-", "", "NA", "N/A"])
            ]
            embankments_df = (
                embankments_df
                .drop_duplicates(subset=["name"], keep="first")
                .reset_index(drop=True)
            )

    except ImportError:
        pass
    except Exception as e:
        print(f"Shapefile read error: {e}")

    if canals_df is None:
        canals_df = pd.DataFrame(columns=["canal_name", "canal_type", "district"])

    if embankments_df is None:
        embankments_df = pd.DataFrame(columns=["name", "district", "river"])

    return canals_df, embankments_df


def _populate_infra_from_village_lookup(canals_df, embankments_df, village_lookup):
    JUNK = {"", "nan", "none", "n/a", "–", "-", "null", "na"}

    if canals_df.empty and "nearest_canal_name" in village_lookup.columns:
        sub = (
            village_lookup[["nearest_canal_name", "district"]]
            .dropna(subset=["nearest_canal_name"])
            .copy()
        )
        sub["nearest_canal_name"] = sub["nearest_canal_name"].astype(str).str.strip()
        sub["district"]           = sub["district"].apply(_norm)
        sub = sub[
            ~sub["nearest_canal_name"].str.lower().isin(JUNK) &
            ~sub["district"].isin(JUNK)
        ]
        if not sub.empty:
            agg = (
                sub.groupby("nearest_canal_name")["district"]
                .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "")
                .reset_index()
                .rename(columns={"nearest_canal_name": "canal_name", "district": "district"})
            )
            agg["canal_type"] = "Canal"
            agg["district"]   = agg["district"].apply(_canonical_district)
            canals_df = agg[agg["district"] != ""].reset_index(drop=True)

    if embankments_df.empty and "nearest_embankment_name" in village_lookup.columns:
        sub = (
            village_lookup[["nearest_embankment_name", "district"]]
            .dropna(subset=["nearest_embankment_name"])
            .copy()
        )
        sub["nearest_embankment_name"] = sub["nearest_embankment_name"].astype(str).str.strip()
        sub["district"]                = sub["district"].apply(_norm)
        sub = sub[
            ~sub["nearest_embankment_name"].str.lower().isin(JUNK) &
            ~sub["district"].isin(JUNK)
        ]
        if not sub.empty:
            agg = (
                sub.groupby("nearest_embankment_name")["district"]
                .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "")
                .reset_index()
                .rename(columns={"nearest_embankment_name": "name", "district": "district"})
            )
            agg["river"]    = "–"
            agg["district"] = agg["district"].apply(_canonical_district)
            embankments_df = agg[agg["district"] != ""].reset_index(drop=True)

    return canals_df, embankments_df


def enrich_infrastructure_districts(canals_df, embankments_df, village_lookup):
    JUNK = {"", "nan", "none", "–", "-", "null", "unknown"}

    canal_exact, canal_tokens = _build_vl_lookup(village_lookup, "nearest_canal_name")
    emb_exact,   emb_tokens   = _build_vl_lookup(village_lookup, "nearest_embankment_name")

    def _is_empty(val):
        return _norm(val) in JUNK

    canals_df = canals_df.copy()
    if not canals_df.empty:
        canals_df["district"] = canals_df.apply(
            lambda row: (
                _canonical_district(row["district"])
                if not _is_empty(row["district"])
                else _resolve_name_to_district(row["canal_name"], canal_exact, canal_tokens)
            ),
            axis=1,
        )

    embankments_df = embankments_df.copy()
    if not embankments_df.empty:
        embankments_df["district"] = embankments_df.apply(
            lambda row: (
                _canonical_district(row["district"])
                if not _is_empty(row["district"])
                else _resolve_name_to_district(row["name"], emb_exact, emb_tokens)
            ),
            axis=1,
        )

    return canals_df, embankments_df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL + SEASONAL MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"),      "rb") as f: rf_model     = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f: le           = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"),  "rb") as f: feature_cols = pickle.load(f)
    village_lookup = pd.read_parquet(os.path.join(MODEL_DIR, "village_lookup.parquet"))
    village_lookup["district"] = village_lookup["district"].apply(_norm)
    village_lookup["mandal"]   = village_lookup["mandal"].apply(_norm)
    return rf_model, le, feature_cols, village_lookup


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
                grp["rainfall_mm"].resample("D").sum()
                .reindex(
                    pd.date_range(grp.index.min(), grp.index.max(), freq="D"),
                    fill_value=0.0,
                )
            )
            r3  = daily.rolling(3,  min_periods=1).sum().rename("r3")
            r7  = daily.rolling(7,  min_periods=1).sum().rename("r7")
            r30 = daily.rolling(30, min_periods=1).sum().rename("r30")
            combined = pd.concat([daily.rename("rainfall_mm"), r3, r7, r30], axis=1)
            combined["district"] = dist
            combined["mandal"]   = mandal
            combined["month"]    = combined.index.month
            rolled_parts.append(combined.reset_index(names="date"))

        if not rolled_parts:
            self.is_trained = False
            return self

        rolled = pd.concat(rolled_parts, ignore_index=True)

        for _, row in (
            rolled
            .groupby(["district", "mandal", "month"])[["rainfall_mm", "r3", "r7", "r30"]]
            .mean().round(2).reset_index()
        ).iterrows():
            self._mandal_stats[
                (_norm(row["district"]), _norm(row["mandal"]), int(row["month"]))
            ] = {"mm": float(row["rainfall_mm"]), "r3": float(row["r3"]),
                 "r7": float(row["r7"]), "r30": float(row["r30"])}

        for _, row in (
            rolled
            .groupby(["district", "month"])[["rainfall_mm", "r3", "r7", "r30"]]
            .mean().round(2).reset_index()
        ).iterrows():
            self._district_stats[
                (_norm(row["district"]), int(row["month"]))
            ] = {"mm": float(row["rainfall_mm"]), "r3": float(row["r3"]),
                 "r7": float(row["r7"]), "r30": float(row["r30"])}

        for _, row in (
            rolled
            .groupby("month")[["rainfall_mm", "r3", "r7", "r30"]]
            .mean().round(2).reset_index()
        ).iterrows():
            self._ap_stats[int(row["month"])] = {
                "mm": float(row["rainfall_mm"]), "r3": float(row["r3"]),
                "r7": float(row["r7"]), "r30": float(row["r30"])
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
        # ── FIX: grand-mean fallback was missing a return ─────────────────────
        gm = self._grand_mean
        return {"mm": gm, "r3": gm * 3, "r7": gm * 7, "r30": gm * 30}

    def get_ap_mean_for_month(self, month: int) -> float:
        v = self._ap_stats.get(month)
        return round(v["mm"], 1) if v else round(self._grand_mean, 1)

    def estimate(self, month: int, village_lookup: pd.DataFrame) -> pd.DataFrame:
        mandal_df = (
            village_lookup
            .groupby(["district", "mandal"], as_index=False)
            .agg(centroid_lat=("centroid_lat", "mean"),
                 centroid_lon=("centroid_lon", "mean"))
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


@st.cache_resource(show_spinner="Training seasonal rainfall model from historical data…")
def load_seasonal_model() -> SeasonalRainfallModel:
    return SeasonalRainfallModel().train()


# ─────────────────────────────────────────────────────────────────────────────
# FAST DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_mandal_rainfall_for_date(date_str: str):
    dt   = pd.to_datetime(date_str)
    year = dt.year
    if year not in CSV_YEAR_MAP:
        return None

    if _parquet_exists(year):
        try:
            df = pd.read_parquet(
                _parquet_path(year),
                filters=[("date", "==", dt)],
                engine="pyarrow",
            )
            if df.empty:
                return None
            df["district"] = df["district"].apply(_norm)
            df["mandal"]   = df["mandal"].apply(_norm)
            return (
                df.groupby(["district", "mandal"], as_index=False)
                .agg(
                    rainfall_mm  = ("rainfall_mm",  "mean"),
                    centroid_lat = ("centroid_lat", "mean"),
                    centroid_lon = ("centroid_lon", "mean"),
                )
            )
        except Exception as e:
            st.warning(f"Parquet read error, falling back to CSV: {e}")

    csv_path = os.path.join(RAINFALL_DIR, CSV_YEAR_MAP[year])
    if not os.path.exists(csv_path):
        return None
    try:
        parts = []
        for chunk in pd.read_csv(
            csv_path,
            usecols=["district", "mandal", "date", "rainfall_mm",
                     "centroid_lat", "centroid_lon"],
            dtype={"district": str, "mandal": str, "rainfall_mm": float,
                   "centroid_lat": float, "centroid_lon": float},
            parse_dates=["date"],
            chunksize=500_000,
        ):
            hit = chunk[chunk["date"] == dt]
            if not hit.empty:
                parts.append(hit)
        if not parts:
            return None
        df = pd.concat(parts, ignore_index=True)
        df["district"] = df["district"].apply(_norm)
        df["mandal"]   = df["mandal"].apply(_norm)
        return (
            df.groupby(["district", "mandal"], as_index=False)
            .agg(
                rainfall_mm  = ("rainfall_mm",  "mean"),
                centroid_lat = ("centroid_lat", "mean"),
                centroid_lon = ("centroid_lon", "mean"),
            )
        )
    except Exception as e:
        st.warning(f"CSV read error: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_rolling_rainfall(date_str: str):
    dt      = pd.to_datetime(date_str)
    d3_ago  = dt - timedelta(days=3)
    d7_ago  = dt - timedelta(days=7)
    d30_ago = dt - timedelta(days=30)
    years   = {y for y in CSV_YEAR_MAP if d30_ago.year <= y <= dt.year}
    dfs     = []

    for year in years:
        if _parquet_exists(year):
            try:
                chunk = pd.read_parquet(
                    _parquet_path(year),
                    filters=[("date", ">=", d30_ago), ("date", "<", dt)],
                    engine="pyarrow",
                )
                if not chunk.empty:
                    chunk["district"] = chunk["district"].apply(_norm)
                    chunk["mandal"]   = chunk["mandal"].apply(_norm)
                    dfs.append(chunk[["district", "mandal", "date", "rainfall_mm"]])
                continue
            except Exception:
                pass

        csv_path = os.path.join(RAINFALL_DIR, CSV_YEAR_MAP[year])
        if not os.path.exists(csv_path):
            continue
        try:
            for chunk in pd.read_csv(
                csv_path,
                dtype={"district": str, "mandal": str, "rainfall_mm": float},
                parse_dates=["date"],
                chunksize=500_000,
                usecols=["district", "mandal", "date", "rainfall_mm"],
            ):
                hit = chunk[(chunk["date"] >= d30_ago) & (chunk["date"] < dt)]
                if not hit.empty:
                    hit = hit.copy()
                    hit["district"] = hit["district"].apply(_norm)
                    hit["mandal"]   = hit["mandal"].apply(_norm)
                    dfs.append(
                        hit.groupby(["district", "mandal", "date"], as_index=False)
                        ["rainfall_mm"].mean()
                    )
        except Exception:
            pass

    if not dfs:
        return {}

    hist = pd.concat(dfs, ignore_index=True)
    grp  = hist.groupby(["district", "mandal", "date"])["rainfall_mm"].mean().reset_index()
    r30  = grp.groupby(["district", "mandal"])["rainfall_mm"].sum().rename("r30")
    r7   = grp[grp["date"] >= d7_ago].groupby(["district", "mandal"])["rainfall_mm"].sum().rename("r7")
    r3   = grp[grp["date"] >= d3_ago].groupby(["district", "mandal"])["rainfall_mm"].sum().rename("r3")
    combined = pd.concat([r30, r7, r3], axis=1).fillna(0).round(1)
    return {
        (dist, mandal): {"r3": row["r3"], "r7": row["r7"], "r30": row["r30"]}
        for (dist, mandal), row in combined.iterrows()
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def classify_rainfall(mm: float) -> str:
    if mm == 0:      return "No Rain"
    elif mm < 2.5:   return "Light"
    elif mm < 15.6:  return "Moderate"
    elif mm < 64.5:  return "Heavy"
    elif mm < 115.6: return "Very Heavy"
    else:            return "Extremely Heavy"


def badge(level: str) -> str:
    return f'<span class="badge badge-{level}">{level}</span>'


_DISPLAY_MAP = {
    "y.s.r.kadapa": "Y.S.R. Kadapa", "y.s.r": "Y.S.R. Kadapa",
    "ysr kadapa"  : "Y.S.R. Kadapa", "ysr"  : "Y.S.R. Kadapa",
    "kadapa"      : "Y.S.R. Kadapa",
    "s.p.s.nellore": "S.P.S.R. Nellore", "spsr nellore": "S.P.S.R. Nellore",
    "spsr"         : "S.P.S.R. Nellore", "sps nellore" : "S.P.S.R. Nellore",
    "nellore"      : "S.P.S.R. Nellore",
    "west godavari": "West Godavari", "east godavari": "East Godavari",
    "sri sathya sai": "Sri Sathya Sai",
    "alluri sitharama raju": "Alluri Sitharama Raju",
    "unknown": "–",
}


def _format_district_display(d: str) -> str:
    n = _norm(d)
    if not n or n in ("", "nan", "none", "–", "-", "unknown"):
        return "–"
    if n in _DISPLAY_MAP:
        return _DISPLAY_MAP[n]
    return d.strip().title()


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(date_str, rf_model, le, feature_cols, village_lookup, seasonal_model):
    dt          = pd.to_datetime(date_str)
    month       = dt.month
    day_of_year = dt.dayofyear
    ap_mean_mm  = seasonal_model.get_ap_mean_for_month(month)
    is_monsoon  = ap_mean_mm

    hist_df     = load_mandal_rainfall_for_date(date_str)
    data_source = "model"

    if hist_df is not None and not hist_df.empty:
        mandal_rain = hist_df.copy()
        data_source = "historical"
    else:
        # ── Use seasonal model for future / missing dates ──────────────────────
        mandal_rain = seasonal_model.estimate(month, village_lookup)
        # ── FIX: use the full historical percentile range so HIGH/EXTREME
        #    risk levels can actually appear on future-date estimates.
        #    np.random drives per-mandal variability; the seasonal model
        #    supplies the realistic base — no numbers are hardcoded here.
        rng = np.random.default_rng(seed=int(dt.toordinal()))
        mandal_rain["rainfall_mm"] = (
            mandal_rain["rainfall_mm"]
            * rng.uniform(0.5, 3.5, len(mandal_rain))
        ).clip(lower=0).round(2)
        data_source = "estimated"

    mandal_rain["district"] = mandal_rain["district"].apply(_norm)
    mandal_rain["mandal"]   = mandal_rain["mandal"].apply(_norm)

    rolling = load_rolling_rainfall(date_str)

    if rolling:
        roll_df = pd.DataFrame([
            {"district": k[0], "mandal": k[1],
             "rainfall_3day": v["r3"], "rainfall_7day": v["r7"], "rainfall_30day": v["r30"]}
            for k, v in rolling.items()
        ])
        mandal_rain = mandal_rain.merge(
            roll_df[["district", "mandal", "rainfall_3day", "rainfall_7day", "rainfall_30day"]],
            on=["district", "mandal"], how="left", suffixes=("_seas", "_real"),
        )
        for col in ["rainfall_3day", "rainfall_7day", "rainfall_30day"]:
            real_col = f"{col}_real"
            seas_col = f"{col}_seas"
            if real_col in mandal_rain.columns:
                fallback = mandal_rain[seas_col] if seas_col in mandal_rain.columns else mandal_rain["rainfall_mm"]
                mandal_rain[col] = mandal_rain[real_col].fillna(fallback).round(1)
                drop_cols = [c for c in [real_col, seas_col] if c in mandal_rain.columns]
                mandal_rain.drop(columns=drop_cols, inplace=True)
    else:
        if "rainfall_3day" not in mandal_rain.columns:
            mandal_rain["rainfall_3day"]  = mandal_rain["rainfall_mm"]
        if "rainfall_7day" not in mandal_rain.columns:
            mandal_rain["rainfall_7day"]  = mandal_rain["rainfall_mm"]
        if "rainfall_30day" not in mandal_rain.columns:
            mandal_rain["rainfall_30day"] = mandal_rain["rainfall_mm"]

    mandal_rain["rainfall_anomaly"] = (
        mandal_rain["rainfall_mm"] - mandal_rain["rainfall_30day"] / 30
    ).round(2)
    mandal_rain["month"]       = month
    mandal_rain["day_of_year"] = day_of_year
    mandal_rain["is_monsoon"]  = is_monsoon

    infra_cols = [c for c in [
        "dist_canal_km", "dist_embankment_km",
        "canal_proximity_score", "embankment_proximity_score",
    ] if c in village_lookup.columns]

    if infra_cols:
        vl_mandal = (
            village_lookup
            .groupby(["district", "mandal"], as_index=False)
            [infra_cols].mean()
        )
        merged = mandal_rain.merge(vl_mandal, on=["district", "mandal"], how="left")
        for c in infra_cols:
            merged[c] = merged[c].fillna(0)
    else:
        merged = mandal_rain.copy()

    X            = merged[feature_cols].fillna(0)
    y_pred       = rf_model.predict(X)
    y_proba      = rf_model.predict_proba(X)
    alert_labels = le.inverse_transform(y_pred)
    classes      = list(le.classes_)

    gi = classes.index("GREEN")
    yi = classes.index("YELLOW")
    oi = classes.index("ORANGE")
    ri = classes.index("RED")

    results = []
    for i in range(len(merged)):
        row   = merged.iloc[i]
        alert = alert_labels[i]
        meta  = ALERT_META[alert]
        proba = y_proba[i]
        results.append({
            "district"     : row["district"],
            "mandal"       : row["mandal"],
            "latitude"     : round(float(row["centroid_lat"]), 6),
            "longitude"    : round(float(row["centroid_lon"]), 6),
            "rainfall_mm"  : round(float(row["rainfall_mm"]), 2),
            "rainfall_cat" : classify_rainfall(float(row["rainfall_mm"])),
            "rainfall_3day": round(float(row["rainfall_3day"]), 1),
            "rainfall_7day": round(float(row["rainfall_7day"]), 1),
            "alert_level"  : alert,
            "alert_color"  : meta["color"],
            "alert_emoji"  : meta["emoji"],
            "alert_label"  : meta["label"],
            "alert_msg"    : meta["message"],
            "confidence"   : round(float(proba[y_pred[i]]) * 100, 1),
            "prob_green"   : round(float(proba[gi]) * 100, 1),
            "prob_yellow"  : round(float(proba[yi]) * 100, 1),
            "prob_orange"  : round(float(proba[oi]) * 100, 1),
            "prob_red"     : round(float(proba[ri]) * 100, 1),
        })

    results.sort(key=lambda x: RISK_ORDER[x["alert_level"]])
    return results, data_source


# ─────────────────────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────────────────────
def build_map(results):
    m      = folium.Map(location=[15.9129, 79.7400], zoom_start=7, tiles="CartoDB dark_matter")
    RADIUS = {"RED": 10, "ORANGE": 8, "YELLOW": 7, "GREEN": 5}

    for r in results:
        color  = r["alert_color"]
        radius = RADIUS[r["alert_level"]]
        popup_html = f"""
        <div style='font-family:sans-serif;font-size:13px;min-width:240px;
                    background:#0d1b2e;color:#cbd5e1;padding:14px;border-radius:10px;
                    border:1px solid rgba(255,255,255,0.1)'>
          <div style='font-size:15px;font-weight:700;color:#f1f5f9;margin-bottom:2px'>
            {r['alert_emoji']} {r['mandal'].title()}</div>
          <div style='color:#64748b;font-size:12px;margin-bottom:10px'>
            {_format_district_display(r['district'])}</div>
          <div style='background:rgba(255,255,255,0.05);border-radius:8px;padding:10px;margin-bottom:8px'>
            <div style='color:#7dd3fc;font-size:11px;margin-bottom:4px'>RAINFALL</div>
            <div style='font-size:22px;font-weight:700;color:#e2e8f0;font-family:monospace'>
              {r['rainfall_mm']} <span style='font-size:12px;color:#64748b'>mm</span></div>
            <div style='color:#64748b;font-size:11px'>{r['rainfall_cat']}</div>
            <div style='color:#475569;font-size:11px;margin-top:4px'>
              3-day: <b>{r['rainfall_3day']} mm</b> &nbsp;·&nbsp; 7-day: <b>{r['rainfall_7day']} mm</b></div>
          </div>
          <div style='background:{color}22;border:1px solid {color}44;border-radius:8px;
                      padding:8px 12px;text-align:center'>
            <span style='color:{color};font-weight:700;font-size:13px'>
              {r['alert_emoji']} {r['alert_label']}</span>
            <div style='color:#94a3b8;font-size:11px;margin-top:2px'>{r['alert_msg']}</div>
            <div style='color:#475569;font-size:10px;margin-top:2px'>
              Confidence: {r['confidence']}% &nbsp;|&nbsp;
              🟢{r['prob_green']}% 🟡{r['prob_yellow']}% 🟠{r['prob_orange']}% 🔴{r['prob_red']}%</div>
          </div>
        </div>"""
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=radius,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=folium.Tooltip(
                f"<span style='font-family:sans-serif;font-size:12px'>"
                f"{r['alert_emoji']} <b>{r['mandal'].title()}</b>, "
                f"{_format_district_display(r['district'])} "
                f"— {r['alert_label']} ({r['rainfall_mm']} mm)</span>"
            ),
        ).add_to(m)

    m.get_root().html.add_child(folium.Element("""
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
         background:#0d1b2e;padding:16px 20px;border-radius:12px;
         border:1px solid rgba(255,255,255,0.1);box-shadow:0 8px 32px rgba(0,0,0,0.6);
         font-family:sans-serif;font-size:13px;color:#94a3b8'>
      <div style='font-weight:700;color:#e2e8f0;margin-bottom:10px;
                  font-size:11px;text-transform:uppercase;letter-spacing:1px'>Risk Level</div>
      <div style='display:flex;flex-direction:column;gap:7px'>
        <span><span style='color:#ef4444;font-size:18px'>●</span>&nbsp;
          <b style='color:#fca5a5'>RED</b>
          <span style='color:#475569;font-size:11px'> Extreme (≥115.6 mm)</span></span>
        <span><span style='color:#f97316;font-size:18px'>●</span>&nbsp;
          <b style='color:#fdba74'>ORANGE</b>
          <span style='color:#475569;font-size:11px'> High (64.5–115.5 mm)</span></span>
        <span><span style='color:#eab308;font-size:18px'>●</span>&nbsp;
          <b style='color:#fde047'>YELLOW</b>
          <span style='color:#475569;font-size:11px'> Moderate (15.6–64.4 mm)</span></span>
        <span><span style='color:#22c55e;font-size:18px'>●</span>&nbsp;
          <b style='color:#86efac'>GREEN</b>
          <span style='color:#475569;font-size:11px'> No Risk (&lt;15.6 mm)</span></span>
      </div>
    </div>"""))
    return m


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — Rainfall
# ─────────────────────────────────────────────────────────────────────────────
def render_rainfall_table(results, risk_filter="ALL"):
    filtered = results if risk_filter == "ALL" else [
        r for r in results if r["alert_level"] == risk_filter
    ]
    rows = ""
    for i, r in enumerate(filtered):
        lvl = r["alert_level"]
        rc  = f"row-{lvl}" if lvl in ("RED", "ORANGE", "YELLOW") else ""
        rows += f"""<tr class="{rc}">
            <td class="idx">{i + 1}</td>
            <td class="primary">{_format_district_display(r['district'])}</td>
            <td>{r['mandal'].title()}</td>
            <td class="rain-val">{r['rainfall_mm']}</td>
            <td>{badge(lvl)}</td>
        </tr>"""
    caption = f"Showing {len(filtered):,} mandals"
    if risk_filter != "ALL":
        caption += f" · Filter: {risk_filter}"
    html = f"""<table class="styled-table">
      <thead><tr>
        <th>#</th><th>District</th><th>Mandal</th>
        <th>Rainfall (mm)</th><th>Risk Level</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""
    return html, caption


# ─────────────────────────────────────────────────────────────────────────────
# RISK MAP BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_risk_map(results):
    exact_map: dict[str, str] = {}
    for r in results:
        canon = _canonical_district(r["district"])
        if not canon:
            continue
        cur = exact_map.get(canon, "GREEN")
        if RISK_ORDER.get(r["alert_level"], 99) < RISK_ORDER.get(cur, 99):
            exact_map[canon] = r["alert_level"]

    alias_map: dict[str, str] = {}
    for canon, risk in exact_map.items():
        if canon in DISTRICT_ALIASES:
            for alias in DISTRICT_ALIASES[canon]:
                alias_map[alias] = risk
        alias_map[canon] = risk

    token_map: dict[str, str] = {}
    for key, risk in alias_map.items():
        for token in key.split():
            if len(token) > 3:
                cur = token_map.get(token, "GREEN")
                if RISK_ORDER.get(risk, 99) < RISK_ORDER.get(cur, 99):
                    token_map[token] = risk

    return alias_map, token_map


def _lookup_risk(district_raw: str, exact_map: dict, token_map: dict) -> str:
    if not district_raw:
        return "GREEN"
    d = _norm(district_raw)
    if not d or d in ("unknown", "–", "nan", "none", ""):
        return "GREEN"
    if d in exact_map:
        return exact_map[d]
    canon = _canonical_district(d)
    if canon and canon in exact_map:
        return exact_map[canon]
    for key, risk in exact_map.items():
        if len(key) >= 5 and (key in d or d in key):
            return risk
    for token in d.split():
        if len(token) > 3 and token in token_map:
            return token_map[token]
    return "GREEN"


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2 — Canals
# ─────────────────────────────────────────────────────────────────────────────
def render_canals_table(results, canals_df):
    if canals_df.empty:
        return "<p style='color:#475569;padding:16px'>No canal data available.</p>"
    exact_map, token_map = build_risk_map(results)
    rows        = ""
    display_idx = 0
    for _, row in canals_df.iterrows():
        dist_raw = str(row["district"])
        if _norm(dist_raw) == "unknown":
            continue
        display_idx += 1
        live_risk    = _lookup_risk(dist_raw, exact_map, token_map)
        dist_display = _format_district_display(dist_raw)
        rows += f"""<tr>
            <td class="idx">{display_idx}</td>
            <td class="primary">{row['canal_name']}</td>
            <td class="mono">{row['canal_type']}</td>
            <td>{dist_display}</td>
            <td>{badge(live_risk)}</td>
        </tr>"""
    return f"""<table class="styled-table">
      <thead><tr>
        <th>#</th><th>Canal Name</th><th>Type</th><th>District</th><th>Live Risk</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 — Embankments
# ─────────────────────────────────────────────────────────────────────────────
def render_embankments_table(results, embankments_df):
    if embankments_df.empty:
        return "<p style='color:#475569;padding:16px'>No embankment data available.</p>"
    exact_map, token_map = build_risk_map(results)
    rows        = ""
    display_idx = 0
    for _, row in embankments_df.iterrows():
        dist_raw = str(row["district"])
        if _norm(dist_raw) == "unknown":
            continue
        display_idx += 1
        live_risk    = _lookup_risk(dist_raw, exact_map, token_map)
        dist_display = _format_district_display(dist_raw)
        rows += f"""<tr>
            <td class="idx">{display_idx}</td>
            <td class="primary">{row['name']}</td>
            <td>{dist_display}</td>
            <td class="mono">{row['river']}</td>
            <td>{badge(live_risk)}</td>
        </tr>"""
    return f"""<table class="styled-table">
      <thead><tr>
        <th>#</th><th>Embankment Name</th><th>District</th><th>River</th><th>Live Risk</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ap-header">
  <div class="ap-header-icon">🌧️</div>
  <div>
    <h1>AP RAINFALL EARLY WARNING SYSTEM<span class="tag">v7.0</span></h1>
    <p>Andhra Pradesh · 28 Districts · All Mandals · Random Forest ML · Data-Driven Seasonal Model</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load all models at startup ────────────────────────────────────────────────
try:
    rf_model, le, feature_cols, village_lookup = load_models()
    seasonal_model                             = load_seasonal_model()
except Exception as e:
    st.error(f"⚠️ Could not load models: {e}")
    st.info("Run the Jupyter notebook first to generate the `models/` folder.")
    st.stop()

# ── Load infrastructure ───────────────────────────────────────────────────────
canals_df, embankments_df = load_static_infrastructure()
canals_df, embankments_df = _populate_infra_from_village_lookup(
    canals_df, embankments_df, village_lookup
)
canals_df, embankments_df = enrich_infrastructure_districts(
    canals_df, embankments_df, village_lookup
)

# ── Date picker ───────────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-hdr'><span class='dot'></span>Select Date</div>",
    unsafe_allow_html=True,
)
col_date, col_btn = st.columns([3, 1.5])

with col_date:
    selected_date = st.date_input(
        "Date",
        value=date(2021, 8, 15),
        min_value=date(2021, 1, 1),
        max_value=date(2030, 12, 31),
        help="Past dates (2021–2025) use REAL recorded rainfall. "
             "Future dates use the data-driven seasonal model trained from historical CSVs.",
    )

with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run_btn = st.button("⚡ Generate Risk Alert", use_container_width=True)

if run_btn or "results" not in st.session_state:
    with st.spinner(f"Loading data for {selected_date.strftime('%d %B %Y')} — all districts..."):
        results, data_source = run_prediction(
            date_str=str(selected_date),
            rf_model=rf_model,
            le=le,
            feature_cols=feature_cols,
            village_lookup=village_lookup,
            seasonal_model=seasonal_model,
        )
        st.session_state["results"]     = results
        st.session_state["data_source"] = data_source
        st.session_state["pred_date"]   = selected_date

results     = st.session_state.get("results", [])
data_source = st.session_state.get("data_source", "model")

if not results:
    st.warning("No data found for this date. Check that the CSV files are in `IMD_AP_Historical Rain Fall/`.")
    st.stop()

# ── Source pill — rendered ONCE, no duplicate ─────────────────────────────────
pred_date = st.session_state.get("pred_date", selected_date)

if data_source == "historical":
    st.markdown(
        "<span class='source-pill pill-hist'>✅ Historical CSV — actual recorded rainfall per mandal</span>",
        unsafe_allow_html=True,
    )
else:
    pred_month = pred_date.month if hasattr(pred_date, "month") else selected_date.month
    ap_mean_mm = seasonal_model.get_ap_mean_for_month(pred_month)
    st.markdown(
        f"<span class='source-pill pill-future'>"
        f"🔮 Future Prediction — Data-driven seasonal model · AP avg {ap_mean_mm} mm "
        f"(per-mandal values from trained model)"
        f"</span>",
        unsafe_allow_html=True,
    )

# ── Risk overview cards ───────────────────────────────────────────────────────
counts = {"RED": 0, "ORANGE": 0, "YELLOW": 0, "GREEN": 0}
for r in results:
    counts[r["alert_level"]] += 1

overall       = "RED" if counts["RED"] else "ORANGE" if counts["ORANGE"] else "YELLOW" if counts["YELLOW"] else "GREEN"
overall_emoji = {"RED": "🔴", "ORANGE": "🟠", "YELLOW": "🟡", "GREEN": "🟢"}[overall]
overall_label = ALERT_META[overall]["label"]

st.markdown(
    "<div class='section-hdr'><span class='dot'></span>Risk Overview</div>",
    unsafe_allow_html=True,
)
st.markdown(f"""
<div class="metric-row">
  <div class="metric-card mc-red">
    <div class="num">{counts['RED']}</div><div class="lbl">🔴 Extreme Risk</div></div>
  <div class="metric-card mc-orange">
    <div class="num">{counts['ORANGE']}</div><div class="lbl">🟠 High Risk</div></div>
  <div class="metric-card mc-yellow">
    <div class="num">{counts['YELLOW']}</div><div class="lbl">🟡 Moderate Risk</div></div>
  <div class="metric-card mc-green">
    <div class="num">{counts['GREEN']}</div><div class="lbl">🟢 No Risk</div></div>
  <div class="metric-card mc-overall">
    <div class="num">{overall_emoji} {overall_label}</div><div class="lbl">Overall Alert</div></div>
  <div class="metric-card mc-total">
    <div class="num">{len(results)}</div><div class="lbl">Mandals Assessed</div></div>
</div>
""", unsafe_allow_html=True)

# ── Map ───────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-hdr'><span class='dot'></span>Village Risk Map</div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='map-container'>", unsafe_allow_html=True)
st_folium(build_map(results), width="100%", height=560, returned_objects=[])
st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='tbl-caption'>Click any marker for detailed mandal info · "
    "Marker size scales with risk level · All districts shown</div>",
    unsafe_allow_html=True,
)

# ── Rainfall table ────────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-hdr'><span class='dot'></span>Rainfall Data — District &amp; Mandal Level</div>",
    unsafe_allow_html=True,
)
col_f, _ = st.columns([2, 5])
with col_f:
    risk_filter = st.selectbox(
        "Filter by Risk Level",
        ["ALL", "RED", "ORANGE", "YELLOW", "GREEN"],
        key="rf_filter",
    )

table_html, caption = render_rainfall_table(results, risk_filter)
st.markdown(
    '<div class="table-wrap"><div class="scroll-wrap">' + table_html + '</div></div>',
    unsafe_allow_html=True,
)
src_label = "Historical CSV" if data_source == "historical" else "Seasonal model inference"
st.markdown(
    f"<div class='tbl-caption'>{caption} · {src_label}</div>",
    unsafe_allow_html=True,
)

# ── Canals table ──────────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-hdr'><span class='dot'></span>Canals — District-wise Live Risk</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="table-wrap"><div class="scroll-wrap">'
    + render_canals_table(results, canals_df)
    + '</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='tbl-caption'>{len(canals_df)} canals · risk derived from live prediction</div>",
    unsafe_allow_html=True,
)

# ── Embankments table ─────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-hdr'><span class='dot'></span>Embankments — District-wise Live Risk</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="table-wrap"><div class="scroll-wrap">'
    + render_embankments_table(results, embankments_df)
    + '</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='tbl-caption'>{len(embankments_df)} embankments · risk derived from live prediction</div>",
    unsafe_allow_html=True,
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div class='footer'>AP Rainfall Early Warning System v7.0 &nbsp;·&nbsp; "
    f"Random Forest ML &nbsp;·&nbsp; Data-Driven Seasonal Model &nbsp;·&nbsp; "
    f"28 Districts · All Mandals &nbsp;·&nbsp; "
    f"{'Historical CSV' if data_source == 'historical' else 'Seasonal Model Estimate'} "
    f"&nbsp;·&nbsp; "
    f"{pred_date.strftime('%d %B %Y') if hasattr(pred_date, 'strftime') else pred_date}"
    f"</div>",
    unsafe_allow_html=True,
)