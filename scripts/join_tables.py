"""
join_tables.py
==============
F1 Analytics — Join Script
Reads all cleaned CSVs and produces 8 analysis-ready joined tables.

Usage:
    python join_tables.py --input_dir <path_to_cleaned_csvs> --output_dir <path_for_joined_tables>

Defaults (if run from project root):
    --input_dir  ./data/processed
    --output_dir ./data/joined

Output tables:
    J1_master_race_results.csv       — Core race-by-race results (all pillars)
    J2_driver_alpha.csv              — Driver vs teammate alpha analysis (Pillar A)
    J3_qualifying_analysis.csv       — Qualifying performance enriched (Pillar A)
    J4_championship_progression.csv  — Cumulative points race-by-race (Pillar A + Dashboard 4)
    J5_pit_strategy.csv              — Pit stop strategy + race outcomes (Pillar B)
    J6_lap_by_lap.csv                — Lap-level positions + pit events (Pillar B)
    J7_reliability_dnf.csv           — DNF analysis by constructor/era (Pillar C)
    J8_constructor_reliability.csv   — Constructor reliability index by season (Pillar C)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# 0.  Config & helpers
# ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Map clean file names → canonical load names (handles spaces/underscores)
FILE_MAP = {
    "circuits":               ["Clean_Circuits_Data.csv",        "circuits.csv"],
    "constructors":           ["Constructors_Clean_Data.csv",    "constructors.csv"],
    "constructor_results":    ["Constructor_Results_Clean.csv",  "constructor_results.csv"],
    "constructor_standings":  ["Constructor_Standings.csv",      "constructor_standings.csv"],
    "driver_standings":       ["Driver_Standings.csv",           "driver_standings.csv"],
    "drivers":                ["Clean_Drivers_Data.csv",         "drivers.csv"],
    "lap_times":              ["Lap_Times_Clean.csv",            "lap_times.csv"],
    "pit_stops":              ["Cleaned_Pit_Stops.csv",          "pit_stops.csv"],
    "qualifying":             ["Cleaned_Qualifying_Data.csv",    "qualifying.csv"],
    "races":                  ["Cleaned_Races.csv",              "races.csv"],
    "results":                ["Cleaned_Results.csv",            "results.csv"],
    "seasons":                ["Cleaned_Seasons.csv",            "seasons.csv"],
    "status":                 ["Cleaned_Status.csv",             "status.csv"],
}

# Points tables across eras
POINTS_1950_60  = {1:8,  2:6,  3:4,  4:3,  5:2,  6:1}
POINTS_1961_90  = {1:9,  2:6,  3:4,  4:3,  5:2,  6:1}
POINTS_1991_02  = {1:10, 2:6,  3:4,  4:3,  5:2,  6:1}
POINTS_2003_09  = {1:10, 2:8,  3:6,  4:5,  5:4,  6:3, 7:2, 8:1}
POINTS_2010_NOW = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

MECHANICAL_STATUSES = {
    "Engine","Gearbox","Transmission","Clutch","Hydraulics","Electrical",
    "Radiator","Brakes","Differential","Mechanical","Tyre","Driveshaft",
    "Fuel Pressure","Throttle","Steering","Technical","Electronics","Exhaust",
    "Wheel Rim","Water Leak","Fuel Pump","Oil Pressure","Engine Fire","Engine Misfire",
    "Tyre Puncture","Out Of Fuel","Wheel Nut","Pneumatics","Handling","Rear Wing",
    "Wheel Bearing","Fuel System","Oil Line","Fuel Rig","Drivetrain","Ignition",
    "Battery","Halfshaft","Crankshaft","Alternator","Oil Pump","Fuel Leak",
    "Injection","Distributor","Turbo","Cv Joint","Water Pump","Power Unit","Ers",
    "Brake Duct","Overheating","Power Loss","Vibrations","Suspension","Front Wing",
    "Broken Wing","Wheel","Oil Leak","Stalled",
}

ACCIDENT_STATUSES = {
    "Accident","Collision","Spun Off","Collision Damage","Fatal Accident",
    "Fire","Heat Shield Fire","Safety",
}

FINISHED_PREFIXES = ("Finished", "+")


def assign_era(year):
    if year <= 1994:  return "Pre-V10"
    if year <= 2005:  return "V10"
    if year <= 2013:  return "V8"
    return "Hybrid"


def classify_status(s):
    if pd.isna(s): return "Other DNF"
    s_title = str(s).strip().title()
    if s_title.startswith(FINISHED_PREFIXES): return "Finished"
    if s_title in MECHANICAL_STATUSES:        return "Mechanical DNF"
    if s_title in ACCIDENT_STATUSES:          return "Accident DNF"
    return "Other DNF"


def time_str_to_ms(t):
    """Convert 'M:SS.mmm' or 'SS.mmm' to milliseconds."""
    if pd.isna(t): return np.nan
    t = str(t).strip()
    try:
        if ":" in t:
            m, s = t.split(":", 1)
            return (int(m) * 60 + float(s)) * 1000
        return float(t) * 1000
    except Exception:
        return np.nan


def load(input_dir: Path, key: str) -> pd.DataFrame:
    """Load a clean CSV by canonical key, trying multiple filename variants."""
    for fname in FILE_MAP[key]:
        for p in [input_dir / fname, input_dir / fname.replace("_", " ")]:
            if p.exists():
                df = pd.read_csv(p, encoding="latin1", low_memory=False)
                log.info(f"  Loaded {key:25s} ← {p.name}  ({len(df):,} rows)")
                return df
    raise FileNotFoundError(f"Could not find file for key '{key}' in {input_dir}")


def save(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    path = output_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    log.info(f"  ✓ Saved {name:45s}  {len(df):>8,} rows  {len(df.columns)} cols")


# ─────────────────────────────────────────────────────────
# 1.  Load all clean tables
# ─────────────────────────────────────────────────────────

def load_all(input_dir: Path) -> dict:
    log.info("═══ Loading clean tables ═══")
    raw = {k: load(input_dir, k) for k in FILE_MAP}

    # ── Derived columns on raw tables before any joins ──

    # Races: era flag + clean name
    raw["races"]["era"] = raw["races"]["year"].apply(assign_era)
    raw["races"].rename(columns={"name": "race_name"}, inplace=True)

    # Drivers: ensure full_name exists
    if "full_name" not in raw["drivers"].columns:
        raw["drivers"]["full_name"] = (
            raw["drivers"]["forename"].str.strip() + " " + raw["drivers"]["surname"].str.strip()
        )

    # Constructors: clean name column
    raw["constructors"].rename(columns={"name": "constructor_name"}, inplace=True)

    # Circuits: clean name column
    raw["circuits"].rename(columns={"name": "circuit_name"}, inplace=True)

    # Status: add dnf_category
    raw["status"]["dnf_category"] = raw["status"]["status"].apply(classify_status)

    # Results: derived flags
    r = raw["results"]
    r["dnf"]           = r["position"].isna() | ~r["statusId"].isin(
                            raw["status"][raw["status"]["dnf_category"] == "Finished"]["statusId"])
    r["finished_flag"] = ~r["dnf"]
    r["points_scored"] = r["points"] > 0
    r["grid_vs_finish"] = np.where(r["grid"] > 0, r["grid"] - r["positionOrder"], np.nan)
    raw["results"] = r

    # Qualifying: convert time strings → ms
    for col in ["q1", "q2", "q3"]:
        raw["qualifying"][f"{col}_ms"] = raw["qualifying"][col].apply(time_str_to_ms)

    # Pit stops: validate duration
    pit = raw["pit_stops"]
    pit["duration_ms"]   = pit["milliseconds"]
    pit["is_valid_stop"] = pit["duration_ms"].between(2_000, 120_000)
    raw["pit_stops"] = pit

    # Lap times: racing lap flag
    lt = raw["lap_times"]
    lt["is_racing_lap"] = lt["milliseconds"].between(60_000, 300_000)
    raw["lap_times"] = lt

    log.info(f"All {len(raw)} tables loaded.\n")
    return raw


# ─────────────────────────────────────────────────────────
# 2.  J1 — Master Race Results (Pillar A + C + Dashboard 4)
# ─────────────────────────────────────────────────────────

def build_J1_master(raw: dict) -> pd.DataFrame:
    log.info("Building J1_master_race_results ...")

    df = (
        raw["results"]
        .merge(raw["races"][["raceId","race_name","year","round","circuitId","date","era"]],
               on="raceId", how="left")
        .merge(raw["circuits"][["circuitId","circuit_name","location","country","lat","lng"]],
               on="circuitId", how="left")
        .merge(raw["drivers"][["driverId","full_name","nationality","dob","birth_year"]],
               on="driverId", how="left")
        .merge(raw["constructors"][["constructorId","constructor_name","nationality"]],
               on="constructorId", how="left", suffixes=("_driver","_constructor"))
        .merge(raw["status"][["statusId","status","dnf_category"]],
               on="statusId", how="left")
    )

    # KPIs
    df["is_win"]          = (df["positionOrder"] == 1).astype(int)
    df["is_podium"]       = (df["positionOrder"] <= 3).astype(int)
    df["is_points_finish"]= (df["points"] > 0).astype(int)
    df["is_pole"]         = (df["grid"] == 1).astype(int)
    df["is_dnf"]          = df["dnf"].astype(int)

    return df


# ─────────────────────────────────────────────────────────
# 3.  J2 — Driver Alpha (Pillar A)
# ─────────────────────────────────────────────────────────

def build_J2_driver_alpha(raw: dict, J1: pd.DataFrame) -> pd.DataFrame:
    log.info("Building J2_driver_alpha ...")

    # Season-level stats per driver
    season = J1.groupby(["driverId","full_name","year","constructorId","constructor_name","era"]).agg(
        races          = ("raceId",        "count"),
        season_points  = ("points",        "sum"),
        wins           = ("is_win",        "sum"),
        podiums        = ("is_podium",     "sum"),
        dnfs           = ("is_dnf",        "sum"),
        avg_finish_pos = ("positionOrder", "mean"),
        avg_grid_pos   = ("grid",          "mean"),
        avg_grid_delta = ("grid_vs_finish","mean"),
    ).reset_index()

    season["ppr"]       = season["season_points"] / season["races"]
    season["win_rate"]  = season["wins"]   / season["races"]
    season["dnf_rate"]  = season["dnfs"]   / season["races"]

    # Constructor average PPR in the same season (the "Beta")
    constr_avg = season.groupby(["constructorId","year"])["ppr"].mean().reset_index()
    constr_avg.rename(columns={"ppr": "constructor_avg_ppr"}, inplace=True)

    season = season.merge(constr_avg, on=["constructorId","year"], how="left")
    season["driver_alpha"] = season["ppr"] - season["constructor_avg_ppr"]

    # Career summary alpha
    career = season.groupby(["driverId","full_name"]).agg(
        total_seasons  = ("year",           "count"),
        total_races    = ("races",          "sum"),
        career_points  = ("season_points",  "sum"),
        career_wins    = ("wins",           "sum"),
        career_podiums = ("podiums",        "sum"),
        career_dnfs    = ("dnfs",           "sum"),
        career_alpha   = ("driver_alpha",   "mean"),
        avg_ppr        = ("ppr",            "mean"),
    ).reset_index()
    career["career_win_rate"]  = career["career_wins"]  / career["total_races"]
    career["career_dnf_rate"]  = career["career_dnfs"]  / career["total_races"]

    # Merge career stats back for context
    df = season.merge(
        career[["driverId","career_alpha","avg_ppr","total_races","career_wins","career_dnf_rate"]],
        on="driverId", how="left"
    )

    return df


# ─────────────────────────────────────────────────────────
# 4.  J3 — Qualifying Analysis (Pillar A)
# ─────────────────────────────────────────────────────────

def build_J3_qualifying(raw: dict) -> pd.DataFrame:
    log.info("Building J3_qualifying_analysis ...")

    df = (
        raw["qualifying"]
        .merge(raw["races"][["raceId","race_name","year","round","circuitId","era"]], on="raceId", how="left")
        .merge(raw["drivers"][["driverId","full_name","nationality"]], on="driverId", how="left")
        .merge(raw["constructors"][["constructorId","constructor_name"]], on="constructorId", how="left")
        .merge(raw["circuits"][["circuitId","circuit_name","country"]], on="circuitId", how="left")
    )

    # Merge with race result to compare grid vs finish
    res_subset = raw["results"][["raceId","driverId","positionOrder","points","grid","dnf","grid_vs_finish"]]
    df = df.merge(res_subset, on=["raceId","driverId"], how="left")

    # Best qualifying time = best available session
    df["best_qual_ms"] = df[["q3_ms","q2_ms","q1_ms"]].bfill(axis=1).iloc[:, 0]

    # Teammate comparison: within each race × constructor pair
    teammate_agg = (
        df.groupby(["raceId","constructorId"])["q1_ms"]
        .transform("mean")
    )
    df["q1_vs_team_avg_ms"] = df["q1_ms"] - teammate_agg   # negative = faster than team avg

    # Q1→Q3 progression (made it through all sessions)
    df["reached_q2"] = df["q2_ms"].notna().astype(int)
    df["reached_q3"] = df["q3_ms"].notna().astype(int)

    return df


# ─────────────────────────────────────────────────────────
# 5.  J4 — Championship Progression (Pillar A + Dashboard 4)
# ─────────────────────────────────────────────────────────

def build_J4_championship(raw: dict, J1: pd.DataFrame) -> pd.DataFrame:
    log.info("Building J4_championship_progression ...")

    df = J1[["year","raceId","round","race_name","country","era",
             "driverId","full_name","constructorId","constructor_name",
             "points","is_win","is_podium","is_dnf","positionOrder","grid"]].copy()

    df.sort_values(["year","driverId","round"], inplace=True)
    df["cumulative_points"] = df.groupby(["year","driverId"])["points"].cumsum()
    df["cumulative_wins"]   = df.groupby(["year","driverId"])["is_win"].cumsum()

    # Championship position after each round
    df["championship_pos"] = (
        df.groupby(["year","round"])["cumulative_points"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Points gap to leader after each round
    leader_pts = df.groupby(["year","round"])["cumulative_points"].transform("max")
    df["gap_to_leader"] = leader_pts - df["cumulative_points"]

    return df


# ─────────────────────────────────────────────────────────
# 6.  J5 — Pit Strategy (Pillar B)
# ─────────────────────────────────────────────────────────

def build_J5_pit_strategy(raw: dict) -> pd.DataFrame:
    log.info("Building J5_pit_strategy ...")

    # Pit stop summary per driver per race
    pit_summary = (
        raw["pit_stops"][raw["pit_stops"]["is_valid_stop"]]
        .groupby(["raceId","driverId"])
        .agg(
            total_stops    = ("stop",        "max"),
            total_pit_ms   = ("duration_ms", "sum"),
            median_pit_ms  = ("duration_ms", "median"),
            min_pit_ms     = ("duration_ms", "min"),
            first_pit_lap  = ("lap",         "min"),
            last_pit_lap   = ("lap",         "max"),
        )
        .reset_index()
    )
    pit_summary["total_pit_sec"]   = pit_summary["total_pit_ms"]   / 1000
    pit_summary["median_pit_sec"]  = pit_summary["median_pit_ms"]  / 1000

    # Enrich with race + driver + result context
    df = (
        pit_summary
        .merge(raw["races"][["raceId","race_name","year","round","circuitId","era"]], on="raceId", how="left")
        .merge(raw["circuits"][["circuitId","circuit_name","country"]], on="circuitId", how="left")
        .merge(raw["drivers"][["driverId","full_name"]], on="driverId", how="left")
        .merge(
            raw["results"][["raceId","driverId","constructorId","positionOrder","points",
                             "grid","laps","dnf","grid_vs_finish","statusId"]],
            on=["raceId","driverId"], how="left"
        )
        .merge(raw["constructors"][["constructorId","constructor_name"]], on="constructorId", how="left")
        .merge(raw["status"][["statusId","status","dnf_category"]], on="statusId", how="left")
    )

    # Pit efficiency: total pit time as % of race laps (rough proxy)
    df["pit_time_per_stop_sec"] = df["total_pit_sec"] / df["total_stops"]

    # Stint strategy label
    df["strategy"] = df["total_stops"].map({1:"1-Stop",2:"2-Stop",3:"3-Stop"}).fillna("4+ Stops")

    return df


# ─────────────────────────────────────────────────────────
# 7.  J6 — Lap-by-Lap with Pit Events (Pillar B)
# ─────────────────────────────────────────────────────────

def build_J6_lap_by_lap(raw: dict) -> pd.DataFrame:
    log.info("Building J6_lap_by_lap (this may take a moment — 426k rows) ...")

    # Flag which laps had a pit stop
    pit_laps = raw["pit_stops"][["raceId","driverId","lap","duration_ms","is_valid_stop"]].copy()
    pit_laps.rename(columns={"duration_ms":"pit_duration_ms"}, inplace=True)
    pit_laps["pit_stop_lap"] = True

    df = (
        raw["lap_times"]
        .merge(raw["races"][["raceId","year","round","race_name","era"]], on="raceId", how="left")
        .merge(raw["drivers"][["driverId","full_name"]], on="driverId", how="left")
        .merge(
            raw["results"][["raceId","driverId","constructorId"]].drop_duplicates(),
            on=["raceId","driverId"], how="left"
        )
        .merge(raw["constructors"][["constructorId","constructor_name"]], on="constructorId", how="left")
        .merge(pit_laps[["raceId","driverId","lap","pit_duration_ms","pit_stop_lap","is_valid_stop"]],
               on=["raceId","driverId","lap"], how="left")
    )

    df["pit_stop_lap"]  = df["pit_stop_lap"].fillna(False)
    df["lap_time_sec"]  = df["milliseconds"] / 1000

    # Position change lap-over-lap
    df.sort_values(["raceId","driverId","lap"], inplace=True)
    df["prev_position"]   = df.groupby(["raceId","driverId"])["position"].shift(1)
    df["position_change"] = df["prev_position"] - df["position"]   # positive = gained a place

    return df


# ─────────────────────────────────────────────────────────
# 8.  J7 — Reliability & DNF Analysis (Pillar C)
# ─────────────────────────────────────────────────────────

def build_J7_reliability(raw: dict, J1: pd.DataFrame) -> pd.DataFrame:
    log.info("Building J7_reliability_dnf ...")

    df = J1[[
        "resultId","raceId","race_name","year","round","era","date",
        "driverId","full_name","nationality_driver",
        "constructorId","constructor_name","nationality_constructor",
        "circuit_name","country",
        "grid","positionOrder","laps","points",
        "statusId","status","dnf_category",
        "is_dnf","is_win","is_podium",
    ]].copy()

    # Laps completed pct
    total_laps = J1.groupby("raceId")["laps"].max().reset_index()
    total_laps.rename(columns={"laps":"total_race_laps"}, inplace=True)
    df = df.merge(total_laps, on="raceId", how="left")
    df["laps_pct_completed"] = np.where(
        df["total_race_laps"] > 0,
        df["laps"] / df["total_race_laps"] * 100,
        np.nan
    )

    # Estimated points lost (for DNFs only — based on last position)
    POINTS_TABLE = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
    df["potential_points_lost"] = np.where(
        df["is_dnf"] == 1,
        df["positionOrder"].map(POINTS_TABLE).fillna(0),
        0
    )

    return df


# ─────────────────────────────────────────────────────────
# 9.  J8 — Constructor Reliability Index (Pillar C)
# ─────────────────────────────────────────────────────────

def build_J8_constructor_reliability(raw: dict, J7: pd.DataFrame) -> pd.DataFrame:
    log.info("Building J8_constructor_reliability ...")

    season_rel = J7.groupby(["constructorId","constructor_name","year","era"]).agg(
        entries          = ("resultId",               "count"),
        total_dnfs       = ("is_dnf",                 "sum"),
        mechanical_dnfs  = ("dnf_category",           lambda x: (x == "Mechanical DNF").sum()),
        accident_dnfs    = ("dnf_category",            lambda x: (x == "Accident DNF").sum()),
        other_dnfs       = ("dnf_category",            lambda x: (x == "Other DNF").sum()),
        total_points     = ("points",                  "sum"),
        total_wins       = ("is_win",                  "sum"),
        pts_lost_to_dnf  = ("potential_points_lost",   "sum"),
    ).reset_index()

    season_rel["dnf_rate"]           = season_rel["total_dnfs"]      / season_rel["entries"]
    season_rel["mechanical_dnf_rate"]= season_rel["mechanical_dnfs"] / season_rel["entries"]
    season_rel["reliability_index"]  = 1 - season_rel["dnf_rate"]

    # Rolling 3-season reliability per constructor
    season_rel.sort_values(["constructorId","year"], inplace=True)
    season_rel["rolling3_reliability"] = (
        season_rel.groupby("constructorId")["reliability_index"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Era-level reliability per constructor
    era_rel = J7.groupby(["constructorId","constructor_name","era"]).agg(
        era_entries        = ("resultId",   "count"),
        era_dnfs           = ("is_dnf",     "sum"),
        era_mech_dnfs      = ("dnf_category", lambda x: (x == "Mechanical DNF").sum()),
    ).reset_index()
    era_rel["era_dnf_rate"]      = era_rel["era_dnfs"]      / era_rel["era_entries"]
    era_rel["era_reliability"]   = 1 - era_rel["era_dnf_rate"]
    era_rel["era_mech_dnf_rate"] = era_rel["era_mech_dnfs"] / era_rel["era_entries"]

    df = season_rel.merge(
        era_rel[["constructorId","era","era_reliability","era_mech_dnf_rate"]],
        on=["constructorId","era"], how="left"
    )

    return df


# ─────────────────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="F1 Table Joiner")
    parser.add_argument("--input_dir",  type=Path, default=Path("data/processed"),
                        help="Directory containing cleaned CSV files")
    parser.add_argument("--output_dir", type=Path, default=Path("data/joined"),
                        help="Directory to write joined CSV files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log.info(f"Input  : {args.input_dir}")
    log.info(f"Output : {args.output_dir}\n")

    # ── Load ──
    raw = load_all(args.input_dir)

    # ── Build joins ──
    log.info("\n═══ Building joined tables ═══")

    J1 = build_J1_master(raw)
    save(J1, args.output_dir, "J1_master_race_results")

    J2 = build_J2_driver_alpha(raw, J1)
    save(J2, args.output_dir, "J2_driver_alpha")

    J3 = build_J3_qualifying(raw)
    save(J3, args.output_dir, "J3_qualifying_analysis")

    J4 = build_J4_championship(raw, J1)
    save(J4, args.output_dir, "J4_championship_progression")

    J5 = build_J5_pit_strategy(raw)
    save(J5, args.output_dir, "J5_pit_strategy")

    J6 = build_J6_lap_by_lap(raw)
    save(J6, args.output_dir, "J6_lap_by_lap")

    J7 = build_J7_reliability(raw, J1)
    save(J7, args.output_dir, "J7_reliability_dnf")

    J8 = build_J8_constructor_reliability(raw, J7)
    save(J8, args.output_dir, "J8_constructor_reliability")

    # ── Summary ──
    log.info(f"\n{'═'*60}")
    log.info(f"All 8 joined tables written to: {args.output_dir}")
    log.info(f"Total time: {time.time()-t0:.1f}s")
    log.info(f"{'═'*60}")
    log.info("\nTableau connection guide:")
    log.info("  Dashboard 1 — Driver Alpha    → J2_driver_alpha.csv")
    log.info("  Dashboard 2 — Pit Strategy    → J5_pit_strategy.csv + J6_lap_by_lap.csv")
    log.info("  Dashboard 3 — Reliability     → J7_reliability_dnf.csv + J8_constructor_reliability.csv")
    log.info("  Dashboard 4 — Championship    → J4_championship_progression.csv")
    log.info("  All dashboards (base)         → J1_master_race_results.csv")


if __name__ == "__main__":
    main()