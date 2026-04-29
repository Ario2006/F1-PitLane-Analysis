# Data Dictionary — Formula 1 Analytics Project

## How To Use This File

1. Every column used in analysis, KPI calculation, or Tableau dashboarding is documented here.
2. Plain-language descriptions explain what each field means in the context of F1 racing.
3. Cleaning decisions and transformations are recorded in the Cleaning Notes column.
4. Nullable columns, derived fields, and known data quality issues are explicitly flagged.
5. The dictionary covers both the 13 raw source tables and the final `MASTER_tableau_ready.csv` used in Tableau.

---

## Dataset Summary

| Item | Details |
|---|---|
| **Dataset name** | Formula 1 Race Data |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/cjgdev/formula-1-race-data-19502017?select=circuits.csv) |
| **Raw file names** | 13 CSV files: `Cleaned_Results.csv`, `Cleaned_Races.csv`, `Clean_Drivers_Data.csv`, `Constructors_Clean_Data.csv`, `Cleaned_Status.csv`, `Clean_Circuits_Data.csv`, `Cleaned_Qualifying_Data.csv`, `Cleaned_Pit_Stops.csv`, `Lap_Times_Clean.csv`, `Driver_Standings.csv`, `Constructor_Standings.csv`, `Constructor_Results_Clean.csv`, `Cleaned_Seasons.csv` |
| **Master analysis file** | `MASTER_tableau_ready.csv` — all tables joined into one flat file |
| **Coverage** | 1950–2017 (68 seasons, 976 races, 842 drivers, 207 constructors, 72 circuits) |
| **Last updated** | 2025 |
| **Granularity** | One row per driver per race entry (race start attempt). A race with 20 drivers produces 20 rows. |
| **Master file size** | 23,777 rows × 85 columns — 12.7 MB on disk |
| **Pit stop data coverage** | 2011–2017 only (6,251 recorded stops) |
| **Qualifying data coverage** | 1994–2017 (7,516 qualifying entries) |

---

## Section 1 — Raw Source Tables

These are the 13 cleaned CSV files before any joining. Documented here as a record of what each source file contains and what cleaning was applied at the file level.

---

### 1.1 `Cleaned_Results.csv` — 23,777 rows

The primary table. One row per driver per race. The backbone of all three analytical pillars.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `resultId` | int | Unique identifier for each race entry row | `1` | Join key, EDA | Primary key. No nulls. Range: 1–23,781. |
| `raceId` | int | Foreign key linking to `races` table | `18` | Join key, all pillars | No nulls. 976 unique values. |
| `driverId` | int | Foreign key linking to `drivers` table | `1` | Join key, all pillars | No nulls. 842 unique values. |
| `constructorId` | int | Foreign key linking to `constructors` table | `1` | Join key, all pillars | No nulls. 207 unique values. |
| `number` | int | Car number used in the race | `22` | Reference only | Nulls filled with `0` for historical entries where no permanent number existed. |
| `grid` | int | Starting grid position. `0` = pit lane start | `1` | Pillar A, Tableau | No nulls. Range: 0–34. Grid 0 indicates a pit lane start and is excluded from grid delta calculations. |
| `position` | int | Final classified race position. `0` = did not finish | `1` | Pillar C, DNF flag | No nulls. `position == 0` is the reliable DNF indicator in this dataset. Do not confuse with `positionOrder`. |
| `positionText` | string | Position as displayed text. Can be `"R"` (retired), `"D"` (disqualified), `"W"` (withdrew), `"F"` (failed to qualify) | `"1"` | Reference | Not used in numeric calculations. Retained for reference. |
| `positionOrder` | int | Numeric sort order for final classification. Filled for all drivers including DNFs — DNF drivers are ordered after finishers by how far they got | `1` | Pillar A, B, C, Tableau | No nulls. Range: 1–39. Used as finish position in all ranking calculations. Never null. |
| `points` | float | Championship points awarded for this race result | `10.0` | All pillars, KPIs | No nulls. Range: 0–50. Maximum of 50 occurs only in 2014 Abu Dhabi (double-points finale rule). |
| `laps` | int | Number of race laps completed by this driver | `58` | Pillar C | No nulls. Range: 0–200. `0` laps typically indicates an accident on the formation lap. |
| `time` | string | Race finish time or gap to leader. Empty for DNFs and non-finishers | `"34:50.6"` | Not used | Not used in analysis. Retained for reference. Mixed format (absolute time vs gap) makes numeric parsing unreliable. |
| `milliseconds` | int | Total race time in milliseconds. `0` for DNFs | `5690616` | Not used | Not used in analysis. `0` for all non-finishers makes it unreliable as a continuous measure. |
| `fastestLap` | int | Lap number on which this driver set their fastest lap. `0` if not tracked | `39` | Not used | Only populated from 2004 onward. 17,773 rows have `0`. Not used in primary analysis. |
| `rank` | int | Fastest lap rank among all drivers in the race. `0` if not tracked | `2` | Not used | Same coverage limitation as `fastestLap`. Not used. |
| `fastestLapTime` | string | Fastest lap time as `"M:SS.mmm"` string. `"0"` if not tracked | `"01:27.5"` | Not used | Not converted to milliseconds. Not used in analysis. |
| `fastestLapSpeed` | float | Speed in kph at the fastest lap. `0.0` if not tracked | `218.3` | Not used | Range: 89.54–257.32 kph. Only populated post-2004. Not used in primary analysis. |
| `statusId` | int | Foreign key to `status` table — reason for finishing or retiring | `1` | Pillar C, DNF classification | No nulls. 132 unique status IDs. |

---

### 1.2 `Cleaned_Races.csv` — 997 rows

One row per race event. Links raceId to circuit, date, and season metadata.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `raceId` | int | Unique race identifier | `1` | Join key | Primary key. No nulls. |
| `year` | int | Season year | `2008` | All pillars, filters | No nulls. Range: 1950–2017. Note: dataset covers 997 races across 68 seasons. 2018 season is not included. |
| `round` | int | Round number within the season (1 = first race) | `1` | Pillar A (championship), Tableau | No nulls. Range: 1–21. |
| `circuitId` | int | Foreign key to `circuits` table | `1` | Join key | No nulls. |
| `name` | string | Official race name | `"Australian Grand Prix"` | Tableau display | Renamed to `race_name` in master table to avoid column name collision. |
| `date` | string | Race date in `YYYY-MM-DD` format | `"2008-03-16"` | Timeline charts, Tableau | Parsed to datetime type in Tableau. No nulls. |
| `time` | string | Local race start time. Missing for most historical races | `"04:00:00"` | Not used | 731 of 997 rows are null. Dropped in analysis. |
| `url` | string | Wikipedia article URL | `"https://..."` | Not used | Dropped in analysis. Reference only. |

---

### 1.3 `Clean_Drivers_Data.csv` — 841 rows

One row per driver. Contains biographical and nationality information.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `driverId` | int | Unique driver identifier | `1` | Join key | Primary key. No nulls. |
| `driverRef` | string | URL-safe slug for driver name | `"hamilton"` | Not used | Not used in analysis. Reference only. |
| `forename` | string | Driver first name | `"Lewis"` | Driver name construction | Combined with `surname` to create `full_name`. |
| `surname` | string | Driver last name | `"Hamilton"` | Driver name construction | Combined with `forename` to create `full_name`. |
| `full_name` | string | Concatenated display name | `"Lewis Hamilton"` | All pillars, Tableau | Derived field. Created as `forename + ' ' + surname`. Used as the primary driver label across all visualisations. |
| `dob` | string | Date of birth in `YYYY-MM-DD` format | `"1985-01-07"` | Reference | 1 null in dataset. Parsed to datetime where needed. Range of birth years: 1896–1998. |
| `birth_year` | float | Year extracted from `dob` | `1985` | Reference | Derived from `dob`. Useful for era-cohort analysis. |
| `nationality` | string | Driver nationality | `"British"` | Tableau filters | Renamed to `nationality_driver` in master table to distinguish from constructor nationality. 41 unique nationalities. |
| `code` | string | Three-letter driver code | `"HAM"` | Not used | 757 of 841 values are null — only exists for modern drivers with FIA-assigned codes. Dropped in analysis. |
| `number` | int | Permanent driver number | `44` | Not used | 804 of 841 values are null — only assigned since 2014. Dropped in analysis. |
| `url` | string | Wikipedia URL | `"https://..."` | Not used | Dropped in analysis. |

---

### 1.4 `Constructors_Clean_Data.csv` — 208 rows

One row per constructor (team). 207 unique constructors appear in the results data.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `constructorId` | int | Unique constructor identifier | `1` | Join key | Primary key. No nulls. |
| `constructorRef` | string | URL-safe slug | `"mclaren"` | Not used | Not used in analysis. |
| `name` | string | Official constructor name | `"McLaren"` | All pillars, Tableau | Renamed to `constructor_name` in master table. |
| `nationality` | string | Constructor nationality | `"British"` | Tableau filters | Renamed to `nationality_constructor` in master table. 24 unique values. |
| `url` | string | Wikipedia URL | `"https://..."` | Not used | Dropped in analysis. |

---

### 1.5 `Cleaned_Status.csv` — 134 rows

Lookup table mapping `statusId` to a human-readable retirement reason.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `statusId` | int | Unique status identifier | `1` | Join key | Primary key. No nulls. |
| `status` | string | Plain-text description of why the driver finished or retired | `"Finished"` | Pillar C, Tableau tooltip | 134 unique values including `"Finished"`, `"+1 Lap"`, `"Engine"`, `"Accident"`, etc. Used to derive `dnf_category`. |

---

### 1.6 `Clean_Circuits_Data.csv` — 73 rows

One row per circuit. Provides geographic data for map visualisations.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `circuitId` | int | Unique circuit identifier | `1` | Join key | Primary key. No nulls. |
| `circuitRef` | string | URL-safe slug | `"albert_park"` | Not used | Not used. |
| `name` | string | Official circuit name | `"Albert Park Grand Prix Circuit"` | Tableau map | Renamed to `circuit_name` in master table. |
| `location` | string | City or region where circuit is located | `"Melbourne"` | Tableau tooltip | No nulls. 69 unique locations. |
| `country` | string | Country where circuit is located | `"Australia"` | Tableau map, filters | No nulls. 32 unique countries. |
| `lat` | float | Circuit latitude (decimal degrees) | `-37.8497` | Tableau geographic map | No nulls. Range: −37.85 to 57.27. Assigned Geographic Role → Latitude in Tableau. |
| `lng` | float | Circuit longitude (decimal degrees) | `144.968` | Tableau geographic map | No nulls. Range: −118.19 to 144.97. Assigned Geographic Role → Longitude in Tableau. |
| `alt` | float | Altitude in metres | `10.0` | Not used | 72 of 73 values are null. Dropped in analysis. |
| `url` | string | Wikipedia URL | `"https://..."` | Not used | Dropped in analysis. |

---

### 1.7 `Cleaned_Qualifying_Data.csv` — 7,516 rows

One row per driver per qualifying session. Contains lap times from Q1, Q2, and Q3.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `qualifyId` | int | Unique qualifying entry identifier | `1` | Join key | Primary key. No nulls. |
| `raceId` | int | Foreign key to races | `18` | Join key | No nulls. |
| `driverId` | int | Foreign key to drivers | `1` | Join key | No nulls. |
| `constructorId` | int | Foreign key to constructors | `1` | Join key | No nulls. |
| `number` | int | Car number | `22` | Reference | No nulls. |
| `position` | int | Final qualifying classification position | `1` | Pillar A | No nulls. |
| `q1` | string | Q1 lap time as `"M:SS.mmm"` string | `"1:26.572"` | Pillar A | 119 nulls — drivers who did not set a Q1 time (DNS, no time set). Converted to `q1_ms` in master. |
| `q2` | string | Q2 lap time. Null if driver was eliminated in Q1 | `"1:25.187"` | Pillar A | 3,864 nulls — structurally expected. Only top ~15 drivers advance to Q2. Do not impute. Converted to `q2_ms`. |
| `q3` | string | Q3 lap time. Null if driver did not reach Q3 | `"1:26.714"` | Pillar A | 5,338 nulls — structurally expected. Only top 10 drivers reach Q3. Do not impute. Converted to `q3_ms`. |

---

### 1.8 `Cleaned_Pit_Stops.csv` — 6,251 rows

One row per individual pit stop. Covers 2011–2017 only.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `raceId` | int | Foreign key to races | `841` | Join key | No nulls. Only raceIds from 2011–2017. |
| `driverId` | int | Foreign key to drivers | `1` | Join key | No nulls. |
| `stop` | int | Stop number within the race (1 = first stop, 2 = second, etc.) | `1` | Pillar B — total stops | No nulls. Range: 1–6. |
| `lap` | int | Lap number on which the pit stop occurred | `14` | Pillar B — stop timing | No nulls. Range: 1–74. |
| `time` | string | Time of day when the stop occurred (local race time) | `"17:05:23"` | Not used | Not used in analysis — too coarse for lap-level analysis. |
| `duration` | float | Pit stop duration in seconds | `23.32` | Pillar B — pit efficiency | Converted to `duration_ms` (×1000) and cross-validated against `milliseconds` column. Outliers outside 2–120 seconds flagged as invalid. |
| `milliseconds` | int | Pit stop duration in milliseconds (stored value) | `23320` | Pillar B — primary numeric | Used as canonical duration field. Range: 15,049 ms to 70,187 ms after outlier filtering. Stops outside 2,000–120,000 ms flagged `is_valid_stop = False`. |

---

### 1.9 `Lap_Times_Clean.csv` — 426,633 rows

One row per driver per lap. The most granular table in the dataset.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `raceId` | int | Foreign key to races | `841` | Join key (J6) | No nulls. Composite primary key with `driverId` + `lap`. |
| `driverId` | int | Foreign key to drivers | `1` | Join key (J6) | No nulls. |
| `lap` | int | Lap number | `1` | Pillar B — undercut analysis | No nulls. Range: 1–200. |
| `position` | int | Track position (not race classification) at the end of this lap | `3` | Pillar B — position change | No nulls. Range: 1–33. This is on-track position, updated each lap. |
| `time` | string | Lap time as `"M:SS.mmm"` string | `"1:38.109"` | Not used | Dropped. `milliseconds` column is used instead. |
| `milliseconds` | int | Lap time in milliseconds | `98109` | Pillar B — lap pace | No nulls. Range: 0 ms to 15,090,540 ms before filtering. `is_racing_lap` flag applied for laps between 60,000–300,000 ms. |
| `lap_flag` | string | Quality flag indicating lap type | `"normal"` | Pillar B — lap filtering | Values: `"normal"` or `"slow_or_pitlap"`. Used to identify safety car and pit stop laps. |

---

### 1.10 `Driver_Standings.csv` — 31,726 rows

Cumulative championship standings after each race in a season.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `driverStandingsId` | int | Unique standings row identifier | `1` | Not used as key | Primary key. |
| `raceId` | int | Race after which these standings apply | `18` | Join key | No nulls. |
| `driverId` | int | Foreign key to drivers | `1` | Join key | No nulls. |
| `points` | float | Cumulative championship points after this race | `10.0` | Pillar A — championship | No nulls. The championship progression columns (`cumulative_points`, `championship_pos`) in the master are computed from results data directly rather than this table, for consistency. |
| `position` | int | Championship position after this race | `1` | Reference | No nulls. |
| `positionText` | string | Position as display text | `"1"` | Not used | Not used in analysis. |
| `wins` | int | Cumulative wins after this race | `1` | Reference | No nulls. |

---

### 1.11 `Constructor_Standings.csv` — 11,896 rows

Cumulative constructor championship standings after each race.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `constructorStandingsId` | int | Unique identifier | `1` | Not used | Primary key. |
| `raceId` | int | Race after which standings apply | `18` | Reference | No nulls. |
| `constructorId` | int | Foreign key to constructors | `1` | Reference | No nulls. |
| `points` | float | Cumulative constructor championship points | `10.0` | Reference | No nulls. |
| `position` | int | Constructor championship position | `1` | Reference | No nulls. |
| `positionText` | string | Display text | `"1"` | Not used | Not used. |
| `wins` | int | Cumulative constructor wins | `1` | Reference | No nulls. |

---

### 1.12 `Constructor_Results_Clean.csv` — 11,142 rows

Points scored per constructor per race (aggregated from driver results).

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `constructorResultsId` | int | Unique identifier | `1` | Not used | Primary key. |
| `raceId` | int | Foreign key to races | `18` | Reference | No nulls. |
| `constructorId` | int | Foreign key to constructors | `1` | Reference | No nulls. |
| `points` | float | Constructor points for this race | `10.0` | Pillar C — constructor analysis | Nulls filled with `0` (DNS/DSQ = 0 points). The `status` column in this table was 11,125 of 11,142 null — dropped. |

---

### 1.13 `Cleaned_Seasons.csv` — 69 rows

One row per season. Minimal table — year and Wikipedia reference only.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `year` | int | Season year | `1950` | Reference | Primary key. Range: 1950–2018. Note: 2018 season is in this table but no race result data exists for 2018 in `Cleaned_Results.csv`. |
| `url` | string | Wikipedia URL | `"https://..."` | Not used | Dropped in analysis. |

---

## Section 2 — Joined Analytical Tables (J1–J8)

These are the 8 tables produced by `join_tables.py`. Each table has a specific analytical purpose and joins key tables at a consistent granularity.

---

### 2.1 `J1_master_race_results.csv` — 23,777 rows

The core flat table. One row per driver per race entry. Joins results, races, circuits, drivers, constructors, and status.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| All columns from `Cleaned_Results.csv` | — | See Section 1.1 | — | All pillars | Inherited as-is, plus derived flags below. |
| `race_name` | string | Official race name | `"Australian Grand Prix"` | Tableau | Renamed from `name` (races table) to avoid collision. |
| `era` | string | Engine era label | `"V8"` | All pillars, filters | Derived in join script. Values: `Pre-V10` (1950–1994), `V10` (1995–2005), `V8` (2006–2013), `Hybrid` (2014–2017). |
| `circuit_name` | string | Full circuit name | `"Albert Park Grand Prix Circuit"` | Tableau | Renamed from `name` (circuits table). |
| `full_name` | string | Driver full name | `"Lewis Hamilton"` | All pillars, Tableau | From drivers table. No nulls. |
| `nationality_driver` | string | Driver nationality | `"British"` | Tableau | Renamed to avoid collision with constructor nationality. |
| `nationality_constructor` | string | Constructor nationality | `"British"` | Tableau | Renamed to avoid collision with driver nationality. |
| `constructor_name` | string | Constructor display name | `"McLaren"` | All pillars, Tableau | Renamed from `name` (constructors table). |
| `status` | string | Retirement or finish reason | `"Finished"` | Pillar C | From status lookup table. |
| `dnf_category` | string | Grouped retirement reason | `"Mechanical DNF"` | Pillar C, Tableau | Derived via `classify_dnf()` function. Four values: `Finished`, `Mechanical DNF`, `Accident DNF`, `Other DNF`. |
| `is_win` | int | 1 if driver won this race, 0 otherwise | `1` | All pillars, KPIs | Derived: `1 if positionOrder == 1 else 0`. No nulls. |
| `is_podium` | int | 1 if driver finished P1–P3, 0 otherwise | `1` | Pillar A, KPIs | Derived: `1 if positionOrder <= 3 else 0`. No nulls. |
| `is_points_finish` | int | 1 if driver scored points, 0 otherwise | `1` | KPIs | Derived: `1 if points > 0 else 0`. No nulls. |
| `is_pole` | int | 1 if driver started from pole position, 0 otherwise | `1` | Pillar A | Derived: `1 if grid == 1 else 0`. No nulls. |
| `is_dnf` | int | 1 if driver did not finish (retired), 0 otherwise | `0` | Pillar C, KPIs | Derived: `1 if position == 0 else 0`. `position == 0` is the reliable DNF indicator in this dataset. 10,550 of 23,777 entries are DNFs (44.4%). |
| `grid_vs_finish` | float | Places gained or lost from grid to finish. Positive = moved forward | `2.0` | Pillar A — driver alpha | Derived: `grid − positionOrder`. Null for pit lane starts (grid = 0). 6.6% null. Range: −31 to +30. |

---

### 2.2 `J2_driver_alpha.csv` — 3,397 rows (before dedup to 3,058)

One row per driver per season per constructor. Produced by aggregating J1. Contains the core Pillar A KPIs.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `driverId` | int | Driver identifier | `1` | Join key | Composite key with `year` and `constructorId`. |
| `full_name` | string | Driver full name | `"Lewis Hamilton"` | Pillar A, Tableau | No nulls. |
| `year` | int | Season year | `2008` | Join key, filters | No nulls. |
| `constructorId` | int | Constructor identifier | `1` | Join key | No nulls. |
| `constructor_name` | string | Constructor name | `"McLaren"` | Pillar A, Tableau | No nulls. |
| `era` | string | Engine era | `"V8"` | Pillar A, filters | No nulls. |
| `races` | float | Number of races started this season for this constructor | `18.0` | Pillar A, KPIs | No nulls. Range: 1–21. |
| `season_points` | float | Total championship points scored this season | `98.0` | Pillar A, KPIs | No nulls. Range: 0–397. |
| `wins` | float | Race wins this season | `5.0` | Pillar A, KPIs | No nulls. |
| `podiums` | float | Podium finishes (P1–P3) this season | `12.0` | Pillar A | No nulls. |
| `dnfs` | float | DNF count this season | `1.0` | Pillar A | No nulls. |
| `avg_finish_pos` | float | Average finishing position (positionOrder) this season | `5.22` | Pillar A | No nulls. |
| `avg_grid_pos` | float | Average starting grid position this season | `2.5` | Pillar A | No nulls. |
| `avg_grid_delta` | float | Average places gained per race this season | `−1.33` | Pillar A | 1.6% null — seasons where all grid starts were from the pit lane. |
| `ppr` | float | Points Per Race this season. Primary efficiency metric | `5.44` | Pillar A — alpha, KPIs | Derived: `season_points / races`. Range: 0–20.89. No nulls. |
| `win_rate` | float | Fraction of races won this season | `0.278` | Pillar A, KPIs | Derived: `wins / races`. Range: 0–1. No nulls. |
| `dnf_rate` | float | Fraction of races that ended in retirement this season | `0.056` | Pillar A | Derived: `dnfs / races`. Range: 0–1. No nulls. |
| `constructor_avg_ppr` | float | Average PPR of all drivers at the same constructor in the same season. The "Beta" | `4.19` | Pillar A — alpha calculation | Derived via groupby on constructor + year. Represents car performance baseline. No nulls. |
| `driver_alpha` | float | Driver's PPR minus constructor's average PPR. Core Pillar A metric | `1.25` | Pillar A — primary KPI | Derived: `ppr − constructor_avg_ppr`. Positive = outperforming the car. Range: −5.90 to +8.20. No nulls. |
| `career_alpha` | float | Driver's mean alpha averaged across all seasons | `0.615` | Pillar A, Tableau | Derived: `mean(driver_alpha)` across all seasons for this driver. Range: −3.0 to +3.94. No nulls. |
| `avg_ppr` | float | Driver's career average PPR (across all seasons and teams) | `12.30` | Pillar A | Derived: career mean of `ppr`. No nulls. |
| `total_races` | float | Total races started across entire career | `208.0` | Pillar A — minimum threshold filter | No nulls. Range: 1–326. Use `total_races >= 30` filter for meaningful career comparisons. |
| `career_wins` | float | Total career race wins | `62.0` | Pillar A, KPIs | No nulls. |
| `career_dnf_rate` | float | Career fraction of races ending in DNF | `0.120` | Pillar A | No nulls. Range: 0–1. |

---

### 2.3 `J3_qualifying_analysis.csv` — 7,516 rows

One row per driver per qualifying session. Covers 1994–2017 only.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `qualifyId` | int | Unique qualifying entry | `1` | Join key | Primary key. |
| `raceId` | int | Race identifier | `18` | Join key | No nulls. |
| `driverId` | int | Driver identifier | `1` | Join key | No nulls. |
| `constructorId` | int | Constructor identifier | `1` | Join key | No nulls. |
| `q1_ms` | float | Q1 lap time in milliseconds | `86572.0` | Pillar A | Derived from `q1` string via `qual_time_to_ms()`. 119 nulls (DNS or no time set). 68.9% null in master (pre-1994 has no qualifying data). |
| `q2_ms` | float | Q2 lap time in milliseconds | `85187.0` | Pillar A | 3,864 nulls — structural (only top ~15 advance). Do not impute. 84.6% null in master. |
| `q3_ms` | float | Q3 lap time in milliseconds | `86714.0` | Pillar A | 5,338 nulls — structural (only top 10 advance). Do not impute. 90.8% null in master. |
| `best_qual_ms` | float | Best lap time achieved across any qualifying session | `86714.0` | Pillar A | Derived: `min(q3_ms, q2_ms, q1_ms)` using backfill. Best available session time. 68.9% null in master. |
| `q1_vs_team_avg_ms` | float | Q1 time minus team's mean Q1 time in this race. Negative = faster than teammate | `454.0` | Pillar A — teammate comparison | Derived using `groupby(raceId, constructorId).transform('mean')`. Range: −451,843 to +451,843 ms. Extreme outliers exist in early qualifying data. Clip to ±3 seconds in Tableau. 68.9% null in master. |
| `reached_q2` | float | 1 if driver progressed to Q2, 0 otherwise | `1.0` | Pillar A | Derived: `1 if q2_ms is not null else 0`. 68.4% null in master (pre-1994). |
| `reached_q3` | float | 1 if driver progressed to Q3, 0 otherwise | `1.0` | Pillar A | Derived: `1 if q3_ms is not null else 0`. 68.4% null in master. |

---

### 2.4 `J4_championship_progression.csv` — 23,777 rows (23,688 after dedup)

One row per driver per race with running championship totals. Enables the championship bump chart.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `raceId` | int | Race identifier | `18` | Join key | Composite key with `driverId`. 89 duplicate raceId+driverId pairs existed (drivers who entered the same race for two constructors in historical data). Resolved by `drop_duplicates(keep='last')`. |
| `driverId` | int | Driver identifier | `1` | Join key | No nulls. |
| `cumulative_points` | float | Running total of championship points after this race | `10.0` | Pillar A — championship, Tableau | Derived: `groupby(year, driverId).cumsum(points)`. Rows sorted by year + driverId + round before cumsum. Range: 0–397. No nulls. |
| `cumulative_wins` | int | Running total of wins after this race | `1` | Pillar A — championship | Derived: cumulative sum of `is_win`. Range: 0–13. No nulls. |
| `championship_pos` | int | Championship standing position after this race | `1` | Pillar A — bump chart, KPIs | Derived: `rank(cumulative_points, ascending=False, method='min')` within each year + round group. Range: 1–28. No nulls. |
| `gap_to_leader` | float | Points behind the championship leader after this race | `0.0` | Pillar A — championship tension | Derived: `max(cumulative_points in round) − driver's cumulative_points`. `0` for the leader. Range: 0–397. No nulls. |

---

### 2.5 `J5_pit_strategy.csv` — 2,782 rows

One row per driver per race — but only for races where a valid pit stop was recorded. Covers 2011–2017 only.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `raceId` | int | Race identifier | `841` | Join key | No nulls. 2011–2017 only. |
| `driverId` | int | Driver identifier | `1` | Join key | No nulls. |
| `total_stops` | float | Total pit stops made in this race | `2.0` | Pillar B — strategy | Derived: `max(stop)` per raceId+driverId. Range: 1–6. 88.3% null in master (pre-2011 races). |
| `total_pit_ms` | float | Cumulative pit stop time in milliseconds | `46639.0` | Pillar B | Derived: `sum(milliseconds)` for valid stops. |
| `median_pit_ms` | float | Median single-stop duration in milliseconds | `23320.0` | Pillar B | Derived: `median(milliseconds)` for valid stops. |
| `min_pit_ms` | float | Fastest single pit stop in this race for this driver | `21400.0` | Pillar B — crew performance | Derived: `min(milliseconds)` for valid stops. |
| `first_pit_lap` | float | Lap number of the first pit stop | `14.0` | Pillar B — stop timing | Derived: `min(lap)` per raceId+driverId. Range: 1–56. 88.3% null in master. |
| `last_pit_lap` | float | Lap number of the final pit stop | `36.0` | Pillar B | Derived: `max(lap)` per raceId+driverId. Range: 1–74. |
| `total_pit_sec` | float | Cumulative pit time in seconds | `46.64` | Pillar B, Tableau | Derived: `total_pit_ms / 1000`. Range: 18.93–228.0 seconds. |
| `median_pit_sec` | float | Median single-stop duration in seconds | `23.32` | Pillar B — primary pit KPI, Tableau | Derived: `median_pit_ms / 1000`. Range: 15.05–70.19 seconds. 88.3% null in master. |
| `pit_time_per_stop_sec` | float | Average time per stop (total pit time ÷ number of stops) | `23.32` | Pillar B — efficiency | Derived: `total_pit_sec / total_stops`. Range: 8.50–70.19 seconds. |
| `strategy` | string | Categorical pit stop strategy label | `"2-Stop"` | Pillar B — primary grouping, Tableau | Derived: `total_stops` mapped to labels. Values: `1-Stop`, `2-Stop`, `3-Stop`, `4+ Stops`. 88.3% null in master. |

---

### 2.6 `J6_lap_by_lap.csv` — 426,633 rows

One row per driver per lap. The most granular joined table. Used for undercut analysis.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `raceId` | int | Race identifier | `841` | Join key | Composite key with driverId + lap. |
| `driverId` | int | Driver identifier | `1` | Join key | No nulls. |
| `lap` | int | Lap number | `14` | Pillar B | Range: 1–200. |
| `position` | int | On-track position at end of lap | `3` | Pillar B — undercut tracking | No nulls. Range: 1–33. |
| `milliseconds` | int | Lap time in milliseconds | `98109` | Pillar B — pace | No nulls. Range: 0–15,090,540. |
| `is_racing_lap` | bool | True if lap time is within 60,000–300,000 ms (filters out safety car and anomalies) | `True` | Pillar B — lap filtering | Derived. 337 non-racing laps in dataset. Always filter to `is_racing_lap = True` for lap pace analysis. |
| `full_name` | string | Driver full name | `"Lewis Hamilton"` | Tableau | Joined from drivers table. |
| `constructor_name` | string | Constructor name | `"McLaren"` | Tableau | Joined from constructors table. |
| `race_name` | string | Race name | `"Australian Grand Prix"` | Tableau | Joined from races table. |
| `year` | int | Season year | `2011` | Filter | Joined from races. Only 2011–2017 in this table. |
| `era` | string | Engine era | `"V8"` | Filter | Joined from races. Only V8 and Hybrid in this table. |
| `lap_time_sec` | float | Lap time in seconds | `98.11` | Pillar B — pace analysis | Derived: `milliseconds / 1000`. |
| `pit_stop_lap` | bool | True if a valid pit stop occurred on this lap | `False` | Pillar B — pit event marker | Derived from left join with pit_stops. Nulls filled with `False`. |
| `pit_duration_ms` | float | Duration of the pit stop if one occurred on this lap | `23320.0` | Pillar B | Null for non-pit laps. |
| `prev_position` | float | Driver's on-track position on the previous lap | `3.0` | Pillar B — position change | Derived: `groupby(raceId, driverId).position.shift(1)`. Null on lap 1. |
| `position_change` | float | Positions gained or lost vs previous lap. Positive = moved forward | `1.0` | Pillar B — undercut effectiveness | Derived: `prev_position − position`. Range: −20 to +13. Null on lap 1. |

---

### 2.7 `J7_reliability_dnf.csv` — 23,777 rows

One row per driver per race. Same granularity as J1, with additional reliability columns added.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| All columns from J1 | — | See J1 documentation above | — | Pillar C | Inherited with subset of columns selected. |
| `total_race_laps` | int | Maximum laps completed by any driver in this race (the winner's lap count) | `58` | Pillar C | Derived: `groupby(raceId).laps.max()`. Represents the full race distance. No nulls. |
| `laps_pct_completed` | float | Percentage of the race completed before retiring or finishing | `100.0` | Pillar C — DNF timing | Derived: `laps / total_race_laps * 100`. Range: 0.0–100.0. No nulls. Values of 0.0 indicate a lap-1 accident. |
| `potential_points_lost` | float | Estimated championship points the driver would have scored had they finished in their last known position | `0.0` | Pillar C — championship cost of DNF | Derived using era-appropriate points table applied to `positionOrder`. `0` for non-DNF rows. Range: 0–18. Non-zero only for DNF rows where the driver's `positionOrder` was in the points-scoring positions. Note: `positionOrder` for DNFs represents final classification order, not their on-track position at retirement — this is an approximation. |

---

### 2.8 `J8_constructor_reliability.csv` — 1,041 rows

One row per constructor per season. Aggregated from J7. Used for the reliability heatmap and era comparisons.

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| `constructorId` | int | Constructor identifier | `1` | Join key | Composite key with `year`. |
| `constructor_name` | string | Constructor name | `"McLaren"` | Pillar C, Tableau | No nulls. |
| `year` | int | Season year | `2008` | Join key, filters | No nulls. Range: 1950–2017. |
| `era` | string | Engine era | `"V8"` | Pillar C, filters | No nulls. |
| `entries` | int | Total race entries (driver-race starts) for this constructor this season | `36` | Pillar C — denominator | Derived: `count(resultId)`. No nulls. |
| `total_dnfs` | int | Total DNFs for this constructor this season | `5` | Pillar C | Derived: `sum(is_dnf)`. No nulls. |
| `mechanical_dnfs` | int | DNFs caused by mechanical failure | `3` | Pillar C | Derived: count of rows where `dnf_category == 'Mechanical DNF'`. No nulls. |
| `accident_dnfs` | int | DNFs caused by accidents or collisions | `1` | Pillar C | Derived: count of `dnf_category == 'Accident DNF'`. No nulls. |
| `other_dnfs` | int | DNFs due to disqualification, withdrawal, or other reasons | `1` | Pillar C | Derived: count of `dnf_category == 'Other DNF'`. No nulls. |
| `total_points` | float | Total championship points scored this season | `132.0` | Reference | Derived: `sum(points)`. No nulls. |
| `total_wins` | int | Total race wins this season | `7` | Reference | Derived: `sum(is_win)`. No nulls. |
| `pts_lost_to_dnf` | float | Estimated championship points lost due to DNFs this season | `12.0` | Pillar C | Derived: `sum(potential_points_lost)`. No nulls. |
| `dnf_rate` | float | Fraction of race entries that ended in DNF | `0.139` | Pillar C — primary reliability metric | Derived: `total_dnfs / entries`. Range: 0–1. No nulls. |
| `mechanical_dnf_rate` | float | Fraction of entries that ended in mechanical failure | `0.083` | Pillar C | Derived: `mechanical_dnfs / entries`. No nulls. |
| `reliability_index` | float | Complement of DNF rate. 1.0 = perfect reliability, 0.0 = all entries retired | `0.861` | Pillar C — primary KPI, Tableau heatmap | Derived: `1 − dnf_rate`. Range: 0–1. No nulls. |
| `rolling3_reliability` | float | 3-season rolling mean of `reliability_index` for this constructor | `0.814` | Pillar C — trend smoothing | Derived: `rolling(3, min_periods=1).mean()` applied per constructor sorted by year. No nulls. |
| `era_reliability` | float | Mean `reliability_index` across all seasons in this era for this constructor | `0.823` | Pillar C | Derived via groupby on constructorId + era. No nulls. |
| `era_mech_dnf_rate` | float | Mean mechanical DNF rate across all seasons in this era for this constructor | `0.057` | Pillar C | Derived via groupby on constructorId + era. No nulls. |

---

## Section 3 — Master Single Source Table

**File:** `MASTER_tableau_ready.csv` — 23,777 rows × 85 columns — 12.7 MB

This is the single file used in Tableau. All columns from J1 through J8 are flattened into one table joined on `resultId`, `raceId + driverId`, `driverId + year`, and `constructorId + year`. No row inflation: the master preserves exactly 23,777 rows.

**Columns are inherited directly from the joined tables.** For full definitions see sections 2.1–2.8 above. This section documents the join behaviour and any master-specific notes.

### Join logic used to build the master

| Source table | Join type | Join keys | Rows after join | Null implications |
|---|---|---|---|---|
| J1 (base) | — | — | 23,777 | None |
| J2 (driver alpha) | Left join | `driverId + year` | 23,777 | J2 deduplicated: drivers who raced for 2 constructors in one season — kept row with most races started that season |
| J3 (qualifying) | Left join | `raceId + driverId` | 23,777 | 68.9% null — qualifying data begins 1994 |
| J4 (championship) | Left join | `raceId + driverId` | 23,777 | J4 deduplicated: 89 duplicate raceId+driverId pairs (historical double-entry records) resolved by `keep='last'` |
| J5 (pit strategy) | Left join | `raceId + driverId` | 23,777 | 88.3% null — pit data is 2011–2017 only, and only for entries with valid stops |
| J7 (reliability extras) | Left join | `resultId` | 23,777 | 0% null — J7 has same granularity as J1 |
| J8 (constructor reliability) | Left join | `constructorId + year` | 23,777 | 0% null — every constructor-season combination is present |

### Column groups summary for Tableau

| Group | Columns | Null rate | Notes |
|---|---|---|---|
| Identity keys | `resultId`, `raceId`, `driverId`, `constructorId`, `statusId`, `circuitId` | 0% | Safe to use as join keys and filters |
| Race context | `race_name`, `year`, `round`, `date`, `era`, `circuit_name`, `location`, `country`, `lat`, `lng` | 0% | Core dimensions for all dashboards |
| Driver info | `full_name`, `nationality_driver`, `dob`, `birth_year` | 0% | Use `full_name` as primary driver label |
| Constructor info | `constructor_name`, `nationality_constructor` | 0% | Use `constructor_name` as primary team label |
| Race result | `grid`, `positionOrder`, `points`, `laps`, `status`, `dnf_category`, `is_win`, `is_podium`, `is_points_finish`, `is_pole`, `is_dnf`, `grid_vs_finish`, `laps_pct_completed` | 0–6.6% | Core measures for all three dashboards |
| Season KPIs | `ppr`, `win_rate`, `driver_alpha`, `career_alpha`, `constructor_avg_ppr`, `season_points`, `avg_finish_pos` | 0% | Always use `AVG()` in Tableau — values repeat per driver-season |
| Championship | `cumulative_points`, `cumulative_wins`, `championship_pos`, `gap_to_leader` | 0% | For bump chart and championship trajectory |
| Qualifying | `q1_ms`, `q2_ms`, `q3_ms`, `best_qual_ms`, `q1_vs_team_avg_ms`, `reached_q2`, `reached_q3` | 68–91% | Filter to `year >= 1994` before use |
| Pit stops | `total_stops`, `median_pit_sec`, `total_pit_sec`, `first_pit_lap`, `strategy` | 88% | Filter to `year BETWEEN 2011 AND 2017` before use |
| Reliability | `potential_points_lost`, `reliability_index`, `mechanical_dnf_rate`, `rolling3_reliability` | 0% | For reliability heatmap and DNF cost analysis |

---

## Section 4 — Derived Columns

All derived columns were computed in Python (`join_tables.py` and `etl_pipeline.py`) before loading into Tableau.

| Derived Column | Source Table | Logic | Business Meaning |
|---|---|---|---|
| `era` | races | `IF year <= 1994 → 'Pre-V10'; IF 1995–2005 → 'V10'; IF 2006–2013 → 'V8'; IF 2014+ → 'Hybrid'` | Groups seasons by engine regulation era. Enables era-to-era reliability and performance comparisons. |
| `full_name` | drivers | `forename.strip() + ' ' + surname.strip()` | Single display label for driver names across all charts. |
| `race_name` | races | Renamed from `name` | Avoids column collision when multiple tables with a `name` column are joined. |
| `constructor_name` | constructors | Renamed from `name` | Same anti-collision rename. |
| `circuit_name` | circuits | Renamed from `name` | Same anti-collision rename. |
| `nationality_driver` | drivers | Renamed from `nationality` | Distinguishes driver nationality from constructor nationality after join. |
| `nationality_constructor` | constructors | Renamed from `nationality` | Distinguishes constructor nationality from driver nationality after join. |
| `dnf_category` | status | `classify_dnf(status)` → one of: `Finished`, `Mechanical DNF`, `Accident DNF`, `Other DNF` | Collapses 134 individual status strings into 4 analytically meaningful groups for Pillar C. Mechanical = component failure. Accident = crash or collision. Other = administrative (DSQ, DNS, DNC). |
| `is_dnf` | results | `1 if position == 0 else 0` | Binary DNF flag. `position == 0` is the reliable indicator in this dataset. All 10,550 such rows are verified retirements. |
| `is_win` | results | `1 if positionOrder == 1 else 0` | Binary race win flag. Used to compute win rates and KPIs without filtering. |
| `is_podium` | results | `1 if positionOrder <= 3 else 0` | Binary podium flag. P1, P2, P3. |
| `is_points_finish` | results | `1 if points > 0 else 0` | Binary points-scoring flag. Points-scoring positions changed across eras (top 6 pre-2003, top 8 in 2003–2009, top 10 from 2010). |
| `is_pole` | results | `1 if grid == 1 else 0` | Binary pole position flag. |
| `grid_vs_finish` | results | `grid − positionOrder` ; `NULL if grid == 0` | Measures places gained or lost from starting position to finishing position. Positive = moved forward. The primary over/under-performance metric for Pillar A. Null for pit lane starts where grid position is meaningless. |
| `ppr` | J2 (aggregated from J1) | `season_points / races` | Points Per Race. Normalises a driver's scoring output per race regardless of how many races they entered. The primary comparator for driver quality across eras with different season lengths. |
| `constructor_avg_ppr` | J2 | `mean(ppr) grouped by constructorId + year` | The constructor's average PPR across all their drivers in a given season. Represents the car's baseline scoring potential — the "Beta" in the financial alpha analogy. |
| `driver_alpha` | J2 | `ppr − constructor_avg_ppr` | The driver's scoring output above or below their constructor's baseline. Positive alpha = driver extracts more from the car than average. This is the central Pillar A metric, analogous to active return in portfolio management. Range: −5.90 to +8.20. |
| `career_alpha` | J2 | `mean(driver_alpha)` across all seasons | Career-level measure of whether a driver consistently outperformed their machinery throughout their career. Smooths out single exceptional or poor seasons. Range: −3.0 to +3.94. |
| `q1_ms`, `q2_ms`, `q3_ms` | qualifying | `qual_time_to_ms(time_string)` — parses `"M:SS.mmm"` to milliseconds | Enables numeric comparison of qualifying times. String format is unusable for ranking or averaging. |
| `best_qual_ms` | qualifying | `min(q3_ms, q2_ms, q1_ms)` using backfill | Best available qualifying time regardless of which session set it. Allows comparison across drivers who reached different stages. |
| `q1_vs_team_avg_ms` | qualifying | `q1_ms − mean(q1_ms grouped by raceId + constructorId)` | Qualifying time relative to the team's average Q1 time in the same race. Negative = driver was faster than their team average. Captures head-to-head qualifying performance controlling for car pace. |
| `total_stops`, `median_pit_sec`, `strategy` | pit_stops | Aggregated via `groupby(raceId, driverId)` | Race-level pit stop summary. `total_stops = max(stop)`. `median_pit_sec = median(milliseconds) / 1000`. `strategy` categorises by total stops count. |
| `first_pit_lap` | pit_stops | `min(lap)` per raceId + driverId | Identifies when in the race the driver first pitted. Used to analyse early vs late stop strategies. |
| `laps_pct_completed` | J7 | `laps / total_race_laps * 100` | Normalises race completion across different race lengths. A driver retiring on lap 30 of 60 completed 50% regardless of circuit. |
| `potential_points_lost` | J7 | Era-appropriate points table applied to `positionOrder` for DNF rows only | Estimates championship points forfeited due to retirement. Uses the correct points table for each season's era. Non-DNF rows receive `0`. Known limitation: `positionOrder` for DNFs reflects final classification order, not their position on track at the moment of retirement. |
| `reliability_index` | J8 | `1 − dnf_rate` where `dnf_rate = total_dnfs / entries` | A 0–1 score where 1.0 = zero retirements in the season and 0.0 = every entry retired. More intuitive than DNF rate for dashboard colour encoding (green = good). |
| `rolling3_reliability` | J8 | `rolling(3, min_periods=1).mean()` of `reliability_index` per constructor sorted by year | Smooths year-to-year noise in reliability. A single catastrophic season (e.g., new engine introduction) inflates the DNF rate but the rolling average shows the underlying trend. |
| `cumulative_points` | J4 | `groupby(year, driverId).cumsum(points)` sorted by round | Running championship total after each race. Essential for championship trajectory and gap-to-leader calculations. |
| `championship_pos` | J4 | `rank(cumulative_points, ascending=False, method='min')` within year + round groups | Championship standing after each round. Enables the bump chart. `method='min'` assigns tied drivers the same (better) position, matching the official F1 convention. |
| `gap_to_leader` | J4 | `max(cumulative_points in round) − driver's cumulative_points` | Points deficit to the championship leader after each race. Zero for the leader. Quantifies how far a driver is from winning the title at any point in the season. |

---

## Section 5 — Tableau Calculated Fields

These are computed inside Tableau from the master table columns. They are not stored in the CSV.

| Calculated Field | Formula | Business Meaning |
|---|---|---|
| `[DNF Rate %]` | `SUM([is_dnf]) / COUNT([resultId]) * 100` | Overall DNF rate as a percentage. The denominator `COUNT([resultId])` counts all race entries including nulls. |
| `[Mechanical DNF Rate %]` | `SUM(IF [dnf_category]='Mechanical DNF' THEN 1 ELSE 0 END) / COUNT([resultId]) * 100` | Fraction of entries ending in mechanical failure. |
| `[Career Alpha Score]` | `AVG([career_alpha])` | Use `AVG` not `SUM` — the value repeats identically for each row of a driver-season, so summing inflates it. |
| `[Season Alpha]` | `AVG([driver_alpha])` | Season-level alpha. When grouped by driver + year, gives one value per season. |
| `[Points Per Race]` | `AVG([ppr])` | Season PPR. Use `AVG` for same reason as alpha — pre-aggregated value repeats per row. |
| `[Q1 Gap to Team (sec)]` | `AVG([q1_vs_team_avg_ms]) / 1000` | Qualifying advantage in seconds. Negative = faster than teammate average. |
| `[Qualifying Advantage %]` | `SUM(IF [q1_vs_team_avg_ms] < 0 THEN 1 ELSE 0 END) / COUNT([q1_ms]) * 100` | Percentage of sessions where driver was faster than team average. `COUNT([q1_ms])` only counts non-null rows. |
| `[First Stop % of Race]` | `AVG([first_pit_lap] / [laps] * 100)` | When in the race (as a percentage) the first pit stop occurred. |
| `[Pit Stop Entries]` | `COUNT([total_stops])` | Count of non-null pit stop rows. Use as denominator for pit metrics, not `COUNT([resultId])`. |
| `[1-Stop Win Rate %]` | `SUM(IF [strategy]='1-Stop' AND [is_win]=1 THEN 1 ELSE 0 END) / SUM(IF [strategy]='1-Stop' THEN 1 ELSE 0 END) * 100` | Win conversion rate specifically for 1-stop strategies. Expected value: 5.8%. |
| `[Reliability Index]` | `AVG([reliability_index])` | Use `AVG` — pre-aggregated at constructor-season level, repeats per row. |
| `[Points Lost to DNF]` | `SUM([potential_points_lost])` | Use `SUM` — this field is 0 for non-DNF rows, so summing across rows gives the correct total. |
| `[Era Order]` | `IF [era]='Pre-V10' THEN 1 ELSEIF [era]='V10' THEN 2 ELSEIF [era]='V8' THEN 3 ELSE 4 END` | Numeric sort key for eras. Without this, Tableau sorts alphabetically (Hybrid → Pre-V10 → V8 → V10). |
| `[Has Pit Data]` | `NOT ISNULL([total_stops])` | Boolean flag. Use as a filter on any pit stop sheet to exclude the 88.3% of null rows. |
| `[Pit Strategy Label]` | `IF ISNULL([strategy]) THEN 'No Pit Data' ELSE [strategy] END` | Prevents blank/null values appearing in strategy charts for pre-2011 data. |
| `[Is Contender]` | `IF [championship_pos] <= 5 THEN 'Title Contender' ELSE 'Non-Contender' END` | Labels drivers as championship contenders for use in Pillar C points-lost analysis. |

---

## Section 6 — Data Quality Notes

1. **Row count vs race count mismatch.** There are 23,777 result rows but only 976 unique `raceId` values (instead of the expected 997 from `Cleaned_Races.csv`). The 21 missing races had no entries in `Cleaned_Results.csv` — these are early 1950s races with incomplete historical records.

2. **Pit stop coverage is 2011–2017 only.** The Ergast API began recording pit stop data in 2011. All pit stop columns (`total_stops`, `median_pit_sec`, `strategy`, etc.) are null for 88.3% of the master table rows. Apply `year BETWEEN 2011 AND 2017` or `NOT ISNULL([total_stops])` as a filter on every Tableau sheet using pit data.

3. **Qualifying coverage begins 1994.** The three-session format (Q1/Q2/Q3) was introduced in 2006. Pre-2006 qualifying data exists in the source but follows different formats (single-lap shootout, aggregate times). The `q1_ms`, `q2_ms`, `q3_ms` columns are 68–91% null in the master table. Apply `year >= 1994` filter before using any qualifying column.

4. **Driver alpha inflation for mid-season team switches.** When a driver switches constructors mid-season (e.g., Sebastian Vettel at BMW Sauber then Toro Rosso in 2007), J2 originally produced one row per driver-constructor-year combination. The deduplication step in the ETL pipeline keeps the row with the most races started — the shorter stint's alpha score is dropped. Approximately 339 such cases exist in the raw J2 data.

5. **positionOrder for DNF drivers is classification order, not on-track position.** For the `potential_points_lost` estimate, `positionOrder` is used as a proxy for the driver's race position at retirement. However, `positionOrder` for DNFs reflects their classification rank among all non-finishers (ordered by laps completed), which may differ from their actual on-track running position at the moment of failure. This introduces estimation error of 1–3 positions in some cases.

6. **Double-points finale in 2014.** The 2014 Abu Dhabi Grand Prix awarded double points (winner scored 50 instead of 25). This is reflected in the `points` column. The `potential_points_lost` calculation uses `25` as the max value in the Hybrid points table — any driver who retired while leading the 2014 finale would have their potential points underestimated by 25.

7. **Pre-V10 era reliability data skews all-time averages.** The Pre-V10 era (1950–1994) has 14,842 rows — 62.4% of the entire dataset — and a DNF rate of ~56.5%. Any overall statistics on `is_dnf`, `dnf_category`, or `reliability_index` are heavily influenced by this era. Always filter by `era` or `year` when making modern-era claims.

8. **Points system changed six times.** The points awarded for each finishing position changed in 1960, 1961, 1991, 2003, and 2010. The `points` column stores the actual points awarded under the rules of that season. Comparing raw `points` across eras (e.g., a 1962 win scoring 9 points vs a 2015 win scoring 25) requires using `ppr` (Points Per Race) normalised within each season as a relative measure.

9. **Qualifying format changed.** Before 2003, qualifying used aggregate or single-lap formats. The Q1/Q2/Q3 knock-out format began in 2006. `q2_ms` and `q3_ms` nulls before 2006 reflect the absence of those sessions, not missing data.

10. **Extreme outliers in `q1_vs_team_avg_ms`.** Values range from −451,843 to +451,843 ms (approximately ±7.5 minutes). These extreme values occur in pre-modern era races where only one driver per constructor entered, making the team average equal to that driver's time and the gap equal to zero for that driver but producing large gaps when another driver subsequently joins or leaves. Clip to ±3 seconds (±3,000 ms) in any Tableau visualisation.

11. **J4 deduplication removes 89 rows.** The championship progression table had 89 duplicate `raceId + driverId` pairs from historical races where a driver entered under multiple constructor registrations in the same event. The `drop_duplicates(keep='last')` approach retains the last recorded entry. This affects a small number of 1950s–1960s races.

12. **`milliseconds` in results table is 0 for all DNFs.** The `milliseconds` column (total race time) is 0 for every non-finishing driver. It cannot be used for any cross-driver comparison without first filtering to finished entries. The column is retained in the master but not used in any KPI calculation.
