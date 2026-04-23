
# F1 EDA Findings

## Dataset Coverage
- 13 cleaned CSV files covering circuits, drivers, constructors, races, seasons,
  results, qualifying, pit stops, lap times, standings (driver + constructor), and status.

## Key Observations

### Circuits
- Circuits are distributed globally; a small number of countries host multiple venues.
- Altitude varies significantly — some circuits are near sea level while others are at high elevation,
  which can affect engine performance.

### Drivers
- The driver pool is heavily skewed toward European nationalities.
- Birth year distribution reveals waves of driver generations entering the sport.

### Seasons & Races
- The calendar has grown substantially over the decades (from ~8 to 20+ races/season).
- A small number of Grand Prix names dominate the historical count.

### Results
- The majority of race entries score zero points, reflecting that points only go to the top finishers.
- Grid position has a strong positive correlation with finish position, but significant movement occurs.

### Qualifying
- Q2 and Q3 times are missing for earlier seasons (format introduced in 2005).
- Qualifying position is the strongest single predictor of race finish.

### Pit Stops
- Most pit stops are 2–4 seconds (modern era); longer stops suggest mechanical issues.
- Pit stop count per race peaked in the era of tyre changes and has since standardised.

### Lap Times
- Median lap times vary widely by circuit; overall times have trended downward with faster cars.
- A slight performance drop is visible across a stint (tyre degradation).

### Standings
- Championship leader appearances concentrate around a small group of elite drivers.
- Points totals for champions have grown as race calendars expanded.

## Hypotheses for Next Steps
- H1: Drivers with below-median qualifying gaps vs teammate have higher points-per-race.
- H2: Fewer pit stops correlate with better race finish in the hybrid era.
- H3: Mechanical DNF rate has declined each decade due to improved reliability.
- H4: Constructor budget (proxied by championship position) predicts the following season's points.
