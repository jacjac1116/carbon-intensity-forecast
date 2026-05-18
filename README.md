# carbon-intensity-forecast

# UK Grid Carbon Intensity Forecasting System
## Project Specification

---

### Problem Statement

This project builds a multi-horizon forecasting system for Great Britain's electricity grid carbon intensity (gCO₂/kWh) at half-hourly resolution. The system predicts carbon intensity at three explicit forecast horizons — **t+1h**, **t+12h**, and **t+48h** — each representing a distinct operational use case and modelling challenge:

- **t+1h (nowcast):** Near-real-time operational decisions. The model has access to current generation mix, demand, and weather observations. The challenge is capturing rapid transitions (e.g., wind ramps, generator trips).
- **t+12h (intraday):** Scheduling and dispatch planning. The model relies on weather forecasts and expected demand profiles. The challenge is forecasting through diurnal transitions (overnight to morning ramp).
- **t+48h (day-ahead):** Strategic planning and market positioning. The model depends heavily on weather forecast skill and seasonal patterns. The challenge is compounding uncertainty from weather model errors.

Each horizon will produce probabilistic forecasts (10th, 50th, 90th percentiles) using quantile regression, providing operationally meaningful prediction intervals rather than point estimates alone.

The forecasting model is one component of a larger system. The project's primary differentiator is an **automated evaluation and failure analysis pipeline** that systematically identifies when, where, and why the model fails — and an **agentic evaluation layer** that generates hypotheses about failure causes and recommends targeted corrective actions.

---

### Stakeholder Value

**Primary stakeholders:** Distribution Network Operators (DNOs), flexibility service providers, battery storage operators, large industrial consumers, carbon reporting teams.

**Operational value of forecasts:**

Accurate carbon intensity forecasts enable battery operators to schedule charging during low-carbon windows and discharging during high-carbon periods, optimising both cost and emissions. Flexibility providers can time demand-side response to maximise carbon savings. Industrial consumers can shift energy-intensive processes (e.g., steel arc furnaces, data centre batch jobs) to cleaner periods, generating auditable carbon reductions. For DNOs such as Northern Powergrid, understanding when the grid is carbon-heavy informs network constraint management and supports regulatory reporting under Ofgem's net-zero obligations.

**Operational value of the failure analysis system:**

Forecasts are only trustworthy if their failure modes are understood. A model that systematically underpredicts carbon intensity during low-wind winter evenings is operationally dangerous — it would cause battery operators to undercharge when the grid is dirtiest. The evaluation pipeline's role is to surface these systematic errors, quantify their operational impact, and trigger alerts or retraining recommendations before they cause harm. This is where the system goes beyond a research exercise and becomes production-relevant.

---

### Data Sources

| Source | Provides | Access | Notes |
|--------|----------|--------|-------|
| National Grid ESO Carbon Intensity API (carbonintensity.org.uk) | Historical and forecast carbon intensity at national and regional level, generation mix breakdown | REST API (free, no key) | Primary target variable; forecast data used as a benchmark baseline, not as a model feature |
| National Grid ESO Data Portal (data.nationalgrideso.com) | Grid demand (national/regional), generation by fuel type, interconnector flows, system frequency | API / CSV download | Generation mix is the dominant feature set |
| Elexon BMRS (bmreports.com) | Settlement-period demand, generation by fuel type, balancing mechanism data | API (free tier available) | More granular than ESO for some generation data; useful for cross-validation of generation figures |
| Open-Meteo Historical Weather API (open-meteo.com) | Temperature, wind speed (10m/100m), solar irradiance (GHI/DNI), cloud cover, pressure | REST API (free, no key) | Preferred over Met Office for ease of access; covers full UK with hourly resolution |
| PVLive (solar.sheffield.ac.uk/pvlive) | Estimated national and regional solar PV generation | REST API (free) | Direct solar generation estimates, more useful than raw irradiance alone |
| ENTSO-E Transparency Platform (transparency.entsoe.eu) | European interconnector scheduled and actual flows, connected market generation | REST API (free, requires registration) | GB interconnector flows (France, Netherlands, Belgium, Norway) significantly affect the domestic generation mix |
| UK Government / Python libraries | Bank holidays, school holidays, sunset/sunrise times | Static CSV / `holidays` library | Calendar effects on demand patterns which indirectly affect dispatch |

---

### Feature Hypotheses

Each hypothesis is stated with its **mechanism** — the causal chain from feature to target — not just correlation.

**1. Generation Mix (dominant)**
Carbon intensity is arithmetically a function of the generation mix. Each fuel type has a known emission factor (gas ~394, coal ~937, biomass ~120, wind/solar/nuclear ~0 gCO₂/kWh). When wind output is high, gas plants are displaced down the merit order, and intensity drops. When wind drops, gas (and occasionally coal) ramp up to fill the gap. The generation mix is the single most explanatory feature set and will be available at all horizons (as actuals for t+1h, as forecasts for longer horizons).

**2. Interconnector Flows**
When GB imports from France (predominantly nuclear, ~50-60 gCO₂/kWh), it displaces domestic gas generation and lowers intensity. When GB exports or interconnectors are constrained, domestic thermal generation increases. The direction and magnitude of interconnector flows directly affect which domestic generators are dispatched.

**3. Electricity Demand**
Higher demand requires dispatching higher-cost, higher-carbon generators further up the merit order. Peak demand (winter evenings 16:00-19:00) consistently produces elevated intensity because base-load renewables and nuclear are insufficient and gas CCGTs fill the gap. At longer horizons, demand will be represented by forecasted values.

**4. Weather Variables**
Weather affects both supply and demand through distinct mechanisms:
- **Wind speed (100m hub height):** Directly determines wind generation output. Relationship is non-linear (cubic below rated speed, flat above, zero above cut-out ~25m/s).
- **Solar irradiance / cloud cover:** Determines PV output. Relevant primarily in summer daylight hours; negligible in winter evenings.
- **Temperature:** Drives heating demand (inverse relationship below ~15°C in the UK), which increases total demand and pushes up the merit order.
- **Pressure systems:** Persistent high-pressure blocking events correlate with low wind and clear skies (cold winters) or heatwaves (summer), creating sustained periods of high or unusual intensity.

**5. Autoregressive Features**
Carbon intensity exhibits strong temporal autocorrelation. The current intensity is highly predictive of near-future intensity because generator dispatch doesn't change instantaneously — plants take time to ramp up/down, and market dispatch cycles operate on half-hourly settlement periods. Lagged intensity values (t-1, t-2, t-48 for same-time-yesterday) will be particularly important for the t+1h horizon.

**6. Calendar and Temporal Features**
Time-of-day, day-of-week, and holiday indicators capture demand-driven patterns. Weekend demand is lower and flatter, shifting the dispatch stack. Bank holidays resemble Sundays. These are proxy features for demand shape when demand itself is not yet observed.

---

### Evaluation Strategy

#### Metrics

| Metric | Purpose | Notes |
|--------|---------|-------|
| MAE | Primary accuracy metric | Interpretable in gCO₂/kWh; used for headline reporting |
| RMSE | Penalises large errors | Important because large forecast errors have disproportionate operational cost |
| Pinball Loss (τ = 0.1, 0.5, 0.9) | Evaluates probabilistic calibration | Measures whether prediction intervals are well-calibrated |
| Prediction Interval Coverage (PICP) | Checks interval reliability | What fraction of actuals fall within the 10th-90th percentile interval? Target: ~80% |
| Prediction Interval Width (PINAW) | Checks interval sharpness | Narrow intervals are more useful; PICP alone rewards infinitely wide intervals |
| R² | Variance explained | Secondary; included for comparability with other published work |

**MAPE is explicitly excluded.** Carbon intensity regularly approaches zero during high-renewable periods, causing MAPE to diverge. It would produce misleading evaluations precisely when the grid is cleanest.

#### Temporal Validation

All evaluation uses **walk-forward (expanding window) cross-validation** with strict temporal ordering. No future data leaks into any training fold. The validation scheme:

1. Train on months 1–N, validate on month N+1
2. Expand training window, repeat
3. Report performance distribution across folds, not a single held-out score

This captures how the model would perform in a genuine production deployment where it's periodically retrained on accumulating data.

#### Baselines

| Baseline | Description |
|----------|-------------|
| Persistence | Predict that intensity at t+h equals intensity at t (for short horizons) or intensity at t-24h+h (same time yesterday, for longer horizons) |
| 7-day rolling mean | Simple smoothed average for the same half-hour period |
| ESO official forecast | The operational carbon intensity forecast published by National Grid ESO; this is the real-world benchmark to beat |

The model's value is measured as **improvement over the best available baseline**, not absolute accuracy. If the ESO forecast already achieves MAE of X, the model must demonstrably beat X to justify its existence.

#### Stratified Error Analysis

Performance will be broken down across operationally meaningful conditions:

- **By horizon:** t+1h, t+12h, t+48h reported separately — never averaged together
- **By season:** Winter (high demand, low solar), summer (low demand, high solar), shoulder months
- **By time-of-day:** Overnight, morning ramp, afternoon, evening peak
- **By wind regime:** Low wind (<5 GW national), medium (5-15 GW), high (>15 GW)
- **By demand level:** Low, medium, high (terciles)
- **By carbon intensity level:** Low (<100), medium (100-250), high (>250 gCO₂/kWh)
- **By day type:** Weekday, weekend, bank holiday
- **During known stress events:** Identified post-hoc (e.g., named storms, heatwaves, major plant outages)

This stratification is not decorative — each breakdown tests a specific hypothesis about where the model might fail and whether that failure matters operationally.

#### Residual Diagnostics

After each training run, the pipeline will automatically generate:

- Residual time series with rolling bias detection (is the model drifting?)
- Residual vs predicted scatter (is there heteroscedasticity — does error scale with prediction?)
- Autocorrelation function of residuals (are errors serially correlated — is the model missing a temporal pattern?)
- QQ plots (are residuals normally distributed, or are there heavy tails indicating extreme-event failures?)

---

### Automated Failure Analysis Pipeline

This is the system's core differentiator. After each evaluation cycle, the pipeline will:

1. **Detect failure regimes:** Automatically identify contiguous time windows where forecast error exceeds a threshold (e.g., >2σ from mean error). Cluster these windows by feature conditions to identify systematic patterns (e.g., "model consistently underpredicts by 40+ gCO₂/kWh during evening peaks when wind generation drops below 3 GW").

2. **Quantify operational impact:** For each identified failure regime, estimate the operational cost — e.g., "this systematic error pattern affects 12% of evening peak periods and would cause a battery operator to mistime discharge by an average of 2 settlement periods."

3. **Generate structured failure reports:** Produce machine-readable (JSON) and human-readable (markdown) reports summarising: which conditions trigger failures, how severe they are, how frequent they are, and whether they're worsening over time.

4. **Agentic evaluation layer (Phase 2):** An LLM-based agent ingests the failure reports and:
   - Generates hypotheses about root causes (e.g., "the model lacks features to capture rapid wind ramp-down events")
   - Proposes targeted test datasets to validate those hypotheses (e.g., "construct a test set of all periods where wind generation dropped >5 GW within 3 hours")
   - Recommends specific corrective actions (e.g., "add wind forecast gradient as a feature" or "increase training weight on evening peak periods")
   - Logs its reasoning for human review

This pipeline transforms the project from "I built a forecasting model" into "I built a system that understands its own weaknesses and proposes improvements" — which is precisely what the target job description requires.

---

### Known Risks and Challenges

**Data quality and alignment:** Energy settlement data operates on half-hourly settlement periods (not aligned to clock time during BST transitions). Weather data is typically hourly. Aligning these correctly — especially across the October and March clock changes — requires careful timestamp handling. Missing data from API outages must be detected and handled (interpolation for short gaps, exclusion for long gaps), not silently ignored.

**Forecast leakage across horizons:** The feature set must be strictly horizon-appropriate. At t+48h, the model cannot use observed generation mix or actual weather — it must use forecasted equivalents or drop those features entirely. This means the t+48h model is a fundamentally different model from the t+1h model, even if the architecture is shared. Feature auditing for leakage is a hard requirement.

**Distribution shift from grid decarbonisation:** The UK grid's carbon intensity has roughly halved over the past decade. Historical relationships between demand and intensity are non-stationary — five years ago, a given demand level would have produced much higher intensity than today because coal was still in the mix. The model must either account for this trend explicitly or use a training window short enough to reflect current grid conditions. This is also a risk to the evaluation baseline: if ESO's forecast is calibrated to current conditions and the model trains on older data, the comparison is unfair.

**Rare event representation:** Dunkelflaute (sustained low-wind, low-solar) periods, major generator trips, interconnector outages, and extreme temperature events are by definition rare in the training data. The model will likely perform worst exactly when accurate forecasts are most operationally valuable. The failure analysis pipeline is specifically designed to surface these cases, but improving model performance on them may require targeted data augmentation or specialised sub-models.

**API rate limits and data access:** Some data sources (particularly ENTSO-E and Met Office) have rate limits or registration requirements that may constrain the volume or frequency of data collection. The data pipeline must handle retries, caching, and graceful degradation.
