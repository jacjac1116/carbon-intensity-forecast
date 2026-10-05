WITH scored AS (
SELECT
p.horizon_steps,
p.target_time,
p.predicted_at,
a.actual,
p.forecast,
p.q10, p.q50, p.q90,
p.persistence,
a.official_forecast
FROM carbon_forecast.predictions as p
JOIN carbon_forecast.actuals as a
using (target_time)
where a.actual is not NULL
and p.forecast is not NULL
and p.q10 is not NULL
and p.q50 is not NULL
and p.q90 is not NULL
and p.persistence is not NULL
and a.official_forecast is not NULL
)
select
horizon_steps,
COUNT(*) AS n_rows,
ROUND(AVG(ABS(forecast - actual)), 2) AS mae_model,
ROUND(AVG(ABS(q50 - actual)), 2) AS mae_q50,  
ROUND(AVG(ABS(persistence - actual)), 2) AS mae_persistence,
ROUND(AVG(ABS(official_forecast - actual)), 2) AS mae_official,

ROUND(AVG(forecast - actual), 2) AS bias_model,
ROUND(AVG(IF(actual BETWEEN q10 AND q90, 1, 0)), 3) AS picp_80,
ROUND(AVG(IF(ABS(forecast - actual) < ABS(official_forecast - actual), 1, 0)), 3) AS frac_model_beats_official
from scored
order by horizon_steps