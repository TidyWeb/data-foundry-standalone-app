"""Factory Energy Usage.

One row is one site-day: one factory site on one calendar day.  Energy is in
kWh, water in cubic metres, carbon in kg of CO2 (using a *fictional* emission
factor per site), hours are hours.

The story, in the order the code tells it:
    site (size, opening hours, efficiency, tariff) -> weather (heating degree
    days) -> breakdowns (downtime) -> production for the day -> energy
    -> water -> cost and carbon -> load factor -> flags

Energy follows a baseline-plus-production model:
    expected kWh = baseline + kWh per unit x production + kWh per degree-day x HDD
A factory burns energy even when it makes nothing (lighting, heating, idle
machines) - that is the baseline, the non-zero intercept of the line.  A site's
real use is the expectation times its own efficiency and a bit of daily noise;
days more than 10% above expectation are flagged.  Illustrative teaching values.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal

FLAG_ABOVE = 1.10          # flagged: used more than 10% above the expected kWh
EFFICIENT_BELOW = 0.97     # efficient: used at least 3% less than expected
WEEKDAY_PRODUCTION = [1.0, 1.0, 1.0, 1.0, 1.0, 0.65, 0.30]     # Monday ... Sunday


def simulate(rng, n, snapshot):
    # 1. Sites: each has its own size and habits, fixed for every row.
    n_sites = int(np.clip(np.ceil(n / 80), 3, 12))
    scheduled_hours = rng.choice([16.0, 24.0], n_sites, p=[0.5, 0.5])
    units_per_hour = lognormal(rng, 40, 0.5, n_sites)             # production speed
    baseline_kwh = lognormal(rng, 1500, 0.4, n_sites) * scheduled_hours / 24
    kwh_per_unit = rng.uniform(0.6, 2.5, n_sites)
    kwh_per_degree_day = baseline_kwh * rng.uniform(0.010, 0.025, n_sites)   # heating load
    site_efficiency = lognormal(rng, 1.0, 0.08, n_sites)          # >1 means wasteful
    water_base = lognormal(rng, 60, 0.4, n_sites)
    water_per_unit = rng.uniform(0.02, 0.10, n_sites)
    tariff = rng.uniform(0.09, 0.16, n_sites)                     # cost per kWh
    emission_factor = rng.uniform(0.15, 0.30, n_sites)            # fictional kg CO2 per kWh

    # 2. One row per site per day (enough days to cover n rows), oldest day first.
    n_days = int(np.ceil(n / n_sites))
    dates = pd.date_range(snapshot - pd.Timedelta(days=n_days), periods=n_days)
    grid = pd.MultiIndex.from_product([dates, range(n_sites)], names=["date", "site"]).to_frame(index=False)
    rows = len(grid)
    site = grid.site.to_numpy()
    day_of_year = pd.DatetimeIndex(grid.date).dayofyear.to_numpy()
    weekday = pd.DatetimeIndex(grid.date).dayofweek.to_numpy()

    # 3. Weather: cold winters.  Heating degree days measure how far the average
    #    temperature falls below 15.5 C (0 on a warm day).  All sites share the
    #    regional weather, plus a small local difference.
    regional = rng.normal(0, 3, n_days)[np.searchsorted(dates, grid.date)]
    temperature = 11 - 7 * np.cos(2 * np.pi * (day_of_year - 20) / 365) + regional + rng.normal(0, 0.8, rows)
    heating_degree_days = np.round(np.clip(15.5 - temperature, 0, None), 1)

    # 4. Downtime: breakdowns are occasional and last a couple of hours.
    hours_scheduled = scheduled_hours[site]
    breakdowns = rng.poisson(0.5, rows)
    lost = np.array([lognormal(rng, 3.0, 0.6, int(b)).sum() if b else 0.0 for b in breakdowns])
    downtime = np.round(np.minimum(lost, 0.8 * hours_scheduled), 1)
    uptime = np.round(hours_scheduled - downtime, 1)

    # 5. Production for the day: running hours x speed, lower at weekends.
    production = uptime * units_per_hour[site] * np.array(WEEKDAY_PRODUCTION)[weekday] * lognormal(rng, 1.0, 0.07, rows)
    production_units = np.round(production).astype(int)

    # 6. Energy: expectation from the baseline model, then the site's efficiency and noise.
    expected_kwh = (baseline_kwh[site] + kwh_per_unit[site] * production_units
                    + kwh_per_degree_day[site] * heating_degree_days)
    energy = expected_kwh * site_efficiency[site] * lognormal(rng, 1.0, 0.045, rows)
    energy_usage = np.round(energy).astype(int)

    # 7. Water rises with production; cost and carbon are simple multiples of kWh.
    water_usage = np.round(water_base[site] + water_per_unit[site] * production_units
                           * lognormal(rng, 1.0, 0.08, rows)).astype(int)
    energy_cost = np.round(energy_usage * tariff[site], 2)
    carbon_footprint = np.round(energy_usage * emission_factor[site]).astype(int)

    # 8. Load factor = average demand / peak demand.  Steady, fully running days
    #    have a flatter demand curve (higher load factor).
    load_factor = np.round(np.clip(0.50 + 0.32 * uptime / hours_scheduled + rng.normal(0, 0.04, rows),
                                   0.35, 0.98), 2)
    peak_demand_kw = (energy_usage / 24) / load_factor

    df = pd.DataFrame({
        "site_id": [f"S{k + 1}" for k in site],
        "usage_date": grid.date,
        "production_units": production_units,
        "scheduled_hours": hours_scheduled,
        "uptime": uptime,
        "downtime": downtime,
        "load_factor": load_factor,
        "heating_degree_days": heating_degree_days,
        "energy_usage": energy_usage,
        "water_usage": water_usage,
        "carbon_footprint": carbon_footprint,
        "energy_cost": energy_cost,
        "is_efficient": energy_usage < EFFICIENT_BELOW * expected_kwh,
        "is_flagged": energy_usage > FLAG_ABOVE * expected_kwh,
        # hidden helpers
        "expected_kwh": np.round(expected_kwh, 1),
        "expected_kwh_exact": expected_kwh,
        "peak_demand_kw": peak_demand_kw,
        "site_efficiency": site_efficiency[site],
        "tariff": tariff[site],
        "emission_factor": emission_factor[site],
    })
    return df.iloc[:n].reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _demeaned(df, col):
    return df[col] - df.groupby("site_id")[col].transform("mean")


def validate(df, snapshot):
    out = []

    # ---- exact rules ----
    out.append(check("uptime + downtime = scheduled hours",
                     ((df.uptime + df.downtime - df.scheduled_hours).abs() < 1e-6).all()))
    out.append(check("downtime >= 0 and uptime > 0", ((df.downtime >= 0) & (df.uptime > 0)).all()))
    out.append(check("energy_cost = energy_usage x tariff (+-0.005)",
                     ((df.energy_cost - df.energy_usage * df.tariff).abs() <= 0.0051).all()))
    out.append(check("carbon_footprint = energy_usage x emission factor (+-0.5)",
                     ((df.carbon_footprint - df.energy_usage * df.emission_factor).abs() <= 0.5001).all()))
    out.append(check("load_factor = average demand / peak demand, and is between 0 and 1",
                     ((df.load_factor - (df.energy_usage / 24) / df.peak_demand_kw).abs() < 1e-9).all()
                     and df.load_factor.between(0.01, 1).all()))
    out.append(check("is_flagged exactly when energy is more than 10% above the expected kWh",
                     (df.is_flagged == (df.energy_usage > FLAG_ABOVE * df.expected_kwh_exact)).all()))
    out.append(check("is_efficient exactly when energy is at least 3% below the expected kWh",
                     (df.is_efficient == (df.energy_usage < EFFICIENT_BELOW * df.expected_kwh_exact)).all()))
    out.append(check("a day cannot be both flagged and efficient", not (df.is_flagged & df.is_efficient).any()))
    out.append(check("counts and usages are non-negative",
                     ((df.energy_usage > 0) & (df.water_usage >= 0) & (df.production_units >= 0)
                      & (df.heating_degree_days >= 0) & (df.carbon_footprint > 0)).all()))
    out.append(check("one row per site and date", not df.duplicated(["site_id", "usage_date"]).any()))
    out.append(check("dates not after snapshot", (pd.to_datetime(df.usage_date) <= snapshot).all()))
    out.append(check("scheduled hours constant per site",
                     (df.groupby("site_id").scheduled_hours.nunique() == 1).all()
                     and (df.groupby("site_id").tariff.nunique() == 1).all()))

    # ---- relationships ----
    n = len(df)
    c = float(_demeaned(df, "energy_usage").corr(_demeaned(df, "production_units"))) if n >= 30 else 1.0
    out.append(check("within a site, busier days use more energy", c > 0.4, f"corr={c:.2f}", "relationship"))

    ratios = []
    for _, g in df.groupby("site_id"):
        if len(g) >= 15:
            g = g.sort_values("production_units")
            third = len(g) // 3
            ratios.append(g.energy_usage.iloc[:third].median() / g.energy_usage.iloc[-third:].median())
    if ratios:
        r = float(np.mean(ratios))
        out.append(check("quiet days still use a large baseline of energy (between 45% and 97% of busy days)",
                         0.45 < r < 0.97, f"ratio={r:.2f}", "relationship"))

    if n >= 60:
        x = np.column_stack([_demeaned(df, "production_units"), _demeaned(df, "heating_degree_days")])
        beta = np.linalg.lstsq(x, _demeaned(df, "energy_usage").to_numpy(), rcond=None)[0]
        out.append(check("colder days (more heating degree days) use more energy after allowing for production",
                         beta[1] > 0, f"kWh per degree-day={beta[1]:.1f}", "relationship"))

    sites = df.groupby("site_id").agg(flagged=("is_flagged", "mean"), eff=("site_efficiency", "first"))
    if len(sites) >= 3 and sites.eff.max() / sites.eff.min() >= 1.10:     # sites must really differ
        worst, best = sites.eff.idxmax(), sites.eff.idxmin()
        out.append(check("the most wasteful site is flagged more often than the most efficient one",
                         sites.flagged[worst] > sites.flagged[best],
                         f"{sites.flagged[worst]:.2f} vs {sites.flagged[best]:.2f}", "relationship"))
    weekdays = df[pd.to_datetime(df.usage_date).dt.dayofweek < 5]      # weekends are quieter for every site
    c = float(_demeaned(weekdays, "downtime").corr(_demeaned(weekdays, "production_units"))) if len(weekdays) >= 30 else -1.0
    out.append(check("on weekdays, more downtime means fewer units made", c < -0.3, f"corr={c:.2f}",
                     "relationship"))
    c = float(_demeaned(df, "water_usage").corr(_demeaned(df, "production_units"))) if n >= 30 else 1.0
    out.append(check("within a site, busier days use more water", c > 0.5, f"corr={c:.2f}", "relationship"))
    return out


RECIPE = Recipe(
    key="manufacturing_energy",
    title="Factory Energy Usage",
    row_meaning="One row is one factory site on one calendar day.",
    simulate=simulate,
    columns=["site_id", "usage_date", "production_units", "scheduled_hours", "uptime", "downtime",
             "load_factor", "heating_degree_days", "energy_usage", "water_usage",
             "carbon_footprint", "energy_cost", "is_efficient", "is_flagged"],
    core=["site_id", "production_units", "heating_degree_days", "energy_usage"],
    priority=["usage_date", "water_usage", "downtime", "is_flagged", "energy_cost",
              "carbon_footprint", "uptime", "load_factor", "is_efficient", "scheduled_hours"],
    docs={
        "site_id": "Factory site identifier; each site has one row per day.",
        "usage_date": "Calendar date the row describes.",
        "production_units": "Units produced at the site that day.",
        "scheduled_hours": "Hours the site was scheduled to run (16 or 24, fixed per site).",
        "uptime": "Hours the site actually ran (scheduled hours minus downtime).",
        "downtime": "Hours lost to breakdowns during the scheduled time.",
        "load_factor": "Average power demand divided by peak demand for the day (0 to 1); higher means a steadier load.",
        "heating_degree_days": "How cold the day was: 15.5 C minus the average temperature, or 0 on a warm day.",
        "energy_usage": "Electricity used that day in kilowatt-hours (kWh).",
        "water_usage": "Water used that day in cubic metres.",
        "carbon_footprint": "Kilograms of CO2 from the electricity, using a fictional per-site emission factor.",
        "energy_cost": "Electricity bill for the day: kWh x the site's tariff per kWh (plain number).",
        "is_efficient": "True if energy_usage was at least 3% below what the site's baseline model expects for that day.",
        "is_flagged": "True if energy_usage was more than 10% above what the site's baseline model expects for that day.",
    },
    validate=validate,
    date_cols=["usage_date"],
    targets={
        "energy_usage": ["energy_cost", "carbon_footprint", "is_flagged", "is_efficient", "load_factor"],
        "is_flagged": ["energy_usage", "energy_cost", "carbon_footprint", "is_efficient"],
        "production_units": ["uptime", "downtime"],
    },
)
