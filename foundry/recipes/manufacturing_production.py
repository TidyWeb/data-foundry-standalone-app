"""Manufacturing Production Output.

One row is one production line-shift: what one line produced during one shift
on one day.  Rates are percentages (0-100), times are minutes, cycle_time is
seconds per unit.

The story, in the order the code tells it:
    line (ideal cycle time, automated?, maintenance rhythm) -> shift (planned
    minutes) -> breakdowns and changeovers (downtime) -> running speed
    -> total output -> defects -> good units -> the KPIs

This is the standard OEE idea (Overall Equipment Effectiveness):
    availability = operating time / planned time
    performance  = (output x ideal cycle time) / operating time
    quality      = good units / total output
    OEE          = availability x performance x quality
Downtime takes minutes away, running below ideal speed takes units away, and
defects take good units away.  Illustrative teaching values, not real data.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal

SHIFTS = ["day", "evening", "night"]
PLANNED_MINUTES = {"day": 450, "evening": 450, "night": 420}       # 8h less breaks
DOWNTIME_SHIFT_FACTOR = {"day": 1.0, "evening": 1.1, "night": 1.4}
DEFECT_SHIFT_FACTOR = {"day": 1.0, "evening": 1.1, "night": 1.4}
MAINTENANCE_EVERY_DAYS = 28      # each line is serviced every 4 weeks


def simulate(rng, n, snapshot):
    # 1. Lines.  Automated lines run faster and steadier; manual lines are slower.
    n_lines = int(np.clip(n // 40, 4, 12))
    is_automated = np.array([i % 2 == 0 for i in range(n_lines)])
    rng.shuffle(is_automated)
    ideal_cycle = lognormal(rng, 14, 0.4, n_lines) * np.where(is_automated, 0.75, 1.0)
    ideal_cycle = np.round(np.clip(ideal_cycle, 4, 60), 1)               # seconds per unit
    base_speed = np.where(is_automated, 0.93, 0.85) + rng.normal(0, 0.02, n_lines)
    base_defect = np.where(is_automated, 0.012, 0.030) * lognormal(rng, 1.0, 0.25, n_lines)
    breakdown_rate = np.where(is_automated, 1.0, 1.3) * lognormal(rng, 1.0, 0.2, n_lines)
    service_offset = rng.integers(0, MAINTENANCE_EVERY_DAYS, n_lines)

    # 2. One row for every line, shift and day (enough days to cover n rows).
    n_days = int(np.ceil(n / (n_lines * len(SHIFTS))))
    dates = pd.date_range(snapshot - pd.Timedelta(days=n_days), periods=n_days)
    grid = pd.MultiIndex.from_product([dates, range(3), range(n_lines)],
                                      names=["production_date", "shift_no", "line"]).to_frame(index=False)
    rows = len(grid)
    line = grid.line.to_numpy()
    shift = np.array(SHIFTS, dtype=object)[grid.shift_no.to_numpy()]
    day_number = (pd.to_datetime(grid.production_date) - dates[0]).dt.days.to_numpy()

    # 3. Planned time, then downtime.  Downtime events are more frequent on the
    #    night shift and the longer it has been since the line was serviced.
    planned_time = np.array([PLANNED_MINUTES[s] for s in shift])
    days_since_service = (day_number + service_offset[line]) % MAINTENANCE_EVERY_DAYS
    wear_factor = 1 + 1.5 * days_since_service / MAINTENANCE_EVERY_DAYS
    event_rate = 0.5 * breakdown_rate[line] * wear_factor * np.array([DOWNTIME_SHIFT_FACTOR[s] for s in shift])
    events = rng.poisson(event_rate)
    downtime = np.zeros(rows)
    for i in np.flatnonzero(events):                       # each event lasts a skewed number of minutes
        downtime[i] = lognormal(rng, 22, 0.7, int(events[i])).sum()
    downtime = np.minimum(np.round(downtime), 0.6 * planned_time).astype(int)
    operating_time = planned_time - downtime

    # 4. Speed: how close to the ideal cycle time the line ran (never above it).
    speed = base_speed[line] - 0.03 * (shift == "night") + rng.normal(0, 0.04, rows)
    speed = np.clip(speed, 0.5, 1.0)

    # 5. Output: units made while the line was running, at that speed.
    cycle_time = ideal_cycle[line]
    output = np.floor(operating_time * 60 * speed / cycle_time).astype(int)
    capacity = np.floor(planned_time * 60 / cycle_time).astype(int)

    # 6. Defects: rarer on automated lines, likelier on nights, and likelier when
    #    the line is pushed close to its top speed.
    defect_chance = (base_defect[line] * np.array([DEFECT_SHIFT_FACTOR[s] for s in shift])
                     * (1 + 6 * np.clip(speed - 0.85, 0, None)))
    scrap = rng.binomial(output, np.clip(defect_chance, 0, 0.5))
    good_units = output - scrap

    df = pd.DataFrame({
        "line_id": [f"L{k + 1}" for k in line],
        "production_date": grid.production_date,
        "shift": shift,
        "is_automated": is_automated[line],
        "capacity": capacity,
        "cycle_time": cycle_time,
        "planned_time": planned_time,
        "downtime": downtime,
        "output": output,
        "good_units": good_units,
        "defect_rate": np.round(100 * scrap / np.maximum(output, 1), 2),
        "availability": np.round(100 * operating_time / planned_time, 2),
        "throughput": np.round(output / (operating_time / 60), 2),
        # OEE = A x P x Q simplifies to good units x ideal cycle time / planned time
        "oee": np.round(100 * good_units * cycle_time / (planned_time * 60), 2),
        # hidden helpers
        "operating_time": operating_time,
        "scrap_units": scrap,
        "speed_factor": speed,
        "days_since_service": days_since_service,
        "downtime_events": events,
    })
    return df.iloc[:n].reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _demeaned(df, col):
    return df[col] - df.groupby("line_id")[col].transform("mean")


def validate(df, snapshot):
    out = []
    performance = df.output * df.cycle_time / (df.operating_time * 60)
    availability = df.operating_time / df.planned_time
    quality = df.good_units / df.output

    # ---- exact rules ----
    out.append(check("operating time = planned time - downtime, and > 0",
                     ((df.planned_time - df.downtime == df.operating_time) & (df.operating_time > 0)).all()))
    out.append(check("availability = operating time / planned time (+-0.005)",
                     ((df.availability - 100 * availability).abs() <= 0.0051).all()))
    out.append(check("output <= capacity (theoretical maximum for the planned time)",
                     (df.output <= df.capacity).all()))
    out.append(check("output = whole units made at the running speed",
                     (df.output == np.floor(df.operating_time * 60 * df.speed_factor / df.cycle_time)).all()))
    out.append(check("performance (output x cycle time / operating time) is at most 100%",
                     (performance <= 1.0 + 1e-9).all()))
    out.append(check("good units + scrap units = output, good <= output",
                     ((df.good_units + df.scrap_units == df.output) & (df.good_units >= 0)).all()))
    out.append(check("defect_rate = scrap / output x 100 (+-0.005)",
                     ((df.defect_rate - 100 * df.scrap_units / df.output).abs() <= 0.0051).all()))
    out.append(check("throughput = output per operating hour (+-0.005)",
                     ((df.throughput - df.output / (df.operating_time / 60)).abs() <= 0.0051).all()))
    out.append(check("OEE = availability x performance x quality (+-0.02)",
                     ((df.oee - 100 * availability * performance * quality).abs() <= 0.02).all()))
    out.append(check("OEE between 0 and 100", df.oee.between(0, 100).all()))
    out.append(check("line traits constant per line",
                     (df.groupby("line_id").cycle_time.nunique() == 1).all()
                     and (df.groupby("line_id").is_automated.nunique() == 1).all()))
    out.append(check("one row per line, date and shift",
                     not df.duplicated(["line_id", "production_date", "shift"]).any()))
    out.append(check("dates not after snapshot", (pd.to_datetime(df.production_date) <= snapshot).all()))
    out.append(check("shifts valid; planned time matches shift",
                     df["shift"].isin(SHIFTS).all()
                     and (df.planned_time == df["shift"].map(PLANNED_MINUTES)).all()))

    # ---- relationships ----
    by_line = df.groupby("line_id").agg(auto=("is_automated", "first"), oee=("oee", "mean"))
    if by_line.auto.nunique() == 2:
        out.append(check("automated lines have higher average OEE",
                         by_line.oee[by_line.auto].mean() > by_line.oee[~by_line.auto].mean(),
                         "", "relationship"))
    c = float(_demeaned(df, "downtime").corr(_demeaned(df, "output"))) if len(df) >= 30 else -1.0
    out.append(check("within a line, more downtime means less output", c < -0.4,
                     f"corr={c:.2f}", "relationship"))
    night, day = df[df["shift"] == "night"], df[df["shift"] == "day"]
    if len(night) >= 10 and len(day) >= 10:
        out.append(check("night shifts lose more minutes to downtime than day shifts",
                         night.downtime.mean() > day.downtime.mean(),
                         f"{night.downtime.mean():.1f} vs {day.downtime.mean():.1f}", "relationship"))
    c = float(df.days_since_service.corr(df.downtime)) if len(df) >= 30 else 1.0
    out.append(check("downtime creeps up the longer since the line was serviced", c > 0.05,
                     f"corr={c:.2f}", "relationship"))
    c = float(_demeaned(df, "speed_factor").corr(_demeaned(df, "defect_rate"))) if len(df) >= 30 else 1.0
    out.append(check("within a line, running faster goes with a higher defect rate", c > 0.02,
                     f"corr={c:.2f}", "relationship"))
    return out


RECIPE = Recipe(
    key="manufacturing_production",
    title="Manufacturing Production Output",
    row_meaning=("One row is one production line-shift: what one line made during one "
                 "shift on one day."),
    simulate=simulate,
    columns=["line_id", "production_date", "shift", "is_automated", "capacity", "cycle_time",
             "planned_time", "downtime", "output", "good_units", "defect_rate",
             "availability", "throughput", "oee"],
    core=["line_id", "downtime", "output", "oee"],
    priority=["is_automated", "planned_time", "good_units", "defect_rate", "shift",
              "cycle_time", "production_date", "availability", "capacity", "throughput"],
    docs={
        "line_id": "Production line identifier; every line appears in every shift and day.",
        "production_date": "Calendar date of the shift.",
        "shift": "day, evening or night.",
        "is_automated": "True for automated lines (faster and steadier) and False for manual ones.",
        "capacity": "Most units the line could make in the planned time at its ideal cycle time.",
        "cycle_time": "Ideal seconds to make one unit on this line (fixed per line).",
        "planned_time": "Minutes the shift was scheduled to produce (breaks already removed).",
        "downtime": "Minutes lost to breakdowns and changeovers during the shift.",
        "output": "Total units made in the shift, good or scrap.",
        "good_units": "Units that passed quality checks (output minus scrap).",
        "defect_rate": "Scrapped units as a percentage of output (0-100, 2 dp).",
        "availability": "Operating time as a percentage of planned time: (planned - downtime) / planned x 100.",
        "throughput": "Units made per hour of actual running time.",
        "oee": "Overall Equipment Effectiveness (%): availability x performance x quality, i.e. good units x cycle time / planned time.",
    },
    validate=validate,
    date_cols=["production_date"],
    targets={
        "oee": ["good_units", "availability", "defect_rate", "throughput", "output"],
        "good_units": ["output", "defect_rate", "oee"],
        "defect_rate": ["good_units", "oee"],
    },
)
