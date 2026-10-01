"""Employee Performance Reviews.

One row is one employee in one annual review cycle.  The three most recent
yearly reviews (each on 30 June) are covered; an employee appears in every
cycle where they had been employed for at least 6 months and had not yet
left.  Rows are sorted employee, then review date.  Salary is yearly, whole
units, no currency.

The story, in the order the code tells it:
    employees (hire date, ability, team, grade, pay level)
    -> each cycle: training hours
    -> TRUE performance = ability + learning curve with tenure (fast at first,
       then a plateau) + team effect + training + certification
    -> what we can OBSERVE: productivity (true performance + noise)
    -> what the MANAGER says: performance_score 1-5 = an imperfect reading of true
       performance, shifted by the team's manager (lenient or strict) and noise
    -> bonus = salary x a percentage set by the score band
    -> promotion (needs a good score), then the grade goes up next cycle
    -> some people leave after a review; low scorers leave more often

Definitions:
    performance_score  1 = well below expectations ... 5 = outstanding (whole numbers)
    productivity       output index, about 100 for a typical employee
    tenure_months      whole months from hire_date to review_date
    bonus              salary x bonus rate for the score: 1-2 -> 0%, 3 -> 3%,
                       4 -> 7%, 5 -> 12% (rounded to a whole number)
    is_active          True if still employed at the snapshot date
All numbers are illustrative teaching values; scores are deliberately imperfect
(measurement error and rater effects), which is part of what students study.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli, logistic

GRADE_SALARY = {1: 32_000, 2: 42_000, 3: 55_000, 4: 72_000, 5: 95_000}
BONUS_RATE = {1: 0.0, 2: 0.0, 3: 0.03, 4: 0.07, 5: 0.12}
CYCLES = 3
MIN_TENURE_DAYS = 180


def months_between(start, end):
    """Whole months from start to end (both Series of timestamps)."""
    months = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
    return (months - (end.dt.day < start.dt.day)).to_numpy()


def review_dates(snapshot):
    """30 June of each of the last three years that is not after the snapshot."""
    last = pd.Timestamp(year=snapshot.year, month=6, day=30)
    if last > snapshot:
        last = pd.Timestamp(year=snapshot.year - 1, month=6, day=30)
    return [pd.Timestamp(year=last.year - k, month=6, day=30) for k in range(CYCLES - 1, -1, -1)]


def simulate_employees(rng, m, cycles, first_id, first_team):
    """Simulate m employees through all cycles and return their review rows."""
    # 1. Persistent traits: when hired, ability, pay level, team, grade at hire.
    months_at_last = np.maximum(MIN_TENURE_DAYS / 30.4, lognormal(rng, 30, 0.9, m))
    hire_date = pd.Series(cycles[-1] - pd.to_timedelta(np.round(months_at_last * 30.4), unit="D"))
    ability = rng.normal(0, 1, m)
    pay_level = np.exp(rng.normal(0, 0.08, m))
    n_teams = max(1, m // 8)
    team = rng.integers(0, n_teams, m)
    team_effect = rng.normal(0, 0.4, n_teams)             # some teams really are stronger
    leniency = rng.normal(0, 0.35, n_teams)               # some managers score high or low
    grade = np.clip(1 + (months_at_last / 30).astype(int) // 2 + rng.integers(0, 2, m), 1, 4)
    certified = np.zeros(m, dtype=bool)
    employed = np.ones(m, dtype=bool)
    rows = []

    for review_date in cycles:
        # Present this cycle: employed, and hired at least 6 months before.
        days = (review_date - hire_date).dt.days.to_numpy()
        here = employed & (days >= MIN_TENURE_DAYS)
        tenure_months = months_between(hire_date, pd.Series(review_date, index=hire_date.index))
        tenure_years = tenure_months / 12

        # 2. Training hours this year: juniors train more.
        training = np.clip(np.round(lognormal(rng, 22, 0.5, m) * np.where(grade <= 2, 1.3, 1.0)), 0, 120)

        # 3. Certification: earned over time, more likely with training; never lost.
        earn = bernoulli(rng, logistic(-2.2 + 0.03 * (training - 22) + 0.25 * tenure_years))
        certified = certified | (earn & here)

        # 4. TRUE performance (never shown) and the observable productivity.
        learning = 1 - np.exp(-tenure_years / 1.5)          # steep at first, then flat
        true_perf = (ability + 1.2 * learning + 0.5 * team_effect[team]
                     + 0.03 * (training - 22) + 0.25 * certified - 1.0)   # centred so 0 = typical
        productivity = np.round(100 + 11 * true_perf + rng.normal(0, 7, m)).astype(int)

        # 5. The manager's score: imperfect reading + the team's leniency + noise.
        raw = 3.1 + 0.75 * true_perf + leniency[team] + rng.normal(0, 0.55, m)
        score = np.clip(np.round(raw), 1, 5).astype(int)

        # 6. Pay: grade band x personal level x a little seniority; bonus by score band.
        base = np.array([GRADE_SALARY[g] for g in grade], dtype=float)
        salary = np.round(base * pay_level * (1 + 0.02 * np.minimum(tenure_years, 10)) / 100) * 100
        bonus = np.round(salary * np.array([BONUS_RATE[s] for s in score]))

        # 7. Promotion: needs a good score, a year of service and room to grow.
        chance = np.where(score == 5, 0.45, np.where(score == 4, 0.25, 0.0))
        promoted = bernoulli(rng, chance) & (tenure_months >= 12) & (grade < 5)

        rows.append(pd.DataFrame({
            "employee_idx": np.arange(m)[here],
            "team_idx": team[here] + first_team,
            "grade": grade[here], "hire_date": hire_date[here].to_numpy(),
            "review_date": review_date, "tenure_months": tenure_months[here],
            "training_hours": training[here].astype(int), "is_certified": certified[here],
            "productivity": productivity[here], "performance_score": score[here],
            "salary": salary[here].astype(int), "bonus": bonus[here].astype(int),
            "is_promoted": promoted[here],
            "true_performance": np.round(true_perf[here], 3),
        }))

        # 8. After the review: promoted people move up; weak scorers leave more often.
        grade = grade + (promoted & here)
        p_leave = logistic(-2.4 - 0.6 * (score - 3) - 0.7 * promoted)
        employed = employed & ~(here & bernoulli(rng, p_leave))

    out = pd.concat(rows)
    out["is_active"] = employed[out.employee_idx.to_numpy()]
    out["employee_idx"] = out.employee_idx + first_id
    return out


def simulate(rng, n, snapshot):
    cycles = review_dates(snapshot)
    parts, total, first_id, first_team = [], 0, 0, 0
    while total < n:                                 # add employees until there are enough rows
        m = max(10, int((n - total) / 1.8) + 5)
        part = simulate_employees(rng, m, cycles, first_id, first_team)
        parts.append(part)
        total += len(part)
        first_id += m
        first_team += max(1, m // 8)
    df = pd.concat(parts)
    df["team_id"] = "T" + (df.team_idx + 1).astype(str).str.zfill(3)
    df = df.sort_values(["employee_idx", "review_date"]).reset_index(drop=True)
    df["employee_id"] = "E" + (df.employee_idx + 1).astype(str).str.zfill(5)
    df = df.drop(columns=["employee_idx", "team_idx"])
    return df.iloc[:n].reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    hd, rd = pd.to_datetime(df.hire_date), pd.to_datetime(df.review_date)
    g = df.groupby("employee_id")

    # ---- exact rules ----
    out.append(check("tenure_months = whole months from hire_date to review_date",
                     (months_between(hd, rd) == df.tenure_months).all()))
    out.append(check("hire_date at least 6 months before every review; review_date <= snapshot",
                     ((rd - hd).dt.days >= MIN_TENURE_DAYS).all() and (rd <= snapshot).all()))
    out.append(check("bonus = salary x bonus rate for the score band (rounded)",
                     (df.bonus == np.round(df.salary * df.performance_score.map(BONUS_RATE))).all()))
    out.append(check("performance_score is a whole number 1-5", df.performance_score.between(1, 5).all()))
    out.append(check("grade 1-5 and salary positive", df.grade.between(1, 5).all() and (df.salary > 0).all()))
    out.append(check("training_hours between 0 and 120", df.training_hours.between(0, 120).all()))
    out.append(check("promotion needs score >= 4, tenure >= 12 months, grade < 5",
                     (~df.is_promoted | ((df.performance_score >= 4) & (df.tenure_months >= 12)
                                         & (df.grade < 5))).all()))
    nxt_grade = g.grade.shift(-1)
    promoted_with_next = df.is_promoted & nxt_grade.notna()
    out.append(check("is_promoted => grade is one higher next cycle",
                     (nxt_grade[promoted_with_next] == df.grade[promoted_with_next] + 1).all()))
    stay = ~df.is_promoted & nxt_grade.notna()
    out.append(check("not promoted => same grade next cycle",
                     (nxt_grade[stay] == df.grade[stay]).all()))
    cert_drop = g.is_certified.shift(1).fillna(False).astype(bool) & ~df.is_certified
    out.append(check("a certification is never lost", not cert_drop.any()))
    out.append(check("one hire_date and one is_active per employee",
                     (g.hire_date.nunique() == 1).all() and (g.is_active.nunique() == 1).all()))
    out.append(check("review dates are 30 June, one row per employee per cycle",
                     (rd.dt.month == 6).all() and (rd.dt.day == 30).all()
                     and not df.duplicated(["employee_id", "review_date"]).any()))
    out.append(check("an employee stays in one team (grade never falls)",
                     (g.team_id.nunique() == 1).all() and (g.grade.diff().dropna() >= 0).all()))

    # ---- relationships ----
    c1 = float(df.productivity.corr(df.performance_score, method="spearman"))
    out.append(check("productivity and manager score move together (but imperfectly)",
                     0.25 < c1 < 0.9, f"rank corr={c1:.2f}", "relationship"))
    new = df.productivity[df.tenure_months < 24]
    old = df.productivity[df.tenure_months >= 48]
    if len(new) >= 15 and len(old) >= 15:
        out.append(check("learning curve: median productivity, under 2 years' tenure < 4+ years",
                         new.median() < old.median(),
                         f"{new.median():.0f} vs {old.median():.0f}", "relationship"))
    c3 = float(df.training_hours.corr(df.productivity, method="spearman"))
    out.append(check("more training goes with higher productivity",
                     c3 > 0.05, f"rank corr={c3:.2f}", "relationship"))
    # Rater effect: after accounting for productivity, teams still differ in scores.
    if df.team_id.nunique() >= 4 and len(df) >= 60:
        slope = np.polyfit(df.productivity, df.performance_score, 1)
        resid = df.performance_score - np.polyval(slope, df.productivity)
        between = float(((resid.groupby(df.team_id).transform("mean")) ** 2).sum())
        share = between / float((resid ** 2).sum())
        chance = (df.team_id.nunique() - 1) / (len(df) - 1)
        out.append(check("teams differ in how they score (rater effect beyond productivity)",
                         share > 1.5 * chance, f"share={share:.2f}, chance~{chance:.2f}", "relationship"))
    if df.is_promoted.sum() >= 5:
        out.append(check("promoted rows have higher average scores",
                         df.performance_score[df.is_promoted].mean()
                         > df.performance_score[~df.is_promoted].mean(), "", "relationship"))
    cert = df.is_certified
    if cert.sum() >= 15 and (~cert).sum() >= 15:
        out.append(check("certified employees have higher median productivity",
                         df.productivity[cert].median() > df.productivity[~cert].median(),
                         "", "relationship"))
    return out


RECIPE = Recipe(
    key="hr_performance",
    title="Employee Performance Reviews",
    row_meaning=("One row is one employee in one annual review cycle "
                 "(the three most recent yearly reviews)."),
    simulate=simulate,
    columns=["employee_id", "team_id", "grade", "hire_date", "review_date", "tenure_months",
             "training_hours", "is_certified", "productivity", "performance_score",
             "salary", "bonus", "is_promoted", "is_active"],
    core=["tenure_months", "training_hours", "productivity", "performance_score"],
    priority=["bonus", "salary", "is_promoted", "grade", "is_certified", "team_id",
              "review_date", "hire_date", "is_active", "employee_id"],
    docs={
        "employee_id": "Unique employee identifier; repeats once per review cycle.",
        "team_id": "The team (and manager) the employee belongs to; managers score more leniently or strictly.",
        "grade": "Job grade from 1 (entry) to 5 (most senior); rises by one the cycle after a promotion.",
        "hire_date": "Date the employee joined.",
        "review_date": "Date of the annual review (30 June).",
        "tenure_months": "Whole months from hire_date to review_date.",
        "training_hours": "Hours of training taken in the year before the review.",
        "is_certified": "True once the employee holds a professional certification (never lost).",
        "productivity": "Measured output index for the year (about 100 for a typical employee); a noisy reading of real performance.",
        "performance_score": "Manager's rating from 1 (well below expectations) to 5 (outstanding); a rough, imperfect judgement.",
        "salary": "Yearly base salary at the review, whole units (no currency).",
        "bonus": "Bonus paid = salary x rate for the score (1-2: 0%, 3: 3%, 4: 7%, 5: 12%), whole units.",
        "is_promoted": "True if promoted at this review (needs score of 4 or 5 and a year of service).",
        "is_active": "True if the employee still works there at the snapshot date.",
    },
    validate=validate,
    date_cols=["hire_date", "review_date"],
    targets={
        "performance_score": ["bonus"],
        "is_promoted": ["performance_score", "bonus"],
        "is_active": [],
    },
)
