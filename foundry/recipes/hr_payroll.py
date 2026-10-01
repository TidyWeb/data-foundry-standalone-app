"""Payroll & Compensation.

One row is one monthly payslip for one employee.  The table covers every pay
date since the start of the current pay year (which begins 1 April) up to the
snapshot; pay day is the 25th.  Rows are sorted employee, then pay date.
Everything is FICTIONAL: the tax bands, pension and levy below belong to no
country.  Amounts are plain numbers with no currency.

The story, in the order the code tells it:
    employee (role family, grade, contract hours, hire date, union member?)
    -> annual salary (grade band x role x length of service x personal spread)
    -> for each pay date the employee is on the payroll:
       base pay = salary / 12 x FTE  (pro-rated in the month of hire)
       overtime hours (by role; some people do it month after month)
       overtime pay = hours x hourly wage x multiplier
       bonus (paid in June and December, sized by grade)
       gross pay = base pay + overtime pay + bonus
       deductions = income tax + pension + union dues + levy
       net pay = gross pay - deductions
    -> some people leave part-way through the year (is_active = False)

Fictional rules (per month):
    wage           = salary / (52 x 40)  (hourly rate for a 40-hour week, 2 dp)
    FTE            = contract hours / 40; is_fulltime means 40 contract hours
    overtime rate  = wage x 1.5 for union members, wage x 1.25 for others;
                     managers are not paid overtime
    pension        = 5% of base pay
    union dues     = 1% of base pay, union members only
    levy           = 2% of gross pay, at most 300 a month
    income tax     = 0% on the first 1,000 of (gross - pension), 15% on the next
                     2,000, 25% on the next 3,000, 35% above 6,000
Every line is rounded to 2 decimals (pennies) FIRST, then the totals are added,
so gross_pay - deductions = net_pay holds exactly.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli

ROLES = {   # name: (weight, pay factor, share full-time, union share, overtime hours per month)
    "operations": (0.30, 0.95, 0.85, 0.55, 9.0),
    "admin":      (0.20, 0.90, 0.70, 0.30, 2.5),
    "technical":  (0.22, 1.15, 0.90, 0.15, 4.5),
    "sales":      (0.18, 1.00, 0.85, 0.05, 2.0),
    "management": (0.10, 1.10, 1.00, 0.00, 0.0),
}
GRADE_SALARY = {1: 24_000, 2: 30_000, 3: 38_000, 4: 48_000, 5: 62_000, 6: 80_000}
BONUS_MONTHS_OF_PAY = {1: 0.4, 2: 0.5, 3: 0.7, 4: 1.0, 5: 1.4, 6: 2.0}   # paid in June and December
LEVY_CAP = 30_000       # cents


def pay_dates(snapshot):
    """25th of each month from 1 April of the current pay year to the snapshot."""
    last = pd.Timestamp(snapshot.year, snapshot.month, 25)
    if last > snapshot:
        last = last - pd.DateOffset(months=1)
    start_year = last.year if last.month >= 4 else last.year - 1
    return list(pd.date_range(pd.Timestamp(start_year, 4, 25), last, freq=pd.DateOffset(months=1)))


def income_tax_cents(taxable):
    """Progressive fictional tax on monthly taxable pay, in cents."""
    tax = (0.15 * np.clip(taxable - 100_000, 0, 200_000)
           + 0.25 * np.clip(taxable - 300_000, 0, 300_000)
           + 0.35 * np.maximum(taxable - 600_000, 0))
    return np.round(tax).astype(np.int64)


def build_employees(rng, m, first_id, dates):
    months = len(dates)
    first_day = pd.Timestamp(dates[0].year, dates[0].month, 1)

    # 1. Who they are.
    names = list(ROLES)
    role = pick(rng, names, [ROLES[r][0] for r in names], m)
    is_fulltime = bernoulli(rng, np.array([ROLES[r][2] for r in role]))
    contract_hours = np.where(is_fulltime, 40, rng.choice([16, 20, 24, 30], m))
    fte = contract_hours / 40
    is_union = bernoulli(rng, np.array([ROLES[r][3] for r in role]))

    # 2. Hire date: most joined before this pay year; about 12% joined during it.
    joined_this_year = bernoulli(rng, np.full(m, 0.12))
    service_days = np.ceil(lognormal(rng, 4 * 365, 0.9, m))
    span = (dates[-1] - first_day).days
    hire_date = pd.Series(np.where(joined_this_year,
                                   first_day + pd.to_timedelta(rng.integers(0, span + 1, m), unit="D"),
                                   first_day - pd.to_timedelta(service_days, unit="D")))
    hire_date = pd.to_datetime(hire_date)
    service_years = np.maximum((first_day - hire_date).dt.days.to_numpy() / 365, 0)

    # 3. Grade rises with service (plus luck); managers start higher.  Salary follows.
    grade = 1 + (service_years / 3).astype(int) + rng.integers(0, 2, m)
    grade = np.where(role == "management", np.maximum(grade, 4), np.minimum(grade, 5))
    grade = np.clip(grade, 1, 6)
    step = np.minimum(service_years, 8)
    salary = (np.array([GRADE_SALARY[g] for g in grade]) * np.array([ROLES[r][1] for r in role])
              * (1 + 0.02 * step) * lognormal(rng, 1.0, 0.06, m))
    salary = (np.round(salary / 100) * 100).astype(int)
    wage_c = np.round(salary * 100 / 2080).astype(np.int64)       # hourly wage in cents

    # 4. Which pay dates each person is on the payroll.
    pay = np.array(dates, dtype="datetime64[ns]")
    hired_by = hire_date.to_numpy()[:, None] <= pay[None, :]
    first_index = np.where(hired_by.any(axis=1), hired_by.argmax(axis=1), months)
    last_index = np.full(m, months - 1)
    leaves = bernoulli(rng, np.full(m, 0.10)) & (first_index < months - 1)
    span_left = np.maximum(months - 1 - first_index, 1)
    last_index = np.where(leaves, first_index + rng.integers(0, span_left), last_index)
    k = np.arange(months)[None, :]
    present = (k >= first_index[:, None]) & (k <= last_index[:, None]) & (first_index[:, None] < months)

    # 5. Base pay: salary / 12 x FTE, pro-rated in the month someone starts.
    month_start = np.array([pd.Timestamp(d.year, d.month, 1) for d in dates], dtype="datetime64[ns]")
    month_end = np.array([pd.Timestamp(d.year, d.month, 1) + pd.offsets.MonthEnd(0) for d in dates],
                         dtype="datetime64[ns]")
    days_in_month = (month_end - month_start).astype("timedelta64[D]").astype(int) + 1
    days_worked = (month_end[None, :] - np.maximum(hire_date.to_numpy()[:, None], month_start[None, :])
                   ).astype("timedelta64[D]").astype(int) + 1
    proration = np.clip(days_worked / days_in_month[None, :], 0, 1)
    base_c = np.round(salary[:, None] * 100 / 12 * fte[:, None] * proration).astype(np.int64)

    # 6. Overtime: role-dependent, with a personal habit that persists and rare busy spells.
    habit = lognormal(rng, 1.0, 0.5, m)
    busy = np.where(bernoulli(rng, np.full((m, months), 0.06)), 3.0, 1.0)
    per_month = np.array([ROLES[r][4] for r in role])
    overtime_hours = rng.poisson(per_month[:, None] * habit[:, None] * busy)
    multiplier = np.where(is_union, 1.5, 1.25)
    overtime_c = np.round(overtime_hours * wage_c[:, None] * multiplier[:, None]).astype(np.int64)

    # 7. Bonus: paid on the June and December pay dates to people hired 6+ months ago.
    is_bonus_month = np.array([d.month in (6, 12) for d in dates])[None, :]
    eligible = (pay[None, :] - hire_date.to_numpy()[:, None]) >= np.timedelta64(182, "D")
    performance = lognormal(rng, 1.0, 0.3, m)
    months_of_pay = np.array([BONUS_MONTHS_OF_PAY[g] for g in grade]) * performance
    bonus = np.round(salary / 12 * fte * months_of_pay)[:, None] * (is_bonus_month & eligible)

    # 8. Gross, deductions and net, all in whole cents.
    gross_c = base_c + overtime_c + bonus.astype(np.int64) * 100
    pension_c = np.round(0.05 * base_c).astype(np.int64)
    union_c = np.round(0.01 * base_c).astype(np.int64) * is_union[:, None]
    levy_c = np.minimum(np.round(0.02 * gross_c).astype(np.int64), LEVY_CAP)
    tax_c = income_tax_cents(gross_c - pension_c)
    deductions_c = tax_c + pension_c + union_c + levy_c
    net_c = gross_c - deductions_c

    def flat(a):
        return np.broadcast_to(np.asarray(a), (m, months))[present]

    def per_employee(a):
        return flat(np.asarray(a)[:, None])

    ids = np.array([f"E{first_id + i + 1:05d}" for i in range(m)], dtype=object)
    last_paid = pd.to_datetime(np.array(dates, dtype="datetime64[ns]")[np.clip(last_index, 0, months - 1)])
    df = pd.DataFrame({
        "employee_id": per_employee(ids),
        "role_family": per_employee(role),
        "is_fulltime": per_employee(is_fulltime),
        "is_unionized": per_employee(is_union),
        "is_active": per_employee(last_index == months - 1),
        "hire_date": per_employee(hire_date.to_numpy()),
        "payment_date": flat(np.broadcast_to(pay[None, :], (m, months))),
        "salary": per_employee(salary),
        "wage": per_employee(wage_c) / 100,
        "base_pay": flat(base_c) / 100,
        "overtime_hours": flat(overtime_hours),
        "overtime_pay": flat(overtime_c) / 100,
        "bonus": flat(bonus).astype(int),
        "gross_pay": flat(gross_c) / 100,
        "deductions": flat(deductions_c) / 100,
        "net_pay": flat(net_c) / 100,
        # hidden helpers
        "tax": flat(tax_c) / 100, "pension": flat(pension_c) / 100,
        "union_dues": flat(union_c) / 100, "levy": flat(levy_c) / 100,
        "contract_hours": per_employee(contract_hours),
        "proration": flat(proration),
        "grade": per_employee(grade),
        "last_paid_date": per_employee(last_paid.to_numpy()),
    })
    return df


def simulate(rng, n, snapshot):
    dates = pay_dates(snapshot)
    parts, total, first_id = [], 0, 0
    while total < n:                                    # add employees until there are enough rows
        m = max(10, int((n - total) / max(1, len(dates)) * 1.2) + 3)
        part = build_employees(rng, m, first_id, dates)
        parts.append(part)
        total += len(part)
        first_id += m
    df = pd.concat(parts).reset_index(drop=True)
    for c in ["hire_date", "payment_date", "last_paid_date"]:
        df[c] = pd.to_datetime(df[c])
    df["ytd_gross"] = np.round(df.groupby("employee_id").gross_pay.cumsum(), 2)
    df["last_pay_date"] = dates[-1]
    return df.iloc[:n].reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def cents(series):
    return np.round(series.to_numpy() * 100).astype(np.int64)


def validate(df, snapshot):
    out = []
    pd_ = pd.to_datetime(df.payment_date)
    g = df.groupby("employee_id")

    # ---- exact rules (money compared in whole cents) ----
    out.append(check("gross_pay = base_pay + overtime_pay + bonus",
                     (cents(df.gross_pay) == cents(df.base_pay) + cents(df.overtime_pay) + df.bonus * 100).all()))
    out.append(check("deductions = tax + pension + union_dues + levy",
                     (cents(df.deductions) == cents(df.tax) + cents(df.pension)
                      + cents(df.union_dues) + cents(df.levy)).all()))
    out.append(check("net_pay = gross_pay - deductions (to the penny)",
                     (cents(df.net_pay) == cents(df.gross_pay) - cents(df.deductions)).all()))
    out.append(check("wage = salary / (52 x 40), 2 dp",
                     ((df.wage - df.salary / 2080).abs() <= 0.0051).all()))
    fte = df.contract_hours / 40
    expected_base = df.salary / 12 * fte * df.proration
    out.append(check("base_pay = salary / 12 x FTE x proration (2 dp)",
                     ((df.base_pay - expected_base).abs() <= 0.0051).all()))
    out.append(check("base_pay never exceeds salary / 12 x FTE (+0.01)",
                     (df.base_pay <= df.salary / 12 * fte + 0.011).all()))
    mult = np.where(df.is_unionized, 1.5, 1.25)
    out.append(check("overtime_pay = hours x wage x multiplier (1.5 union, 1.25 other; +-0.01)",
                     ((df.overtime_pay - df.overtime_hours * df.wage * mult).abs() <= 0.011).all()))
    out.append(check("managers have no overtime",
                     (df.overtime_hours[df.role_family == "management"] == 0).all()))
    out.append(check("pension = 5% of base_pay; union dues = 1% of base_pay for members only",
                     ((df.pension - 0.05 * df.base_pay).abs() <= 0.011).all()
                     and ((df.union_dues - 0.01 * df.base_pay * df.is_unionized).abs() <= 0.011).all()))
    out.append(check("levy = 2% of gross_pay, capped at 300",
                     ((df.levy - np.minimum(0.02 * df.gross_pay, 300)).abs() <= 0.011).all()))
    taxable = (cents(df.gross_pay) - cents(df.pension)).astype(float)
    out.append(check("income tax follows the fictional bands (+-0.01)",
                     ((cents(df.tax) - income_tax_cents(taxable)).__abs__() <= 1).all()))
    out.append(check("bonus paid only in June and December, never negative",
                     ((df.bonus == 0) | pd_.dt.month.isin([6, 12])).all() and (df.bonus >= 0).all()))
    out.append(check("ytd_gross is the running total of gross_pay per employee",
                     ((df.ytd_gross - g.gross_pay.cumsum()).abs() <= 0.011).all()))
    out.append(check("pay dates are the 25th, not after the snapshot, and on/after hire_date",
                     (pd_.dt.day == 25).all() and (pd_ <= snapshot).all()
                     and (pd.to_datetime(df.hire_date) <= pd_).all()))
    out.append(check("one payslip per employee per pay date",
                     not df.duplicated(["employee_id", "payment_date"]).any()))
    out.append(check("is_fulltime = 40 contract hours",
                     (df.is_fulltime == (df.contract_hours == 40)).all()))
    out.append(check("is_active = still on the latest pay date; leavers stop being paid",
                     (df.is_active == (pd.to_datetime(df.last_paid_date) == pd.to_datetime(df.last_pay_date))).all()
                     and (pd_ <= pd.to_datetime(df.last_paid_date)).all()))
    out.append(check("salary, hours and amounts non-negative",
                     (df.salary > 0).all() and (df.overtime_hours >= 0).all() and (df.net_pay > 0).all()))
    out.append(check("employee traits constant", (g.role_family.nunique() == 1).all()
                     and (g.salary.nunique() == 1).all() and (g.is_unionized.nunique() == 1).all()))

    # ---- relationships ----
    service = (pd_ - pd.to_datetime(df.hire_date)).dt.days
    c1 = float(service.corr(df.salary, method="spearman"))
    out.append(check("longer service goes with higher salary",
                     c1 > 0.15, f"rank corr={c1:.2f}", "relationship"))
    med = df.groupby("role_family").salary.median()
    counts = df.groupby("role_family").size()
    if {"management", "admin"} <= set(med.index) and counts["management"] >= 6 and counts["admin"] >= 6:
        out.append(check("median salary: management > admin",
                         med["management"] > med["admin"], "", "relationship"))
    ot = df.groupby("role_family").overtime_hours.mean()
    if {"operations", "sales"} <= set(ot.index) and counts["operations"] >= 10 and counts["sales"] >= 10:
        out.append(check("mean overtime hours: operations > sales",
                         ot["operations"] > ot["sales"], "", "relationship"))
    prev = g.overtime_hours.shift(1)
    both = prev.notna()
    if both.sum() >= 30:
        c2 = float(df.overtime_hours[both].corr(prev[both]))
        out.append(check("overtime is habitual: correlated with the employee's previous month",
                         c2 > 0.1, f"corr={c2:.2f}", "relationship"))
    rate = df.tax / df.gross_pay
    c3 = float(rate.corr(df.gross_pay, method="spearman"))
    out.append(check("progressive tax: effective tax rate rises with gross_pay",
                     c3 > 0.5, f"rank corr={c3:.2f}", "relationship"))
    ft, pt = df.base_pay[df.is_fulltime], df.base_pay[~df.is_fulltime]
    if len(ft) >= 10 and len(pt) >= 10:
        out.append(check("part-timers have lower median base_pay than full-timers",
                         pt.median() < ft.median(), "", "relationship"))
    return out


RECIPE = Recipe(
    key="hr_payroll",
    title="Payroll & Compensation",
    row_meaning=("One row is one monthly payslip for one employee, for every pay date "
                 "since the start of the current pay year (1 April)."),
    simulate=simulate,
    columns=["employee_id", "role_family", "is_fulltime", "is_unionized", "is_active",
             "hire_date", "payment_date", "salary", "wage", "base_pay", "overtime_hours",
             "overtime_pay", "bonus", "gross_pay", "deductions", "net_pay"],
    core=["role_family", "salary", "gross_pay", "net_pay"],
    priority=["base_pay", "deductions", "overtime_hours", "overtime_pay", "bonus",
              "payment_date", "is_fulltime", "wage", "is_unionized", "hire_date",
              "employee_id", "is_active"],
    docs={
        "employee_id": "Unique employee identifier; repeats once per monthly payslip.",
        "role_family": "operations, admin, technical, sales or management.",
        "is_fulltime": "True if the contract is 40 hours a week; part-timers work 16-30.",
        "is_unionized": "True if the employee is a union member (pays dues and gets the higher overtime rate).",
        "is_active": "True if the employee is still on the payroll at the latest pay date.",
        "hire_date": "Date the employee joined.",
        "payment_date": "Pay day (the 25th of the month).",
        "salary": "Contracted yearly salary for a full-time equivalent, whole units (no currency); part-timers are paid pro rata.",
        "wage": "Hourly rate = salary / (52 x 40), 2 dp.",
        "base_pay": "Basic pay for the month = salary / 12 x FTE (part-time fraction), pro-rated in the month of hire; 2 dp.",
        "overtime_hours": "Overtime hours worked in the month (managers have none).",
        "overtime_pay": "Overtime hours x wage x 1.5 (union) or 1.25 (non-union), 2 dp.",
        "bonus": "Bonus paid this month, whole units; only in June and December, sized by grade.",
        "gross_pay": "Total pay before deductions = base_pay + overtime_pay + bonus.",
        "deductions": "Total deducted = fictional income tax + pension (5%) + union dues (1%, members) + levy (2%, capped at 300).",
        "net_pay": "Take-home pay = gross_pay - deductions, exact to the penny.",
    },
    validate=validate,
    date_cols=["hire_date", "payment_date"],
    targets={
        "net_pay": ["gross_pay", "deductions"],
        "gross_pay": ["net_pay", "deductions"],
        "overtime_hours": ["overtime_pay", "gross_pay", "net_pay"],
    },
)
