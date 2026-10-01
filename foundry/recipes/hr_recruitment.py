"""Recruitment & Hiring.

One row is one job vacancy (requisition) that has been FILLED, described
through the hiring funnel that filled it and the journey of the candidate
who was hired.  Only filled vacancies are included, so every column is
defined for every row (there is no hire date for a vacancy still open).
Hires happened in the last 24 months.  Salaries are yearly, whole units
(no currency).

The story, in the order the code tells it:
    department and seniority -> market salary for that kind of role
    -> the salary the company put in the offer (a bit above or below market)
    -> how many people applied (remote roles attract far more; senior roles fewer)
    -> how many were interviewed -> how many offers had to be made
       (each offer is accepted with a probability that rises with pay vs market)
    -> the hired candidate was the m-th best interviewee, m = offers made
       (so their interview score is high, but lower when the first choices said no)
    -> time from application to interview, and from interview to accepted offer
       (referred candidates skip queues; senior roles take longer; each rejected
       offer costs more days)
    -> dates, worked backwards from the hire date

Definitions:
    applicants       = applications received for the vacancy up to the hire
    interviewed      = candidates interviewed for it
    offers_made      = offers made until one was accepted (1 = first choice said yes)
    interview_score  = the hired candidate's interview score, 1 to 10
    time_to_hire     = days from the hired candidate's application to hire_date
    hire_date        = the day the offer was accepted
(Time-to-fill, measured from when the vacancy opened, is a hidden helper.)
All numbers are illustrative teaching values.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli, logistic

DEPARTMENTS = {                       # name: (weight, typical junior-to-mid market salary)
    "engineering": (1.3, 68_000), "sales": (1.2, 52_000), "marketing": (0.8, 50_000),
    "finance": (0.7, 58_000), "operations": (1.0, 46_000), "support": (1.0, 38_000),
}
LEVELS = ["junior", "mid", "senior", "lead"]
LEVEL_WEIGHT = [0.35, 0.35, 0.22, 0.08]
LEVEL_SALARY = {"junior": 0.75, "mid": 1.0, "senior": 1.35, "lead": 1.75}
LEVEL_RANK = {name: i for i, name in enumerate(LEVELS)}
WINDOW_DAYS = 730


def simulate(rng, n, snapshot):
    # 1. Department, seniority and the market salary for that kind of role.
    names = list(DEPARTMENTS)
    department = pick(rng, names, [DEPARTMENTS[d][0] for d in names], n)
    seniority = pick(rng, LEVELS, LEVEL_WEIGHT, n)
    level = np.array([LEVEL_RANK[s] for s in seniority])
    market = np.array([DEPARTMENTS[d][1] * LEVEL_SALARY[s]
                       for d, s in zip(department, seniority)]) * lognormal(rng, 1.0, 0.05, n)

    # 2. The offer: usually near the market, sometimes generous or stingy.
    #    Rounded to the nearest 500.
    pay_ratio = np.exp(rng.normal(0.02, 0.08, n))
    salary_offered = (np.round(market * pay_ratio / 500) * 500).astype(int)

    # 3. Remote roles and referrals.  Remote roles draw a much bigger pool.
    is_remote = bernoulli(rng, np.where(department == "support", 0.20, 0.35))
    #    Referrals are common for junior/mid roles and rare for lead roles.
    is_referred = bernoulli(rng, np.array([0.30, 0.25, 0.15, 0.08])[level])

    # 4. Applicants (senior roles get fewer), then how many get interviewed.
    applicants = np.round(lognormal(rng, 45, 0.55, n) * np.where(is_remote, 1.8, 1.0)
                          * 0.7 ** level).astype(int)
    applicants = np.maximum(applicants, 4)
    interviewed = np.clip(np.round(applicants * rng.uniform(0.06, 0.14, n)), 3, 15).astype(int)

    # 5. Offers.  Each offer is accepted with a probability that rises when
    #    the salary is above market and falls for senior roles (more options).
    p_accept = logistic(0.7 + 8.0 * np.log(salary_offered / market) - 0.25 * level)
    offers_made = rng.geometric(p_accept)                    # offers until the first "yes"
    interviewed = np.maximum(interviewed, offers_made)       # you cannot offer to more people than you met
    applicants = np.maximum(applicants, interviewed)

    # 6. Interview scores: every interviewee has a score; we hire the offers_made-th best.
    widest = int(interviewed.max())
    scores = np.clip(np.round(rng.normal(6.0, 1.6, (n, widest)), 1), 1.0, 10.0)
    scores[np.arange(widest)[None, :] >= interviewed[:, None]] = -1     # padding, not real people
    best_first = -np.sort(-scores, axis=1)
    interview_score = best_first[np.arange(n), offers_made - 1]

    # 7. Delays in days.  Referred candidates skip the queue; senior roles are slower;
    #    every extra offer means another round of negotiation.
    screen_days = np.ceil(lognormal(rng, 14, 0.4, n) * (1 + 0.20 * level)
                          * np.where(is_referred, 0.6, 1.0)).astype(int)
    offer_days = 2 + np.array([np.ceil(lognormal(rng, 5, 0.5, m)).sum() for m in offers_made]).astype(int)
    time_to_hire = screen_days + offer_days
    posting_delay = np.ceil(lognormal(rng, 8, 0.6, n)).astype(int)    # days before the hired person applied

    # 8. Dates, worked backwards from the hire date.
    hire_date = pd.Series(snapshot - pd.to_timedelta(rng.integers(0, WINDOW_DAYS + 1, n), unit="D"))
    application_date = hire_date - pd.to_timedelta(time_to_hire, unit="D")
    interview_date = application_date + pd.to_timedelta(screen_days, unit="D")
    opened_date = application_date - pd.to_timedelta(posting_delay, unit="D")

    df = pd.DataFrame({
        "department": department,
        "seniority": seniority,
        "is_remote": is_remote,
        "is_referred": is_referred,
        "salary_offered": salary_offered,
        "applicants": applicants,
        "interviewed": interviewed,
        "offers_made": offers_made.astype(int),
        "interview_score": interview_score,
        "application_date": application_date,
        "interview_date": interview_date,
        "hire_date": hire_date,
        "time_to_hire": time_to_hire,
        # hidden helpers
        "opened_date": opened_date,
        "market_salary": np.round(market).astype(int),
        "p_accept": np.round(p_accept, 3),
    })
    df = df.sort_values("hire_date").reset_index(drop=True)
    df.insert(0, "requisition_id", [f"R{i + 1:04d}" for i in range(n)])
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    op, ap, iv, hd = (pd.to_datetime(df[c]) for c in
                      ["opened_date", "application_date", "interview_date", "hire_date"])

    # ---- exact rules ----
    out.append(check("opened <= application <= interview <= hire <= snapshot",
                     ((op <= ap) & (ap <= iv) & (iv < hd) & (hd <= snapshot)).all()))
    out.append(check("time_to_hire = hire_date - application_date (days)",
                     ((hd - ap).dt.days == df.time_to_hire).all()))
    out.append(check("hire_date inside the last 24 months",
                     (hd >= snapshot - pd.Timedelta(days=WINDOW_DAYS)).all()))
    out.append(check("funnel: applicants >= interviewed >= offers_made >= 1",
                     ((df.applicants >= df.interviewed) & (df.interviewed >= df.offers_made)
                      & (df.offers_made >= 1)).all()))
    out.append(check("interview_score between 1 and 10", df.interview_score.between(1, 10).all()))
    out.append(check("salary_offered is a positive multiple of 500", ((df.salary_offered > 0)
                     & (df.salary_offered % 500 == 0)).all()))
    out.append(check("time_to_hire at least 3 days", (df.time_to_hire >= 3).all()))
    out.append(check("categories valid", df.department.isin(DEPARTMENTS).all()
                     and df.seniority.isin(LEVELS).all()))
    out.append(check("requisition_id unique", df.requisition_id.is_unique))
    out.append(check("salary within 50% of the market rate for the role",
                     (df.salary_offered / df.market_salary).between(0.5, 1.6).all()))

    # ---- relationships ----
    rank = df.seniority.map(LEVEL_RANK)
    c = float(np.log(df.salary_offered).corr(rank))
    out.append(check("more senior roles pay more (log salary vs level)",
                     c > 0.6, f"corr={c:.2f}", "relationship"))
    rem = df.applicants[df.is_remote]
    loc = df.applicants[~df.is_remote]
    if len(rem) >= 8 and len(loc) >= 8:
        out.append(check("remote roles attract more applicants (median)",
                         rem.median() > loc.median(), "", "relationship"))
    ref = df.time_to_hire[df.is_referred]
    non = df.time_to_hire[~df.is_referred]
    if len(ref) >= 8 and len(non) >= 8:
        out.append(check("referred hires are quicker (median time_to_hire)",
                         ref.median() < non.median(), "", "relationship"))
    c2 = float(rank.corr(df.time_to_hire, method="spearman"))
    out.append(check("more senior roles take longer to fill",
                     c2 > 0.1, f"rank corr={c2:.2f}", "relationship"))
    ratio = np.log(df.salary_offered / df.market_salary)
    c3 = float(ratio.corr(df.offers_made, method="spearman"))
    out.append(check("paying above market means fewer offers are needed",
                     c3 < -0.1, f"rank corr={c3:.2f}", "relationship"))
    c4 = float(df.offers_made.corr(df.interview_score, method="spearman"))
    out.append(check("when first choices decline, the hired candidate scored lower",
                     c4 < -0.05, f"rank corr={c4:.2f}", "relationship"))
    return out


RECIPE = Recipe(
    key="hr_recruitment",
    title="Recruitment & Hiring",
    row_meaning=("One row is one job vacancy that has been filled, showing the "
                 "hiring funnel behind it and the journey of the person hired."),
    simulate=simulate,
    columns=["requisition_id", "department", "seniority", "is_remote", "is_referred",
             "salary_offered", "applicants", "interviewed", "offers_made",
             "interview_score", "application_date", "interview_date", "hire_date",
             "time_to_hire"],
    core=["seniority", "salary_offered", "applicants", "time_to_hire"],
    priority=["department", "is_remote", "is_referred", "offers_made", "interview_score",
              "interviewed", "hire_date", "application_date", "interview_date",
              "requisition_id"],
    docs={
        "requisition_id": "Unique identifier of the filled vacancy.",
        "department": "Hiring department (engineering, sales, marketing, finance, operations, support).",
        "seniority": "Level of the role: junior, mid, senior or lead (ordered).",
        "is_remote": "True if the role could be done remotely; these attract many more applicants.",
        "is_referred": "True if the person hired was referred by an employee; referred candidates move faster.",
        "salary_offered": "Yearly salary in the accepted offer, whole units (nearest 500, no currency).",
        "applicants": "Number of applications received for the vacancy up to the hire.",
        "interviewed": "Number of candidates interviewed for the vacancy.",
        "offers_made": "Offers made until one was accepted (1 means the first choice accepted).",
        "interview_score": "The hired candidate's interview score, 1 to 10 (one decimal).",
        "application_date": "Date the hired candidate applied.",
        "interview_date": "Date the hired candidate was interviewed.",
        "hire_date": "Date the offer was accepted and the person was hired.",
        "time_to_hire": "Days from the hired candidate's application to hire_date.",
    },
    validate=validate,
    date_cols=["application_date", "interview_date", "hire_date"],
    targets={
        "time_to_hire": ["hire_date", "application_date", "interview_date"],
        "offers_made": ["hire_date", "time_to_hire"],
        "is_referred": [],
    },
)
