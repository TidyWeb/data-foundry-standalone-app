"""Loan & Credit Risk.

One row is one APPROVED loan, observed at the snapshot date.  Applications
that were declined are not in the table (so this is a "selection bias"
lesson: you only see the borrowers the lender said yes to).  Every column is
defined for every row.  Money is in plain whole numbers (no currency);
credit_score uses a FICTIONAL 300-850 scale.

The story, in the order the code tells it:
    applicants (income, credit score, existing debt, purpose, amount, term)
    -> approval scorecard (uses application data only; some noise)
    -> for approved loans: interest rate = base + risk premium (application data only)
    -> monthly repayment from the standard amortisation formula
    -> loan date and due date
    -> default risk from a logistic model, then WHEN a defaulter stops paying
    -> payments made so far, outstanding balance, overdue state at the snapshot

Definitions:
    debt_ratio       = existing monthly debt payments / monthly income (before this loan)
    interest_rate    = yearly rate in percent
    repayment_amount = the fixed monthly payment  P r / (1 - (1 + r)^-n), r = monthly rate, 2 dp
    is_default       = the borrower stopped paying (missed payments from some month on)
                       and that month has already happened by the snapshot.
                       Recent loans have had less time to default (censoring).
    is_overdue       = the loan is behind on payments at the snapshot
                       (every defaulted loan is overdue; a few others are just late).
    outstanding_balance = what is still owed at the snapshot: last month's balance
                       x (1 + monthly rate) - payment, for each month; months that
                       were not paid add interest but no payment.
All probabilities are teaching values, not real lending data.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli, logistic

# purpose: (weight, typical loan as a share of yearly income, allowed terms in months, rate premium)
PURPOSES = {
    "car":                (0.30, 0.30, [36, 48, 60], 0.0),
    "home_improvement":   (0.20, 0.25, [36, 60],     0.3),
    "debt_consolidation": (0.25, 0.30, [24, 36, 60], 0.8),
    "education":          (0.10, 0.35, [48, 60],     0.2),
    "personal":           (0.15, 0.12, [12, 24, 36], 1.5),
}
BASE_RATE = 4.5           # percent a year for a safe borrower


def draw_applicants(rng, m):
    """m loan applications, decided by a simple scorecard."""
    income = np.round(lognormal(rng, 42_000, 0.5, m) / 100) * 100
    score = 680 + 75 * rng.normal(0, 1, m) + 12 * np.log(income / 42_000)
    score = np.clip(np.round(score), 300, 850).astype(int)
    # People with better credit tend to carry less debt.
    debt_ratio = np.clip(rng.normal(0.28, 0.10, m) - 0.0012 * (score - 680), 0.02, 0.70)
    purpose = pick(rng, list(PURPOSES), [PURPOSES[p][0] for p in PURPOSES], m)
    typical = np.array([PURPOSES[p][1] for p in purpose])
    amount = np.round(income * typical * lognormal(rng, 1.0, 0.4, m) / 100) * 100
    amount = np.clip(amount, 500, 100_000).astype(int)
    term = np.array([rng.choice(PURPOSES[p][2]) for p in purpose])
    # Scorecard: good score helps; heavy existing debt and a big loan relative to income hurt.
    card = ((score - 620) / 40 - 6 * (debt_ratio - 0.3) - 3 * (amount / income - 0.3)
            + rng.normal(0, 0.6, m))
    return pd.DataFrame({"annual_income": income.astype(int), "credit_score": score,
                         "debt_ratio": np.round(debt_ratio, 2), "loan_purpose": purpose,
                         "loan_amount": amount, "term_months": term.astype(int),
                         "scorecard": np.round(card, 3)})


def months_passed(start, snapshot):
    """Whole months from start to the snapshot (a monthly payment day has passed)."""
    start = pd.to_datetime(start)
    months = (snapshot.year - start.dt.year) * 12 + (snapshot.month - start.dt.month)
    return (months - (snapshot.day < start.dt.day)).to_numpy()


def simulate(rng, n, snapshot):
    # 1. Applicants until enough are approved (about half are).
    batches, approved = [], 0
    while approved < n:
        batch = draw_applicants(rng, n + 60)
        batches.append(batch[batch.scorecard > 0])
        approved += len(batches[-1])
    df = pd.concat(batches).iloc[:n].reset_index(drop=True)
    score = df.credit_score.to_numpy()
    dti = df.debt_ratio.to_numpy()
    amount = df.loan_amount.to_numpy().astype(float)
    term = df.term_months.to_numpy()
    income = df.annual_income.to_numpy().astype(float)

    # 2. Interest rate from application data only: worse credit, more debt,
    #    longer term and riskier purpose all cost more.
    premium = np.array([PURPOSES[p][3] for p in df.loan_purpose])
    rate = (BASE_RATE + 0.018 * (750 - score) + 4.0 * dti + 0.012 * term + premium
            + rng.normal(0, 0.4, n))
    rate = np.round(np.clip(rate, 3.0, 28.0), 2)
    r = rate / 1200                                        # monthly rate

    # 3. Fixed monthly repayment (standard amortisation).
    payment = np.round(amount * r / (1 - (1 + r) ** (-term.astype(float))), 2)

    # 4. Dates: loans were taken out between 5 years and 45 days ago.
    days_ago = rng.integers(45, 5 * 365 + 1, n)
    loan_date = pd.Series(snapshot - pd.to_timedelta(days_ago, unit="D"))
    due_date = loan_date.copy()
    for t in np.unique(term):
        rows = term == t
        due_date[rows] = loan_date[rows] + pd.DateOffset(months=int(t))
    elapsed = np.minimum(months_passed(loan_date, snapshot), term)   # payments that have fallen due

    # 5. Default risk (logistic), driven by score, existing debt, and how heavy
    #    the payment is relative to income, plus unobserved life events.
    payment_to_income = payment / (income / 12)
    logit = (-2.4 - 0.012 * (score - 680) + 3.5 * (dti - 0.28)
             + 5.0 * (payment_to_income - 0.12) + rng.normal(0, 0.5, n))
    will_default = bernoulli(rng, logistic(logit))
    stop_month = np.maximum(1, np.ceil(term * rng.beta(2, 3, n))).astype(int)   # first unpaid month
    is_default = will_default & (stop_month <= elapsed)

    # 6. Late (not defaulted) borrowers missed the most recent payment.
    still_running = elapsed < term
    p_late = logistic(-3.2 - 0.008 * (score - 680) + 2.0 * (dti - 0.28))
    is_late = bernoulli(rng, p_late) & ~is_default & still_running & (elapsed >= 1)
    is_overdue = is_default | is_late

    # 7. Payments made and the balance at the snapshot.
    payments_made = np.where(is_default, stop_month - 1,
                             np.where(is_late, elapsed - 1, elapsed))
    growth = (1 + r) ** payments_made
    balance_after_paid = amount * growth - payment * (growth - 1) / r
    balance = balance_after_paid * (1 + r) ** (elapsed - payments_made)   # unpaid months add interest
    balance = np.where(payments_made >= term, 0.0, np.maximum(balance, 0.0))

    df["loan_date"] = loan_date
    df["due_date"] = pd.to_datetime(due_date)
    df["interest_rate"] = rate
    df["repayment_amount"] = payment
    df["outstanding_balance"] = np.round(balance, 2)
    df["is_overdue"] = is_overdue
    df["is_default"] = is_default
    # hidden helpers
    df["payments_due"] = elapsed.astype(int)
    df["payments_made"] = payments_made.astype(int)
    df["default_prob"] = np.round(logistic(logit), 4)
    df = df.sort_values("loan_date").reset_index(drop=True)
    df.insert(0, "loan_id", [f"L{i + 1:05d}" for i in range(n)])
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    r = df.interest_rate / 1200
    n = df.term_months
    ld, dd = pd.to_datetime(df.loan_date), pd.to_datetime(df.due_date)

    # ---- exact rules ----
    formula = df.loan_amount * r / (1 - (1 + r) ** (-n))
    out.append(check("repayment_amount = P r / (1 - (1+r)^-n) (2 dp)",
                     ((df.repayment_amount - formula).abs() <= 0.006).all()))
    expected_due = pd.Series([a + pd.DateOffset(months=int(t)) for a, t in zip(ld, n)])
    out.append(check("due_date = loan_date + term_months", (dd.reset_index(drop=True) == expected_due).all()))
    out.append(check("loan_date is before the snapshot", (ld < snapshot).all()))
    growth = (1 + r) ** df.payments_made
    paid_balance = df.loan_amount * growth - df.repayment_amount * (growth - 1) / r
    expected = paid_balance * (1 + r) ** (df.payments_due - df.payments_made)
    expected = expected.where(df.payments_made < n, 0.0).clip(lower=0)
    out.append(check("outstanding_balance follows the balance recurrence (+-0.01)",
                     ((df.outstanding_balance - expected).abs() <= 0.011).all()))
    out.append(check("0 <= payments_made <= payments_due <= term_months",
                     ((df.payments_made >= 0) & (df.payments_made <= df.payments_due)
                      & (df.payments_due <= n)).all()))
    out.append(check("paid-up loans owe nothing; unpaid loans still owe something",
                     ((df.outstanding_balance == 0) == (df.payments_made == n)).all()))
    out.append(check("is_default => is_overdue", (~df.is_default | df.is_overdue).all()))
    out.append(check("is_overdue = behind on payments (payments_made < payments_due)",
                     (df.is_overdue == (df.payments_made < df.payments_due)).all()))
    out.append(check("a loan that reached its due date and is not overdue is fully paid",
                     ((dd > snapshot) | df.is_overdue | (df.outstanding_balance == 0)).all()))
    out.append(check("credit_score integer in 300-850", df.credit_score.between(300, 850).all()))
    out.append(check("debt_ratio in [0.02, 0.70]", df.debt_ratio.between(0.02, 0.70).all()))
    out.append(check("interest_rate in [3, 28]", df.interest_rate.between(3, 28).all()))
    out.append(check("loan_amount >= 500, term is a whole number of months in 12-60",
                     (df.loan_amount >= 500).all() and n.between(12, 60).all()))
    out.append(check("every loan here was approved by the scorecard", (df.scorecard > 0).all()))
    out.append(check("loan_id unique", df.loan_id.is_unique))

    # ---- relationships ----
    c1 = float(df.credit_score.corr(df.interest_rate))
    out.append(check("better credit score -> lower interest rate",
                     c1 < -0.5, f"corr={c1:.2f}", "relationship"))
    c2 = float(np.log(df.annual_income).corr(np.log(df.loan_amount)))
    out.append(check("higher income -> larger loan",
                     c2 > 0.3, f"corr={c2:.2f}", "relationship"))
    c3 = float(df.debt_ratio.corr(df.interest_rate))
    out.append(check("more existing debt -> higher interest rate",
                     c3 > 0.15, f"corr={c3:.2f}", "relationship"))
    c4 = float(df.credit_score.corr(df.debt_ratio))
    out.append(check("better credit score goes with a lower debt ratio",
                     c4 < -0.15, f"corr={c4:.2f}", "relationship"))
    d = df[df.is_default]
    if len(d) >= 5 and (~df.is_default).sum() >= 20:
        out.append(check("defaulters have a lower mean credit score",
                         d.credit_score.mean() < df.credit_score[~df.is_default].mean(),
                         "", "relationship"))
        out.append(check("defaulters carry more existing debt (mean debt_ratio)",
                         d.debt_ratio.mean() > df.debt_ratio[~df.is_default].mean(),
                         "", "relationship"))
    return out


RECIPE = Recipe(
    key="finance_lending",
    title="Loan & Credit Risk",
    row_meaning=("One row is one approved loan, observed at the snapshot date; "
                 "declined applications are not included."),
    simulate=simulate,
    columns=["loan_id", "loan_purpose", "annual_income", "credit_score", "debt_ratio",
             "loan_amount", "term_months", "interest_rate", "repayment_amount",
             "loan_date", "due_date", "outstanding_balance", "is_overdue", "is_default"],
    core=["credit_score", "loan_amount", "interest_rate", "is_default"],
    priority=["debt_ratio", "annual_income", "term_months", "loan_purpose", "repayment_amount",
              "loan_date", "due_date", "is_overdue", "outstanding_balance", "loan_id"],
    docs={
        "loan_id": "Unique loan identifier.",
        "loan_purpose": "What the borrower said the loan was for (car, home_improvement, debt_consolidation, education, personal).",
        "annual_income": "Borrower's yearly income, whole units of money (no currency).",
        "credit_score": "Borrower's credit score on a fictional 300-850 scale (higher is safer).",
        "debt_ratio": "Existing monthly debt payments divided by monthly income, before this loan (e.g. 0.30 = 30%).",
        "loan_amount": "Amount borrowed (principal), whole units, multiples of 100.",
        "term_months": "Length of the loan in months.",
        "interest_rate": "Yearly interest rate in percent, set from the borrower's application data only.",
        "repayment_amount": "Fixed monthly payment (2 dp) that repays the loan and interest over term_months.",
        "loan_date": "Date the loan was paid out.",
        "due_date": "Date the final payment is due = loan_date + term_months (can be in the future).",
        "outstanding_balance": "Amount still owed at the snapshot date (2 dp); 0 if fully repaid. Only known after the loan starts.",
        "is_overdue": "True if the loan is behind on payments at the snapshot; every defaulted loan is overdue.",
        "is_default": "True if the borrower stopped paying and that point has already passed by the snapshot.",
    },
    validate=validate,
    date_cols=["loan_date", "due_date"],
    targets={
        "is_default": ["is_overdue", "outstanding_balance"],
        "is_overdue": ["is_default", "outstanding_balance"],
        "interest_rate": ["repayment_amount"],
    },
)
