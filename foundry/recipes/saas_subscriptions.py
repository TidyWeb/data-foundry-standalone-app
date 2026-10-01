"""Customer Churn & Subscriptions.

One row is one customer-month: a single billing month of one subscription.
A customer contributes one row per month until they cancel (their last row)
or until the snapshot date (still subscribed).  Every visible column is
always defined, so there is no blank cancellation_date; you can find each
customer's last month with is_churned or by looking at their final row.

The story, in the order the code tells it:
    customer -> plan, seats, billing term -> trial or not
    -> each month: engagement drifts -> support tickets -> seat changes
    -> billing (MRR) -> renewal date -> did they cancel this month?

Engagement is an evolving state (it drifts month to month, it is not redrawn).
Low engagement and many support tickets raise the chance of cancelling;
long-standing customers and bigger plans are stickier.  Cohort measures such
as churn rate and retention are NOT stored per row: students calculate them
with groupby.  All numbers are illustrative teaching values.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, pick, bernoulli, logistic, lognormal

# plan: (price per seat per month if billed monthly, if billed annually,
#        typical starting seats, churn adjustment on the log-odds scale,
#        typical engagement boost)
PLANS = {
    "starter":  (12, 10, 4,  0.4, 0),
    "growth":   (25, 21, 12, 0.0, 5),
    "business": (45, 38, 35, -0.6, 10),
}
PLAN_NAMES = list(PLANS)
MONTHS_BACK = 20          # customers signed up within the last 20 months
TRIAL_SHARE = 0.55        # share of monthly customers who start with a free trial month


def month_date(month_index, day):
    """Turn (year*12 + month-1, day-of-month) arrays into dates."""
    return pd.to_datetime(pd.DataFrame({"year": month_index // 12,
                                        "month": month_index % 12 + 1,
                                        "day": day}))


def make_customers(rng, n_cust, snapshot):
    """Step 1: who the customers are and when they signed up."""
    plan = pick(rng, PLAN_NAMES, [0.5, 0.35, 0.15], n_cust)
    typical_seats = np.array([PLANS[p][2] for p in plan], dtype=float)
    seats = np.maximum(1, np.round(lognormal(rng, typical_seats, 0.5))).astype(int)

    # Bigger plans are more often on annual contracts (a 12-month term).
    p_annual = np.where(plan == "business", 0.35, np.where(plan == "growth", 0.20, 0.08))
    annual = bernoulli(rng, p_annual)
    # Only monthly customers get a free trial month first.
    has_trial = (~annual) & bernoulli(rng, np.full(n_cust, TRIAL_SHARE))

    # Signup date: a random month in the last 20 months, on a day 1-28
    # (never later than today for customers who signed up this month).
    snap_index = snapshot.year * 12 + snapshot.month - 1
    months_ago = rng.integers(0, MONTHS_BACK, n_cust)
    day = rng.integers(1, 29, n_cust)
    day = np.where(months_ago == 0, np.minimum(day, snapshot.day), day)

    # Each customer's typical engagement (0-100): bigger plans engage more.
    boost = np.array([PLANS[p][4] for p in plan])
    typical_engagement = 52 + boost + rng.normal(0, 12, n_cust)
    return dict(plan=plan, seats=seats, annual=annual, has_trial=has_trial,
                signup_index=snap_index - months_ago, day=day, months_ago=months_ago,
                typical_engagement=typical_engagement)


def simulate_months(rng, c):
    """Step 2: walk forward one month at a time for every customer at once."""
    n_cust = len(c["plan"])
    alive = np.ones(n_cust, dtype=bool)
    engagement = c["typical_engagement"] + rng.normal(0, 8, n_cust)
    seats = c["seats"].copy()
    term = np.where(c["annual"], 12, 1)
    plan_effect = np.array([PLANS[p][3] for p in c["plan"]])
    monthly_price = np.array([PLANS[p][1] if a else PLANS[p][0]
                              for p, a in zip(c["plan"], c["annual"])])
    pieces = []

    for k in range(MONTHS_BACK):
        here = alive & (k <= c["months_ago"])       # still subscribed and month k has happened
        if not here.any():
            break
        if k > 0:
            # Engagement drifts: pulled back toward the customer's typical level, plus a shock.
            engagement = 0.8 * engagement + 0.2 * c["typical_engagement"] + rng.normal(0, 7, n_cust)
            # Happy customers add seats; unhappy ones sometimes remove them.
            grow = bernoulli(rng, np.where(engagement > 65, 0.10, 0.03))
            shrink = bernoulli(rng, np.where(engagement < 40, 0.07, 0.01))
            step = np.maximum(1, np.round(0.15 * seats)).astype(int)
            seats = np.maximum(1, seats + step * grow - step * (shrink & ~grow))
        score = np.clip(np.round(engagement), 0, 100).astype(int)

        # Support tickets: more seats means more tickets; disengaged users hit more problems.
        ticket_rate = 0.15 + 0.06 * np.sqrt(seats) + 1.5 * np.maximum(0, (50 - score) / 50)
        tickets = rng.poisson(ticket_rate)

        is_trial = c["has_trial"] & (k == 0)
        z = (score - 55) / 15                       # engagement in "standard deviations"

        # Trial months: convert to paying with a chance that rises with engagement.
        p_cancel_trial = 1 - logistic(0.6 + 0.9 * z)
        # Paying monthly customers: hazard falls with tenure, engagement, bigger plans.
        logit = (-3.3 + plan_effect - 0.8 * z + 0.3 * np.minimum(tickets, 5)
                 + 0.9 * np.exp(-k / 3.0))
        p_cancel_monthly = logistic(logit)
        # Annual customers can only cancel at renewal (the last month of each 12).
        p_cancel_annual = logistic(-1.9 + plan_effect - 0.8 * z + 0.3 * np.minimum(tickets, 5))
        can_cancel_annual = (k + 1) % 12 == 0

        p_cancel = np.where(is_trial, p_cancel_trial, p_cancel_monthly)
        p_cancel = np.where(c["annual"], p_cancel_annual * can_cancel_annual, p_cancel)
        churned = here & bernoulli(rng, p_cancel)

        pieces.append(pd.DataFrame({
            "cust": np.flatnonzero(here), "tenure_months": k,
            "engagement_score": score[here], "support_tickets": tickets[here],
            "seats": seats[here], "is_trial": is_trial[here],
            "price_per_seat": monthly_price[here], "is_churned": churned[here],
            "next_renewal_months": ((k // term[here]) + 1) * term[here],
        }))
        alive &= ~churned
    return pd.concat(pieces, ignore_index=True)


def simulate(rng, n, snapshot):
    n_cust = n // 4 + 10
    while True:
        c = make_customers(rng, n_cust, snapshot)
        rows = simulate_months(rng, c)
        if len(rows) >= n:
            break
        n_cust *= 2

    cust = rows["cust"].to_numpy()
    idx = c["signup_index"][cust]
    day = c["day"][cust]
    # Money: a trial month is free; otherwise seats x price per seat, whole units.
    mrr = np.where(rows.is_trial, 0, rows.seats * rows.price_per_seat).astype(int)
    df = pd.DataFrame({
        "customer_id": [f"C{i + 1:04d}" for i in cust],
        "billing_date": month_date(idx + rows.tenure_months.to_numpy(), day),
        "signup_date": month_date(idx, day),
        "plan": c["plan"][cust],
        "billing_term": np.where(c["annual"][cust], "annual", "monthly"),
        "seats": rows.seats.to_numpy(),
        "price_per_seat": rows.price_per_seat.to_numpy(),
        "tenure_months": rows.tenure_months.to_numpy(),
        "engagement_score": rows.engagement_score.to_numpy(),
        "support_tickets": rows.support_tickets.to_numpy(),
        "is_trial": rows.is_trial.to_numpy(),
        "monthly_recurring_revenue": mrr,
        "renewal_date": month_date(idx + rows.next_renewal_months.to_numpy(), day),
        "is_churned": rows.is_churned.to_numpy(),
    })
    df = df.sort_values(["customer_id", "billing_date"]).reset_index(drop=True)
    return df.iloc[:n].reset_index(drop=True)      # trim so every identity holds on visible rows


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    bd, sd, rd = (pd.to_datetime(df[c]) for c in ("billing_date", "signup_date", "renewal_date"))
    term = np.where(df.billing_term == "annual", 12, 1)

    # ---- exact rules ----
    out.append(check("MRR = seats x price_per_seat, or 0 in a trial month",
                     (df.monthly_recurring_revenue == np.where(df.is_trial, 0, df.seats * df.price_per_seat)).all()))
    months_since_signup = (bd.dt.year - sd.dt.year) * 12 + (bd.dt.month - sd.dt.month)
    out.append(check("tenure_months = whole months from signup_date to billing_date",
                     (df.tenure_months == months_since_signup).all()))
    out.append(check("billing_date is on the signup day and never after the snapshot",
                     ((bd.dt.day == sd.dt.day) & (bd <= snapshot)).all()))
    renew_gap = (rd.dt.year - bd.dt.year) * 12 + (rd.dt.month - bd.dt.month)
    out.append(check("renewal_date is the next signup-anniversary multiple of the term",
                     ((renew_gap > 0) & (renew_gap <= term)
                      & ((months_since_signup + renew_gap) % term == 0)).all()))
    out.append(check("trial only in tenure month 0 on monthly terms",
                     (~df.is_trial | ((df.tenure_months == 0) & (df.billing_term == "monthly"))).all()))
    out.append(check("annual customers only churn in their renewal month",
                     (~(df.is_churned & (df.billing_term == "annual"))
                      | ((df.tenure_months + 1) % 12 == 0)).all()))
    last_row = ~df.duplicated("customer_id", keep="last")
    out.append(check("a churned row is the customer's last row (no rows after churn)",
                     (~df.is_churned | last_row).all()))
    out.append(check("rows for a customer are consecutive months from 0",
                     (df.groupby("customer_id").tenure_months.apply(lambda s: (s.diff().dropna() == 1).all()
                                                                  and s.iloc[0] == 0)).all()))
    out.append(check("customer keeps one plan, term and signup_date",
                     (df.groupby("customer_id")[["plan", "billing_term", "signup_date"]].nunique() == 1).all().all()))
    out.append(check("engagement 0-100, seats >= 1, tickets >= 0",
                     df.engagement_score.between(0, 100).all() and (df.seats >= 1).all()
                     and (df.support_tickets >= 0).all()))

    # ---- relationships ----
    gone = df.is_churned
    if gone.sum() >= 6:
        gap = df.engagement_score[~gone].mean() - df.engagement_score[gone].mean()
        out.append(check("churned months had lower engagement than retained months",
                         gap >= 3, f"gap={gap:.1f}", "relationship"))
    lag = df.groupby("customer_id").engagement_score.shift(1)
    both = lag.notna()
    if both.sum() >= 30:
        r = df.engagement_score[both].corr(lag[both])
        out.append(check("engagement is persistent from month to month",
                         r > 0.4, f"lag-1 corr={r:.2f}", "relationship"))
    starter = df.seats[df.plan == "starter"]
    business = df.seats[df.plan == "business"]
    if len(starter) >= 8 and len(business) >= 8:
        out.append(check("median seats: business > starter",
                         business.median() > starter.median(), "", "relationship"))
    low, high = df[df.engagement_score < 45], df[df.engagement_score >= 60]
    if len(low) >= 15 and len(high) >= 15:
        out.append(check("low-engagement months have more support tickets than high-engagement ones",
                         low.support_tickets.mean() > high.support_tickets.mean(),
                         f"{low.support_tickets.mean():.2f} vs {high.support_tickets.mean():.2f}",
                         "relationship"))
    return out


RECIPE = Recipe(
    key="saas_subscriptions",
    title="Customer Churn & Subscriptions",
    row_meaning=("One row is one customer-month: one billing month of one software "
                 "subscription, from signup until the customer cancels or today."),
    simulate=simulate,
    columns=["customer_id", "billing_date", "signup_date", "tenure_months", "plan",
             "billing_term", "seats", "price_per_seat", "is_trial",
             "monthly_recurring_revenue", "renewal_date", "engagement_score",
             "support_tickets", "is_churned"],
    core=["plan", "seats", "engagement_score", "monthly_recurring_revenue", "is_churned"],
    priority=["support_tickets", "customer_id", "billing_date", "tenure_months", "is_trial",
              "billing_term", "price_per_seat", "signup_date", "renewal_date"],
    docs={
        "customer_id": "Unique customer identifier; a customer has one row per billing month.",
        "billing_date": "Date this month's bill was raised (the signup day-of-month, every month).",
        "signup_date": "Date the customer first signed up.",
        "tenure_months": "Whole months since signup_date (0 = the first month).",
        "plan": "Subscription plan: starter, growth or business.",
        "billing_term": "monthly, or annual (a 12-month contract that can only be cancelled at renewal).",
        "seats": "Number of paid user seats this month.",
        "price_per_seat": "Price of one seat per month on this customer's plan and term, in plain currency units.",
        "is_trial": "True for a free trial month (revenue is 0 in that month).",
        "monthly_recurring_revenue": "Revenue billed for the month: seats x price_per_seat (0 during a trial), in plain currency units.",
        "renewal_date": "Next date the contract renews (next month for monthly, next anniversary for annual).",
        "engagement_score": "How actively the customer used the product this month, 0 (dormant) to 100 (very active).",
        "support_tickets": "Number of support tickets the customer opened this month.",
        "is_churned": "True if the customer cancelled at the end of this month; it is their last row.",
    },
    validate=validate,
    date_cols=["billing_date", "signup_date", "renewal_date"],
    targets={
        "is_churned": [],
        "monthly_recurring_revenue": ["price_per_seat", "is_trial"],
    },
)
