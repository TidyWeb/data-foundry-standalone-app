"""Customer Support Tickets.

One row is one support ticket that has been RESOLVED.  Tickets still open at
today's snapshot are left out, so every column is always defined (no blank
resolved_at).  All times are calendar time, in whole minutes.

The story, in the order the code tells it:
    when the ticket arrives -> customer plan -> category -> priority
    -> how complex the problem is -> escalated? -> first response
    -> handling and resolution -> SLA breach -> reopened? -> satisfaction

Complexity is hidden but drives escalations and handling time.  Priority sets
how fast agents respond and work.  Tickets arriving out of office hours wait
for the next working morning.  Satisfaction falls with slow resolution but
is noisy: customers are not perfectly rational.  Teaching values only.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli, logistic, ordinal

PLANS = ["starter", "growth", "business"]
PLAN_SEVERITY = {"starter": 0.0, "growth": 0.2, "business": 0.6}

# category: (share of tickets, typical hands-on work in minutes, seriousness)
CATEGORIES = {
    "billing":         (0.20, 30, 0.3),
    "how-to":          (0.25, 20, -0.6),
    "bug":             (0.25, 120, 0.5),
    "integration":     (0.12, 180, 0.2),
    "account-access":  (0.13, 15, 0.6),
    "feature-request": (0.05, 25, -1.0),
}
PRIORITIES = ["low", "normal", "high", "urgent"]
# priority: (typical minutes to first response, calendar stretch on hands-on work, SLA target in minutes)
PRIORITY_RULES = {
    "urgent": (12, 1.5, 480),        # SLA: resolved within 8 hours
    "high":   (30, 4.0, 2880),       # 2 days
    "normal": (75, 12.0, 4320),      # 3 days
    "low":    (150, 30.0, 10080),    # 7 days
}
HOUR_WEIGHTS = [1, 1, 1, 1, 1, 1, 2, 4, 8, 10, 10, 9, 7, 8, 9, 8, 7, 5, 3, 2, 2, 2, 1, 1]
WINDOW_DAYS = 120


def next_opening(created):
    """Minutes each ticket waits until the support desk opens (Mon-Fri, 09:00-17:00)."""
    day = created.dt.normalize()
    weekday = created.dt.dayofweek.to_numpy()
    hour = created.dt.hour.to_numpy()
    is_open = (weekday < 5) & (hour >= 9) & (hour < 17)
    add_days = np.where((weekday < 5) & (hour < 9), 0, 1)
    next_weekday = (weekday + add_days) % 7
    add_days = add_days + np.where(next_weekday == 5, 2, np.where(next_weekday == 6, 1, 0))
    opens_at = day + pd.to_timedelta(add_days, unit="D") + pd.Timedelta(hours=9)
    wait = (opens_at - created).dt.total_seconds().to_numpy() / 60
    return np.where(is_open, 0.0, wait)


def make_tickets(rng, m, snapshot):
    # 1. Arrival: busier on weekdays and during working hours.
    days_back = np.arange(1, WINDOW_DAYS + 1)
    day_dates = snapshot - pd.to_timedelta(days_back, unit="D")
    day_weight = np.where(day_dates.dayofweek >= 5, 0.3, 1.0)
    day_pick = rng.choice(WINDOW_DAYS, m, p=day_weight / day_weight.sum())
    hour = rng.choice(24, m, p=np.array(HOUR_WEIGHTS) / sum(HOUR_WEIGHTS))
    minute = rng.integers(0, 60, m)
    created = pd.Series(day_dates[day_pick]) + pd.to_timedelta(hour * 60 + minute, unit="m")

    # 2. Customer plan, category, then priority (serious categories and big customers rank higher).
    plan = pick(rng, PLANS, [0.5, 0.35, 0.15], m)
    category = pick(rng, list(CATEGORIES), [v[0] for v in CATEGORIES.values()], m)
    seriousness = (np.array([CATEGORIES[c][2] for c in category])
                   + np.array([PLAN_SEVERITY[p] for p in plan]) + rng.normal(0, 1.0, m))
    priority = ordinal(seriousness, [-0.2, 0.6, 1.3], PRIORITIES)

    # 3. Complexity (hidden) -> escalations.
    complexity = lognormal(rng, 1.0, 0.5, m)
    hard_category = np.isin(category, ["bug", "integration"])
    p_escalate = logistic(-2.3 + 1.0 * np.log(complexity) + 0.7 * hard_category
                          + 0.5 * (priority == "urgent"))
    escalations = bernoulli(rng, p_escalate).astype(int)
    escalations += escalations * bernoulli(rng, np.full(m, 0.25))    # a few are escalated twice

    # 4. First response: priority sets the speed; out-of-hours tickets wait for the desk to open
    #    (urgent tickets are covered by an on-call agent, so they do not wait).
    response_median = np.array([PRIORITY_RULES[p][0] for p in priority], dtype=float)
    rush = np.where(created.dt.hour.between(9, 11) | (created.dt.dayofweek == 0), 1.3, 1.0)
    wait_for_desk = np.where(priority == "urgent", 0.0, next_opening(created))
    response = np.maximum(1, np.round(lognormal(rng, response_median * rush, 0.6) + wait_for_desk))

    # 5. Hands-on work, then the calendar time it takes (waiting on the customer, queues, nights).
    work_minutes = np.array([CATEGORIES[c][1] for c in category]) * complexity
    stretch = np.array([PRIORITY_RULES[p][1] for p in priority])
    elapsed = work_minutes * stretch * (1 + 0.6 * escalations) * lognormal(rng, 1.0, 0.5, m)
    resolution = (response + np.maximum(5, np.round(elapsed))).astype(int)
    target = np.array([PRIORITY_RULES[p][2] for p in priority])
    breached = resolution > target

    # 6. Reopened (rare; more likely if hard, escalated or late), then satisfaction.
    p_reopen = logistic(-3.0 + 0.6 * np.log(complexity) + 0.4 * (escalations > 0) + 0.5 * breached)
    reopened = bernoulli(rng, p_reopen)
    latent = (4.6 - 0.25 * np.log2(resolution / 60) - 0.9 * breached - 0.3 * escalations
              - 0.7 * reopened + rng.normal(0, 0.8, m))
    satisfaction = np.clip(np.round(latent), 1, 5).astype(int)

    return pd.DataFrame({
        "customer_plan": plan, "category": category, "priority": priority,
        "created_at": created, "response_time": response.astype(int),
        "resolution_time": resolution, "escalations": escalations,
        "is_reopened": reopened, "sla_breached": breached, "satisfaction_score": satisfaction,
        "sla_target": target, "complexity": np.round(complexity, 3),      # hidden
    })


def simulate(rng, n, snapshot):
    m = 2 * n + 30
    while True:
        df = make_tickets(rng, m, snapshot)
        df["resolved_at"] = df.created_at + pd.to_timedelta(df.resolution_time, unit="m")
        df = df[df.resolved_at <= snapshot]           # only tickets already resolved today
        if len(df) >= n:
            break
        m *= 2
    df = df.sort_values("created_at").iloc[:n].reset_index(drop=True)
    df["first_response_at"] = df.created_at + pd.to_timedelta(df.response_time, unit="m")
    df["is_escalated"] = df.escalations > 0
    df.insert(0, "ticket_id", [f"T{i + 1:05d}" for i in range(len(df))])
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    c, f, r = (pd.to_datetime(df[x]) for x in ("created_at", "first_response_at", "resolved_at"))
    minutes = lambda a, b: ((b - a).dt.total_seconds() / 60).round().astype(int)

    # ---- exact rules ----
    out.append(check("created_at <= first_response_at <= resolved_at <= snapshot",
                     ((c <= f) & (f <= r) & (r <= snapshot)).all()))
    out.append(check("response_time = minutes from created_at to first_response_at",
                     (df.response_time == minutes(c, f)).all()))
    out.append(check("resolution_time = minutes from created_at to resolved_at",
                     (df.resolution_time == minutes(c, r)).all()))
    out.append(check("response_time >= 1 and resolution_time > response_time",
                     ((df.response_time >= 1) & (df.resolution_time > df.response_time)).all()))
    out.append(check("is_escalated <=> escalations > 0", (df.is_escalated == (df.escalations > 0)).all()))
    out.append(check("escalations between 0 and 2", df.escalations.between(0, 2).all()))
    limits = {p: v[2] for p, v in PRIORITY_RULES.items()}
    out.append(check("sla_breached <=> resolution_time > SLA target for the priority",
                     (df.sla_breached == (df.resolution_time > df.priority.map(limits))).all()))
    out.append(check("satisfaction_score is a whole number 1-5", df.satisfaction_score.between(1, 5).all()))
    out.append(check("categories valid",
                     df.category.isin(CATEGORIES).all() and df.priority.isin(PRIORITIES).all()
                     and df.customer_plan.isin(PLANS).all()))
    out.append(check("ticket_id unique and rows ordered by created_at",
                     df.ticket_id.is_unique and c.is_monotonic_increasing))

    # ---- relationships ----
    med = df.groupby("priority").resolution_time.median()
    counts = df.priority.value_counts()
    if counts.get("urgent", 0) >= 8 and counts.get("low", 0) >= 8:
        out.append(check("median resolution_time: urgent < low",
                         med["urgent"] < med["low"], f"{med['urgent']:.0f} vs {med['low']:.0f}", "relationship"))
    cat = df.groupby("category").resolution_time.median()
    counts = df.category.value_counts()
    if counts.get("bug", 0) >= 8 and counts.get("how-to", 0) >= 8:
        out.append(check("bugs take longer to resolve than how-to questions (median)",
                         cat["bug"] > cat["how-to"], "", "relationship"))
    rho = df.resolution_time.rank().corr(df.satisfaction_score.rank())
    out.append(check("slower resolution goes with lower satisfaction",
                     rho < -0.15, f"rank corr={rho:.2f}", "relationship"))
    hard = df.escalations > 0
    if hard.sum() >= 6:
        out.append(check("escalated tickets take longer (median resolution_time)",
                         df.resolution_time[hard].median() > df.resolution_time[~hard].median(),
                         "", "relationship"))
    normal = df[df.priority != "urgent"]
    out_of_hours = ~((c.dt.dayofweek < 5) & (c.dt.hour >= 9) & (c.dt.hour < 17))
    oh = out_of_hours[normal.index]
    if oh.sum() >= 8 and (~oh).sum() >= 8:
        out.append(check("non-urgent tickets created out of hours wait longer for a first response",
                         normal.response_time[oh].median() > normal.response_time[~oh].median(),
                         "", "relationship"))
    return out


RECIPE = Recipe(
    key="saas_support",
    title="Customer Support Tickets",
    row_meaning=("One row is one customer support ticket that has already been "
                 "resolved (tickets still open today are not included)."),
    simulate=simulate,
    columns=["ticket_id", "customer_plan", "category", "priority", "created_at",
             "first_response_at", "resolved_at", "response_time", "resolution_time",
             "escalations", "is_escalated", "is_reopened", "sla_breached",
             "satisfaction_score"],
    core=["category", "priority", "resolution_time", "satisfaction_score"],
    priority=["response_time", "sla_breached", "escalations", "customer_plan", "created_at",
              "is_reopened", "ticket_id", "is_escalated", "resolved_at", "first_response_at"],
    docs={
        "ticket_id": "Unique ticket identifier, numbered in the order tickets were created.",
        "customer_plan": "Plan of the customer who raised the ticket: starter, growth or business.",
        "category": "Type of problem: billing, how-to, bug, integration, account-access or feature-request.",
        "priority": "Urgency assigned to the ticket: low, normal, high or urgent (ordered).",
        "created_at": "Date and time the ticket was opened.",
        "first_response_at": "Date and time of the first reply from a support agent.",
        "resolved_at": "Date and time the ticket was marked resolved.",
        "response_time": "Minutes from created_at to first_response_at (calendar minutes, including nights and weekends).",
        "resolution_time": "Minutes from created_at to resolved_at (calendar minutes, including nights and weekends).",
        "escalations": "How many times the ticket was passed to a senior engineer (0, 1 or 2).",
        "is_escalated": "True if the ticket was escalated at least once.",
        "is_reopened": "True if the customer reopened the ticket after it was first resolved.",
        "sla_breached": "True if resolution_time was longer than the target for the priority (urgent 8 hours, high 2 days, normal 3 days, low 7 days).",
        "satisfaction_score": "Customer survey rating after resolution, 1 (very unhappy) to 5 (very happy).",
    },
    validate=validate,
    datetime_cols=["created_at", "first_response_at", "resolved_at"],
    targets={
        "sla_breached": ["resolution_time", "resolved_at", "satisfaction_score"],
        "satisfaction_score": ["is_reopened", "sla_breached"],
    },
)
