"""Product Usage Analytics.

One row is one user-day: a day on which one user opened the product at least
once, summarised from the sessions they had that day.  Days with no session
have no row (so no blank cells).  Users are followed from their activation
date up to today (at most 30 days).

The story, in the order the code tells it:
    accounts (seat_count) -> users in those accounts -> each user's habits
    (how often they visit, how long they stay, how many features they use)
    -> each day: does the user show up? (weekday, mood of the week, fading)
    -> sessions -> session lengths, API calls, features -> day totals
    -> usage_score, power-user and API-limit flags

Sessions are simulated one by one and then added up, so the day totals are
exactly the sum of their sessions.  Session lengths are heavy-tailed: most
sessions are short, a few last hours.  All numbers are illustrative.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, bernoulli

WINDOW_DAYS = 30          # users activated within the last 30 days
N_FEATURES = 12           # the product has 12 features
API_LIMIT = 250           # calls per user per day before the rate limit is hit
WEEKDAY_FACTOR = np.array([1.0, 1.1, 1.1, 1.05, 0.9, 0.25, 0.2])   # Monday ... Sunday


def make_users(rng, n_users, snapshot):
    """Steps 1-2: accounts, then users with persistent habits."""
    n_acc = max(2, n_users // 5)
    seat_count = np.maximum(2, np.round(lognormal(rng, 12, 0.9, n_acc))).astype(int)
    account = rng.choice(n_acc, n_users, p=seat_count / seat_count.sum())   # big accounts have more users

    age = rng.integers(4, WINDOW_DAYS, n_users)              # days since activation
    feature_popularity = np.sort(rng.uniform(0.2, 1.0, N_FEATURES))[::-1]   # some features are used by everyone
    breadth = rng.beta(2, 4, n_users)                        # how many features this user likes to try
    return dict(
        account=account, seat_count=seat_count[account], age=age,
        activation_date=snapshot - pd.to_timedelta(age, unit="D"),
        activity_rate=rng.beta(2.5, 3.5, n_users),           # chance of showing up on a normal weekday
        sessions_per_day=lognormal(rng, 1.4, 0.5, n_users),
        median_session_sec=lognormal(rng, 420, 0.5, n_users),
        # Bigger accounts wire the product into other systems: more API calls per session.
        api_per_session=lognormal(rng, 8, 0.5, n_users) * (1 + 0.6 * np.log(seat_count[account])),
        fading=bernoulli(rng, np.full(n_users, 0.25)),       # some users drift away over the month
        feature_prob=np.clip(breadth[:, None] * feature_popularity[None, :] * 1.6, 0.02, 0.95),
        usual_hour=rng.normal(13, 2.5, n_users),
    )


def simulate_days(rng, u, snapshot):
    """Steps 3-4: decide which days each user shows up, then simulate the sessions."""
    n_users = len(u["age"])
    days = np.arange(WINDOW_DAYS)
    start = pd.DatetimeIndex(u["activation_date"]).to_numpy().astype("datetime64[D]")
    dates = start[:, None] + days.astype("timedelta64[D]")[None, :]
    weekday = (dates.astype(int) + 3) % 7                    # 1970-01-01 was a Thursday; Monday = 0
    still_open = days[None, :] <= u["age"][:, None]          # only days from activation to today

    # A slowly changing "mood": a mean-reverting random walk, plus steady decay for fading users.
    mood = np.zeros((n_users, WINDOW_DAYS))
    level = rng.normal(0, 0.2, n_users)
    for d in range(WINDOW_DAYS):
        level = 0.85 * level + rng.normal(0, 0.15, n_users)
        mood[:, d] = level - 0.04 * d * u["fading"]
    chance = np.clip(u["activity_rate"][:, None] * WEEKDAY_FACTOR[weekday] * np.exp(mood), 0.01, 0.97)
    active = bernoulli(rng, chance) & still_open
    user_of_row, day_of_row = np.nonzero(active)             # sorted by user, then date
    n_rows = len(user_of_row)

    # Sessions: at least one on an active day; heavier days have more.
    mult = np.exp(mood[user_of_row, day_of_row])
    sessions = 1 + rng.poisson(u["sessions_per_day"][user_of_row] * 0.6 * mult)
    row_of_session = np.repeat(np.arange(n_rows), sessions.astype(np.intp))
    user_of_session = user_of_row[row_of_session]

    # Each session: duration (heavy-tailed seconds), API calls (overdispersed), start time.
    seconds = np.clip(np.round(lognormal(rng, u["median_session_sec"][user_of_session], 0.9)), 10, 7200)
    call_mean = u["api_per_session"][user_of_session]
    calls = rng.poisson(rng.gamma(0.8, call_mean / 0.8))
    start_minute = np.clip(rng.normal(u["usual_hour"][user_of_session] * 60, 150), 360, 1290)

    # Add the sessions up into one row per user-day.
    session_length = np.bincount(row_of_session, weights=seconds, minlength=n_rows).astype(int)
    api_calls = np.bincount(row_of_session, weights=calls, minlength=n_rows).astype(int)
    last_start = pd.Series(start_minute).groupby(row_of_session).max().round().to_numpy()

    # Features: a user tries each feature with their own propensity; more sessions, more chances.
    p_used = 1 - (1 - u["feature_prob"][user_of_row]) ** sessions[:, None]
    features_used = bernoulli(rng, p_used).sum(axis=1)

    activity_date = pd.DatetimeIndex(dates[user_of_row, day_of_row].astype("datetime64[ns]"))
    return pd.DataFrame({
        "user_num": user_of_row, "activity_date": activity_date, "sessions": sessions,
        "session_length": session_length, "api_calls": api_calls,
        "features_used": features_used,
        "last_login": activity_date + pd.to_timedelta(last_start, unit="m"),
    })


def simulate(rng, n, snapshot):
    n_users = n // 4 + 10
    while True:
        u = make_users(rng, n_users, snapshot)
        rows = simulate_days(rng, u, snapshot)
        if len(rows) >= n:
            break
        n_users *= 2

    who = rows.user_num.to_numpy()
    df = pd.DataFrame({
        "user_id": [f"U{i + 1:04d}" for i in who],
        "account_id": [f"A{i + 1:03d}" for i in u["account"][who]],
        "seat_count": u["seat_count"][who],
        "activation_date": u["activation_date"][who],
        "activity_date": rows.activity_date.to_numpy(),
        "sessions": rows.sessions.to_numpy(),
        "session_length": rows.session_length.to_numpy(),
        "api_calls": rows.api_calls.to_numpy(),
        "features_used": rows.features_used.to_numpy(),
        "last_login": rows.last_login.to_numpy(),
        "activity_rate": np.round(u["activity_rate"][who], 3),      # hidden
    })
    df = df.iloc[:n].reset_index(drop=True)          # trim first: the flag below depends on earlier rows

    # Calculated columns (each has a stated rule).
    df["feature_adoption"] = np.round(df.features_used / N_FEATURES, 2)
    minutes_part = np.minimum(df.session_length / 3600, 1)
    sessions_part = np.minimum(df.sessions / 6, 1)
    api_part = np.minimum(df.api_calls / 200, 1)
    df["usage_score"] = np.round(100 * (0.35 * minutes_part + 0.25 * sessions_part
                                        + 0.25 * df.feature_adoption + 0.15 * api_part)).astype(int)
    df["is_over_api_limit"] = df.api_calls > API_LIMIT

    # Power user (as of this day): active on at least 60% of the days since activation, 5+ active days.
    active_days = df.groupby("user_id").cumcount() + 1
    days_open = (df.activity_date - df.activation_date).dt.days + 1
    df["is_power_user"] = (active_days >= 5) & (active_days / days_open >= 0.6)
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    ad, act = pd.to_datetime(df.activity_date), pd.to_datetime(df.activation_date)

    # ---- exact rules ----
    out.append(check("activation_date <= activity_date <= snapshot", ((act <= ad) & (ad <= snapshot)).all()))
    out.append(check("last_login is on activity_date", (pd.to_datetime(df.last_login).dt.normalize() == ad).all()))
    out.append(check("feature_adoption = features_used / 12 (2 dp)",
                     (df.feature_adoption == np.round(df.features_used / N_FEATURES, 2)).all()))
    out.append(check("features_used between 0 and 12", df.features_used.between(0, N_FEATURES).all()))
    out.append(check("an active day has >= 1 session and >= 10 seconds",
                     ((df.sessions >= 1) & (df.session_length >= 10 * df.sessions)).all()))
    out.append(check("each session is at most 2 hours", (df.session_length <= 7200 * df.sessions).all()))
    minutes_part = np.minimum(df.session_length / 3600, 1)
    expected = np.round(100 * (0.35 * minutes_part + 0.25 * np.minimum(df.sessions / 6, 1)
                               + 0.25 * df.feature_adoption + 0.15 * np.minimum(df.api_calls / 200, 1)))
    out.append(check("usage_score = weighted index (0-100)",
                     ((df.usage_score == expected) & df.usage_score.between(0, 100)).all()))
    out.append(check("is_over_api_limit = api_calls > 250", (df.is_over_api_limit == (df.api_calls > API_LIMIT)).all()))
    out.append(check("one row per user per day", not df.duplicated(["user_id", "activity_date"]).any()))
    out.append(check("a user keeps one account, seat_count and activation_date",
                     (df.groupby("user_id")[["account_id", "seat_count", "activation_date"]].nunique() == 1).all().all()))
    active_days = df.groupby("user_id").cumcount() + 1
    days_open = (ad - act).dt.days + 1
    out.append(check("is_power_user = >=5 active days and >=60% of days since activation",
                     (df.is_power_user == ((active_days >= 5) & (active_days / days_open >= 0.6))).all()))

    # ---- relationships ----
    if len(df) >= 60:
        weekend = ad.dt.dayofweek >= 5
        out.append(check("weekends are quiet: under 20% of user-days",
                         weekend.mean() < 0.20, f"share={weekend.mean():.2f}", "relationship"))
    out.append(check("session_length is right-skewed (mean > 1.15 x median)",
                     df.session_length.mean() > 1.15 * df.session_length.median(),
                     f"{df.session_length.mean() / df.session_length.median():.2f}", "relationship"))
    rho = df.seat_count.rank().corr(df.api_calls.rank())
    out.append(check("bigger accounts make more API calls", rho > 0.03, f"rank corr={rho:.2f}", "relationship"))
    per_user = df.groupby("user_id").agg(rate=("activity_rate", "first"), days=("activity_date", "count"),
                                         first=("activation_date", "first"), last=("activity_date", "max"))
    per_user["share"] = per_user.days / ((per_user["last"] - per_user["first"]).dt.days + 1)
    if len(per_user) >= 8:
        r = per_user.rate.corr(per_user.share)
        out.append(check("users with a higher activity habit are active on more of their days",
                         r > 0.3, f"corr={r:.2f}", "relationship"))
    r2 = df.usage_score.corr(df.sessions)
    out.append(check("more sessions, higher usage_score", r2 > 0.3, f"corr={r2:.2f}", "relationship"))
    return out


RECIPE = Recipe(
    key="saas_usage",
    title="Product Usage Analytics",
    row_meaning=("One row is one user-day: a day on which one user opened the product "
                 "at least once, summarised from that day's sessions."),
    simulate=simulate,
    columns=["user_id", "account_id", "seat_count", "activation_date", "activity_date",
             "sessions", "session_length", "api_calls", "features_used", "feature_adoption",
             "usage_score", "last_login", "is_power_user", "is_over_api_limit"],
    core=["seat_count", "sessions", "session_length", "api_calls", "usage_score"],
    priority=["feature_adoption", "activity_date", "user_id", "is_power_user", "features_used",
              "account_id", "is_over_api_limit", "last_login", "activation_date"],
    docs={
        "user_id": "Unique user identifier; a user has one row for each day they were active.",
        "account_id": "The customer account (company) the user belongs to.",
        "seat_count": "Number of paid seats the user's account has (account size).",
        "activation_date": "Date the user was activated (their first possible day of use).",
        "activity_date": "The calendar day this row summarises.",
        "sessions": "Number of sessions (visits) the user had that day.",
        "session_length": "Total time in the product that day, in seconds, summed over all sessions.",
        "api_calls": "Total API calls made that day, summed over all sessions.",
        "features_used": "How many of the product's 12 features the user used that day.",
        "feature_adoption": "features_used divided by 12, rounded to 2 decimals (0 to 1).",
        "usage_score": "0-100 index: 35% time in product (capped at 1 hour), 25% sessions (capped at 6), 25% feature_adoption, 15% API calls (capped at 200).",
        "last_login": "Date and time of the user's last session start that day.",
        "is_power_user": "True if, up to this day, the user was active on at least 60% of days since activation (with 5 or more active days).",
        "is_over_api_limit": "True if the day's api_calls went over the limit of 250 calls per user per day.",
    },
    validate=validate,
    date_cols=["activation_date", "activity_date"],
    datetime_cols=["last_login"],
    targets={
        "is_power_user": ["sessions", "usage_score"],
        "is_over_api_limit": ["api_calls"],
    },
)
