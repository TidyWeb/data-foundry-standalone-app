"""Investment Portfolio Returns.

One row is one holding (one asset inside one portfolio) at the end of one
month.  Each portfolio is followed for 9 consecutive months ending at the
latest month end on or before the snapshot.  Rows are sorted portfolio, then
month, then asset, so a portfolio-month is a block of consecutive rows.

Units (all fictional, plain numbers, no currency):
    monthly_return = the holding's total return over the month, in percent
                     (price change plus income; NOT annualised)
    risk           = annualised volatility in percent: the standard deviation
                     of the asset's PREVIOUS 12 monthly returns x sqrt(12).
                     It only uses months before the row's month (no look-ahead).
    allocation     = the holding's share of the portfolio at month end, percent
    valuation      = market value of the holding at month end
    portfolio_value= market value of the whole portfolio at month end
                     (= the sum of its holdings' valuations)
    dividend_yield = the asset's yearly income as a percent of its price

The story, in the order the code tells it:
    one market return series shared by everybody
    -> a universe of 40 assets, each with a class, a beta (sensitivity to the
       market), an alpha (its own average edge) and its own noise
    -> each asset's monthly returns:  alpha + beta x market + class factor + noise
       (an asset has ONE return history; every portfolio holding it sees it)
    -> portfolios pick holdings and starting weights
    -> monthly loop: values grow with returns, weights drift; active portfolios
       rebalance back to their target weights every quarter
    -> flags: diversified? flagged for review?

Real markets are close to unpredictable month to month, so past returns are a
poor guide to next month's return.  All numbers are teaching values.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli, random_dates

MONTHS = 9              # visible months per portfolio
HISTORY = 12            # earlier months used only to measure risk
MARKET_MEAN, MARKET_SD = 0.009, 0.038       # monthly market return

# class: (assets in universe, beta range, alpha sd, own noise sd, yield range %, class-factor sd)
CLASSES = {
    "equity":    (20, (0.7, 1.5), 0.002, 0.050, (0.5, 3.5), 0.015),
    "bond":      (8,  (0.0, 0.3), 0.001, 0.012, (2.5, 5.0), 0.010),
    "property":  (6,  (0.5, 1.0), 0.002, 0.030, (3.0, 5.5), 0.015),
    "commodity": (6,  (0.1, 0.5), 0.002, 0.050, (0.0, 0.0), 0.030),
}
# how much of each class a portfolio style holds: equity, bond, property, commodity
STYLES = {"aggressive": [0.70, 0.10, 0.10, 0.10],
          "balanced":   [0.45, 0.30, 0.15, 0.10],
          "cautious":   [0.20, 0.55, 0.15, 0.10]}


def simulate(rng, n, snapshot):
    months_total = HISTORY + MONTHS
    class_names = list(CLASSES)

    # 1. One market series for the whole world (fat-tailed: occasional bad months).
    market = MARKET_MEAN + MARKET_SD * rng.standard_t(5, months_total) * np.sqrt(3 / 5)
    class_factor = {c: rng.normal(0, CLASSES[c][5], months_total) for c in class_names}

    # 2. The asset universe and each asset's return history.
    asset_class, beta, dividend_yield = [], [], []
    returns = []
    for c in class_names:
        count, beta_range, alpha_sd, noise_sd, yield_range, _ = CLASSES[c]
        for _ in range(count):
            b = rng.uniform(*beta_range)
            alpha = rng.normal(0.001, alpha_sd)
            noise = noise_sd * rng.uniform(0.7, 1.3)
            returns.append(alpha + b * market + class_factor[c] + rng.normal(0, noise, months_total))
            asset_class.append(c)
            beta.append(b)
            dividend_yield.append(round(rng.uniform(*yield_range), 2))
    returns = np.clip(np.array(returns), -0.6, 0.6)          # (assets, months)
    n_assets = len(asset_class)
    asset_class = np.array(asset_class, dtype=object)

    # Risk for visible month t = st.dev. of the 12 returns BEFORE it, annualised.
    risk = np.zeros((n_assets, MONTHS))
    for t in range(MONTHS):
        window = returns[:, t:t + HISTORY]
        risk[:, t] = window.std(axis=1, ddof=1) * np.sqrt(12) * 100

    # 3. Portfolios: style, number of holdings, holdings, starting weights.
    k_options = [4, 5, 6, 7, 8, 9, 10]
    n_portfolios = -(-n // (MONTHS * 4)) + 1          # each has at least 4 holdings
    while True:
        k = pick(rng, k_options, [1, 1.5, 2, 2, 1.5, 1, 1], n_portfolios).astype(int)
        if (k * MONTHS).sum() >= n:
            break
        n_portfolios += 1
    P, K = n_portfolios, max(k_options)
    style = pick(rng, list(STYLES), [0.3, 0.45, 0.25], P)
    is_active = bernoulli(rng, np.full(P, 0.45))                  # active = rebalances every quarter
    spread = np.where(bernoulli(rng, np.full(P, 0.30)), 1.5, 6.0)  # low = concentrated bets
    start_value = lognormal(rng, 250_000, 0.7, P)

    holding = np.zeros((P, K), dtype=int)
    target = np.zeros((P, K))
    valid = np.zeros((P, K), dtype=bool)
    for p in range(P):
        class_weight = np.array(STYLES[style[p]])
        chance = np.array([class_weight[class_names.index(c)] / CLASSES[c][0] for c in asset_class])
        chosen = rng.choice(n_assets, k[p], replace=False, p=chance / chance.sum())
        holding[p, :k[p]] = chosen
        target[p, :k[p]] = rng.dirichlet(np.full(k[p], spread[p]))
        valid[p, :k[p]] = True

    # 4. Monthly loop: grow each holding, then (for active portfolios) rebalance.
    value_start = np.zeros((P, MONTHS, K))
    value_end = np.zeros((P, MONTHS, K))
    ret = np.zeros((P, MONTHS, K))
    current = np.round(start_value[:, None] * target)
    for t in range(MONTHS):
        if t > 0 and t % 3 == 0:
            total = current.sum(axis=1, keepdims=True)
            rebalanced = np.round(total * target)
            current = np.where(is_active[:, None], rebalanced, current)
        r = returns[holding, HISTORY + t]
        r = np.where(valid, r, 0.0)
        value_start[:, t] = current
        current = np.round(current * (1 + r))
        value_end[:, t] = current
        ret[:, t] = r
    portfolio_value = value_end.sum(axis=2, keepdims=True)
    weight = value_end / np.maximum(portfolio_value, 1)
    max_weight = weight.max(axis=2)                                   # (P, months)

    # 5. Dates: month ends; purchase date is earlier than the first month shown.
    last = (snapshot + pd.offsets.MonthEnd(0))
    if last > snapshot:
        last = last - pd.offsets.MonthEnd(1)
    month_ends = [last - pd.offsets.MonthEnd(MONTHS - 1 - t) for t in range(MONTHS)]
    first_review = month_ends[0]
    purchase = random_dates(rng, first_review - pd.Timedelta(days=1800),
                            first_review - pd.Timedelta(days=45), P * K).to_numpy().reshape(P, K)

    # 6. Flatten in portfolio -> month -> holding order, dropping padding.
    mask = np.broadcast_to(valid[:, None, :], (P, MONTHS, K))

    def flat(a):
        return np.asarray(a)[mask]

    def over_holdings(a):            # (P, K) -> (P, months, K)
        return np.broadcast_to(np.asarray(a)[:, None, :], (P, MONTHS, K))

    def over_portfolios(a):          # (P,) -> (P, months, K)
        return np.broadcast_to(np.asarray(a)[:, None, None], (P, MONTHS, K))

    asset_index = np.broadcast_to(holding[:, None, :], (P, MONTHS, K))
    t_index = np.broadcast_to(np.arange(MONTHS)[None, :, None], (P, MONTHS, K))
    pid = np.array([f"PF{i + 1:03d}" for i in range(P)], dtype=object)
    aid = np.array([f"A{i + 1:02d}" for i in range(n_assets)], dtype=object)

    allocation = np.round(100 * weight, 2)
    monthly_return = np.round(100 * ret, 2)
    n_hold = over_portfolios(k)
    max_w = np.broadcast_to(max_weight[:, :, None], (P, MONTHS, K))
    is_diversified = (n_hold >= 8) & (max_w < 0.25)
    is_flagged = (allocation > 30) | (monthly_return <= -10)

    df = pd.DataFrame({
        "portfolio_id": flat(over_portfolios(pid)),
        "asset_id": flat(aid[asset_index]),
        "asset_class": flat(asset_class[asset_index]),
        "review_date": pd.to_datetime(flat(np.array(month_ends, dtype="datetime64[ns]")[t_index])),
        "purchase_date": pd.to_datetime(flat(over_holdings(purchase))),
        "allocation": flat(allocation),
        "valuation": flat(value_end).astype(int),
        "portfolio_value": flat(np.broadcast_to(portfolio_value, (P, MONTHS, K))).astype(int),
        "monthly_return": flat(monthly_return),
        "risk": np.round(flat(risk[asset_index, t_index]), 2),
        "dividend_yield": flat(np.array(dividend_yield)[asset_index]),
        "is_diversified": flat(is_diversified),
        "is_active": flat(over_portfolios(is_active)),
        "is_flagged": flat(is_flagged),
        # hidden helpers
        "value_start": flat(value_start).astype(int),
        "return_exact": flat(ret),
        "n_holdings": flat(n_hold).astype(int),
        "max_weight": flat(max_w),
        "market_return": np.round(100 * flat(market[HISTORY + t_index]), 2),
        "style": flat(over_portfolios(style)),
    })
    return df.iloc[:n].reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _complete_blocks(df):
    """Portfolio-months whose holdings are all present (the last block may be cut)."""
    size = df.groupby(["portfolio_id", "review_date"]).asset_id.transform("size")
    return df[size == df.n_holdings]


def validate(df, snapshot):
    out = []
    rd = pd.to_datetime(df.review_date)
    full = _complete_blocks(df)
    blocks = full.groupby(["portfolio_id", "review_date"])

    # ---- exact rules ----
    out.append(check("valuation = value_start x (1 + return) (+-1 and rounding)",
                     ((df.valuation - df.value_start * (1 + df.return_exact)).abs() <= 1).all()))
    out.append(check("monthly_return is the exact return in percent (2 dp)",
                     ((df.monthly_return - 100 * df.return_exact).abs() <= 0.0051).all()))
    if len(full):
        out.append(check("portfolio_value = sum of holding valuations (complete blocks)",
                         (blocks.valuation.sum() == blocks.portfolio_value.first()).all()))
        out.append(check("allocations sum to 100 (+-0.05, complete blocks)",
                         ((blocks.allocation.sum() - 100).abs() <= 0.05).all()))
        start_total = blocks.value_start.sum()
        end_total = blocks.valuation.sum()
        weights_start = full.value_start / full.groupby(["portfolio_id", "review_date"]).value_start.transform("sum")
        portfolio_return = (weights_start * full.return_exact).groupby(
            [full.portfolio_id, full.review_date]).sum()
        out.append(check("portfolio return = sum(start weight x holding return) = change in portfolio value",
                         ((end_total / start_total - 1 - portfolio_return).abs() <= 0.0005).all()))
    out.append(check("allocation = 100 x valuation / portfolio_value (2 dp)",
                     ((df.allocation - 100 * df.valuation / df.portfolio_value).abs() <= 0.006).all()))
    out.append(check("month-to-month: value_start = last valuation unless rebalanced quarter",
                     _chain_ok(df)))
    out.append(check("risk >= 0 and dividend_yield >= 0", (df.risk >= 0).all() and (df.dividend_yield >= 0).all()))
    out.append(check("is_diversified = 8+ holdings and largest weight < 25%",
                     (df.is_diversified == ((df.n_holdings >= 8) & (df.max_weight < 0.25))).all()))
    out.append(check("is_flagged = allocation > 30 or monthly_return <= -10",
                     (df.is_flagged == ((df.allocation > 30) | (df.monthly_return <= -10))).all()))
    out.append(check("purchase_date is before every review_date and review_date is a month end, <= snapshot",
                     (pd.to_datetime(df.purchase_date) < rd).all() and (rd <= snapshot).all()
                     and (rd == rd + pd.offsets.MonthEnd(0)).all()))
    out.append(check("an asset has the same return in every portfolio that holds it in a month",
                     (df.groupby(["asset_id", "review_date"]).return_exact.nunique() == 1).all()))
    out.append(check("holdings, class, yield, activity constant per portfolio/asset",
                     (df.groupby("portfolio_id").is_active.nunique() == 1).all()
                     and (df.groupby("asset_id").asset_class.nunique() == 1).all()
                     and (df.groupby("asset_id").dividend_yield.nunique() == 1).all()))
    out.append(check("valuation > 0 and allocation in (0, 100]",
                     (df.valuation >= 0).all() and (df.allocation <= 100).all()))

    # ---- relationships ----
    eq, bd = df[df.asset_class == "equity"], df[df.asset_class == "bond"]
    if len(eq) >= 15 and len(bd) >= 8:
        def slope(g):
            m = g.market_return
            return float(((g.monthly_return - g.monthly_return.mean()) * (m - m.mean())).sum()
                         / max(((m - m.mean()) ** 2).sum(), 1e-9))
        se, sb = slope(eq), slope(bd)
        out.append(check("equities respond more strongly to the market than bonds (slope)",
                         se > sb + 0.3 and se > 0.5, f"equity {se:.2f}, bond {sb:.2f}", "relationship"))
        med = df.groupby("asset_class").risk.median()
        out.append(check("median risk: equity > bond",
                         med["equity"] > med["bond"], "", "relationship"))
        dy = df.groupby("asset_class").dividend_yield.median()
        out.append(check("median dividend_yield: bond > equity",
                         dy["bond"] > dy["equity"], "", "relationship"))

    passive = df[~df.is_active].copy()
    if len(passive) > 30:
        passive = passive.sort_values(["portfolio_id", "asset_id", "review_date"])
        change = passive.groupby(["portfolio_id", "asset_id"]).allocation.diff()
        both = change.notna()
        if both.sum() > 20:
            c = float(change[both].corr(passive.monthly_return[both]))
            out.append(check("buy-and-hold: weights drift towards the holdings that did well",
                             c > 0.3, f"corr={c:.2f}", "relationship"))

    beta_gap = df.groupby("asset_id").agg(r=("monthly_return", "mean"), c=("asset_class", "first"))
    if (beta_gap.c == "equity").sum() >= 3 and (beta_gap.c == "bond").sum() >= 2:
        sd = df.groupby("asset_class").monthly_return.std()
        out.append(check("returns are more spread out for equities than bonds",
                         sd["equity"] > sd["bond"], "", "relationship"))
    return out


def _chain_ok(df):
    """For a holding, this month's starting value equals last month's ending
    value, except when an active portfolio rebalances (every third month)."""
    d = df.sort_values(["portfolio_id", "asset_id", "review_date"])
    g = d.groupby(["portfolio_id", "asset_id"])
    previous_end = g.valuation.shift(1)
    month_no = g.cumcount()
    check_rows = previous_end.notna() & ~((month_no % 3 == 0) & d.is_active)
    return bool((d.value_start[check_rows] == previous_end[check_rows]).all())


RECIPE = Recipe(
    key="finance_investment_portfolio",
    title="Investment Portfolio Returns",
    row_meaning=("One row is one holding (an asset inside a portfolio) at the end "
                 "of one month; each portfolio is followed for 9 months."),
    simulate=simulate,
    columns=["portfolio_id", "asset_id", "asset_class", "review_date", "purchase_date",
             "allocation", "valuation", "portfolio_value", "monthly_return", "risk",
             "dividend_yield", "is_diversified", "is_active", "is_flagged"],
    core=["asset_class", "allocation", "valuation", "monthly_return"],
    priority=["review_date", "risk", "portfolio_id", "asset_id", "portfolio_value",
              "dividend_yield", "is_active", "is_diversified", "is_flagged", "purchase_date"],
    docs={
        "portfolio_id": "Identifier of the portfolio; it repeats for every holding and month.",
        "asset_id": "Identifier of the asset held; the same asset can appear in several portfolios with the same return.",
        "asset_class": "equity, bond, property or commodity.",
        "review_date": "Month-end date the figures are measured at.",
        "purchase_date": "Date the holding was first bought (before the first review shown).",
        "allocation": "The holding's share of the portfolio at month end, in percent (a portfolio-month sums to 100).",
        "valuation": "Market value of the holding at month end, whole units of money (no currency).",
        "portfolio_value": "Market value of the whole portfolio at month end (sum of its holdings' valuations).",
        "monthly_return": "The holding's total return over that month in percent (not annualised); always at least -60 and at most +60.",
        "risk": "Annualised volatility in percent: standard deviation of the asset's previous 12 monthly returns x sqrt(12); uses only earlier months.",
        "dividend_yield": "The asset's yearly income as a percent of its price (0 for commodities).",
        "is_diversified": "True if the portfolio holds at least 8 assets and none is 25% or more of its value that month.",
        "is_active": "True if the portfolio is actively managed (rebalanced to target weights every quarter); False means buy-and-hold.",
        "is_flagged": "True if the holding is flagged for review: over 30% of the portfolio or lost 10% or more in the month.",
    },
    validate=validate,
    date_cols=["review_date", "purchase_date"],
    targets={
        "monthly_return": ["valuation", "allocation", "is_flagged", "portfolio_value"],
        "is_flagged": ["allocation", "monthly_return"],
        "is_diversified": [],
    },
)
