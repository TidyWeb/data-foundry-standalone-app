"""Company Financial Performance.

One row is one company in one calendar quarter (a company-quarter panel).
Each company appears in 12 consecutive quarters, ending at the latest quarter
that has finished by the snapshot date.  Rows are sorted company, then time.

Which "profit" is which (money columns are plain whole numbers, no currency):
    revenue          = sales in the quarter
    cost_of_goods    = direct cost of what was sold
    operating_expense= running costs of the business (staff, rent, marketing);
                       it does NOT include cost_of_goods, interest or tax
    operating_income = revenue - cost_of_goods - operating_expense
                       (profit from operations, before interest and tax)
    interest_and_tax = interest paid on debt + income tax
    profit           = operating_income - interest_and_tax   (NET profit)
    operating_margin = 100 x operating_income / revenue      (percent)
    margin           = 100 x profit / revenue                (net margin, percent)

The story, in the order the code tells it:
    company traits (sector, size, growth, cost structure)
    -> quarterly revenue (growth + seasonality + a shared economy shock + noise)
    -> cost of goods -> operating expense (part fixed, part variable)
    -> operating income -> interest and tax -> profit -> margins
    -> cash: profit minus investment = free cash flow -> burn rate, cash balance
    -> flags (profitable? public? audited?)

Because part of the cost base is fixed, growing companies see their margins
improve (operating leverage) and shrinking ones see them squeezed.  Profit is
not cash: a fast-growing company can be profitable and still burn cash.
All numbers are illustrative teaching values, not real company data.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli, logistic

QUARTERS = 12          # quarters per company
TAX_RATE = 0.25        # fictional flat tax on positive pre-tax profit (no tax on losses)

# sector: (weight, typical quarterly revenue, growth per quarter, gross margin,
#          variable opex share of revenue, fixed opex as share of typical revenue,
#          net investment share of revenue, seasonal effect for quarters 1-4)
SECTORS = {
    "software":      (1.0, 8_000_000, 0.035, 0.75, 0.25, 0.35, 0.05, [0.00, 0.00, 0.00, 0.03]),
    "retail":        (1.2, 25_000_000, 0.010, 0.32, 0.08, 0.17, 0.03, [-0.06, -0.04, -0.02, 0.12]),
    "manufacturing": (1.0, 15_000_000, 0.012, 0.28, 0.06, 0.15, 0.06, [0.00, 0.02, 0.00, -0.02]),
    "services":      (1.0, 6_000_000, 0.020, 0.42, 0.12, 0.22, 0.02, [0.02, 0.00, -0.02, 0.00]),
    "healthcare":    (0.8, 10_000_000, 0.020, 0.55, 0.18, 0.28, 0.05, [0.03, 0.00, -0.03, 0.00]),
}


def latest_quarter_end(snapshot):
    """The most recent quarter end on or before the snapshot."""
    end = snapshot + pd.offsets.QuarterEnd(0)
    if end > snapshot:
        end = end - pd.offsets.QuarterEnd(1)
    return end.normalize()


def simulate(rng, n, snapshot):
    n_companies = -(-n // QUARTERS)          # enough companies to cover n rows
    C, Q = n_companies, QUARTERS

    # 1. Company traits: sector, size, growth tendency and cost structure.
    names = list(SECTORS)
    sector = pick(rng, names, [SECTORS[s][0] for s in names], C)
    par = {i: np.array([SECTORS[s][i] for s in sector]) for i in range(1, 7)}
    base_revenue = lognormal(rng, par[1], 0.8, C)             # size varies a lot
    growth = par[2] + rng.normal(0, 0.012, C)                 # this company's growth tendency
    gross_margin = np.clip(par[3] + rng.normal(0, 0.05, C), 0.10, 0.90)
    variable_opex = par[4] * np.exp(rng.normal(0, 0.15, C))
    fixed_opex = par[5] * np.exp(rng.normal(0, 0.15, C)) * base_revenue
    invest_share = par[6]
    interest_per_quarter = np.round(rng.uniform(0.0, 0.03, C) * base_revenue)   # debt cost
    opening_cash = base_revenue * rng.uniform(0.3, 1.5, C)

    # 2. Calendar: the last quarter ends at the latest finished quarter.
    last = latest_quarter_end(snapshot)
    quarter_ends = [last - pd.offsets.QuarterEnd(Q - 1 - t) for t in range(Q)]
    quarter_number = np.array([d.quarter for d in quarter_ends])          # 1..4

    # 3. Revenue: trend + seasonality + a shock all companies share (the
    #    economy) + company noise that carries forward from quarter to quarter.
    economy = np.cumsum(rng.normal(0, 0.02, Q))
    own_shock = np.cumsum(rng.normal(0, 0.05, (C, Q)), axis=1)
    season = np.array([SECTORS[s][7] for s in sector])[:, quarter_number - 1]
    t = np.arange(Q)
    log_revenue = (np.log(base_revenue)[:, None] + growth[:, None] * t
                   + economy + own_shock + season)
    revenue = np.maximum(np.round(np.exp(log_revenue)), 1000)
    previous = np.column_stack([revenue[:, 0] / (1 + growth), revenue[:, :-1]])

    # 4. Costs.  Cost of goods follows revenue at the company's gross margin;
    #    operating expense = fixed part (creeps up 1% a quarter) + variable part.
    gm_quarter = gross_margin[:, None] + rng.normal(0, 0.015, (C, Q))
    cost_of_goods = np.round(revenue * (1 - gm_quarter))
    fixed_part = fixed_opex[:, None] * 1.01 ** t
    operating_expense = np.round((fixed_part + variable_opex[:, None] * revenue)
                                 * np.exp(rng.normal(0, 0.03, (C, Q))))

    # 5. Profit chain, using whole numbers so every identity is exact.
    operating_income = revenue - cost_of_goods - operating_expense
    interest = np.tile(interest_per_quarter[:, None], (1, Q))
    tax = np.round(TAX_RATE * np.maximum(operating_income - interest, 0))
    interest_and_tax = interest + tax
    profit = operating_income - interest_and_tax

    # 6. Cash.  Growth ties up cash (stock, customers paying late), so net
    #    investment = a share of revenue + 25% of the revenue increase.
    net_investment = np.round(invest_share[:, None] * revenue + 0.25 * (revenue - previous))
    free_cash_flow = profit - net_investment
    burn_rate = np.maximum(-free_cash_flow, 0)
    cash_open = np.zeros((C, Q))
    equity_raise = np.zeros((C, Q))
    cash_close = np.zeros((C, Q))
    cash = np.round(opening_cash)
    for q in range(Q):                       # cash carries forward quarter by quarter
        cash_open[:, q] = cash
        shortfall = 0.05 * revenue[:, q] - (cash + free_cash_flow[:, q])
        equity_raise[:, q] = np.round(np.maximum(shortfall, 0))   # owners top up if cash runs dry
        cash = cash + free_cash_flow[:, q] + equity_raise[:, q]
        cash_close[:, q] = cash

    # 7. Flags.  Bigger companies are more often public.  Public companies
    #    must be audited; large private ones usually are too.
    size = np.log(base_revenue / 1e7)
    is_public = bernoulli(rng, logistic(-0.8 + 1.2 * size))
    is_audited = is_public | bernoulli(rng, logistic(-1.5 + 0.9 * size))

    def flat(a):
        return np.asarray(a).ravel()

    def per_company(a):
        return np.repeat(np.asarray(a), Q)

    df = pd.DataFrame({
        "company_id": per_company([f"C{i + 1:03d}" for i in range(C)]),
        "sector": per_company(sector),
        "quarter_end": pd.to_datetime(np.tile(quarter_ends, C)),
        "revenue": flat(revenue).astype(int),
        "cost_of_goods": flat(cost_of_goods).astype(int),
        "operating_expense": flat(operating_expense).astype(int),
        "operating_income": flat(operating_income).astype(int),
        "operating_margin": np.round(100 * flat(operating_income) / flat(revenue), 2),
        "interest_and_tax": flat(interest_and_tax).astype(int),
        "profit": flat(profit).astype(int),
        "margin": np.round(100 * flat(profit) / flat(revenue), 2),
        "burn_rate": flat(burn_rate).astype(int),
        "is_profitable": flat(profit) > 0,
        "is_audited": per_company(is_audited),
        "is_public": per_company(is_public),
        # hidden helpers
        "interest": flat(interest).astype(int),
        "tax": flat(tax).astype(int),
        "free_cash_flow": flat(free_cash_flow).astype(int),
        "cash_open": flat(cash_open).astype(int),
        "equity_raise": flat(equity_raise).astype(int),
        "cash_close": flat(cash_close).astype(int),
        "previous_revenue": flat(previous).astype(int),
    })
    return df.iloc[:n].reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    qe = pd.to_datetime(df.quarter_end)

    # ---- exact rules ----
    out.append(check("operating_income = revenue - cost_of_goods - operating_expense",
                     (df.operating_income == df.revenue - df.cost_of_goods - df.operating_expense).all()))
    out.append(check("interest_and_tax = interest + tax",
                     (df.interest_and_tax == df.interest + df.tax).all()))
    out.append(check("profit = operating_income - interest_and_tax",
                     (df.profit == df.operating_income - df.interest_and_tax).all()))
    out.append(check("no tax when pre-tax profit <= 0",
                     (df.tax[(df.operating_income - df.interest) <= 0] == 0).all()))
    out.append(check("tax is 25% of positive pre-tax profit (+-1)",
                     ((df.tax - TAX_RATE * (df.operating_income - df.interest).clip(lower=0)).abs() <= 1).all()))
    out.append(check("margin = 100 x profit / revenue (2 dp)",
                     ((df.margin - 100 * df.profit / df.revenue).abs() <= 0.006).all()))
    out.append(check("operating_margin = 100 x operating_income / revenue (2 dp)",
                     ((df.operating_margin - 100 * df.operating_income / df.revenue).abs() <= 0.006).all()))
    out.append(check("is_profitable = profit > 0", (df.is_profitable == (df.profit > 0)).all()))
    out.append(check("burn_rate = max(0, -free_cash_flow)",
                     (df.burn_rate == (-df.free_cash_flow).clip(lower=0)).all()))
    out.append(check("cash_close = cash_open + free_cash_flow + equity_raise",
                     (df.cash_close == df.cash_open + df.free_cash_flow + df.equity_raise).all()))
    out.append(check("cash_open = previous quarter's cash_close (same company)",
                     (df.groupby("company_id").cash_close.shift(1).dropna()
                      == df.cash_open[df.groupby("company_id").cumcount() > 0]).all()))
    out.append(check("revenue, costs, cash_close positive / non-negative",
                     (df.revenue > 0).all() and (df.cost_of_goods >= 0).all()
                     and (df.operating_expense >= 0).all() and (df.cash_close >= 0).all()))
    out.append(check("quarter_end is a calendar quarter end, not after snapshot",
                     (qe == qe + pd.offsets.QuarterEnd(0)).all() and (qe <= snapshot).all()))
    step = qe.groupby(df.company_id).diff().dropna()
    out.append(check("quarters are consecutive within a company (89-92 days)",
                     step.dt.days.between(89, 92).all()))
    out.append(check("public companies are always audited", (~df.is_public | df.is_audited).all()))
    out.append(check("company traits constant across rows",
                     (df.groupby("company_id").sector.nunique() == 1).all()
                     and (df.groupby("company_id").is_public.nunique() == 1).all()))
    out.append(check("sector valid", df.sector.isin(SECTORS).all()))

    # ---- relationships ----
    df = df.assign(g=np.log(df.revenue / df.previous_revenue))
    df = df.assign(d_om=df.operating_margin - df.groupby("company_id").operating_margin.shift(1))
    ok = df.dropna(subset=["d_om"])
    corr = float(ok.g.corr(ok.d_om)) if len(ok) > 10 else float("nan")
    out.append(check("operating leverage: revenue growth lifts operating margin",
                     corr > 0.15, f"corr={corr:.2f}", "relationship"))

    gm = (df.revenue - df.cost_of_goods) / df.revenue
    med = gm.groupby(df.sector).median()
    if {"software", "retail"} <= set(med.index) and \
            (df.sector == "software").sum() >= 8 and (df.sector == "retail").sum() >= 8:
        out.append(check("median gross margin: software > retail",
                         med["software"] > med["retail"], "", "relationship"))

    lagged = df.groupby("company_id").margin.shift(1)
    both = df.margin.notna() & lagged.notna()
    rho = float(df.margin[both].corr(lagged[both])) if both.sum() > 10 else float("nan")
    out.append(check("a company's margin resembles its own last quarter's",
                     rho > 0.5, f"corr={rho:.2f}", "relationship"))

    gap = df.profit - df.free_cash_flow
    c2 = float(gap.corr(df.revenue - df.previous_revenue))
    out.append(check("growth ties up cash: profit minus cash flow rises with revenue growth",
                     c2 > 0.3, f"corr={c2:.2f}", "relationship"))

    by_company = df.groupby("company_id").agg(r=("revenue", "mean"), p=("is_public", "first"))
    if by_company.p.sum() >= 3 and (~by_company.p).sum() >= 3:
        out.append(check("public companies are larger (median revenue)",
                         by_company.r[by_company.p].median() > by_company.r[~by_company.p].median(),
                         "", "relationship"))

    retail = df[df.sector == "retail"]
    if retail.company_id.nunique() >= 3:
        q4 = retail.revenue[qe[retail.index].dt.quarter == 4].mean()
        other = retail.revenue[qe[retail.index].dt.quarter != 4].mean()
        out.append(check("retail revenue is higher in Q4 (seasonality)",
                         q4 > other, "", "relationship"))
    return out


RECIPE = Recipe(
    key="finance_company_performance",
    title="Company Financial Performance",
    row_meaning=("One row is one company in one calendar quarter; each company "
                 "appears in 12 consecutive quarters."),
    simulate=simulate,
    columns=["company_id", "sector", "quarter_end", "revenue", "cost_of_goods",
             "operating_expense", "operating_income", "operating_margin",
             "interest_and_tax", "profit", "margin", "burn_rate",
             "is_profitable", "is_audited", "is_public"],
    core=["sector", "revenue", "operating_income", "profit", "margin"],
    priority=["quarter_end", "company_id", "cost_of_goods", "operating_expense",
              "operating_margin", "interest_and_tax", "burn_rate", "is_public",
              "is_profitable", "is_audited"],
    docs={
        "company_id": "Identifier of the company; the same company repeats across 12 quarters.",
        "sector": "Industry sector (software, retail, manufacturing, services or healthcare).",
        "quarter_end": "Last day of the calendar quarter the figures cover.",
        "revenue": "Sales in the quarter, whole units of money (no currency).",
        "cost_of_goods": "Direct cost of the goods or services sold in the quarter.",
        "operating_expense": "Running costs (staff, rent, marketing); excludes cost_of_goods, interest and tax.",
        "operating_income": "Operating profit before interest and tax = revenue - cost_of_goods - operating_expense.",
        "operating_margin": "Operating income as a percentage of revenue (2 dp).",
        "interest_and_tax": "Interest paid plus income tax for the quarter; tax is a fictional 25% of positive pre-tax profit.",
        "profit": "Net profit = operating_income - interest_and_tax; can be negative.",
        "margin": "Net margin: profit as a percentage of revenue (2 dp).",
        "burn_rate": "Cash the company burned in the quarter (its negative free cash flow); 0 if it generated cash.",
        "is_profitable": "True if profit is above zero.",
        "is_audited": "True if the accounts were externally audited (all public companies are; large private ones usually).",
        "is_public": "True if the company is listed on a stock exchange (more common for bigger companies).",
    },
    validate=validate,
    date_cols=["quarter_end"],
    targets={
        "is_profitable": ["profit", "margin"],
        "profit": ["margin", "is_profitable", "interest_and_tax"],
        "burn_rate": [],
    },
)
