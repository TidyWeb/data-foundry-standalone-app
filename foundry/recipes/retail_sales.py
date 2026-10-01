"""Retail Product Sales.

One row is one order line: one product bought on one order.  Several lines can
share an order_id (same customer, same date).  Prices are plain numbers.

The story, in the order the code tells it:
    catalogue (products, list prices, unit costs) -> calendar (busy days)
    -> orders (date, customer type, number of lines) -> product per line
    -> promotion and discount -> quantity (demand responds to net price)
    -> revenue and margin -> returned? gift-wrapped?

Demand is a constant-elasticity curve: a 30% discount lifts expected sales by
roughly (1 / 0.7) ^ elasticity, and cheap everyday categories are more price
sensitive than electronics.  Illustrative teaching values, not real data.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli

# category: (median list price, unit cost as share of price, price elasticity,
#            base demand (extra units per line), chance a line is returned)
CATEGORIES = {
    "electronics": (120, 0.78, 1.4, 0.50, 0.09),
    "home":        (45,  0.55, 1.9, 1.00, 0.05),
    "clothing":    (35,  0.48, 2.3, 1.40, 0.16),
    "toys":        (25,  0.55, 2.1, 1.20, 0.04),
    "beauty":      (18,  0.38, 2.1, 1.50, 0.03),
    "books":       (12,  0.60, 1.8, 1.10, 0.02),
}
PRODUCTS_PER_CATEGORY = 8

# customer type: (share, extra lines per order, quantity factor, gift-wrap factor)
SEGMENTS = {
    "occasional": (0.55, 0.6, 1.0, 1.0),
    "regular":    (0.35, 1.2, 1.1, 1.0),
    "trade":      (0.10, 2.5, 1.6, 0.1),
}
DISCOUNT_DEPTHS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
DISCOUNT_WEIGHTS = [0.30, 0.20, 0.20, 0.15, 0.10, 0.05]
# how busy each calendar month is (November and December peak, January sales)
MONTH_BUSY = {1: 1.0, 2: 0.8, 3: 0.85, 4: 0.9, 5: 0.9, 6: 0.95,
              7: 1.0, 8: 0.9, 9: 0.9, 10: 1.0, 11: 1.6, 12: 1.9}
WINDOW_DAYS = 365


def build_catalogue(rng):
    """Every product has a fixed list price and a fixed unit cost."""
    rows = []
    for category, (median_price, cost_share, *_rest) in CATEGORIES.items():
        for i in range(PRODUCTS_PER_CATEGORY):
            price = max(2.0, round(lognormal(rng, median_price, 0.5))) - 0.01
            unit_cost = price * cost_share * lognormal(rng, 1.0, 0.08)
            unit_cost = round(min(unit_cost, 0.95 * price), 2)   # never dearer than the price
            rows.append((f"{category[:3]}{i + 1:02d}", category, price, unit_cost))
    return pd.DataFrame(rows, columns=["product_id", "category", "price", "unit_cost"])


def simulate(rng, n, snapshot):
    catalogue = build_catalogue(rng)
    names = list(CATEGORIES)

    # 1. Calendar: busy months and weekends attract more orders.
    days = pd.date_range(snapshot - pd.Timedelta(days=WINDOW_DAYS - 1), snapshot)
    day_weight = np.array([MONTH_BUSY[d.month] * (1.25 if d.dayofweek >= 5 else 1.0)
                           for d in days])

    # 2. Orders.  n orders always give at least n lines; we keep the first n lines.
    order_date = days[rng.choice(len(days), size=n, p=day_weight / day_weight.sum())]
    seg_names = list(SEGMENTS)
    segment = pick(rng, seg_names, [SEGMENTS[s][0] for s in seg_names], n)
    extra_lines = np.array([SEGMENTS[s][1] for s in segment])
    lines_per_order = 1 + rng.poisson(extra_lines)

    line = pd.DataFrame({"order_number": np.repeat(np.arange(n), lines_per_order.astype(np.intp))})
    line["order_id"] = ["O" + str(100000 + k) for k in line.order_number]
    line = line.iloc[:n].copy()
    k = line.order_number.to_numpy()
    line["order_date"] = order_date[k]
    line["customer_segment"] = segment[k]
    # basket_size = how many lines the order has (counted on the rows we keep).
    line["basket_size"] = line.groupby("order_number").order_number.transform("size")

    # 3. Which product is on each line (category shares differ a little).
    category = pick(rng, names, [1.0, 1.2, 1.4, 0.9, 1.1, 1.0], n)
    product_index = rng.integers(0, PRODUCTS_PER_CATEGORY, n)
    product_id = np.array([f"{c[:3]}{i + 1:02d}" for c, i in zip(category, product_index)])
    catalogue = catalogue.set_index("product_id")
    line["product_id"] = product_id
    line["price"] = catalogue.price.loc[product_id].to_numpy()
    line["unit_cost"] = catalogue.unit_cost.loc[product_id].to_numpy()
    line["product_category"] = category
    price = line.price.to_numpy()

    # 4. Promotion: more likely in clearance months and for clothing; depth varies.
    month = pd.DatetimeIndex(line.order_date).month.to_numpy()
    p_sale = (0.12 + 0.20 * np.isin(month, [1, 7, 11]) + 0.10 * (category == "clothing")
              + 0.06 * (category == "toys") * np.isin(month, [11, 12]))
    is_on_sale = bernoulli(rng, p_sale)
    depth = pick(rng, DISCOUNT_DEPTHS, DISCOUNT_WEIGHTS, n).astype(float)
    discount = np.where(is_on_sale, depth, 0.0)
    net_price = np.round(price * (1 - discount), 2)

    # 5. Quantity: demand falls as the net price rises above the category norm.
    #    Expected extra units = base demand x customer factor x (net / median)^-elasticity.
    median_price = np.array([CATEGORIES[c][0] for c in category], dtype=float)
    elasticity = np.array([CATEGORIES[c][2] for c in category])
    base_demand = np.array([CATEGORIES[c][3] for c in category])
    quantity_factor = np.array([SEGMENTS[s][2] for s in line.customer_segment])
    demand = base_demand * quantity_factor * (net_price / median_price) ** (-elasticity)
    quantity = 1 + rng.poisson(np.minimum(demand, 25))

    # 6. Money.  Revenue = quantity x net price; margin = (revenue - cost) / revenue.
    revenue = np.round(quantity * net_price, 2)
    unit_cost = line.unit_cost.to_numpy()
    profit_margin = np.round((revenue - quantity * unit_cost) / revenue, 2)

    # 7. Returns are likelier for clothing and deep discounts; gifts in December.
    p_return = (np.array([CATEGORIES[c][4] for c in category]) + 0.06 * (discount >= 0.25)
                + 0.03 * (quantity >= 4))
    is_returned = bernoulli(rng, p_return)
    gift_factor = np.array([SEGMENTS[s][3] for s in line.customer_segment])
    gift_lift = np.where(month == 12, 4.0, 1.0) * np.where(np.isin(category, ["toys", "beauty", "books"]), 1.5, 1.0)
    is_giftwrapped = bernoulli(rng, np.minimum(0.05 * gift_factor * gift_lift, 0.8))

    df = pd.DataFrame({
        "order_id": line.order_id.to_numpy(),
        "order_date": pd.to_datetime(line.order_date.to_numpy()),
        "customer_segment": line.customer_segment.to_numpy(),
        "product_category": category,
        "price": price,
        "is_on_sale": is_on_sale,
        "discount": discount,
        "quantity": quantity.astype(int),
        "basket_size": line.basket_size.to_numpy().astype(int),
        "unit_cost": unit_cost,
        "revenue": revenue,
        "profit_margin": profit_margin,
        "is_returned": is_returned,
        "is_giftwrapped": is_giftwrapped,
        # hidden helpers
        "product_id": line.product_id.to_numpy(),
        "net_price": net_price,
    })
    return df.sort_values(["order_date", "order_id"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _demeaned(df, col, by):
    return df[col] - df.groupby(by)[col].transform("mean")


def validate(df, snapshot):
    out = []
    dates = pd.to_datetime(df.order_date)

    # ---- exact rules ----
    out.append(check("net_price = price x (1 - discount), 2 dp",
                     ((df.net_price - df.price * (1 - df.discount)).abs() <= 0.006).all()))
    out.append(check("revenue = quantity x net_price (+-0.005)",
                     ((df.revenue - df.quantity * df.net_price).abs() <= 0.005).all()))
    out.append(check("profit_margin = (revenue - quantity x unit_cost) / revenue (+-0.005)",
                     ((df.profit_margin - (df.revenue - df.quantity * df.unit_cost) / df.revenue)
                      .abs() <= 0.0051).all()))
    out.append(check("is_on_sale exactly when discount > 0", (df.is_on_sale == (df.discount > 0)).all()))
    out.append(check("discount between 0 and 0.4", df.discount.between(0, 0.4).all()))
    out.append(check("quantity >= 1", (df.quantity >= 1).all()))
    out.append(check("price and unit_cost positive; cost below list price",
                     ((df.price > 0) & (df.unit_cost > 0) & (df.unit_cost < df.price)).all()))
    out.append(check("dates not after snapshot", (dates <= snapshot).all()))
    sizes = df.groupby("order_id").size()
    out.append(check("basket_size = number of lines in the order",
                     (df.basket_size == df.order_id.map(sizes)).all()))
    out.append(check("lines in one order share date and customer_segment",
                     (df.groupby("order_id").order_date.nunique() == 1).all()
                     and (df.groupby("order_id").customer_segment.nunique() == 1).all()))
    out.append(check("each product has one list price and one unit cost",
                     (df.groupby("product_id").price.nunique() == 1).all()
                     and (df.groupby("product_id").unit_cost.nunique() == 1).all()))
    out.append(check("categories valid", df.product_category.isin(CATEGORIES).all()
                     and df.customer_segment.isin(SEGMENTS).all()))

    # ---- relationships ----
    # Within a category, deeper discount -> more units (demand responds to price).
    on = df[df.is_on_sale]
    off = df[~df.is_on_sale]
    if len(on) >= 10 and len(off) >= 10:
        lift = float(on.quantity.mean() / off.quantity.mean())
        d_disc = _demeaned(df, "discount", "product_category")
        d_qty = _demeaned(df, "quantity", "product_category")
        c = float(d_disc.corr(d_qty))
        out.append(check("discounted lines sell more units than full-price lines",
                         lift > 1.05 and c > 0.02, f"lift={lift:.2f} corr={c:.2f}", "relationship"))

    med = df.groupby("product_category").profit_margin.median()
    if {"electronics", "beauty"} <= set(med.index):
        out.append(check("median margin: beauty > electronics", med["beauty"] > med["electronics"],
                         "", "relationship"))

    if len(on) >= 10 and len(off) >= 10:
        out.append(check("discounted lines have lower margin",
                         on.profit_margin.median() < off.profit_margin.median(), "", "relationship"))

    trade = df[df.customer_segment == "trade"]
    occ = df[df.customer_segment == "occasional"]
    if len(trade) >= 8 and len(occ) >= 8:
        out.append(check("trade customers buy bigger quantities than occasional ones",
                         trade.quantity.mean() > occ.quantity.mean(),
                         f"{trade.quantity.mean():.2f} vs {occ.quantity.mean():.2f}", "relationship"))

    hi = df[df.product_category.isin(["clothing", "electronics"])]
    lo = df[df.product_category.isin(["books", "toys", "beauty"])]
    if len(hi) >= 20 and len(lo) >= 20:
        out.append(check("clothing and electronics are returned more often than books, toys, beauty",
                         hi.is_returned.mean() > lo.is_returned.mean(),
                         f"{hi.is_returned.mean():.2f} vs {lo.is_returned.mean():.2f}", "relationship"))

    busy = dates.dt.month.isin([11, 12])
    window_months = pd.date_range(snapshot - pd.Timedelta(days=WINDOW_DAYS - 1), snapshot)
    expected_share = window_months.month.isin([11, 12]).mean()
    if len(df) >= 100:
        out.append(check("November and December carry more than an even share of orders",
                         busy.mean() > expected_share * 1.15,
                         f"{busy.mean():.2f} vs even {expected_share:.2f}", "relationship"))
    return out


RECIPE = Recipe(
    key="retail_sales",
    title="Retail Product Sales",
    row_meaning=("One row is one order line: one product bought on one order "
                 "(an order can have several lines)."),
    simulate=simulate,
    columns=["order_id", "order_date", "customer_segment", "product_category", "price",
             "is_on_sale", "discount", "quantity", "basket_size", "unit_cost", "revenue",
             "profit_margin", "is_returned", "is_giftwrapped"],
    core=["product_category", "price", "discount", "quantity"],
    priority=["revenue", "order_date", "unit_cost", "profit_margin", "customer_segment",
              "order_id", "basket_size", "is_on_sale", "is_returned", "is_giftwrapped"],
    docs={
        "order_id": "Order identifier; all lines of one order share it.",
        "order_date": "Date the order was placed.",
        "customer_segment": "occasional, regular or trade (business buyer).",
        "product_category": "Product category: electronics, home, clothing, toys, beauty or books.",
        "price": "List price of one unit before any discount (plain number).",
        "is_on_sale": "True if the line was sold on promotion (discount above 0).",
        "discount": "Share taken off the list price, from 0 to 0.4 (0.25 = 25% off).",
        "quantity": "Units of the product on this line.",
        "basket_size": "Number of lines on the order this row belongs to.",
        "unit_cost": "What one unit costs the shop to buy in (plain number).",
        "revenue": "Money taken for the line: quantity x price x (1 - discount), 2 dp.",
        "profit_margin": "(revenue - quantity x unit_cost) / revenue, 2 dp; negative means sold at a loss.",
        "is_returned": "True if the customer sent the goods back.",
        "is_giftwrapped": "True if the line was gift-wrapped (common in December).",
    },
    validate=validate,
    date_cols=["order_date"],
    targets={
        "is_returned": ["revenue", "profit_margin"],
        "quantity": ["revenue", "profit_margin"],
        "revenue": ["quantity", "profit_margin"],
    },
)
