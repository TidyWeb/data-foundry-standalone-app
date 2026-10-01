"""Car Insurance Claims.

One row is one claim, joined to the 12-month policy it was made on.  A policy
with two claims appears twice (same policy_id, same premium, same excess).
Policies with no claims are simulated behind the scenes - that is what keeps
claim frequency realistic - but they are not rows here.

The story, in the order the code tells it:
    policy (start date, driver age band, vehicle value, no-claims years, excess)
    -> risk score -> how many claims (a Poisson count; risky drivers claim more)
    -> per claim: date, damage type -> repair estimate (skewed, scales with car value)
    -> total loss? -> approved? -> payout = amount - excess -> disputed?

Premium is set from rating factors only (age, car, mileage, no-claims years,
excess), never from what later happens.  Claim size and claim count are
separate processes: this is the classic "frequency x severity" model.
All numbers are illustrative teaching values, not real insurance data.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli

# age band: (weight, claim risk factor, premium factor, mean no-claims years, max no-claims years)
AGE_BANDS = {
    "18-24": (0.08, 2.2, 1.90, 1.0, 6),
    "25-34": (0.20, 1.2, 1.15, 3.5, 9),
    "35-49": (0.32, 1.0, 1.00, 6.0, 15),
    "50-64": (0.25, 0.85, 0.90, 8.0, 20),
    "65+":   (0.15, 1.1, 1.05, 9.0, 20),
}
# vehicle group: (weight, median value, claim risk factor, premium factor)
GROUPS = {
    "small":       (0.30, 6000, 0.85, 0.85),
    "family":      (0.35, 12000, 1.00, 1.00),
    "performance": (0.15, 22000, 1.40, 1.35),
    "luxury":      (0.20, 35000, 1.20, 1.50),
}
# damage type: (weight, typical repair as a share of vehicle value, spread)
DAMAGE = {
    "windscreen":      (0.22, 0.035, 0.35),
    "minor collision": (0.38, 0.10, 0.55),
    "major collision": (0.12, 0.42, 0.45),
    "vandalism":       (0.10, 0.08, 0.50),
    "weather":         (0.13, 0.14, 0.60),
    "fire":            (0.05, 0.45, 0.60),
}
EXCESSES = [100, 250, 500, 750, 1000]
EXCESS_WEIGHTS = [0.10, 0.30, 0.35, 0.15, 0.10]

BASE_CLAIM_RATE = 0.10        # claims per policy-year for an average driver
BASE_PREMIUM = 380
TERM_DAYS = 365               # each policy runs 12 months
TOTAL_LOSS_SHARE = 0.70       # repair estimate >= 70% of value -> written off
WINDOW_DAYS = 3 * 365         # policies started in the last 3 years


def make_policies(rng, count, snapshot, first_number):
    """A batch of policy terms, including the ones that never claim."""
    band = pick(rng, list(AGE_BANDS), [v[0] for v in AGE_BANDS.values()], count)
    group = pick(rng, list(GROUPS), [v[0] for v in GROUPS.values()], count)
    age_risk = np.array([AGE_BANDS[b][1] for b in band])
    age_premium = np.array([AGE_BANDS[b][2] for b in band])
    ncd_mean = np.array([AGE_BANDS[b][3] for b in band])
    ncd_max = np.array([AGE_BANDS[b][4] for b in band])
    group_risk = np.array([GROUPS[g][2] for g in group])
    group_premium = np.array([GROUPS[g][3] for g in group])
    group_value = np.array([GROUPS[g][1] for g in group], dtype=float)

    vehicle_value = np.maximum(1000, np.round(lognormal(rng, group_value, 0.30) / 100) * 100).astype(int)
    annual_mileage = np.round(lognormal(rng, 8000, 0.40, count) / 500) * 500
    no_claims_years = np.minimum(rng.poisson(ncd_mean), ncd_max).astype(int)
    deductible = pick(rng, EXCESSES, EXCESS_WEIGHTS, count).astype(int)
    deductible = np.where(band == "18-24", np.maximum(deductible, 500), deductible)

    # Policy term: starts sometime in the last 3 years and runs 12 months.
    start = pd.Series(snapshot - pd.to_timedelta(rng.integers(1, WINDOW_DAYS + 1, count), unit="D"))
    end = start + pd.Timedelta(days=TERM_DAYS - 1)

    # Premium uses rating factors only.
    mileage_factor = (annual_mileage / 8000) ** 0.35
    ncd_factor = np.exp(-0.06 * no_claims_years)
    excess_factor = (500 / deductible) ** 0.15
    premium = (BASE_PREMIUM * age_premium * group_premium * mileage_factor * ncd_factor
               * excess_factor * lognormal(rng, 1.0, 0.05, count))

    # True risk also depends on things the insurer cannot see (driver quality).
    unseen_quality = rng.gamma(4.0, 0.25, count)
    relative_risk = age_risk * group_risk * mileage_factor * ncd_factor * unseen_quality
    days_observed = (end.clip(upper=snapshot) - start).dt.days + 1   # a policy still running counts part of a year
    claim_count = rng.poisson(BASE_CLAIM_RATE * relative_risk * days_observed / 365)

    # Policy numbers follow start date.
    order = np.argsort(start.to_numpy(), kind="stable")
    policies = pd.DataFrame({
        "policy_id": [f"POL{first_number + i + 1:05d}" for i in np.argsort(order)],
        "driver_age_band": band, "vehicle_group": group, "vehicle_value": vehicle_value,
        "annual_mileage": annual_mileage.astype(int), "no_claims_years": no_claims_years,
        "premium": np.round(premium).astype(int), "deductible": deductible,
        "policy_start_date": start, "policy_end_date": end, "claim_count": claim_count,
    })
    return policies


def simulate(rng, n, snapshot):
    # Keep adding batches of policies until they have produced at least n claims.
    batches, claims_so_far, policies_so_far = [], 0, 0
    while claims_so_far < n:
        batch = make_policies(rng, max(500, 12 * n), snapshot, policies_so_far)
        policies_so_far += len(batch)
        claims_so_far += int(batch.claim_count.sum())
        batches.append(batch)
    policies = pd.concat(batches, ignore_index=True)

    # One row per claim: repeat each policy row claim_count times.
    df = policies.loc[policies.index.repeat(policies.claim_count)].reset_index(drop=True)
    m = len(df)

    # Claim date: any day of the policy term that has already happened.
    last_day = df.policy_end_date.clip(upper=snapshot)
    span = (last_day - df.policy_start_date).dt.days.to_numpy() + 1
    claim_date = df.policy_start_date + pd.to_timedelta(np.floor(rng.random(m) * span).astype(int), unit="D")

    # Damage type, then repair estimate: a share of the car's value, right-skewed.
    damage_type = pick(rng, list(DAMAGE), [v[0] for v in DAMAGE.values()], m)
    share = np.array([DAMAGE[d][1] for d in damage_type])
    spread = np.array([DAMAGE[d][2] for d in damage_type])
    value = df.vehicle_value.to_numpy()
    estimate = value * share * np.exp(rng.normal(0, spread))
    repair_estimate = np.round(np.clip(estimate, 50, 1.25 * value)).astype(int)

    # Total loss: the repair would cost most of what the car is worth.
    is_total_loss = repair_estimate >= TOTAL_LOSS_SHARE * value
    approved_amount = np.where(is_total_loss, value, repair_estimate)

    # Approval: claims in the first 30 days of a policy are checked harder.
    early_claim = (claim_date - df.policy_start_date).dt.days.to_numpy() < 30
    p_approve = np.where(damage_type == "windscreen", 0.98, 0.95)
    p_approve = np.where(early_claim, 0.60, p_approve)
    is_approved = bernoulli(rng, p_approve)

    # Payout: approved amount less the excess, never below zero; nothing if refused.
    deductible = df.deductible.to_numpy()
    claim_amount = np.where(is_approved, np.maximum(0, approved_amount - deductible), 0).astype(int)

    # Disputes follow from what happened: refusals, write-offs and big payouts.
    p_dispute = (0.03 + 0.55 * (~is_approved) + 0.08 * is_total_loss + 0.05 * (claim_amount > 5000))
    is_disputed = bernoulli(rng, p_dispute)

    df = df.assign(claim_date=claim_date, damage_type=damage_type, repair_estimate=repair_estimate,
                   claim_amount=claim_amount, is_total_loss=is_total_loss,
                   is_approved=is_approved, is_disputed=is_disputed,
                   approved_amount=approved_amount.astype(int))
    df = df.sort_values(["claim_date", "policy_id"], kind="stable").iloc[:n].reset_index(drop=True)
    return df.drop(columns=["claim_count"])


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []

    # ---- exact rules ----
    out.append(check("claim_date inside policy_start_date .. policy_end_date",
                     ((df.claim_date >= df.policy_start_date) & (df.claim_date <= df.policy_end_date)).all()))
    out.append(check("claim_date not after the snapshot", (df.claim_date <= snapshot).all()))
    out.append(check("policy_end_date = policy_start_date + 364 days (12-month term)",
                     (df.policy_end_date - df.policy_start_date == pd.Timedelta(days=TERM_DAYS - 1)).all()))
    same = df.groupby("policy_id")[["driver_age_band", "vehicle_value", "no_claims_years", "premium",
                                    "deductible", "policy_start_date"]].nunique().max().max()
    out.append(check("policy details are identical on every claim of the same policy", same == 1))
    out.append(check("is_total_loss = repair_estimate >= 70% of vehicle_value",
                     (df.is_total_loss == (df.repair_estimate >= TOTAL_LOSS_SHARE * df.vehicle_value)).all()))
    out.append(check("approved amount = vehicle_value if total loss, else repair_estimate",
                     (df.approved_amount == np.where(df.is_total_loss, df.vehicle_value, df.repair_estimate)).all()))
    paid = np.where(df.is_approved, np.maximum(0, df.approved_amount - df.deductible), 0)
    out.append(check("claim_amount = max(0, approved amount - deductible) if approved, else 0",
                     (df.claim_amount == paid).all()))
    out.append(check("claim_amount never exceeds vehicle_value", (df.claim_amount <= df.vehicle_value).all()))
    out.append(check("repair_estimate between 50 and 1.25 x vehicle_value",
                     ((df.repair_estimate >= 50) & (df.repair_estimate <= 1.25 * df.vehicle_value)).all()))
    out.append(check("no_claims_years within the age band's maximum",
                     all((df.no_claims_years[df.driver_age_band == b] <= AGE_BANDS[b][4]).all() for b in AGE_BANDS)))
    out.append(check("deductible is a standard excess; drivers aged 18-24 pay at least 500",
                     df.deductible.isin(EXCESSES).all() and (df.deductible[df.driver_age_band == "18-24"] >= 500).all()))
    out.append(check("premium > 0 and vehicle_value >= 1000", ((df.premium > 0) & (df.vehicle_value >= 1000)).all()))
    out.append(check("categories valid",
                     df.driver_age_band.isin(AGE_BANDS).all() and df.damage_type.isin(DAMAGE).all()
                     and df.vehicle_group.isin(GROUPS).all()))

    # ---- relationships (tendencies) ----
    log_estimate = np.log(df.repair_estimate)
    log_value = np.log(df.vehicle_value)
    e_dev = log_estimate - log_estimate.groupby(df.damage_type).transform("mean")
    v_dev = log_value - log_value.groupby(df.damage_type).transform("mean")
    corr = float(e_dev.corr(v_dev))
    out.append(check("within a damage type, dearer cars have bigger repair estimates",
                     corr >= 0.6, f"corr={corr:.2f}", "relationship"))

    med = df.groupby("damage_type").repair_estimate.median()
    sizes = df.damage_type.value_counts()
    if all(sizes.get(t, 0) >= 5 for t in ["windscreen", "minor collision", "major collision"]):
        out.append(check("median repair estimate: windscreen < minor collision < major collision",
                         med["windscreen"] < med["minor collision"] < med["major collision"], "", "relationship"))

    ratio = float(df.repair_estimate.mean() / df.repair_estimate.median())
    out.append(check("repair estimates are right-skewed (mean well above median)",
                     ratio >= 1.5, f"mean/median={ratio:.2f}", "relationship"))

    prem = df.groupby("driver_age_band").premium.median()
    counts = df.driver_age_band.value_counts()
    if counts.get("18-24", 0) >= 6 and counts.get("35-49", 0) >= 6:
        out.append(check("median premium: age 18-24 > age 35-49",
                         prem["18-24"] > prem["35-49"], "", "relationship"))

    lp = np.log(df.premium)
    lp_dev = lp - lp.groupby(df.driver_age_band).transform("mean")
    n_dev = df.no_claims_years - df.no_claims_years.groupby(df.driver_age_band).transform("mean")
    c2 = float(lp_dev.corr(n_dev))
    out.append(check("within an age band, more no-claims years means a lower premium",
                     c2 <= -0.2, f"corr={c2:.2f}", "relationship"))

    early = (df.claim_date - df.policy_start_date).dt.days < 30
    if early.sum() >= 6:
        out.append(check("claims in the first 30 days are approved less often",
                         df.is_approved[early].mean() < df.is_approved[~early].mean(), "", "relationship"))
    return out


RECIPE = Recipe(
    key="automotive_insurance",
    title="Car Insurance Claims",
    row_meaning=("One row is one claim on a 12-month car insurance policy; "
                 "a policy with several claims appears several times."),
    simulate=simulate,
    columns=["policy_id", "driver_age_band", "vehicle_value", "no_claims_years", "premium",
             "deductible", "policy_start_date", "policy_end_date", "claim_date",
             "damage_type", "repair_estimate", "claim_amount", "is_total_loss",
             "is_approved", "is_disputed"],
    core=["driver_age_band", "vehicle_value", "damage_type", "repair_estimate", "claim_amount"],
    priority=["deductible", "is_total_loss", "premium", "no_claims_years", "claim_date",
              "is_approved", "is_disputed", "policy_id", "policy_start_date", "policy_end_date"],
    docs={
        "policy_id": "Identifier of the insurance policy; several claims can share one policy_id.",
        "driver_age_band": "Age group of the main driver: 18-24, 25-34, 35-49, 50-64 or 65+.",
        "vehicle_value": "Market value of the insured car when the policy started, whole units.",
        "no_claims_years": "Years the driver had gone without a claim when the policy started.",
        "premium": "Price of the 12-month policy, set from rating factors only, whole units.",
        "deductible": "The excess: the amount the policyholder pays first on any claim, whole units.",
        "policy_start_date": "First day the policy was in force.",
        "policy_end_date": "Last day the policy is in force (start + 364 days; may be in the future).",
        "claim_date": "Date the incident happened; always inside the policy term.",
        "damage_type": "Cause of the claim: windscreen, minor collision, major collision, vandalism, weather or fire.",
        "repair_estimate": "Assessor's estimate of the repair cost, whole units.",
        "claim_amount": "Amount the insurer paid out: value (if total loss) or repair estimate, minus the deductible, floored at 0; 0 if refused.",
        "is_total_loss": "True if the repair estimate is at least 70% of the vehicle's value (the car is written off).",
        "is_approved": "True if the insurer accepted the claim (claims in a policy's first 30 days are checked harder).",
        "is_disputed": "True if the policyholder disputed the decision or amount (more likely after a refusal or a write-off).",
    },
    validate=validate,
    date_cols=["policy_start_date", "policy_end_date", "claim_date"],
    targets={
        "claim_amount": ["repair_estimate", "is_approved", "is_total_loss", "is_disputed"],
        "is_total_loss": ["repair_estimate", "claim_amount"],
        "is_approved": ["claim_amount", "is_disputed"],
    },
)
