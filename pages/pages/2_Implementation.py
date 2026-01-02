import streamlit as st

st.set_page_config(
    page_title="Implementation",
    layout="wide"
)

st.title("Implementation Guide")

st.markdown("""
## Purpose

This page explains **how to use the strategy correctly**.
Read this before running any backtests or live analysis.

---

## Step 1 — Regime Filter

- The strategy monitors a portfolio index relative to a moving average.
- When price is **above** the moving average → **RISK-ON**
- When price is **below** the moving average → **RISK-OFF**

There are no discretionary overrides.

---

## Step 2 — Risk-On Behavior

When the strategy is RISK-ON:

- Capital is allocated to the risk-on sleeve
- A **quarterly target-growth rule** is applied
- Excess gains are trimmed
- Shortfalls are replenished (if capital is available)

Rebalancing only occurs at **true calendar quarter-ends**.

---

## Step 3 — Risk-Off Behavior

When the strategy is RISK-OFF:

- Risk-on capital is frozen
- Portfolio moves fully to the defensive sleeve
- No trading occurs until the regime flips back to RISK-ON

---

## Step 4 — Execution Discipline

To implement this strategy correctly:

- Do not override signals
- Do not rebalance mid-quarter
- Do not optimize parameters during live use
- Follow the dollar amounts exactly

Consistency matters more than optimization.

---

## Important Notes

- This tool is **educational and analytical**
- Past performance does not guarantee future results
- This is **not financial advice**
""")
