import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.title("Sigma Strategy")

st.markdown("""
## What this site is

This website provides a **rules-based portfolio strategy** designed to help users
understand and implement a disciplined, systematic investment process.

This tool is **educational and analytical**, not advisory.

---

## Core Principles

- No discretionary decisions
- No intraday trading
- No curve-fitting during live use
- Follow the model outputs exactly

---

## How to use this site

1. Read the **About** and **Implementation** pages
2. Understand the risk and assumptions
3. Proceed to the **Backtesting** page only if comfortable

---

## Disclaimer

This software is provided **as-is**.
Past performance does not guarantee future results.
Nothing here constitutes financial advice.
""")
