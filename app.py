import streamlit as st

st.set_page_config(
    page_title="Sigma Strategy",
    layout="wide"
)

st.title("Sigma Strategy")

st.markdown("""
## Welcome

This site provides a **rules-based portfolio strategy** designed to help users
understand and implement a disciplined, systematic investment process.

Please read the information below **before** running any analysis.

---

### What this is
- A regime-based allocation system
- Quarterly target-growth rebalancing
- Explicit risk-off behavior
- Fully systematic (no discretion)

---

### How to use this site
1. Read **Implementation**
2. Understand risks and assumptions
3. Proceed to **Backtest** when ready

---

### Disclaimer
This tool is **educational and analytical only**.
Past performance does not guarantee future results.
Nothing here constitutes financial advice.
""")

st.info("Use the sidebar to navigate through the sections.")