import streamlit as st

st.set_page_config(page_title="Help", layout="wide")

st.title("Help")

st.markdown("""
## **Formatting**

**Tickers & Weights (.5 = 50%)**  
- Inputs must be comma separated between tickers and weights.
- The order of tickers must correspond to the same order of weights.

**Example:**  
A portfolio allocation of 75% TQQQ and 25% AGG should be entered as:

- **Tickers:** `TQQQ, AGG`  
- **Weights:** `.75, .25`

---

## **User Inputs**

**Backtest Start Date**  
User selected beginning date (default set to `1900-01-01`).  
The default allows the model to backtest from when all tickers have available price data.

**Backtest End Date (Optional)**  
Used to evaluate strategy performance over specific time segments.  
Leaving this blank will use the most recent available closing prices.

**Official Strategy Inception Date**  
The actual date the user began real-world strategy implementation.

**Benchmark Ticker (Your Performance Graph & Table)**  
User inputted benchmark ticker used for comparison against the Sigma and Buy & Hold strategies.

**Risk On Allocation**  
Input desired ticker(s) to construct the Risk On portfolio  
(e.g., `TQQQ` for a 9sig-style strategy).

**Risk Off Allocation**  
Input desired ticker(s) to construct the Risk Off portfolio  
(e.g., `AGG` for a 9sig-style strategy).

**Annual Portfolio Drag**  
Used to simulate decay in leveraged funds (e.g., `TQQQ`).  
If not applicable, leave this input at `0`.

**Portfolio Value at Last Rebalance**  
Assuming true calendar quarter-end rebalancing, input the total portfolio value at the prior quarter end.

**Portfolio Value Today**  
Input the current total portfolio value.  
If today is not a quarter-end rebalance date, this input is used only to track intra-quarter progress.

---

## **Controls**

**Save Configuration**  
Allows logged-in users to save all inputs for future sessions.

**Reset to Default Configuration**  
Resets all inputs back to default values  
(e.g., 9sig defaults such as `TQQQ`, `AGG`, etc.).

**Run**  
Once all inputs are entered correctly, select **Run** to generate model outputs.
""")

st.markdown("---")
st.caption("Disclaimer: Past performance does not indicate future results. The information provided on this website is for educational purposes only and is not intended as financial advice. No guarantees are made regarding the accuracy or completeness of the data and computations provided. Always seek the advice of your financial advisor or other qualified financial services provider regarding any investment.")
