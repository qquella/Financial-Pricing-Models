# Binomial Pricer GUI

Run locally (Python 3.x, Tkinter + Matplotlib):
```
python binomial_gui.py
```
The window opens with sample values:
S0=100, K=100, r=0.05, q=0.0, T=2, N=2, u=1.1, d=0.9, Call, European.

Switch to "Use σ (CRR)" to enter volatility σ and let the app compute u,d.
Click **Compute** to update numbers and redraw the stock price tree.
Use **Save Tree…** to export the image.
