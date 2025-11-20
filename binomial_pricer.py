import itertools
import math

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Common utilities ----------


def stdnorm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def payoff(S, K, option):
    return max(S - K, 0.0) if option == "call" else max(K - S, 0.0)


def payoff_gap(S, K_pay, K_trig, option):
    if option == "call":
        return (S - K_pay) if (S > K_trig) else 0.0
    else:
        return (K_pay - S) if (S < K_trig) else 0.0


def crr_ud(sigma, dt):
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    return u, d


def risk_neutral_p(r, q, dt, u, d):
    # p = (e^{(r - q)Δt} - d)/(u - d)
    a = math.exp((r - q) * dt)
    return (a - d) / (u - d)


def stock_tree(S0, u, d, N):
    return [[S0 * (u**j) * (d ** (i - j)) for j in range(i + 1)] for i in range(N + 1)]


# ---------- Vanilla binomial (European/American) ----------


def option_tree(S_tree, K, r, q, dt, u, d, option, exercise):
    N = len(S_tree) - 1
    V = [[0.0] * (i + 1) for i in range(N + 1)]
    # terminal
    for j, S in enumerate(S_tree[-1]):
        V[-1][j] = payoff(S, K, option)
    p = risk_neutral_p(r, q, dt, u, d)
    disc = math.exp(-r * dt)
    # backward
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * V[i + 1][j + 1] + (1 - p) * V[i + 1][j])
            if exercise == "american":
                V[i][j] = max(cont, payoff(S_tree[i][j], K, option))
            else:
                V[i][j] = cont
    return V


def option_tree_gap(S_tree, K_pay, K_trig, r, q, dt, u, d, option):
    N = len(S_tree) - 1
    V = [[0.0] * (i + 1) for i in range(N + 1)]
    for j, S in enumerate(S_tree[-1]):
        V[-1][j] = payoff_gap(S, K_pay, K_trig, option)
    p = risk_neutral_p(r, q, dt, u, d)
    disc = math.exp(-r * dt)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            V[i][j] = disc * (p * V[i + 1][j + 1] + (1 - p) * V[i + 1][j])
    return V


# ---------- Barrier (European, discrete monitoring) ----------


def _ups_to_hit(S0, H, u):
    if u <= 1.0:
        raise ValueError("u must be > 1 for up-barrier")
    if S0 >= H:
        return 0
    return int(math.ceil(math.log(H / S0, u)))


def _downs_to_hit(S0, H, d):
    if d >= 1.0:
        raise ValueError("d must be < 1 for down-barrier")
    if S0 <= H:
        return 0
    return int(math.ceil(math.log(H / S0, d)))


def barrier_out_tree(S0, K, r, q, T, N, u, d, option, barrier_type, H):
    dt = T / N
    S = stock_tree(S0, u, d, N)
    V = [[0.0] * (i + 1) for i in range(N + 1)]
    p = risk_neutral_p(r, q, dt, u, d)
    disc = math.exp(-r * dt)

    if barrier_type == "up-and-out":
        k = _ups_to_hit(S0, H, u)
        for j, St in enumerate(S[-1]):
            V[-1][j] = 0.0 if j >= k else payoff(St, K, option)
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                if j >= k:
                    V[i][j] = 0.0
                else:
                    V[i][j] = disc * (p * V[i + 1][j + 1] + (1 - p) * V[i + 1][j])

    elif barrier_type == "down-and-out":
        m = _downs_to_hit(S0, H, d)
        for j, St in enumerate(S[-1]):
            downs = N - j
            V[-1][j] = 0.0 if downs >= m else payoff(St, K, option)
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                downs = i - j
                if downs >= m:
                    V[i][j] = 0.0
                else:
                    V[i][j] = disc * (p * V[i + 1][j + 1] + (1 - p) * V[i + 1][j])
    else:
        raise ValueError("barrier_out_tree only handles up-and-out or down-and-out")
    return S, V


def barrier_price_euro(S0, K, r, q, T, N, u, d, option, barrier_type, H):
    dt = T / N
    if barrier_type in ("up-and-out", "down-and-out"):
        S_tree, V_out = barrier_out_tree(
            S0, K, r, q, T, N, u, d, option, barrier_type, H
        )
        price = V_out[0][0]
        formula = (
            "Barrier OUT (discrete):\n"
            "p = (e^{(r−q)Δt} − d)/(u−d)\n"
            "V_{N,j} = payoff(S_{N,j},K) if alive else 0\n"
            "V_{i,j} = e^{−rΔt}[ p V_{i+1,j+1} + (1−p)V_{i+1,j} ]"
        )
        return price, S_tree, V_out, "out", formula
    else:
        S_tree = stock_tree(S0, u, d, N)
        V_van = option_tree(S_tree, K, r, q, dt, u, d, option, "european")
        out_type = "up-and-out" if "up" in barrier_type else "down-and-out"
        _, V_out = barrier_out_tree(S0, K, r, q, T, N, u, d, option, out_type, H)
        price = V_van[0][0] - V_out[0][0]
        formula = "Barrier IN via parity: Price_IN = Price_vanilla − Price_OUT"
        return price, S_tree, None, "in", formula


# ---------- Plotting / export ----------


def build_frames(S_tree, V_tree):
    rows = []
    for i, level in enumerate(S_tree):
        for j, S in enumerate(level):
            rows.append({"t": i, "j": j, "S": S, "V": V_tree[i][j]})
    return pd.DataFrame(rows).sort_values(["t", "j"]).reset_index(drop=True)


def plot_tree_with_values(S_tree, V_tree, title, outfile_or_buffer, option_color="red"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, level in enumerate(S_tree[:-1]):
        for j, _ in enumerate(level):
            x0, y0 = i, 2 * j - i
            x1, y1 = i + 1, 2 * (j + 1) - (i + 1)
            x2, y2 = i + 1, 2 * j - (i + 1)
            ax.plot([x0, x1], [y0, y1])
            ax.plot([x0, x2], [y0, y2])
    for i, level in enumerate(S_tree):
        for j, S in enumerate(level):
            V = V_tree[i][j]
            ax.text(i + 0.02, 2 * j - i + 0.22, f"{S:.2f}", va="center", fontsize=10)
            ax.text(
                i + 0.02,
                2 * j - i - 0.18,
                f"({V:.2f})",
                va="center",
                fontsize=9,
                color=option_color,
            )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outfile_or_buffer, dpi=150, bbox_inches="tight")
    if hasattr(outfile_or_buffer, "write"):
        outfile_or_buffer.seek(0)
    return outfile_or_buffer


def plot_tree_stock_only(S_tree, title, outfile_or_buffer):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, level in enumerate(S_tree[:-1]):
        for j, _ in enumerate(level):
            x0, y0 = i, 2 * j - i
            x1, y1 = i + 1, 2 * (j + 1) - (i + 1)
            x2, y2 = i + 1, 2 * j - (i + 1)
            ax.plot([x0, x1], [y0, y1])
            ax.plot([x0, x2], [y0, y2])
    for i, level in enumerate(S_tree):
        for j, S in enumerate(level):
            ax.text(i + 0.02, 2 * j - i, f"{S:.2f}", va="center", fontsize=10)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outfile_or_buffer, dpi=150, bbox_inches="tight")
    if hasattr(outfile_or_buffer, "write"):
        outfile_or_buffer.seek(0)
    return outfile_or_buffer


def export_frames_to_files(df, csv_path, xlsx_path):
    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Tree")
    return csv_path, xlsx_path


# ---------- Greeks wrappers (vanilla) ----------


def price_once(S0, K, r, q, T, N, u, d, option, exercise):
    dt = T / N
    S_tree = stock_tree(S0, u, d, N)
    V_tree = option_tree(S_tree, K, r, q, dt, u, d, option, exercise)
    return V_tree[0][0], S_tree, V_tree


def greeks(
    S0, K, r, q, T, N, u=None, d=None, sigma=None, option="call", exercise="european"
):
    if (u is None or d is None) and sigma is None:
        raise ValueError("Provide sigma or both u and d for greeks()")
    dt = T / N
    if u is None or d is None:
        u, d = crr_ud(sigma, dt)
    price0, S_tree, V_tree = price_once(S0, K, r, q, T, N, u, d, option, exercise)

    epsS = 0.01 * S0 if S0 != 0 else 0.01
    p_up, *_ = price_once(S0 + epsS, K, r, q, T, N, u, d, option, exercise)
    p_dn, *_ = price_once(S0 - epsS, K, r, q, T, N, u, d, option, exercise)
    delta = (p_up - p_dn) / (2 * epsS)
    gamma = (p_up - 2 * price0 + p_dn) / (epsS**2)

    h = min(T / 100.0 if T > 0 else 1 / 365.0, 1 / 365.0)
    p_t_up, *_ = price_once(S0, K, r, q, T + h, N, u, d, option, exercise)
    p_t_dn, *_ = price_once(S0, K, r, q, max(T - h, 1e-6), N, u, d, option, exercise)
    theta = (p_t_up - p_t_dn) / (2 * h)

    epsr = 1e-4
    p_r_up, *_ = price_once(S0, K, r + epsr, q, T, N, u, d, option, exercise)
    p_r_dn, *_ = price_once(S0, K, r - epsr, q, T, N, u, d, option, exercise)
    rho = (p_r_up - p_r_dn) / (2 * epsr)

    if sigma is not None:
        epsv = 0.01
        u_up, d_up = crr_ud(sigma + epsv, dt)
        u_dn, d_dn = crr_ud(max(sigma - epsv, 1e-6), dt)
    else:
        lam = 0.01
        u_up, d_up = u * math.exp(lam), d / math.exp(lam)
        u_dn, d_dn = u / math.exp(lam), d * math.exp(lam)
    p_v_up, *_ = price_once(S0, K, r, q, T, N, u_up, d_up, option, exercise)
    p_v_dn, *_ = price_once(S0, K, r, q, T, N, u_dn, d_dn, option, exercise)
    vega = (p_v_up - p_v_dn) / (2 * (0.01))

    omega = delta * (S0 / price0) if price0 != 0 else float("inf")

    formula = (
        "Binomial model formulas:\n"
        f"Δt = T/N = {dt:.6f}\n"
        "p = (e^{(r−q)Δt} − d)/(u − d)\n"
        "V_{N,j} = payoff(S_{N,j}, K)\n"
        "V_{i,j} = e^{−rΔt} [ p V_{i+1,j+1} + (1−p) V_{i+1,j} ]   (European)\n"
        "V_{i,j} = max( intrinsic , continuation )                (American)\n"
        "delta ≈ (V(S+ε) − V(S−ε)) / (2ε)\n"
        "gamma ≈ (V(S+ε) − 2V(S) + V(S−ε)) / ε²\n"
        "theta ≈ (V(T+ε) − V(T−ε)) / (2ε)\n"
        "rho   ≈ (V(r+ε) − V(r−ε)) / (2ε)\n"
        "vega  ≈ (V(σ+ε) − V(σ−ε)) / (2ε)\n"
        "omega = delta * S0 / Price\n"
    )

    return {
        "price": price0,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "omega": omega,
        "S_tree": S_tree,
        "V_tree": V_tree,
        "u": u,
        "d": d,
        "dt": dt,
        "pu": risk_neutral_p(r, q, dt, u, d),
        "pd": 1 - risk_neutral_p(r, q, dt, u, d),
        "formula": formula,
    }


# ---------- Asian options (European) ----------


def enumerate_paths(S0, u, d, N, p):
    for bits in itertools.product([0, 1], repeat=N):
        prob = 1.0
        S = S0
        prices = [S0]
        for b in bits:
            if b == 1:
                S = S * u
                prob *= p
            else:
                S = S * d
                prob *= 1 - p
            prices.append(S)
        yield prices, prob


def asian_price(S0, K, r, q, T, N, u, d, option, kind="arith"):
    dt = T / N
    p = risk_neutral_p(r, q, dt, u, d)
    exp_payoff = 0.0
    for path, prob in enumerate_paths(S0, u, d, N, p):
        if kind == "arith":
            A = sum(path) / (N + 1)
        else:
            prod = 1.0
            for x in path:
                prod *= x
            A = prod ** (1.0 / (N + 1))
        exp_payoff += payoff(A, K, option) * prob
    return math.exp(-r * T) * exp_payoff


def asian_greeks(
    S0, K, r, q, T, N, u=None, d=None, sigma=None, option="call", kind="arith"
):
    dt = T / N
    if (u is None or d is None) and sigma is None:
        raise ValueError("Provide sigma or both u and d for asian_greeks()")
    if u is None or d is None:
        u, d = crr_ud(sigma, dt)

    def P(S0_, K_, r_, q_, T_, u_, d_):
        return asian_price(S0_, K_, r_, q_, T_, N, u_, d_, option, kind)

    price0 = P(S0, K, r, q, T, u, d)
    epsS = 0.01 * S0 if S0 != 0 else 0.01
    p_up = P(S0 + epsS, K, r, q, T, u, d)
    p_dn = P(S0 - epsS, K, r, q, T, u, d)
    delta = (p_up - p_dn) / (2 * epsS)
    gamma = (p_up - 2 * price0 + p_dn) / (epsS**2)
    h = min(T / 100.0 if T > 0 else 1 / 365.0, 1 / 365.0)
    theta = (P(S0, K, r, q, T + h, u, d) - P(S0, K, r, q, max(T - h, 1e-6), u, d)) / (
        2 * h
    )
    epsr = 1e-4
    rho = (P(S0, K, r + epsr, q, T, u, d) - P(S0, K, r - epsr, q, T, u, d)) / (2 * epsr)
    if sigma is not None:
        epsv = 0.01
        u_up, d_up = crr_ud(sigma + epsv, dt)
        u_dn, d_dn = crr_ud(max(sigma - epsv, 1e-6), dt)
    else:
        lam = 0.01
        u_up, d_up = u * math.exp(lam), d / math.exp(lam)
        u_dn, d_dn = u / math.exp(lam), d * math.exp(lam)
    vega = (P(S0, K, r, q, T, u_up, d_up) - P(S0, K, r, q, T, u_dn, d_dn)) / (
        2 * (0.01)
    )
    S_tree = stock_tree(S0, u, d, N)
    omega = delta * (S0 / price0) if price0 != 0 else float("inf")
    formula = (
        "Asian option (binomial, discrete paths):\n"
        "Arithmetic avg A = (S_0 + ... + S_N)/(N+1)\n"
        "Geometric  avg G = (Π S_t)^{1/(N+1)}\n"
        "Price = e^{−rT} Σ prob(path) * payoff(avg)\n"
        "Greeks via finite differences; omega = delta * S0 / Price"
    )
    return {
        "price": price0,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "omega": omega,
        "S_tree": S_tree,
        "u": u,
        "d": d,
        "dt": dt,
        "pu": risk_neutral_p(r, q, dt, u, d),
        "pd": 1 - risk_neutral_p(r, q, dt, u, d),
        "formula": formula,
    }


def asian_paths_dataframe(S0, K, r, q, T, N, u, d, option, kind="arith"):
    dt = T / N
    p = risk_neutral_p(r, q, dt, u, d)
    rows = []
    for idx, (path, prob) in enumerate(enumerate_paths(S0, u, d, N, p)):
        if kind == "arith":
            A = sum(path) / (N + 1)
        else:
            prod = 1.0
            for x in path:
                prod *= x
            A = prod ** (1.0 / (N + 1))
        rows.append(
            {
                "path_index": idx,
                "prob": prob,
                "average": A,
                "payoff": payoff(A, K, option),
                "discounted": math.exp(-r * T) * payoff(A, K, option),
                "prices": path,
            }
        )
    return pd.DataFrame(rows)


# ---------- Binary (digital) options: cash-or-nothing & asset-or-nothing (European) ----------


def digital_terminal_payoff(S, K, option, kind="cash", cashQ=1.0):
    # Threshold convention: ">" for call, "<" for put
    if kind == "cash":
        if option == "call":
            return cashQ if (S > K) else 0.0
        else:
            return cashQ if (S < K) else 0.0
    elif kind == "asset":
        if option == "call":
            return S if (S > K) else 0.0
        else:
            return S if (S < K) else 0.0
    else:
        raise ValueError("kind must be 'cash' or 'asset'")


def digital_tree_euro(S0, K, r, q, T, N, u, d, option, kind="cash", cashQ=1.0):
    dt = T / N
    S_tree = stock_tree(S0, u, d, N)
    V = [[0.0] * (i + 1) for i in range(N + 1)]
    for j, S in enumerate(S_tree[-1]):
        V[-1][j] = digital_terminal_payoff(S, K, option, kind, cashQ)
    p = risk_neutral_p(r, q, dt, u, d)
    if p < 0 or p > 1:
        raise ValueError("Arbitrage check failed: p not in [0,1].")
    disc = math.exp(-r * dt)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            V[i][j] = disc * (p * V[i + 1][j + 1] + (1 - p) * V[i + 1][j])
    price = V[0][0]
    return price, S_tree, V


def digital_greeks(
    S0, K, r, q, T, N, u=None, d=None, sigma=None, option="call", kind="cash", cashQ=1.0
):
    dt = T / N
    if (u is None or d is None) and sigma is None:
        raise ValueError("Provide sigma or both u and d for digital_greeks()")
    if u is None or d is None:
        u, d = crr_ud(sigma, dt)

    def P(S0_, K_, r_, q_, T_, u_, d_):
        price, *_ = digital_tree_euro(
            S0_, K_, r_, q_, T_, N, u_, d_, option, kind, cashQ
        )
        return price

    price0, S_tree, V_tree = digital_tree_euro(
        S0, K, r, q, T, N, u, d, option, kind, cashQ
    )

    epsS = 0.01 * S0 if S0 != 0 else 0.01
    delta = (P(S0 + epsS, K, r, q, T, u, d) - P(S0 - epsS, K, r, q, T, u, d)) / (
        2 * epsS
    )
    gamma = (
        P(S0 + epsS, K, r, q, T, u, d) - 2 * price0 + P(S0 - epsS, K, r, q, T, u, d)
    ) / (epsS**2)

    h = min(T / 100.0 if T > 0 else 1 / 365.0, 1 / 365.0)
    theta = (P(S0, K, r, q, T + h, u, d) - P(S0, K, r, q, max(T - h, 1e-6), u, d)) / (
        2 * h
    )

    epsr = 1e-4
    rho = (P(S0, K, r + epsr, q, T, u, d) - P(S0, K, r - epsr, q, T, u, d)) / (2 * epsr)

    if sigma is not None:
        epsv = 0.01
        u_up, d_up = crr_ud(sigma + epsv, dt)
        u_dn, d_dn = crr_ud(max(sigma - epsv, 1e-6), dt)
    else:
        lam = 0.01
        u_up, d_up = u * math.exp(lam), d / math.exp(lam)
        u_dn, d_dn = u / math.exp(lam), d * math.exp(lam)
    vega = (P(S0, K, r, q, T, u_up, d_up) - P(S0, K, r, q, T, u_dn, d_dn)) / (
        2 * (0.01)
    )

    omega = delta * (S0 / price0) if price0 != 0 else float("inf")

    formula = (
        "Binary option (European):\n"
        "Terminal payoff:\n"
        "  Cash-or-Nothing Call:  Q·1{S_T > K}\n"
        "  Cash-or-Nothing Put:   Q·1{S_T < K}\n"
        "  Asset-or-Nothing Call: S_T·1{S_T > K}\n"
        "  Asset-or-Nothing Put:  S_T·1{S_T < K}\n"
        "Backward induction (risk-neutral):\n"
        "  p = (e^{(r−q)Δt} − d)/(u−d),  V_{i,j} = e^{−rΔt}[ pV_{i+1,j+1} + (1−p)V_{i+1,j} ]\n"
        "Greeks via finite differences; omega = delta * S0 / Price\n"
        "Threshold convention uses strict > for calls and < for puts."
    )

    return {
        "price": price0,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "omega": omega,
        "S_tree": S_tree,
        "V_tree": V_tree,
        "u": u,
        "d": d,
        "dt": dt,
        "pu": risk_neutral_p(r, q, dt, u, d),
        "pd": 1 - risk_neutral_p(r, q, dt, u, d),
        "formula": formula,
    }


# ---------- Black–Scholes (for /bs tab) ----------


def bs_d1_d2(S0, K, r, q, T, sigma):
    if T <= 0 or sigma <= 0 or S0 <= 0 or K <= 0:
        raise ValueError("S0,K,sigma,T must be positive")
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_prices(S0, K, r, q, T, sigma):
    d1, d2 = bs_d1_d2(S0, K, r, q, T, sigma)
    Nd1, Nd2 = stdnorm_cdf(d1), stdnorm_cdf(d2)
    Nmd1, Nmd2 = stdnorm_cdf(-d1), stdnorm_cdf(-d2)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    call = disc_q * S0 * Nd1 - disc_r * K * Nd2
    put = disc_r * K * Nmd2 - disc_q * S0 * Nmd1
    delta_c = disc_q * Nd1
    delta_p = disc_q * (Nd1 - 1.0)
    omega_call = delta_c * (S0 / call) if call != 0 else float("inf")
    omega_put = delta_p * (S0 / put) if put != 0 else float("inf")
    formula = (
        "Black–Scholes (with q):\n"
        "d1 = [ln(S0/K) + (r − q + 0.5σ²)T]/(σ√T)\n"
        "d2 = d1 − σ√T\n"
        "Call = S0 e^{−qT} N(d1) − K e^{−rT} N(d2)\n"
        "Put  = K e^{−rT} N(−d2) − S0 e^{−qT} N(−d1)\n"
        "Delta_call = e^{−qT} N(d1)\n"
        "Delta_put  = e^{−qT}(N(d1) − 1)\n"
        "Omega_call = Delta_call * S0 / Call\n"
        "Omega_put  = Delta_put  * S0 / Put\n"
        "Parity: C − P = S0 e^{−qT} − K e^{−rT}"
    )
    return {
        "call": call,
        "put": put,
        "d1": d1,
        "d2": d2,
        "Nd1": Nd1,
        "Nd2": Nd2,
        "N(-d1)": Nmd1,
        "N(-d2)": Nmd2,
        "delta_call": delta_c,
        "delta_put": delta_p,
        "omega_call": omega_call,
        "omega_put": omega_put,
        "disc_r": disc_r,
        "disc_q": disc_q,
        "formula": formula,
    }


def bs_parity_call_from_put(S0, K, r, q, T, put_price):
    return put_price + S0 * math.exp(-q * T) - K * math.exp(-r * T)


def bs_parity_put_from_call(S0, K, r, q, T, call_price):
    return call_price - S0 * math.exp(-q * T) + K * math.exp(-r * T)
