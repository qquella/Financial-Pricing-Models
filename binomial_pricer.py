import io
import itertools
import math

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Common utilities ----------


def stdnorm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def stdnorm_pdf(x):
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


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
    # European gap at maturity (no early exercise)
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


# ---------- Barrier (European) with discrete monitoring ----------
# Uses "up/down" hit thresholds in terms of # of ups/downs, so each node is either fully alive or fully knocked-out.


def _ups_to_hit(S0, H, u):
    if u <= 1.0:
        raise ValueError("u must be > 1 for up-barrier calculations")
    if S0 >= H:
        return 0
    # ceil(log_u(H/S0))
    return int(math.ceil(math.log(H / S0, u)))


def _downs_to_hit(S0, H, d):
    if d >= 1.0:
        raise ValueError("d must be < 1 for down-barrier calculations")
    if S0 <= H:
        return 0
    # ceil(log_d(H/S0)) with d<1
    return int(math.ceil(math.log(H / S0, d)))


def barrier_out_tree(S0, K, r, q, T, N, u, d, option, barrier_type, H):
    dt = T / N
    S = stock_tree(S0, u, d, N)
    V = [[0.0] * (i + 1) for i in range(N + 1)]
    p = risk_neutral_p(r, q, dt, u, d)
    disc = math.exp(-r * dt)

    if barrier_type == "up-and-out":
        k = _ups_to_hit(S0, H, u)
        # terminal
        for j, St in enumerate(S[-1]):
            if j >= k:  # barrier touched sometime
                V[-1][j] = 0.0
            else:
                V[-1][j] = payoff(St, K, option)
        # backward
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                if j >= k:
                    V[i][j] = 0.0
                else:
                    V[i][j] = disc * (p * V[i + 1][j + 1] + (1 - p) * V[i + 1][j])

    elif barrier_type == "down-and-out":
        m = _downs_to_hit(S0, H, d)
        # At time i with j ups => downs = i-j
        for j, St in enumerate(S[-1]):
            downs = N - j
            if downs >= m:
                V[-1][j] = 0.0
            else:
                V[-1][j] = payoff(St, K, option)
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
    # For "in" types, use in-out parity: IN = vanilla - OUT (discrete-monitoring consistent with our OUT)
    dt = T / N
    if barrier_type in ("up-and-out", "down-and-out"):
        S_tree, V_out = barrier_out_tree(
            S0, K, r, q, T, N, u, d, option, barrier_type, H
        )
        price = V_out[0][0]
        return price, S_tree, V_out, "out"
    elif barrier_type in ("up-and-in", "down-and-in"):
        # vanilla
        S_tree = stock_tree(S0, u, d, N)
        V_van = option_tree(S_tree, K, r, q, dt, u, d, option, "european")
        # out component
        out_type = "up-and-out" if "up" in barrier_type else "down-and-out"
        _, V_out = barrier_out_tree(S0, K, r, q, T, N, u, d, option, out_type, H)
        price = V_van[0][0] - V_out[0][0]
        return price, S_tree, None, "in"
    else:
        raise ValueError("Unknown barrier_type")


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


# ---------- Greeks wrappers ----------


def price_once(S0, K, r, q, T, N, u, d, option, exercise):
    dt = T / N
    p = risk_neutral_p(r, q, dt, u, d)
    if p < 0 or p > 1:
        raise ValueError("Arbitrage check failed (p not in [0,1]).")
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

    return {
        "price": price0,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "S_tree": S_tree,
        "V_tree": V_tree,
        "u": u,
        "d": d,
        "dt": dt,
        "pu": risk_neutral_p(r, q, dt, u, d),
        "pd": 1 - risk_neutral_p(r, q, dt, u, d),
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
    if p < 0 or p > 1:
        raise ValueError("Arbitrage check failed (p not in [0,1]).")
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
    return {
        "price": price0,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "S_tree": S_tree,
        "u": u,
        "d": d,
        "dt": dt,
        "pu": risk_neutral_p(r, q, dt, u, d),
        "pd": 1 - risk_neutral_p(r, q, dt, u, d),
    }


def asian_paths_dataframe(S0, K, r, q, T, N, u, d, option, kind="arith"):
    """Return all paths with prob, average, and payoff (useful for export)."""
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
        "disc_r": disc_r,
        "disc_q": disc_q,
    }


def bs_parity_call_from_put(S0, K, r, q, T, put_price):
    return put_price + S0 * math.exp(-q * T) - K * math.exp(-r * T)


def bs_parity_put_from_call(S0, K, r, q, T, call_price):
    return call_price - S0 * math.exp(-q * T) + K * math.exp(-r * T)
