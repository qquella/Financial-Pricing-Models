import base64
import io
import math

import matplotlib
import pandas as pd
from flask import Flask, make_response, render_template_string, request, send_file

matplotlib.use("Agg")

from binomial_pricer import (
    asian_greeks,
    asian_paths_dataframe,
    barrier_price_euro,
    bs_parity_call_from_put,
    bs_parity_put_from_call,
    bs_prices,
    build_frames,
    crr_ud,
    greeks,
    option_tree,
    option_tree_gap,
    payoff,
    plot_tree_stock_only,
    plot_tree_with_values,
    risk_neutral_p,
    stock_tree,
)

app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<title>Option Toolkit</title>
<h2>Financial Option Models Toolkit</h2>
<p>By Quella @2025</p>
<p>
  <a href="/">Binomial & Exotics</a> |
  <a href="/bs">Black–Scholes Parity</a>
</p>

<form method="post">
  <fieldset>
    <legend>Inputs</legend>
    S0: <input name="S0" value="{{S0}}">
    K: <input name="K" value="{{K}}">
    r: <input name="r" value="{{r}}">
    q: <input name="q" value="{{q}}">
    T: <input name="T" value="{{T}}">
    N: <input name="N" value="{{N}}"><br><br>

    Mode:
      <label><input type="radio" name="mode" value="ud" {% if mode=='ud' %}checked{% endif %}> u/d</label>
      <label><input type="radio" name="mode" value="sigma" {% if mode=='sigma' %}checked{% endif %}> sigma (CRR)</label><br>
    σ: <input name="sigma" value="{{sigma}}">
    u: <input name="u" value="{{u}}">
    d: <input name="d" value="{{d}}"><br><br>

    Payoff:
      <select name="payoff">
        <option value="vanilla" {% if payoff=='vanilla' %}selected{% endif %}>Vanilla (spot at T)</option>
        <option value="asian_arith" {% if payoff=='asian_arith' %}selected{% endif %}>Asian (Arithmetic; avg includes S0)</option>
        <option value="asian_geom" {% if payoff=='asian_geom' %}selected{% endif %}>Asian (Geometric; avg includes S0)</option>
        <option value="barrier" {% if payoff=='barrier' %}selected{% endif %}>Barrier (European)</option>
        <option value="gap" {% if payoff=='gap' %}selected{% endif %}>Gap (European)</option>
      </select><br><br>

    Barrier: type <select name="barrier_type">
      {% for t in ["up-and-out","down-and-out","up-and-in","down-and-in"] %}
        <option value="{{t}}" {% if barrier_type==t %}selected{% endif %}>{{t}}</option>
      {% endfor %}
    </select>
    H: <input name="H" value="{{H}}"><br><br>

    Gap: K_pay <input name="K_pay" value="{{K_pay}}">, K_trig <input name="K_trig" value="{{K_trig}}"><br><br>

    Option:
      <label><input type="radio" name="option" value="call" {% if option=='call' %}checked{% endif %}> Call</label>
      <label><input type="radio" name="option" value="put" {% if option=='put' %}checked{% endif %}> Put</label>
    Exercise:
      <label><input type="radio" name="exercise" value="european" {% if exercise=='european' %}checked{% endif %}> European</label>
      <label><input type="radio" name="exercise" value="american" {% if exercise=='american' %}checked{% endif %}> American</label><br><br>
    <button type="submit">Compute</button>
  </fieldset>
</form>

{% if result %}
<h3>Results</h3>
<pre>{{result}}</pre>

<h3>Tree</h3>
<img src="data:image/png;base64,{{image_b64}}" alt="tree">

<form method="post" action="/download/csv">
  {{hidden_inputs|safe}}
  <button type="submit">Download CSV</button>
</form>
<form method="post" action="/download/xlsx">
  {{hidden_inputs|safe}}
  <button type="submit">Download Excel</button>
</form>
{% endif %}
"""

TEMPLATE_BS = """
<!doctype html>
<title>Financial Option Models Toolkit: Black–Scholes Parity</title>
<h2>Black–Scholes Put–Call Parity</h2>
<p>By Quella @2025</p>
<p>
  <a href="/">Binomial & Exotics</a> |
  <a href="/bs">Black–Scholes Parity</a>
</p>
<form method="post">
  <fieldset>
    <legend>Inputs</legend>
    S0: <input name="S0" value="{{S0}}">
    K: <input name="K" value="{{K}}">
    r: <input name="r" value="{{r}}">
    q: <input name="q" value="{{q}}">
    T: <input name="T" value="{{T}}">
    σ: <input name="sigma" value="{{sigma}}"><br><br>
    Option:
      <label><input type="radio" name="option" value="call" {% if option=='call' %}checked{% endif %}> Call</label>
      <label><input type="radio" name="option" value="put" {% if option=='put' %}checked{% endif %}> Put</label>
    <button type="submit">Compute</button>
  </fieldset>
</form>

{% if result %}
<h3>Results</h3>
<pre>{{result}}</pre>
{% endif %}
"""


def hidden_fields(f):
    return "\n".join(
        f'<input type="hidden" name="{k}" value="{v}">' for k, v in f.items()
    )


def compute_from_form(f):
    S0 = float(f.get("S0", 100))
    K = float(f.get("K", 100))
    r = float(f.get("r", 0.05))
    q = float(f.get("q", 0.0))
    T = float(f.get("T", 2))
    N = int(float(f.get("N", 2)))
    mode = f.get("mode", "ud")
    option = f.get("option", "call")
    exercise = f.get("exercise", "european")
    payoff_style = f.get("payoff", "vanilla")
    sigma = f.get("sigma", "0.2")
    u = f.get("u", "1.1")
    d = f.get("d", "0.9")
    sigma = float(sigma) if sigma not in ("", None) else None
    u = float(u) if u not in ("", None) else None
    d = float(d) if d not in ("", None) else None
    H = float(f.get("H", "120"))
    btype = f.get("barrier_type", "up-and-out")
    K_pay = float(f.get("K_pay", "100"))
    K_trig = float(f.get("K_trig", "110"))

    if mode == "sigma":
        u, d = crr_ud(sigma, T / N)

    if payoff_style == "vanilla":
        g = greeks(S0, K, r, q, T, N, u=u, d=d, option=option, exercise=exercise)
        df = build_frames(g["S_tree"], g["V_tree"])
        buf = io.BytesIO()
        plot_tree_with_values(
            g["S_tree"],
            g["V_tree"],
            f"Stock/Option tree (N={N})",
            buf,
            option_color="red",
        )
        buf.seek(0)
        image_b64 = base64.b64encode(buf.read()).decode("ascii")
        name = f"Vanilla {exercise} {option}"
        result = []
        result.append(name)
        result.append(f"Price: {g['price']:.6f}")
        result.append(
            f"u={g['u']:.6f}, d={g['d']:.6f}, dt={g['dt']:.6f}, p_u={g['pu']:.6f}, p_d={g['pd']:.6f}"
        )
        result.append(
            f"Delta={g['delta']:.6f}, Gamma={g['gamma']:.6f}, Theta/yr={g['theta']:.6f}, Vega≈per 1%={g['vega']:.6f}, Rho={g['rho']:.6f}"
        )
        return g, df, "\n".join(result), image_b64

    elif payoff_style in ("asian_arith", "asian_geom"):
        kind = "arith" if payoff_style == "asian_arith" else "geom"
        g = asian_greeks(
            S0, K, r, q, T, N, u=u, d=d, sigma=sigma, option=option, kind=kind
        )
        df = asian_paths_dataframe(S0, K, r, q, T, N, u, d, option, kind)
        buf = io.BytesIO()
        plot_tree_stock_only(g["S_tree"], f"Stock tree (Asian; N={N})", buf)
        buf.seek(0)
        image_b64 = base64.b64encode(buf.read()).decode("ascii")
        nm = "Arithmetic" if kind == "arith" else "Geometric"
        result = [
            f"Asian ({nm}) European {option}",
            f"Price: {g['price']:.6f}",
            f"u={g['u']:.6f}, d={g['d']:.6f}, dt={g['dt']:.6f}, p_u={g['pu']:.6f}, p_d={g['pd']:.6f}",
            f"Delta={g['delta']:.6f}, Gamma={g['gamma']:.6f}, Theta/yr={g['theta']:.6f}, Vega≈per 1%={g['vega']:.6f}, Rho={g['rho']:.6f}",
            "(Node option values are path-dependent; tree shows S only.)",
        ]
        return g, df, "\n".join(result), image_b64

    elif payoff_style == "barrier":
        price, S_tree, V_tree, kind = barrier_price_euro(
            S0, K, r, q, T, N, u, d, option, btype, H
        )
        if kind == "out":
            df = build_frames(S_tree, V_tree)
            buf = io.BytesIO()
            plot_tree_with_values(
                S_tree, V_tree, f"Barrier {btype} (N={N})", buf, option_color="red"
            )
            buf.seek(0)
            image_b64 = base64.b64encode(buf.read()).decode("ascii")
            result = [
                f"Barrier {btype} European {option}",
                f"H={H}",
                f"Price: {price:.6f}",
                "(Knocked-out nodes show 0).",
            ]
        else:
            df = pd.DataFrame()  # nothing useful nodewise
            buf = io.BytesIO()
            plot_tree_stock_only(S_tree, f"Stock tree (Barrier knock-in; N={N})", buf)
            buf.seek(0)
            image_b64 = base64.b64encode(buf.read()).decode("ascii")
            result = [
                f"Barrier {btype} European {option}",
                f"H={H}",
                f"Price: {price:.6f}",
                "(Knock-in via parity: IN = Vanilla − OUT). Node values omitted.",
            ]
        return {"price": price, "S_tree": S_tree}, df, "\n".join(result), image_b64

    elif payoff_style == "gap":
        S_tree = stock_tree(S0, u, d, N)
        dt = T / N
        V_tree = option_tree_gap(S_tree, K_pay, K_trig, r, q, dt, u, d, option)
        df = build_frames(S_tree, V_tree)
        buf = io.BytesIO()
        plot_tree_with_values(
            S_tree, V_tree, f"Gap {option} (N={N})", buf, option_color="red"
        )
        buf.seek(0)
        image_b64 = base64.b64encode(buf.read()).decode("ascii")
        result = [
            f"Gap European {option}",
            f"K_pay={K_pay}, K_trig={K_trig}",
            f"Price: {V_tree[0][0]:.6f}",
        ]
        result = result + "\n\nFormulas used:\n" + formula
        return (
            {"price": V_tree[0][0], "S_tree": S_tree},
            df,
            "\n".join(result),
            image_b64,
        )

    else:
        raise ValueError("Unknown payoff style")


@app.route("/", methods=["GET", "POST"])
def index():
    defaults = {
        "S0": "100",
        "K": "100",
        "r": "0.05",
        "q": "0.0",
        "T": "2",
        "N": "2",
        "mode": "ud",
        "sigma": "0.2",
        "u": "1.1",
        "d": "0.9",
        "option": "call",
        "exercise": "european",
        "payoff": "vanilla",
        "H": "120",
        "barrier_type": "up-and-out",
        "K_pay": "100",
        "K_trig": "110",
    }
    f = defaults.copy()
    if request.method == "POST":
        f.update(request.form)
        g, df, result, image_b64 = compute_from_form(f)
        return render_template_string(
            TEMPLATE,
            **f,
            result=result,
            image_b64=image_b64,
            hidden_inputs=hidden_fields(f),
        )
    else:
        return render_template_string(
            TEMPLATE, **f, result=None, image_b64=None, hidden_inputs=""
        )


@app.route("/bs", methods=["GET", "POST"])
def bs_page():
    defaults = {
        "S0": "100",
        "K": "100",
        "r": "0.05",
        "q": "0.0",
        "T": "1",
        "sigma": "0.2",
        "option": "call",
    }
    f = defaults.copy()
    if request.method == "POST":
        f.update(request.form)
        S0 = float(f["S0"])
        K = float(f["K"])
        r = float(f["r"])
        q = float(f["q"])
        T = float(f["T"])
        sigma = float(f["sigma"])
        side = f.get("option", "call")
        res = bs_prices(S0, K, r, q, T, sigma)
        call_bs, put_bs = res["call"], res["put"]
        d1, d2, Nd1, Nd2 = res["d1"], res["d2"], res["Nd1"], res["Nd2"]
        delta_c, delta_p = res["delta_call"], res["delta_put"]
        disc_r, disc_q = res["disc_r"], res["disc_q"]
        call_par = bs_parity_call_from_put(S0, K, r, q, T, put_bs)
        put_par = bs_parity_put_from_call(S0, K, r, q, T, call_bs)
        lines = []
        lines.append("Black–Scholes (d1,d2) + Parity check\n")
        lines.append(f"d1 = {d1:.6f},  d2 = {d2:.6f}")
        lines.append(f"N(d1) = {Nd1:.6f},  N(d2) = {Nd2:.6f}")
        lines.append(f"Delta_call = {delta_c:.6f},  Delta_put = {delta_p:.6f}")
        lines.append(
            f"disc_r = e^(-rT) = {disc_r:.6f},  disc_q = e^(-qT) = {disc_q:.6f}\n"
        )
        lines.append(f"BS Call (d1,d2) = {call_bs:.6f}")
        lines.append(f"BS Put  (d1,d2) = {put_bs:.6f}")
        lines.append(
            f"Call via Parity (using Put) = {call_par:.6f}  -> diff {call_par - call_bs:+.6e}"
        )
        lines.append(
            f"Put  via Parity (using Call) = {put_par:.6f}  -> diff {put_par - put_bs:+.6e}"
        )
        lines.append(
            f"\nPut–Call Parity: C − P ?= S0 e^(−qT) − K e^(−rT)  -> {call_bs - put_bs:.6f}  vs  {S0*disc_q - K*disc_r:.6f}"
        )
        return render_template_string(TEMPLATE_BS, **f, result="\n".join(lines))
    else:
        return render_template_string(TEMPLATE_BS, **f, result=None)


@app.post("/download/csv")
def dl_csv():
    f = request.form.to_dict()
    g, df, *_ = compute_from_form(f)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    resp = make_response(buf.getvalue())
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=binomial_tree.csv"
    return resp


@app.post("/download/xlsx")
def dl_xlsx():
    f = request.form.to_dict()
    g, df, *_ = compute_from_form(f)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name=(
                "Tree"
                if f.get("payoff", "vanilla") in ("vanilla", "barrier", "gap")
                else "Paths"
            ),
        )
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="binomial_tree.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
