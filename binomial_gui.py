import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use("TkAgg")
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from binomial_pricer import (
    asian_greeks,
    asian_paths_dataframe,
    asian_price,
    barrier_price_euro,
    bs_parity_call_from_put,
    bs_parity_put_from_call,
    bs_prices,
    build_frames,
    crr_ud,
    export_frames_to_files,
    greeks,
    option_tree,
    option_tree_gap,
    payoff,
    plot_tree_stock_only,
    plot_tree_with_values,
    price_once,
    risk_neutral_p,
    stock_tree,
)

APP_TITLE = "Option Toolkit (Binomial + BS Parity)"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1150x760")
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        self.binomial_tab = ttk.Frame(nb)
        nb.add(self.binomial_tab, text="Binomial & Exotics")
        self.bs_tab = ttk.Frame(nb)
        nb.add(self.bs_tab, text="Black–Scholes Parity")

        self._build_binomial(self.binomial_tab)
        self._build_bs(self.bs_tab)

    # -------- Binomial tab --------
    def _build_binomial(self, root):
        left = ttk.Frame(root, padding=12)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(root, padding=12)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mode
        mode_frame = ttk.LabelFrame(left, text="Parameterization", padding=8)
        mode_frame.pack(fill=tk.X, pady=(0, 8))
        self.mode_var = tk.StringVar(value="ud")
        ttk.Radiobutton(
            mode_frame,
            text="Use u/d",
            variable=self.mode_var,
            value="ud",
            command=lambda: self._set_mode("ud"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(
            mode_frame,
            text="Use σ (CRR)",
            variable=self.mode_var,
            value="sigma",
            command=lambda: self._set_mode("sigma"),
        ).pack(side=tk.LEFT)

        # Inputs
        inputs = ttk.LabelFrame(left, text="Inputs", padding=8)
        inputs.pack(fill=tk.X)
        self._ent = {}

        def add_row(label, key, default, width=12):
            row = ttk.Frame(inputs)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=16, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            e = ttk.Entry(row, textvariable=var, width=width)
            e.pack(side=tk.LEFT)
            self._ent[key] = (e, var)

        add_row("S0", "S0", 100)
        add_row("K", "K", 100)
        add_row("r (cont.)", "r", 0.05)
        add_row("q (yield)", "q", 0.0)
        add_row("T (years)", "T", 2)
        add_row("N (steps)", "N", 2)
        add_row("σ (vol)", "sigma", 0.2)
        add_row("u", "u", 1.1)
        add_row("d", "d", 0.9)

        # Payoff style
        style = ttk.LabelFrame(left, text="Payoff", padding=8)
        style.pack(fill=tk.X, pady=(8, 0))
        self.payoff_var = tk.StringVar(value="vanilla")
        for text, val in [
            ("Vanilla (spot at T)", "vanilla"),
            ("Asian (Arithmetic)", "asian_arith"),
            ("Asian (Geometric)", "asian_geom"),
            ("Barrier (European)", "barrier"),
            ("Gap (European)", "gap"),
        ]:
            ttk.Radiobutton(
                style,
                text=text,
                variable=self.payoff_var,
                value=val,
                command=self._payoff_changed,
            ).pack(anchor="w")

        # Type + Exercise
        ty = ttk.LabelFrame(left, text="Type", padding=8)
        ty.pack(fill=tk.X, pady=(8, 0))
        self.opt_var = tk.StringVar(value="call")
        ttk.Radiobutton(ty, text="Call", variable=self.opt_var, value="call").pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Radiobutton(ty, text="Put", variable=self.opt_var, value="put").pack(
            side=tk.LEFT
        )
        ex = ttk.LabelFrame(left, text="Exercise", padding=8)
        ex.pack(fill=tk.X, pady=(8, 8))
        self.ex_var = tk.StringVar(value="european")
        self.rb_eur = ttk.Radiobutton(
            ex, text="European", variable=self.ex_var, value="european"
        )
        self.rb_am = ttk.Radiobutton(
            ex, text="American", variable=self.ex_var, value="american"
        )
        self.rb_eur.pack(side=tk.LEFT, padx=(0, 8))
        self.rb_am.pack(side=tk.LEFT)

        # Barrier controls
        barr = ttk.LabelFrame(left, text="Barrier params", padding=8)
        barr.pack(fill=tk.X, pady=(4, 0))
        self.barrier_type = tk.StringVar(value="up-and-out")
        ttk.Label(barr, text="Type").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            barr,
            textvariable=self.barrier_type,
            values=["up-and-out", "down-and-out", "up-and-in", "down-and-in"],
            width=14,
            state="readonly",
        ).grid(row=0, column=1)
        ttk.Label(barr, text="H (barrier)").grid(row=1, column=0, sticky="w")
        self.ent_H = ttk.Entry(barr, width=12)
        self.ent_H.insert(0, "120")
        self.ent_H.grid(row=1, column=1)

        # Gap controls
        gap = ttk.LabelFrame(left, text="Gap params", padding=8)
        gap.pack(fill=tk.X, pady=(4, 8))
        ttk.Label(gap, text="K_pay").grid(row=0, column=0, sticky="w")
        self.ent_Kpay = ttk.Entry(gap, width=12)
        self.ent_Kpay.insert(0, "100")
        self.ent_Kpay.grid(row=0, column=1)
        ttk.Label(gap, text="K_trig").grid(row=1, column=0, sticky="w")
        self.ent_Ktrg = ttk.Entry(gap, width=12)
        self.ent_Ktrg.insert(0, "110")
        self.ent_Ktrg.grid(row=1, column=1)

        # Buttons
        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btns, text="Compute", command=self.compute_binom).pack(side=tk.LEFT)
        ttk.Button(btns, text="Save Tree…", command=self.save_tree).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(btns, text="Export CSV/XLSX…", command=self.export_data).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(btns, text="Reset to Sample", command=self._load_defaults).pack(
            side=tk.LEFT
        )

        self.status = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status, foreground="#555").pack(
            fill=tk.X, pady=(8, 0)
        )

        # Outputs
        self.results_box = tk.Text(right, height=16, width=100)
        self.results_box.pack(fill=tk.X)
        fig_frame = ttk.LabelFrame(right, text="Tree", padding=8)
        fig_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._set_mode("ud")
        self._payoff_changed()

    def _set_mode(self, mode):
        sigma_state = "normal" if mode == "sigma" else "disabled"
        ud_state = "disabled" if mode == "sigma" else "normal"
        for key in ("sigma",):
            self._ent[key][0].configure(state=sigma_state)
        for key in ("u", "d"):
            self._ent[key][0].configure(state=ud_state)
        self.mode_var.set(mode)

    def _payoff_changed(self):
        pf = self.payoff_var.get()
        if pf in ("asian_arith", "asian_geom"):
            self.ex_var.set("european")
            self.rb_am.configure(state="disabled")
        elif pf in ("barrier", "gap"):
            self.ex_var.set("european")
            self.rb_am.configure(state="disabled")
        else:
            self.rb_am.configure(state="normal")

    def _load_defaults(self):
        defaults = {
            "S0": 100,
            "K": 100,
            "r": 0.05,
            "q": 0.0,
            "T": 2,
            "N": 2,
            "sigma": 0.2,
            "u": 1.1,
            "d": 0.9,
        }
        for k, v in defaults.items():
            self._ent[k][1].set(str(v))
        self.opt_var.set("call")
        self.ex_var.set("european")
        self.payoff_var.set("vanilla")
        self.barrier_type.set("up-and-out")
        self.ent_H.delete(0, tk.END)
        self.ent_H.insert(0, "120")
        self.ent_Kpay.delete(0, tk.END)
        self.ent_Kpay.insert(0, "100")
        self.ent_Ktrg.delete(0, tk.END)
        self.ent_Ktrg.insert(0, "110")
        self._set_mode("ud")
        self._payoff_changed()
        self.status.set("Loaded sample values.")

    def _parse_common(self):
        S0 = float(self._ent["S0"][1].get())
        K = float(self._ent["K"][1].get())
        r = float(self._ent["r"][1].get())
        q = float(self._ent["q"][1].get())
        T = float(self._ent["T"][1].get())
        N = int(float(self._ent["N"][1].get()))
        if N <= 0:
            raise ValueError("N must be positive")
        if self.mode_var.get() == "sigma":
            sigma = float(self._ent["sigma"][1].get())
            u, d = crr_ud(sigma, T / N)
        else:
            u = float(self._ent["u"][1].get())
            d = float(self._ent["d"][1].get())
            sigma = None
        return S0, K, r, q, T, N, u, d, sigma

    def compute_binom(self):
        try:
            S0, K, r, q, T, N, u, d, sigma = self._parse_common()
            option = self.opt_var.get()
            exercise = self.ex_var.get()
            style = self.payoff_var.get()

            if style == "vanilla":
                g = greeks(
                    S0, K, r, q, T, N, u=u, d=d, option=option, exercise=exercise
                )
                S_tree = g["S_tree"]
                V_tree = g["V_tree"]
                txt = []
                txt += [f"Vanilla {exercise} {option}", f"Price: {g['price']:.6f}"]
                txt += [
                    f"u={g['u']:.6f}, d={g['d']:.6f}, dt={g['dt']:.6f}, p_u={g['pu']:.6f}, p_d={g['pd']:.6f}"
                ]
                txt += [
                    f"Delta={g['delta']:.6f}, Gamma={g['gamma']:.6f}, Theta/yr={g['theta']:.6f}, Vega≈per 1%={g['vega']:.6f}, Rho={g['rho']:.6f}"
                ]
                txt += [f"Omega={g['omega']:.6f}  (omega = delta * S0 / price)"]
                txt += ["", "Formulas used:", g["formula"]]
                self._render_tree(S_tree, V_tree, N, txt, show_values=True)

            elif style in ("asian_arith", "asian_geom"):
                kind = "arith" if style == "asian_arith" else "geom"
                g = asian_greeks(
                    S0, K, r, q, T, N, u=u, d=d, sigma=sigma, option=option, kind=kind
                )
                S_tree = g["S_tree"]
                txt = []
                nm = "Arithmetic" if kind == "arith" else "Geometric"
                txt += [
                    f"Asian ({nm}) European {option} (avg includes S0..S_N)",
                    f"Price: {g['price']:.6f}",
                    f"u={g['u']:.6f}, d={g['d']:.6f}, dt={g['dt']:.6f}, p_u={g['pu']:.6f}, p_d={g['pd']:.6f}",
                    f"Delta={g['delta']:.6f}, Gamma={g['gamma']:.6f}, Theta/yr={g['theta']:.6f}, Vega≈per 1%={g['vega']:.6f}, Rho={g['rho']:.6f}",
                    "(Node option values are path-dependent and omitted from the tree.)",
                ]
                txt += [f"Omega={g['omega']:.6f}  (omega = delta * S0 / price)"]
                txt += ["", "Formulas used:", g["formula"]]
                self._render_tree(S_tree, None, N, txt, show_values=False, asian=True)
                self._last_df = asian_paths_dataframe(
                    S0, K, r, q, T, N, u, d, option, kind
                )
                self._last_mode = "asian"

            elif style == "barrier":
                H = float(self.ent_H.get())
                btype = self.barrier_type.get()
                price, S_tree, V_tree, kind, formula = barrier_price_euro(
                    S0, K, r, q, T, N, u, d, option, btype, H
                )
                txt = [
                    f"Barrier {btype} European {option}",
                    f"H={H}",
                    f"Price: {price:.6f}",
                ]
                txt += ["", "Formulas used:", formula]
                if kind == "out":
                    dt = T / N
                    pu = risk_neutral_p(r, q, dt, u, d)
                    pd = 1 - pu
                    txt += [
                        f"u={u:.6f}, d={d:.6f}, dt={dt:.6f}, p_u={pu:.6f}, p_d={pd:.6f}",
                        "(Red numbers are option values; knocked-out nodes show 0)",
                    ]
                    self._render_tree(S_tree, V_tree, N, txt, show_values=True)
                    self._last_df = build_frames(S_tree, V_tree)
                    self._last_mode = "vanilla"
                else:
                    txt += [
                        "(Knock-in price computed via in–out parity: IN = Vanilla − OUT)",
                        "(Node values are path-dependent and omitted.)",
                    ]
                    self._render_tree(S_tree, None, N, txt, show_values=False)
                    self._last_df = None
                    self._last_mode = "asian"

            elif style == "gap":
                K_pay = float(self.ent_Kpay.get())
                K_trg = float(self.ent_Ktrg.get())
                S_tree = stock_tree(S0, u, d, N)
                dt = T / N
                V_tree = option_tree_gap(S_tree, K_pay, K_trg, r, q, dt, u, d, option)
                pu = risk_neutral_p(r, q, dt, u, d)
                pd = 1 - pu
                txt = [
                    f"Gap European {option}",
                    f"K_pay={K_pay}, K_trig={K_trg}",
                    f"Price: {V_tree[0][0]:.6f}",
                    f"u={u:.6f}, d={d:.6f}, dt={dt:.6f}, p_u={pu:.6f}, p_d={pd:.6f}",
                ]
                txt += [
                    "",
                    "Formulas used:",
                    "Gap payoff (call): (S_T − K_pay) if S_T > K_trig else 0",
                    "Gap payoff (put):  (K_pay − S_T) if S_T < K_trig else 0",
                    "Then discounted backward in binomial tree like European.",
                ]
                self._render_tree(S_tree, V_tree, N, txt, show_values=True)
                self._last_df = build_frames(S_tree, V_tree)
                self._last_mode = "vanilla"

            else:
                raise ValueError("Unknown payoff style")

            self.status.set("Computed successfully.")
        except Exception as e:
            self.status.set(f"Error: {e}")
            messagebox.showerror("Error", str(e))

    def _render_tree(self, S_tree, V_tree, N, lines, show_values=True, asian=False):
        self.results_box.delete("1.0", tk.END)
        self.results_box.insert("1.0", "\n".join(lines))
        self.ax.clear()
        for i, level in enumerate(S_tree[:-1]):
            for j, _ in enumerate(level):
                x0, y0 = i, 2 * j - i
                x1, y1 = i + 1, 2 * (j + 1) - (i + 1)
                x2, y2 = i + 1, 2 * j - (i + 1)
                self.ax.plot([x0, x1], [y0, y1])
                self.ax.plot([x0, x2], [y0, y2])
        for i, level in enumerate(S_tree):
            for j, S in enumerate(level):
                if show_values and V_tree is not None:
                    self.ax.text(
                        i + 0.02, 2 * j - i + 0.22, f"{S:.2f}", va="center", fontsize=10
                    )
                    self.ax.text(
                        i + 0.02,
                        2 * j - i - 0.18,
                        f"({V_tree[i][j]:.2f})",
                        va="center",
                        fontsize=9,
                        color="red",
                    )
                else:
                    self.ax.text(
                        i + 0.02, 2 * j - i, f"{S:.2f}", va="center", fontsize=10
                    )
        ttl = "Stock/Option tree" if show_values else "Stock tree"
        self.ax.set_title(f"{ttl} (N={N})")
        self.ax.axis("off")
        self.canvas.draw()

    def save_tree(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All files", "*.*")],
            title="Save tree image",
        )
        if not path:
            return
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        self.status.set(f"Saved tree to: {path}")

    def export_data(self):
        try:
            df = getattr(self, "_last_df", None)
            if df is None:
                raise RuntimeError("No data to export for this payoff selection")
            base = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                title="Export CSV/XLSX (choose CSV name)",
            )
            if not base:
                return
            csv_path = base
            xlsx_path = base.rsplit(".", 1)[0] + ".xlsx"
            if self._last_mode == "vanilla":
                from binomial_pricer import export_frames_to_files

                export_frames_to_files(df, csv_path, xlsx_path)
            else:
                df.to_csv(csv_path, index=False)
                with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Paths")
            self.status.set(f"Saved {csv_path} and {xlsx_path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    # -------- Black–Scholes tab --------
    def _build_bs(self, root):
        frm = ttk.Frame(root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(frm)
        top.pack(fill=tk.X)
        self.bs_ent = {}

        def add(label, key, default):
            r = ttk.Frame(top)
            r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=label, width=18, anchor="w").pack(side=tk.LEFT)
            v = tk.StringVar(value=str(default))
            e = ttk.Entry(r, textvariable=v, width=14)
            e.pack(side=tk.LEFT)
            self.bs_ent[key] = (e, v)

        add("S0", "S0", 100)
        add("K", "K", 100)
        add("r (cont.)", "r", 0.05)
        add("q (yield)", "q", 0.0)
        add("T (years)", "T", 1.0)
        add("σ (vol)", "sigma", 0.2)

        self.bs_type = tk.StringVar(value="call")
        typ = ttk.LabelFrame(frm, text="Option", padding=8)
        typ.pack(fill=tk.X, pady=(6, 0))
        ttk.Radiobutton(typ, text="Call", variable=self.bs_type, value="call").pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Radiobutton(typ, text="Put", variable=self.bs_type, value="put").pack(
            side=tk.LEFT
        )

        ttk.Button(frm, text="Compute (BS + Parity)", command=self.compute_bs).pack(
            pady=8
        )

        self.bs_box = tk.Text(frm, height=20, width=110)
        self.bs_box.pack(fill=tk.BOTH, expand=True)

    def compute_bs(self):
        try:
            S0 = float(self.bs_ent["S0"][1].get())
            K = float(self.bs_ent["K"][1].get())
            r = float(self.bs_ent["r"][1].get())
            q = float(self.bs_ent["q"][1].get())
            T = float(self.bs_ent["T"][1].get())
            sigma = float(self.bs_ent["sigma"][1].get())
            side = self.bs_type.get()

            res = bs_prices(S0, K, r, q, T, sigma)
            call_bs, put_bs = res["call"], res["put"]
            d1, d2, Nd1, Nd2 = res["d1"], res["d2"], res["Nd1"], res["Nd2"]
            delta_c, delta_p = res["delta_call"], res["delta_put"]
            disc_r, disc_q = res["disc_r"], res["disc_q"]
            # Parity method
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
                f"\nPut–Call Parity should hold: C − P ?= S0 e^(−qT) − K e^(−rT)  -> {call_bs - put_bs:.6f}  vs  {S0*disc_q - K*disc_r:.6f}"
            )
            lines.append("")
            lines.append("Formulas used:")
            lines.append(res["formula"])
            self.bs_box.delete("1.0", tk.END)
            self.bs_box.insert("1.0", "\n".join(lines))
        except Exception as e:
            messagebox.showerror("BS Error", str(e))


if __name__ == "__main__":
    App().mainloop()
