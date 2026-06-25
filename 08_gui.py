# -*- coding: utf-8 -*-
"""
FA-GPC Compressive Strength Predictor
Model: CatBoost · Boruta-selected features · 90% CV Prediction Interval
Dataset: 145 samples · 11 literature sources
"""

import os, threading
import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import messagebox
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'Data_FAGPC.xlsx')
BORUTA_CSV = os.path.join(BASE_DIR, 'results', 'tables', 'table_boruta.csv')

SEED = 0

# ── Colour palette (aligned with paper figures) ────────────────────────────────
C = {
    'header':   '#1A3A5C',        # deep navy — main brand colour
    'g_fa':     '#1A3A5C',        # fly ash source group
    'g_act':    '#16A085',        # alkali activator group
    'g_agg':    '#7D3C98',        # aggregate & curing group
    'fc':       '#C0392B',        # result card accent
    'btn_p':    '#1A3A5C', 'btn_ph': '#2C4F7C',
    'btn_e':    '#D4831A', 'btn_eh': '#B8690E',
    'btn_c':    '#546E7A', 'btn_ch': '#37474F',
    'bg':       '#F5F6FA',
    'card':     '#FFFFFF',
    'border':   '#DDE1E9',
    'secondary':'#4A6572',
    'unit':     '#546E7A',
    'auto_bg':  '#EEF1F5',
    'auto_txt': '#546E7A',
    'annot':    '#607D8B',
    'idle':     '#90A4AE',
    'warn':     '#E65100',
}

# ── Example values (training medians) ─────────────────────────────────────────
EXAMPLE = {
    'FA': '444.7', 'SiO2': '51.1', 'Al2O3': '18.2',
    'Coarse': '1102.8', 'Fine': '594.0',
    'NaOH': '75.3', 'NaOH_M': '12',
    'Water': '143.8', 'Temp': '0',
}

# ── Training ranges for out-of-range warning ───────────────────────────────────
RANGES = {
    'FA':      (254.5, 600.0),  'SiO2':  (36.2, 75.66),
    'Al2O3':   (9.2,   31.25),  'Coarse':(554.0, 1684.0),
    'Fine':    (500.0, 706.0),  'NaOH':  (11.78, 198.0),
    'NaOH_M':  (8,     20),     'Water': (37.4,  206.8),
    'Temp':    (0,     100),
}


# ── Grade helper ───────────────────────────────────────────────────────────────
def _fc_grade(v: float) -> str:
    if v < 20:  return "Structural grade not met  (< 20 MPa)"
    if v < 30:  return "Moderate geopolymer strength"
    if v < 42:  return "High geopolymer strength"
    return "Very high geopolymer strength"


# ── Background model loading ───────────────────────────────────────────────────
def load_and_train():
    df = pd.read_excel(DATA_PATH)
    df['SiO2_Al2O3_ratio'] = df['SiO2'] / df['Al2O3']
    df['NaOH_Temp']        = df['NaOH (M)'] * df['Temperature']
    df['Chem_balance']     = df['Na2SiO3/NaOH'] * df['AA/FA']

    boruta = pd.read_csv(BORUTA_CSV)
    selected = boruta[boruta['Status'] == 'Confirmed']['Feature'].tolist()

    X_raw = df[selected].values
    y     = df["fc'"].values

    X_tr_raw, _, y_tr, _ = train_test_split(
        X_raw, y, test_size=0.20, random_state=SEED)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)

    model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        random_seed=SEED, verbose=0, allow_writing_files=False)
    model.fit(X_tr, y_tr)

    # CV residuals for 90% PI
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    residuals = []
    for ti, vi in kf.split(X_tr):
        cb_f = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                  random_seed=SEED, verbose=0, allow_writing_files=False)
        cb_f.fit(X_tr[ti], y_tr[ti])
        residuals.extend((y_tr[vi] - cb_f.predict(X_tr[vi])).tolist())

    q05 = float(np.percentile(residuals, 5))
    q95 = float(np.percentile(residuals, 95))
    return model, scaler, selected, q05, q95


# ══════════════════════════════════════════════════════════════════════════════
# Application
# ══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("FA-GPC Compressive Strength Predictor")
        self.geometry("1060x760")
        self.minsize(940, 700)
        self.configure(fg_color=C['bg'])
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self._model = self._scaler = self._selected = None
        self._q05 = self._q95 = None
        self._ready = False
        self._build()
        threading.Thread(target=self._train_bg, daemon=True).start()

    # ── background training ───────────────────────────────────────────────────
    def _train_bg(self):
        self._set_status("Loading data and training CatBoost model…", C['annot'])
        try:
            mdl, sc, sel, q05, q95 = load_and_train()
            self._model, self._scaler = mdl, sc
            self._selected = sel
            self._q05, self._q95 = q05, q95
            self._ready = True
            self.after(0, lambda: self._set_status(
                "Model ready  —  CatBoost · Boruta features · 90% CV-PI", C['g_act']))
        except Exception as ex:
            self.after(0, lambda: self._set_status(f"Error: {ex}", C['warn']))

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self._build_banner()
        self._build_body()
        self._build_statusbar()

    # ── banner ────────────────────────────────────────────────────────────────
    def _build_banner(self):
        bn = ctk.CTkFrame(self, fg_color=C['header'], corner_radius=0, height=82)
        bn.grid(row=0, column=0, sticky='ew')
        bn.grid_propagate(False)
        bn.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(bn,
                     text="FA-GPC Compressive Strength Predictor",
                     font=ctk.CTkFont("Arial", 24, "bold"),
                     text_color="#FFFFFF").grid(row=0, column=0, pady=(16, 3))
        ctk.CTkLabel(bn,
                     text="Fly Ash Geopolymer Concrete  ·  "
                          "CatBoost · 145 samples · 11 sources  ·  90% Prediction Interval",
                     font=ctk.CTkFont("Arial", 12),
                     text_color="#90A4AE").grid(row=1, column=0, pady=(0, 12))

    # ── body ──────────────────────────────────────────────────────────────────
    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color=C['bg'], corner_radius=0)
        body.grid(row=1, column=0, sticky='nsew', padx=16, pady=12)
        body.grid_rowconfigure(1, weight=1)
        body.grid_columnconfigure(0, weight=1)
        self._build_input_card(body)
        self._build_output_card(body)

    # ── input card ────────────────────────────────────────────────────────────
    def _build_input_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=C['card'], corner_radius=12,
                            border_width=1, border_color=C['border'])
        card.grid(row=0, column=0, sticky='ew', pady=(0, 12))
        card.grid_columnconfigure(0, weight=3)
        card.grid_columnconfigure(1, weight=0)
        card.grid_columnconfigure(2, weight=4)
        card.grid_columnconfigure(3, weight=0)
        card.grid_columnconfigure(4, weight=3)

        self._build_group_fa(card, col=0)
        ctk.CTkFrame(card, fg_color=C['border'], width=1).grid(
            row=0, column=1, sticky='ns', padx=2, pady=14)
        self._build_group_activator(card, col=2)
        ctk.CTkFrame(card, fg_color=C['border'], width=1).grid(
            row=0, column=3, sticky='ns', padx=2, pady=14)
        self._build_group_aggregate(card, col=4)

        # Buttons
        btn_frame = ctk.CTkFrame(card, fg_color='transparent')
        btn_frame.grid(row=1, column=0, columnspan=5, pady=(4, 16))

        ctk.CTkButton(btn_frame, text="PREDICT", width=148, height=46,
                      font=ctk.CTkFont("Arial", 14, "bold"),
                      fg_color=C['btn_p'], hover_color=C['btn_ph'],
                      corner_radius=8, command=self._predict
                      ).pack(side='left', padx=8)
        ctk.CTkButton(btn_frame, text="EXAMPLE", width=118, height=46,
                      font=ctk.CTkFont("Arial", 13),
                      fg_color=C['btn_e'], hover_color=C['btn_eh'],
                      corner_radius=8, command=self._example
                      ).pack(side='left', padx=8)
        ctk.CTkButton(btn_frame, text="CLEAR", width=108, height=46,
                      font=ctk.CTkFont("Arial", 13),
                      fg_color=C['btn_c'], hover_color=C['btn_ch'],
                      corner_radius=8, command=self._clear
                      ).pack(side='left', padx=8)
        ctk.CTkLabel(btn_frame,
                     text="  * required input     auto = derived automatically",
                     font=ctk.CTkFont("Arial", 11),
                     text_color=C['annot']).pack(side='left', padx=14)

    # ── group: Fly Ash Source ─────────────────────────────────────────────────
    def _build_group_fa(self, parent, col):
        grp = ctk.CTkFrame(parent, fg_color='transparent')
        grp.grid(row=0, column=col, sticky='nsew', padx=14, pady=12)
        grp.grid_columnconfigure(1, weight=1)
        self._group_hdr(grp, "FLY ASH SOURCE", C['g_fa'])

        rows = [
            ('FA Content *',      'kg/m³',   '254–600'),
            ('SiO₂ Content *',    '%',        '36.2–75.7'),
            ('Al₂O₃ Content *',   '%',        '9.2–31.3'),
        ]
        self._e_fa_c, self._e_sio2, self._e_al2o3 = [
            self._input_row(grp, lbl, unit, hint, row=i+1)
            for i, (lbl, unit, hint) in enumerate(rows)
        ]

        # Auto-derived: SiO2/Al2O3 ratio
        ctk.CTkLabel(grp, text="SiO₂/Al₂O₃ Ratio",
                     font=ctk.CTkFont("Arial", 12),
                     text_color=C['secondary']).grid(
            row=4, column=0, sticky='e', padx=(0, 8), pady=6)
        self._lbl_ratio = self._auto_field(grp, "—", width=92)
        self._lbl_ratio.grid(row=4, column=1, padx=4, pady=6)
        ctk.CTkLabel(grp, text="auto",
                     font=ctk.CTkFont("Arial", 10),
                     text_color=C['annot']).grid(row=4, column=2, sticky='w', padx=(4, 0))

        for e in (self._e_sio2, self._e_al2o3):
            e.bind("<KeyRelease>", self._update_auto)

    # ── group: Alkali Activator ───────────────────────────────────────────────
    def _build_group_activator(self, parent, col):
        grp = ctk.CTkFrame(parent, fg_color='transparent')
        grp.grid(row=0, column=col, sticky='nsew', padx=14, pady=12)
        grp.grid_columnconfigure(1, weight=1)
        self._group_hdr(grp, "ALKALI ACTIVATOR", C['g_act'])

        rows = [
            ('NaOH Content *',    'kg/m³',  '11.8–198'),
            ('NaOH Molarity *',   'M',      '8/10/12/14/15/16/20'),
            ('Na₂SiO₃/NaOH',     '—',      '1.0–8.8   (ratio)'),
            ('AA/FA Ratio',       '—',      '0.09–0.93'),
            ('Water *',           'kg/m³',  '37–207'),
        ]
        (self._e_naoh, self._e_naoh_m, self._e_ratio,
         self._e_aafa, self._e_water) = [
            self._input_row(grp, lbl, unit, hint, row=i+1)
            for i, (lbl, unit, hint) in enumerate(rows)
        ]

        # Auto-derived: NaOH × Temperature
        ctk.CTkLabel(grp, text="NaOH(M) × Temp",
                     font=ctk.CTkFont("Arial", 12),
                     text_color=C['secondary']).grid(
            row=6, column=0, sticky='e', padx=(0, 8), pady=6)
        self._lbl_nht = self._auto_field(grp, "—", width=92)
        self._lbl_nht.grid(row=6, column=1, padx=4, pady=6)
        ctk.CTkLabel(grp, text="auto",
                     font=ctk.CTkFont("Arial", 10),
                     text_color=C['annot']).grid(row=6, column=2, sticky='w', padx=(4, 0))

        self._e_naoh_m.bind("<KeyRelease>", self._update_auto)

    # ── group: Aggregate & Curing ─────────────────────────────────────────────
    def _build_group_aggregate(self, parent, col):
        grp = ctk.CTkFrame(parent, fg_color='transparent')
        grp.grid(row=0, column=col, sticky='nsew', padx=14, pady=12)
        grp.grid_columnconfigure(1, weight=1)
        self._group_hdr(grp, "AGGREGATE & CURING", C['g_agg'])

        rows = [
            ('Coarse Aggregate *', 'kg/m³', '554–1684'),
            ('Fine Aggregate *',   'kg/m³', '500–706'),
            ('Curing Temperature *','°C',   '0 = ambient'),
        ]
        (self._e_coarse, self._e_fine, self._e_temp) = [
            self._input_row(grp, lbl, unit, hint, row=i+1)
            for i, (lbl, unit, hint) in enumerate(rows)
        ]

        self._e_temp.bind("<KeyRelease>", self._update_auto)

        # Note
        ctk.CTkLabel(grp,
                     text="Temperature = 0  →  ambient curing\n"
                          "(78 / 145 samples in dataset)",
                     font=ctk.CTkFont("Arial", 10),
                     text_color=C['annot'], justify='left').grid(
            row=4, column=0, columnspan=3, sticky='w', padx=4, pady=(8, 0))

    # ── output card ───────────────────────────────────────────────────────────
    def _build_output_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=C['card'], corner_radius=12,
                            border_width=1, border_color=C['border'])
        card.grid(row=1, column=0, sticky='nsew')
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(card, fg_color=C['fc'], corner_radius=8, height=42)
        hdr.grid(row=0, column=0, sticky='ew', padx=12, pady=(12, 8))
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="Compressive Strength  f'c",
                     font=ctk.CTkFont("Arial", 15, "bold"),
                     text_color="white").place(relx=0.5, rely=0.5, anchor='center')

        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.grid(row=1, column=0, sticky='nsew')
        inner.grid_columnconfigure(0, weight=1)
        inner.grid_columnconfigure(1, weight=1)

        # Left: numeric result
        left = ctk.CTkFrame(inner, fg_color='transparent')
        left.grid(row=0, column=0, sticky='nsew', padx=(20, 10), pady=10)
        left.grid_columnconfigure(0, weight=1)

        self._lbl_val = ctk.CTkLabel(left, text="—",
                                      font=ctk.CTkFont("Arial", 56, "bold"),
                                      text_color=C['fc'])
        self._lbl_val.grid(row=0, column=0, pady=(10, 0))

        ctk.CTkLabel(left, text="MPa",
                     font=ctk.CTkFont("Arial", 18),
                     text_color=C['secondary']).grid(row=1, column=0, pady=(0, 6))

        self._lbl_pi = ctk.CTkLabel(left, text="90% PI:  —",
                                     font=ctk.CTkFont("Arial", 14),
                                     text_color=C['secondary'])
        self._lbl_pi.grid(row=2, column=0, pady=(0, 8))

        # Right: grade + feature summary
        right = ctk.CTkFrame(inner, fg_color=C['auto_bg'], corner_radius=8)
        right.grid(row=0, column=1, sticky='nsew', padx=(10, 20), pady=10)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="Assessment",
                     font=ctk.CTkFont("Arial", 12, "bold"),
                     text_color=C['g_fa']).grid(row=0, column=0, pady=(12, 4))

        self._lbl_grade = ctk.CTkLabel(right, text="—",
                                        font=ctk.CTkFont("Arial", 13, slant="italic"),
                                        text_color=C['idle'], wraplength=280)
        self._lbl_grade.grid(row=1, column=0, padx=14, pady=(0, 8))

        self._lbl_derived = ctk.CTkLabel(right, text="",
                                          font=ctk.CTkFont("Arial", 11),
                                          text_color=C['annot'], justify='left')
        self._lbl_derived.grid(row=2, column=0, padx=14, pady=(0, 14))

    # ── status bar ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = ctk.CTkFrame(self, fg_color=C['header'], corner_radius=0, height=28)
        bar.grid(row=2, column=0, sticky='ew')
        bar.grid_propagate(False)
        self._status_lbl = ctk.CTkLabel(bar, text="Initializing…",
                                         font=ctk.CTkFont("Arial", 10),
                                         text_color="#90A4AE")
        self._status_lbl.place(relx=0.012, rely=0.5, anchor='w')

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _group_hdr(parent, text, color):
        hdr = ctk.CTkFrame(parent, fg_color=color, corner_radius=6, height=30)
        hdr.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 8))
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text=text,
                     font=ctk.CTkFont("Arial", 11, "bold"),
                     text_color="white").place(relx=0.5, rely=0.5, anchor='center')

    @staticmethod
    def _auto_field(parent, text, width=92):
        frm = ctk.CTkFrame(parent, fg_color=C['auto_bg'], corner_radius=6,
                           border_width=1, border_color=C['border'],
                           width=width, height=34)
        frm.grid_propagate(False)
        lbl = ctk.CTkLabel(frm, text=text,
                           font=ctk.CTkFont("Arial", 12),
                           text_color=C['auto_txt'])
        lbl.place(relx=0.5, rely=0.5, anchor='center')
        frm._inner = lbl
        return frm

    def _input_row(self, parent, label, unit, hint, row):
        ctk.CTkLabel(parent, text=label,
                     font=ctk.CTkFont("Arial", 12, "bold"),
                     text_color=C['g_fa'] if 'SiO₂' in label or 'Al₂O₃' in label
                                          or 'FA' in label
                                          else C['secondary'],
                     anchor='e').grid(row=row, column=0, sticky='e',
                                      padx=(0, 8), pady=5)
        ent = ctk.CTkEntry(parent, width=92, height=34,
                           font=ctk.CTkFont("Arial", 12),
                           corner_radius=6, border_color=C['border'])
        ent.grid(row=row, column=1, padx=4, pady=5)
        ctk.CTkLabel(parent, text=f"{unit}  {hint}",
                     font=ctk.CTkFont("Arial", 10),
                     text_color=C['annot']).grid(row=row, column=2,
                                                  sticky='w', padx=(4, 0))
        return ent

    def _set_status(self, msg: str, color: str = '#90A4AE'):
        self._status_lbl.configure(text=msg, text_color=color)

    # ── auto-update derived fields ────────────────────────────────────────────
    def _update_auto(self, _event=None):
        try:
            s = float(self._e_sio2.get())
            a = max(float(self._e_al2o3.get()), 0.01)
            self._lbl_ratio._inner.configure(text=f"{s/a:.3f}")
        except ValueError:
            self._lbl_ratio._inner.configure(text="—")

        try:
            m  = float(self._e_naoh_m.get())
            t  = float(self._e_temp.get())
            self._lbl_nht._inner.configure(text=f"{m * t:.1f}")
        except ValueError:
            self._lbl_nht._inner.configure(text="—")

    # ── predict ───────────────────────────────────────────────────────────────
    def _predict(self):
        if not self._ready:
            messagebox.showwarning("Not Ready", "Model is still loading, please wait.")
            return

        try:
            raw = {
                'FA':     float(self._e_fa_c.get()),
                'SiO2':   float(self._e_sio2.get()),
                'Al2O3':  float(self._e_al2o3.get()),
                'Coarse': float(self._e_coarse.get()),
                'Fine':   float(self._e_fine.get()),
                'NaOH':   float(self._e_naoh.get()),
                'NaOH_M': float(self._e_naoh_m.get()),
                'Water':  float(self._e_water.get()),
                'Temp':   float(self._e_temp.get()),
            }
        except ValueError:
            messagebox.showerror("Input Error",
                                 "All fields marked * must contain numeric values.")
            return

        warns = [f"  • {k}: {v:.2g} outside [{lo}, {hi}]"
                 for k, v in raw.items()
                 for lo, hi in [RANGES[k]]
                 if not (lo <= v <= hi)]
        if warns:
            msg = "The following inputs are outside the training range:\n" + \
                  "\n".join(warns) + "\n\nProceed with extrapolated prediction?"
            if not messagebox.askyesno("Out-of-Range Warning", msg):
                return

        # Build feature vector
        al2o3 = max(raw['Al2O3'], 0.01)
        feat_map = {
            'Coarse aggregate':  raw['Coarse'],
            'FA':                raw['FA'],
            'SiO2_Al2O3_ratio':  raw['SiO2'] / al2o3,
            'Al2O3':             raw['Al2O3'],
            'SiO2':              raw['SiO2'],
            'NaOH (M)':          raw['NaOH_M'],
            'NaOH':              raw['NaOH'],
            'Fine aggregate':    raw['Fine'],
            'Water':             raw['Water'],
            'NaOH_Temp':         raw['NaOH_M'] * raw['Temp'],
        }
        X_in = np.array([[feat_map[f] for f in self._selected]])
        X_sc = self._scaler.transform(X_in)
        y_hat = float(self._model.predict(X_sc)[0])
        lo    = max(0.0, y_hat + self._q05)
        hi    = y_hat + self._q95

        self._lbl_val.configure(text=f"{y_hat:.2f}")
        self._lbl_pi.configure(
            text=f"90% PI:   [{lo:.2f},  {hi:.2f}] MPa")
        self._lbl_grade.configure(text=_fc_grade(y_hat),
                                   text_color=C['fc'] if y_hat >= 30 else C['warn'])

        self._lbl_derived.configure(
            text=f"SiO₂/Al₂O₃  = {raw['SiO2']/al2o3:.3f}\n"
                 f"NaOH(M)×T    = {raw['NaOH_M']*raw['Temp']:.1f}")
        self._update_auto()
        self._set_status(
            f"Predicted  —  f'c = {y_hat:.2f} MPa   PI = [{lo:.2f}, {hi:.2f}]   "
            f"NaOH = {raw['NaOH_M']} M   T = {raw['Temp']}°C",
            C['g_act'])

    # ── example / clear ───────────────────────────────────────────────────────
    def _example(self):
        self._clear()
        for ent, key in [
            (self._e_fa_c,   'FA'),    (self._e_sio2,   'SiO2'),
            (self._e_al2o3,  'Al2O3'), (self._e_coarse, 'Coarse'),
            (self._e_fine,   'Fine'),  (self._e_naoh,   'NaOH'),
            (self._e_naoh_m, 'NaOH_M'),(self._e_water,  'Water'),
            (self._e_temp,   'Temp'),
        ]:
            ent.insert(0, EXAMPLE[key])
        # Na2SiO3/NaOH and AA/FA examples
        self._e_ratio.insert(0, '2.5')
        self._e_aafa.insert(0, '0.53')
        self._update_auto()
        self._set_status("Example values loaded (training medians).", C['annot'])

    def _clear(self):
        for ent in (self._e_fa_c, self._e_sio2, self._e_al2o3,
                    self._e_coarse, self._e_fine, self._e_naoh,
                    self._e_naoh_m, self._e_ratio, self._e_aafa,
                    self._e_water, self._e_temp):
            ent.delete(0, 'end')
        self._lbl_ratio._inner.configure(text="—")
        self._lbl_nht._inner.configure(text="—")
        self._lbl_val.configure(text="—")
        self._lbl_pi.configure(text="90% PI:  —")
        self._lbl_grade.configure(text="—", text_color=C['idle'])
        self._lbl_derived.configure(text="")
        self._set_status("Cleared.", C['annot'])


if __name__ == '__main__':
    app = App()
    app.mainloop()
