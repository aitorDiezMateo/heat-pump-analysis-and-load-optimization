from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_OUTPUT_DIR = PROJECT_ROOT / "figures" 


@dataclass
class LoadShiftingConfig:
    pcp: float = 0.6
    psc: float = 1.4
    eff_ant: float = 1.05
    cc: float = 0.1
    cr: float = 0.3
    iter_max: int = 1000
    uv: float = 0.1
    opening_time_delta: int = 2


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"] + plt.rcParams.get("font.serif", []),
            "font.size": 12,
            "axes.titlesize": 28,
            "axes.labelsize": 24,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "figure.figsize": (10, 6),
        }
    )


def ensure_datetime(df: pd.DataFrame, column: str = "datetime") -> pd.DataFrame:
    if column not in df.columns:
        raise KeyError(f"Missing required column: '{column}'")
    out = df.copy()
    out[column] = pd.to_datetime(out[column])
    out = out.sort_values(column).reset_index(drop=True)
    return out


def load_price_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    required_cols = ["datetime", "price", "EE_total", "E_shifted"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in price data: {', '.join(missing)}")
    df = ensure_datetime(df)
    if "date" not in df.columns:
        df["date"] = df["datetime"].dt.date.astype(str)

    et_cols = ["ET_BC1_Frio", "ET_BC1_Calor", "ET_BC2_Frio", "ET_BC2_Calor", "ET_BC3_Frio", "ET_BC3_Calor"]
    if all(col in df.columns for col in et_cols):
        df["ET_total"] = df[et_cols].sum(axis=1)

    return df


def mark_openning_closing_time(df: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    out = df.copy()
    out["openning_time"] = np.nan
    out["closing_time"] = np.nan

    # Use a temporary cleaned series only to detect opening/closing hours,
    # preserving the original EE_total values used by the optimization.
    detect = out.copy()
    series = detect["EE_total"]
    cond1 = (series > threshold) & (series.shift(1) < threshold) & (series.shift(-1) < threshold)
    cond2 = (
        (series.shift(1) < threshold)
        & (series > threshold)
        & (series.shift(-1) > threshold)
        & (series.shift(-2) < threshold)
    )
    cond3 = (
        (series.shift(2) < threshold)
        & (series.shift(1) > threshold)
        & (series > threshold)
        & (series.shift(-1) < threshold)
    )

    detect["EE_total"] = np.where(cond1 | cond2 | cond3, 0.0, series)

    for _, group in detect.groupby("date"):
        active = group[group["EE_total"] > threshold]
        if active.empty:
            continue
        opening_hour = active.iloc[0]["datetime"].hour
        closing_hour = active.iloc[-1]["datetime"].hour
        out.loc[group.index, "openning_time"] = opening_hour
        out.loc[group.index, "closing_time"] = closing_hour

    return out


def run_load_shifting_price_algorithm(
    df_original: pd.DataFrame, config: LoadShiftingConfig
) -> Tuple[pd.DataFrame, int, List[pd.Series]]:
    cols = ["datetime", "price", "EE_total", "date"]
    df = df_original[cols].copy()
    df = mark_openning_closing_time(df)

    ae_max_incr = (config.psc - 1.0) * float(df["EE_total"].max())
    ae_max_reduc = (1.0 - config.pcp) * float(df["EE_total"].max())

    df["E_max"] = df["EE_total"] + ae_max_incr
    df["E_min"] = (df["EE_total"] - ae_max_reduc).clip(lower=0.0)

    for idx, row in df.iterrows():
        current_hour = int(row["datetime"].hour)
        opening_time = row["openning_time"]
        closing_time = row["closing_time"]

        if pd.isna(opening_time) or pd.isna(closing_time):
            df.loc[idx, "E_max"] = row["EE_total"]
            df.loc[idx, "E_min"] = row["EE_total"]
            continue

        if current_hour < int(opening_time) - config.opening_time_delta or current_hour > int(closing_time):
            df.loc[idx, "E_max"] = row["EE_total"]
            df.loc[idx, "E_min"] = row["EE_total"]

    for _, group in df.groupby("date"):
        if (group["EE_total"] < 0.1).all():
            df.loc[group.index, "E_max"] = df.loc[group.index, "EE_total"]
            df.loc[group.index, "E_min"] = df.loc[group.index, "EE_total"]

    df["E_cons"] = df["EE_total"].copy()
    df["E_ant"] = 0.0
    df["E_ret"] = 0.0

    iteration = 0
    prev_sum_ae: Optional[float] = None
    history: List[pd.Series] = []

    while True:
        iteration += 1

        df["is_local_max"] = 0
        df["grad_loc"] = 0
        df["Acc"] = 0
        df["AE"] = 0.0
        df["AE_ant"] = 0.0
        df["AE_ret"] = 0.0

        left_has_room = df["E_cons"].shift(1) < df["E_max"].shift(1)
        right_has_room = df["E_cons"].shift(-1) < df["E_max"].shift(-1)
        left_price_ok = df["price"].shift(1) <= df["price"]
        right_price_ok = df["price"].shift(-1) <= df["price"]

        is_local_max = (
            (df["E_cons"] > df["E_cons"].shift(1))
            & (df["E_cons"] > df["E_cons"].shift(-1))
            & (df["E_cons"] > df["E_min"])
            & ((left_has_room & left_price_ok) | (right_has_room & right_price_ok))
        )
        is_local_max.iloc[0] = False
        is_local_max.iloc[-1] = False
        df["is_local_max"] = is_local_max.astype(int)

        a = (df["E_cons"] > df["E_cons"].shift(1)) & left_price_ok
        b = left_has_room
        c = df["E_cons"] > df["E_min"]
        d = (df["E_cons"] > df["E_cons"].shift(-1)) & right_price_ok
        e = right_has_room
        grad_loc = (a & b & c) | (d & e & c)
        grad_loc.iloc[0] = False
        grad_loc.iloc[-1] = False
        df["grad_loc"] = grad_loc.astype(int)

        candidate_mask = (df["is_local_max"] == 1) | (df["grad_loc"] == 1)

        for idx, row in df.iterrows():
            if not candidate_mask.iloc[idx]:
                continue

            if idx > 0:
                a_val = row["E_cons"] - row["E_min"]
                b_val = df.loc[idx - 1, "E_max"] - df.loc[idx - 1, "E_cons"]
                c_val = (row["E_cons"] - df.loc[idx - 1, "E_cons"]) / 2.0
                ae_ant = max(min(a_val, b_val, c_val), 0.0)
                if ae_ant < config.uv or df.loc[idx - 1, "price"] > df.loc[idx, "price"]:
                    ae_ant = 0.0
                df.loc[idx, "AE_ant"] = ae_ant

            if idx < len(df) - 1:
                a_val = row["E_cons"] - row["E_min"]
                b_val = df.loc[idx + 1, "E_max"] - df.loc[idx + 1, "E_cons"]
                c_val = (row["E_cons"] - df.loc[idx + 1, "E_cons"]) / 2.0
                ae_ret = max(min(a_val, b_val, c_val), 0.0)
                if ae_ret < config.uv or df.loc[idx + 1, "price"] > df.loc[idx, "price"]:
                    ae_ret = 0.0
                df.loc[idx, "AE_ret"] = ae_ret

        for idx in df.index:
            if not candidate_mask.iloc[idx]:
                continue
            df.loc[idx, "Acc"] = 1 if df.loc[idx, "AE_ret"] > df.loc[idx, "AE_ant"] else -1

        for idx in df.index:
            if df.loc[idx, "Acc"] == 1:
                df.loc[idx, "AE_ret"] = config.cr * df.loc[idx, "AE_ret"]
                df.loc[idx, "AE_ant"] = 0.0
                df.loc[idx, "AE"] = df.loc[idx, "AE_ret"]
            elif df.loc[idx, "Acc"] == -1:
                df.loc[idx, "AE_ant"] = config.cr * df.loc[idx, "AE_ant"]
                df.loc[idx, "AE_ret"] = 0.0
                df.loc[idx, "AE"] = df.loc[idx, "AE_ant"]

        mask1 = (df["Acc"] == 1) & (df["E_ant"] == 0)
        mask2 = (df["Acc"] == 1) & (df["E_ant"].abs() > 0) & (df["E_ant"].abs() <= df["E_ret"].abs())
        mask3 = (df["Acc"] == 1) & (df["E_ant"].abs() > 0) & (df["E_ant"].abs() > df["E_ret"].abs())
        mask4 = (df["Acc"] == -1) & (df["E_ret"] == 0)
        mask5 = (df["Acc"] == -1) & (df["E_ret"].abs() > 0) & (df["E_ret"].abs() <= df["E_ant"].abs())
        mask6 = (df["Acc"] == -1) & (df["E_ret"].abs() > 0) & (df["E_ret"].abs() > df["E_ant"].abs())

        for idx in range(1, len(df) - 1):
            if mask1.iloc[idx]:
                df.loc[idx, "E_cons"] = df.loc[idx, "E_cons"] - df.loc[idx, "AE_ret"]
                df.loc[idx + 1, "E_cons"] = df.loc[idx + 1, "E_cons"] + df.loc[idx, "AE_ret"]
                df.loc[idx, "E_ret"] = df.loc[idx, "E_ret"] - df.loc[idx, "AE_ret"]
            elif mask2.iloc[idx]:
                df.loc[idx, "E_cons"] = df.loc[idx, "E_cons"] - df.loc[idx, "E_ant"]
                df.loc[idx + 1, "E_cons"] = (
                    df.loc[idx + 1, "E_cons"] + (1.0 / config.eff_ant) * df.loc[idx, "E_ant"]
                )
                df.loc[idx, "AE_ret"] = df.loc[idx, "AE_ret"] - df.loc[idx, "E_ant"]
                df.loc[idx, "E_ant"] = 0.0
            elif mask3.iloc[idx]:
                df.loc[idx, "E_cons"] = df.loc[idx, "E_cons"] - df.loc[idx, "AE_ret"]
                df.loc[idx + 1, "E_cons"] = (
                    df.loc[idx + 1, "E_cons"] + (1.0 / config.eff_ant) * df.loc[idx, "AE_ret"]
                )
                df.loc[idx, "E_ant"] = df.loc[idx, "E_ant"] - df.loc[idx, "AE_ret"]
            elif mask4.iloc[idx]:
                df.loc[idx, "E_cons"] = df.loc[idx, "E_cons"] - df.loc[idx, "AE_ant"]
                df.loc[idx - 1, "E_cons"] = (
                    df.loc[idx - 1, "E_cons"] + config.eff_ant * df.loc[idx, "AE_ant"]
                )
                df.loc[idx, "E_ant"] = df.loc[idx, "E_ant"] + df.loc[idx, "AE_ant"]
            elif mask5.iloc[idx]:
                df.loc[idx, "E_cons"] = df.loc[idx, "E_cons"] - df.loc[idx, "E_ret"]
                df.loc[idx - 1, "E_cons"] = (
                    df.loc[idx - 1, "E_cons"] + config.eff_ant * df.loc[idx, "E_ant"]
                )
                df.loc[idx, "E_ant"] = df.loc[idx, "E_ant"] - df.loc[idx, "E_ret"]
                df.loc[idx, "E_ret"] = 0.0
            elif mask6.iloc[idx]:
                df.loc[idx, "E_cons"] = df.loc[idx, "E_cons"] - df.loc[idx, "AE_ant"]
                df.loc[idx - 1, "E_cons"] = (
                    df.loc[idx - 1, "E_cons"] + config.eff_ant * df.loc[idx, "AE_ant"]
                )
                df.loc[idx, "E_ret"] = df.loc[idx, "E_ret"] - df.loc[idx, "AE_ant"]

        history.append(df["E_cons"].copy())

        sum_ae = float(df["AE"].sum())
        if prev_sum_ae is not None and sum_ae <= config.cc * prev_sum_ae:
            break
        prev_sum_ae = sum_ae

        if iteration >= config.iter_max:
            break

    out = df.copy()
    out["E_shifted"] = out["E_cons"]
    return out, iteration, history


def prepare_data_from_price_file(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[object], Optional[float]]:
    required_cols = ["datetime", "price", "EE_total", "E_shifted"]
    optional_cols = ["E_min", "E_max", "openning_time", "closing_time"]

    cols_to_use = required_cols.copy()
    for col in optional_cols:
        if col in df.columns:
            cols_to_use.append(col)

    merged = df[cols_to_use].copy()

    if "ET_total" in df.columns:
        merged["ET_total"] = df["ET_total"]

    merged["date"] = merged["datetime"].dt.date.astype(str)
    merged["cost_EE_total"] = merged["EE_total"] * merged["price"] / 1000.0
    merged["cost_E_shifted"] = merged["E_shifted"] * merged["price"] / 1000.0

    df_ls = merged.copy()
    df_ls["RA_EE_total"] = (
        df_ls.rolling(window=4, on="datetime", center=True, min_periods=1)["E_shifted"].mean()
    )
    df_ls["paid_price"] = df_ls["price"] * df_ls["E_shifted"] / 1000
    df_ls["price_per_kWh"] = df_ls["price"] / 1000
    df_ls["date"] = df_ls["datetime"].dt.strftime("%Y-%m-%d")

    df_orig = merged.copy()
    df_orig["paid_price"] = df_orig["price"] * df_orig["EE_total"] / 1000
    df_orig["price_per_kWh"] = df_orig["price"] / 1000
    df_orig["date"] = df_orig["datetime"].dt.strftime("%Y-%m-%d")

    if "ET_total" in df_orig.columns:
        df_orig["RA_EE_total"] = (
            df_orig.rolling(window=4, on="datetime", center=True, min_periods=1)["EE_total"].mean()
        )
        df_orig["RA_ET_total"] = (
            df_orig.rolling(window=4, on="datetime", center=True, min_periods=1)["ET_total"].mean()
        )
        df_orig["COP"] = df_orig["RA_ET_total"] / df_orig["RA_EE_total"].replace(0.0, np.nan)

        model, cop_cutoff = get_cop_calculation_model(df_orig, degree=5)
        df_ls["COP"] = predict_cop_safe(df_ls, model, cop_cutoff)
        df_orig["COP"] = predict_cop_safe(df_orig, model, cop_cutoff)

        return (
            df_ls.sort_values("datetime").reset_index(drop=True),
            df_orig.sort_values("datetime").reset_index(drop=True),
            model,
            cop_cutoff,
        )

    return (
        df_ls.sort_values("datetime").reset_index(drop=True),
        df_orig.sort_values("datetime").reset_index(drop=True),
        None,
        None,
    )


def fit_cop_model(
    df_original: pd.DataFrame, degree: int = 4
) -> Tuple[object, float, pd.DataFrame]:
    df_fit = df_original.copy()
    df_fit = df_fit[df_fit["RA_EE_total"] > 0.1]
    cutoff = float(df_fit["RA_EE_total"].quantile(0.25))
    df_fit = df_fit[df_fit["RA_EE_total"] >= cutoff]
    df_fit = df_fit.replace([np.inf, -np.inf], np.nan).dropna(subset=["COP_measured"])

    x = df_fit[["RA_EE_total"]].values
    y = df_fit["COP_measured"].values

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x, y)

    return model, cutoff, df_fit


def predict_cop_safe(load_series: pd.Series, model: object, threshold: float) -> pd.Series:
    values = load_series.to_numpy()
    predictions = np.full(values.shape[0], np.nan, dtype=float)
    valid = values >= threshold
    if valid.any():
        predictions[valid] = model.predict(values[valid].reshape(-1, 1))
    return pd.Series(predictions, index=load_series.index)


def compute_variance_pair(original_value: float, shifted_value: float) -> Dict[str, float]:
    delta = shifted_value - original_value
    pct = np.nan if original_value == 0 else (delta / original_value) * 100.0
    return {
        "original": round(float(original_value), 4),
        "shifted": round(float(shifted_value), 4),
        "absolute": round(float(delta), 4),
        "percent": round(float(pct), 4),
    }


def compute_variance_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return {
        "global_energy_kwh": compute_variance_pair(df["EE_total"].sum(), df["E_shifted"].sum()),
        "total_cost_eur": compute_variance_pair(df["cost_EE_total"].sum(), df["cost_E_shifted"].sum()),
        "max_load_kw": compute_variance_pair(df["EE_total"].max(), df["E_shifted"].max()),
    }


def calculate_daily_variance(df: pd.DataFrame, date_value: str) -> Dict[str, Dict[str, float]]:
    date_obj = pd.to_datetime(date_value).date()
    day_df = df[df["datetime"].dt.date == date_obj]
    if day_df.empty:
        return {}
    return compute_variance_metrics(day_df)


def calculate_monthly_variance(df: pd.DataFrame, year_month: str) -> Dict[str, Dict[str, float]]:
    period = pd.Period(year_month, freq="M")
    month_df = df[df["datetime"].dt.to_period("M") == period]
    if month_df.empty:
        return {}
    return compute_variance_metrics(month_df)


def data_span_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Calendar span and coarse coverage for deciding weekly vs daily aggregations."""
    s = pd.to_datetime(df["datetime"], errors="coerce")
    t0, t1 = s.min(), s.max()
    if pd.isna(t0) or pd.isna(t1):
        return {
            "calendar_days_covered": 0,
            "span_hours": 0.0,
            "unique_iso_weeks": 0,
            "datetime_min": None,
            "datetime_max": None,
        }
    calendar_days = int((t1.normalize() - t0.normalize()).days) + 1
    span_hours = float((t1 - t0).total_seconds() / 3600.0)
    unique_weeks = int(s.dt.to_period("W-SUN").nunique())
    return {
        "calendar_days_covered": calendar_days,
        "span_hours": round(span_hours, 2),
        "unique_iso_weeks": unique_weeks,
        "datetime_min": t0.isoformat(),
        "datetime_max": t1.isoformat(),
    }


def insufficient_span_for_weekly_cop(span: Dict[str, Any]) -> bool:
    return int(span.get("calendar_days_covered", 0)) < 7


def default_price_pareto_paper_zooms() -> Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]:
    return (
        (0, 4500, 0, 3.0),
        (0, 800, 0.5, 3.0),
        (1500, 4500, 0, 0.5),
    )


def pick_dashboard_highlight_dates(
    dates_sorted: List[str],
    high_date: Optional[str],
    midday_date: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    auto_high, auto_mid = (None, None)
    if len(dates_sorted) >= 2:
        auto_high, auto_mid = dates_sorted[0], dates_sorted[1]
    elif len(dates_sorted) == 1:
        auto_high = auto_mid = dates_sorted[0]

    if high_date is not None or midday_date is not None:
        return (high_date or auto_high, midday_date or auto_mid)
    return auto_high, auto_mid


def calculate_daily_peak_reduction_quartiles(
    df: pd.DataFrame,
    percentile_cutoff: int = 80,
    min_reduction_percent: float = 0.0,
) -> Dict[str, Optional[float]]:
    daily_max = (
        df.groupby(df["datetime"].dt.date)[["EE_total", "E_shifted"]]
        .max()
        .rename(columns={"EE_total": "original_max", "E_shifted": "shifted_max"})
        .dropna()
    )
    daily_max = daily_max[daily_max["original_max"] >= 10.0]

    if not daily_max.empty:
        q_threshold = np.percentile(daily_max["original_max"], percentile_cutoff)
        daily_max = daily_max[daily_max["original_max"] >= q_threshold]

    if daily_max.empty:
        return {"q1_reduction_percent": None, "q3_reduction_percent": None, "count_days": 0}

    reduction_pct = ((daily_max["original_max"] - daily_max["shifted_max"]) / daily_max["original_max"]) * 100.0
    reduction_abs = daily_max["original_max"] - daily_max["shifted_max"]

    valid = (reduction_abs > 1e-6) & (reduction_pct >= min_reduction_percent)
    reduction_pct = reduction_pct[valid]
    if reduction_pct.empty:
        return {"q1_reduction_percent": None, "q3_reduction_percent": None, "count_days": 0}

    return {
        "q1_reduction_percent": round(float(np.percentile(reduction_pct, 25)), 4),
        "q3_reduction_percent": round(float(np.percentile(reduction_pct, 75)), 4),
        "count_days": int(reduction_pct.shape[0]),
    }


def prepare_visualization_frames(
    df_original: pd.DataFrame,
    df_merged: pd.DataFrame,
    cop_degree: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, object, float]:
    df_ls = df_merged.copy()
    df_ls["RA_EE_total"] = df_ls.rolling(window=4, on="datetime", center=True, min_periods=1)["E_shifted"].mean()
    df_ls["paid_price"] = df_ls["price"] * df_ls["E_shifted"] / 1000
    df_ls["price_per_kWh"] = df_ls["price"] / 1000
    df_ls["date"] = df_ls["datetime"].dt.strftime("%Y-%m-%d")

    df_orig = df_original.copy()
    df_orig["paid_price"] = df_orig["price"] * df_orig["EE_total"] / 1000
    df_orig["price_per_kWh"] = df_orig["price"] / 1000
    df_orig["date"] = df_orig["datetime"].dt.strftime("%Y-%m-%d")

    df_orig["RA_EE_total"] = df_orig.rolling(window=4, on="datetime", center=True, min_periods=1)["EE_total"].mean()
    df_orig["RA_ET_total"] = df_orig.rolling(window=4, on="datetime", center=True, min_periods=1)["ET_total"].mean()
    df_orig["COP"] = df_orig["RA_ET_total"] / df_orig["RA_EE_total"].replace(0.0, np.nan)

    model, cop_cutoff = get_cop_calculation_model(df_orig, degree=cop_degree)
    df_ls["COP"] = predict_cop_safe(df_ls, model, cop_cutoff)
    df_orig["COP"] = predict_cop_safe(df_orig, model, cop_cutoff)

    return df_ls, df_orig, model, cop_cutoff


def get_cop_calculation_model(df: pd.DataFrame, degree: int = 5):
    df_cop = df[df["RA_EE_total"] > 0.1]
    cutoff_threshold = df_cop["RA_EE_total"].quantile(0.25)
    df_cop = df_cop[df_cop["RA_EE_total"] >= cutoff_threshold]
    X = df_cop[["RA_EE_total"]].values
    y = df_cop["COP"].values
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    return model, cutoff_threshold


def predict_cop_safe(df: pd.DataFrame, model: object, threshold: float):
    cop_predictions = np.full(len(df), np.nan)
    valid_mask = df["RA_EE_total"] >= threshold
    if not valid_mask.any():
        return cop_predictions
    X_pred = df.loc[valid_mask, ["RA_EE_total"]].values
    cop_predictions[valid_mask] = model.predict(X_pred)
    return cop_predictions


def cop_vs_time(df_input: pd.DataFrame) -> None:
    df_ls = df_input.copy()
    df_ls = df_ls[df_ls["RA_EE_total"] > 0.1]
    df_ls["hour"] = df_ls["datetime"].dt.hour
    data_to_plot = [df_ls[df_ls["hour"] == h]["COP"].dropna().values for h in range(24)]

    fig, ax = plt.subplots(figsize=(12, 6))
    hours = range(24)
    complex_start, complex_end = 4, 22
    violin_fill = "#a1c8e6"
    line_color = "#444444"
    scatter_color = "#3b4b5b"

    for h in hours:
        hour_data = data_to_plot[h]
        if len(hour_data) == 0:
            continue
        if complex_start <= h <= complex_end:
            vparts = ax.violinplot([hour_data], positions=[h], showmeans=False, showmedians=False, showextrema=False, widths=0.6)
            for b in vparts["bodies"]:
                b.set_facecolor(violin_fill)
                b.set_edgecolor("#2b7bba")
                b.set_linewidth(0.6)
                b.set_alpha(0.95)
                b.set_zorder(1)
            q1, median, q3 = np.percentile(hour_data, [25, 50, 75])
            ax.plot([h, h], [q1, q3], color="#ff0000", lw=2.2, alpha=0.95, zorder=3)
            ax.plot([h - 0.20, h + 0.20], [median, median], color="white", lw=4.0, zorder=6)
            ax.plot([h - 0.20, h + 0.20], [median, median], color=line_color, lw=1.2, zorder=7)
            ax.scatter([h], [median], s=70, color="#ff0000", edgecolor=line_color, linewidths=1.2, zorder=8)
            cap_w = 0.18
            ax.plot([h - cap_w, h + cap_w], [q1, q1], color="#ff0000", lw=2.2, zorder=5)
            ax.plot([h - cap_w, h + cap_w], [q3, q3], color="#ff0000", lw=2.2, zorder=5)
            ax.plot([h - cap_w, h + cap_w], [q1, q1], color=line_color, lw=0.9, zorder=6)
            ax.plot([h - cap_w, h + cap_w], [q3, q3], color=line_color, lw=0.9, zorder=6)
            x_jitter = np.random.uniform(h - 0.15, h + 0.15, size=len(hour_data))
        else:
            x_jitter = np.random.uniform(h - 0.15, h + 0.15, size=len(hour_data))
        ax.scatter(x_jitter, hour_data, alpha=0.35, s=8, color=scatter_color, edgecolor="k", linewidths=0.12, zorder=4)

    ax.set_title("COP vs Hour of Day", pad=20)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("COP")
    ax.set_xticks(list(hours))
    ax.set_xticklabels(list(hours))
    ax.set_xlim(-0.5, 23.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="#dcdcdc")
    ax.xaxis.grid(False)
    sns.despine(trim=True, offset=5)
    plt.tight_layout()
    save_figure(fig, PUBLICATION_OUTPUT_DIR / "fig5_cop_vs_time.svg", show=False)


def plot_cop_pareto_and_weekly_stack(
    df_ls_input: pd.DataFrame,
    df_orig_input: pd.DataFrame,
    *,
    use_daily_cop_panel: bool = False,
) -> None:
    df_ls = df_ls_input.copy()
    df_orig = df_orig_input.copy()

    df_ls = df_ls[df_ls["RA_EE_total"] > 0.1]
    p25 = df_ls["RA_EE_total"].quantile(0.25)
    df_ls = df_ls[df_ls["RA_EE_total"] >= p25]

    df_orig = df_orig[df_orig["RA_EE_total"] > 0.1]
    p25 = df_orig["RA_EE_total"].quantile(0.25)
    df_orig = df_orig[df_orig["RA_EE_total"] >= p25]

    sorted_cop_ls = df_ls["COP"].sort_values(ascending=False).reset_index(drop=True)
    sorted_cop_orig = df_orig["COP"].sort_values(ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
    ax = axes[0]
    ax.plot(sorted_cop_ls.index, sorted_cop_ls, color="#2b7bba", linewidth=3.5, label="Shifted Load")
    ax.plot(sorted_cop_orig.index, sorted_cop_orig, color="#f52121", linewidth=3.5, label="Original Load")
    ax.set_xlabel("Sorted Observations")
    ax.set_ylabel("COP")
    ax.set_title("COP Pareto Curve", pad=20)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_xlim(0, len(sorted_cop_ls))
    ax.legend(frameon=False)

    ax = axes[1]
    if use_daily_cop_panel:
        df_ls_d = df_ls.resample("D", on="datetime").mean(numeric_only=True).reset_index()
        df_orig_d = df_orig.resample("D", on="datetime").mean(numeric_only=True).reset_index()
        ax.plot(df_ls_d["datetime"], df_ls_d["COP"], color="#2b7bba", linewidth=3.5, label="Shifted Load")
        ax.plot(df_orig_d["datetime"], df_orig_d["COP"], color="#f52121", linewidth=3.5, label="Original Load")
        ax.set_xlabel("Date")
        ax.set_ylabel("COP")
        ax.set_title("COP Daily Mean (sample window)", pad=20)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment("right")
    else:
        df_ls_weekly = df_ls.resample("W", on="datetime").mean(numeric_only=True).reset_index()
        df_orig_weekly = df_orig.resample("W", on="datetime").mean(numeric_only=True).reset_index()
        ax.plot(df_ls_weekly.index, df_ls_weekly["COP"], color="#2b7bba", linewidth=3.5, label="Shifted Load")
        ax.plot(df_orig_weekly.index, df_orig_weekly["COP"], color="#f52121", linewidth=3.5, label="Original Load")
        ax.set_xlabel("Week Number")
        ax.set_ylabel("COP")
        ax.set_title("COP Weekly Average", pad=20)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=False)
    save_figure(fig, PUBLICATION_OUTPUT_DIR / "fig6_cop_pareto_and_weekly_stack.svg", show=False)


def plot_price_vs_time(df_ls_input: pd.DataFrame, df_orig_input: pd.DataFrame, x_min=None, x_max=None, y_min=None, y_max=3) -> None:
    df_ls = df_ls_input.copy()
    df_ls = df_ls.resample("W", on="datetime").sum(numeric_only=True).reset_index()
    df_orig = df_orig_input.copy()
    df_orig = df_orig.resample("W", on="datetime").sum(numeric_only=True).reset_index()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(df_orig["datetime"], df_orig["paid_price"], color="#f52121", linewidth=3.5, label="Original Load")
    ax.plot(df_ls["datetime"], df_ls["paid_price"], color="#2b7bba", linewidth=3.5, label="Shifted Load")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price (€)")
    ax.set_title("Price Weekly Total", pad=20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    if x_max is not None and x_min is not None:
        ax.set_xlim(x_min, x_max)
    if y_max is not None and y_min is not None:
        ax.set_ylim(y_min, y_max)
    ax.legend(frameon=False)
    plt.tight_layout()
    save_figure(fig, PUBLICATION_OUTPUT_DIR / "price_weekly_total.svg", show=False)


def plot_multi_price_pareto(
    df_ls_input: pd.DataFrame,
    df_orig_input: pd.DataFrame,
    zoom_orig_lims=None,
    zoom1_lims=None,
    zoom2_lims=None,
) -> None:
    """07-style multi-scale price Pareto plot."""
    df_ls = df_ls_input.copy()
    df_orig = df_orig_input.copy()

    sorted_price_ls = df_ls["paid_price"].sort_values(ascending=False).reset_index(drop=True)
    sorted_price_orig = df_orig["paid_price"].sort_values(ascending=False).reset_index(drop=True)

    layout_structure = [["main", "zoom_top"], ["main", "zoom_btm"]]
    fig, ax_dict = plt.subplot_mosaic(layout_structure, figsize=(14, 14), constrained_layout=True)

    def render_subplot(ax_key, title=None, x_lims=None, y_lims=None, show_legend=False, show_ylabels=True, set_title=True):
        ax = ax_dict[ax_key]
        ax.plot(sorted_price_orig.index, sorted_price_orig, color="#f52121", linewidth=3.5, label="Original Load")
        ax.plot(sorted_price_ls.index, sorted_price_ls, color="#2b7bba", linewidth=3.5, label="Shifted Load")
        if set_title:
            ax.set_title(title, pad=15, fontsize=42)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        if x_lims:
            ax.set_xlim(x_lims)
        else:
            ax.set_xlim(0, len(sorted_price_ls))
        if y_lims:
            ax.set_ylim(y_lims)
        if ax_key == "main":
            ax.set_xlabel("Sorted Observations", fontsize=38)
        if show_ylabels:
            ax.set_ylabel("Price (€)", fontsize=38)
        if show_legend:
            ax.legend(frameon=False, loc="upper right")
        ax.tick_params(axis="both", which="major", labelsize=32)

    z_orig_x = (zoom_orig_lims[0], zoom_orig_lims[1]) if zoom_orig_lims else (0, len(sorted_price_ls))
    z_orig_y = (zoom_orig_lims[2], zoom_orig_lims[3]) if zoom_orig_lims else (0, max(sorted_price_orig.max(), sorted_price_ls.max()))
    render_subplot("main", title="Global Price Pareto Curve", x_lims=z_orig_x, y_lims=z_orig_y, show_legend=True)

    z1_x = (zoom1_lims[0], zoom1_lims[1]) if zoom1_lims else (0, int(len(sorted_price_ls) * 0.25))
    z1_y = (zoom1_lims[2], zoom1_lims[3]) if zoom1_lims else (z_orig_y[1] * 0.5, z_orig_y[1])
    render_subplot("zoom_top", title="Detail: High Price Region", x_lims=z1_x, y_lims=z1_y, show_ylabels=False, set_title=False)

    z2_x = (zoom2_lims[0], zoom2_lims[1]) if zoom2_lims else (int(len(sorted_price_ls) * 0.5), len(sorted_price_ls))
    z2_y = (zoom2_lims[2], zoom2_lims[3]) if zoom2_lims else (0, z_orig_y[1] * 0.2)
    render_subplot("zoom_btm", title="Detail: Low Price Region", x_lims=z2_x, y_lims=z2_y, show_ylabels=False, set_title=False)

    main_ax = ax_dict["main"]
    zoom_top_ax = ax_dict["zoom_top"]
    zoom_btm_ax = ax_dict["zoom_btm"]

    rect_style = dict(fill=True, facecolor="#ffbf00", alpha=0.18, linewidth=5.5, linestyle="--", edgecolor="#bf8000")
    rect_z1 = patches.Rectangle((z1_x[0], z1_y[0]), z1_x[1] - z1_x[0], z1_y[1] - z1_y[0], **rect_style)
    rect_z2 = patches.Rectangle((z2_x[0], z2_y[0]), z2_x[1] - z2_x[0], z2_y[1] - z2_y[0], **rect_style)
    main_ax.add_patch(rect_z1)
    main_ax.add_patch(rect_z2)

    z1_center = (z1_x[0] + (z1_x[1] - z1_x[0]) / 2, z1_y[0] + (z1_y[1] - z1_y[0]) / 2)
    z2_center = (z2_x[0] + (z2_x[1] - z2_x[0]) / 2, z2_y[0] + (z2_y[1] - z2_y[0]) / 2)
    arrow_style = dict(arrowstyle="->", color="#555555", lw=7.0, mutation_scale=35, zorder=10)
    conn_top = patches.ConnectionPatch(
        xyA=z1_center, coordsA=main_ax.transData,
        xyB=(0.0, 0.5), coordsB=zoom_top_ax.transAxes,
        **arrow_style,
    )
    conn_top.set_clip_on(False)
    fig.add_artist(conn_top)

    conn_btm = patches.ConnectionPatch(
        xyA=z2_center, coordsA=main_ax.transData,
        xyB=(0.0, 0.5), coordsB=zoom_btm_ax.transAxes,
        **arrow_style,
    )
    conn_btm.set_clip_on(False)
    fig.add_artist(conn_btm)

    plt.suptitle("Multi-Scale Price Pareto Analysis", y=1.08, fontsize=48)
    save_figure(fig, PUBLICATION_OUTPUT_DIR / "fig7_multi_scale_price_pareto.svg", show=False)


def plot_multi_energy_pareto(
    df_ls_input: pd.DataFrame,
    df_orig_input: pd.DataFrame,
    zoom_orig_lims=None,
    zoom1_lims=None,
    zoom2_lims=None,
) -> None:
    """07-style multi-scale energy Pareto plot."""
    df_ls = df_ls_input.copy()
    df_orig = df_orig_input.copy()

    sorted_price_ls = df_ls["E_shifted"].sort_values(ascending=False).reset_index(drop=True)
    sorted_price_orig = df_orig["EE_total"].sort_values(ascending=False).reset_index(drop=True)

    layout_structure = [["main", "zoom_top"], ["main", "zoom_btm"]]
    fig, ax_dict = plt.subplot_mosaic(layout_structure, figsize=(14, 14), constrained_layout=True)

    def render_subplot(ax_key, title=None, x_lims=None, y_lims=None, show_legend=False, show_ylabels=True, set_title=True):
        ax = ax_dict[ax_key]
        ax.plot(sorted_price_orig.index, sorted_price_orig, color="#f52121", linewidth=3.5, label="Original Load")
        ax.plot(sorted_price_ls.index, sorted_price_ls, color="#2b7bba", linewidth=3.5, label="Shifted Load")
        if set_title:
            ax.set_title(title, pad=15, fontsize=42)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        if x_lims:
            ax.set_xlim(x_lims)
        else:
            ax.set_xlim(0, len(sorted_price_ls))
        if y_lims:
            ax.set_ylim(y_lims)
        if ax_key == "main":
            ax.set_xlabel("Sorted Observations", fontsize=38)
        if show_ylabels:
            ax.set_ylabel("Power Load (kW)", fontsize=38)
        if show_legend:
            ax.legend(frameon=False, loc="upper right")
        ax.tick_params(axis='both', which='major', labelsize=32)

    z_orig_x = (zoom_orig_lims[0], zoom_orig_lims[1]) if zoom_orig_lims else (0, len(sorted_price_ls))
    z_orig_y = (zoom_orig_lims[2], zoom_orig_lims[3]) if zoom_orig_lims else (0, max(sorted_price_orig.max(), sorted_price_ls.max()))
    render_subplot("main", title="Global Load Duration Curve", x_lims=z_orig_x, y_lims=z_orig_y, show_legend=True)

    z1_x = (zoom1_lims[0], zoom1_lims[1]) if zoom1_lims else (0, int(len(sorted_price_ls) * 0.25))
    z1_y = (zoom1_lims[2], zoom1_lims[3]) if zoom1_lims else (z_orig_y[1] * 0.5, z_orig_y[1])
    render_subplot("zoom_top", title="Detail: High Demand Region", x_lims=z1_x, y_lims=z1_y, show_ylabels=False, set_title=False)

    z2_x = (zoom2_lims[0], zoom2_lims[1]) if zoom2_lims else (int(len(sorted_price_ls) * 0.5), len(sorted_price_ls))
    z2_y = (zoom2_lims[2], zoom2_lims[3]) if zoom2_lims else (0, z_orig_y[1] * 0.2)
    render_subplot("zoom_btm", title="Detail: Low Demand Region", x_lims=z2_x, y_lims=z2_y, show_ylabels=False, set_title=False)

    main_ax = ax_dict["main"]
    zoom_top_ax = ax_dict["zoom_top"]
    zoom_btm_ax = ax_dict["zoom_btm"]

    rect_style = dict(fill=True, facecolor="#ffbf00", alpha=0.18, linewidth=5.5, linestyle="--", edgecolor="#bf8000")
    rect_z1 = patches.Rectangle((z1_x[0], z1_y[0]), z1_x[1] - z1_x[0], z1_y[1] - z1_y[0], **rect_style)
    rect_z2 = patches.Rectangle((z2_x[0], z2_y[0]), z2_x[1] - z2_x[0], z2_y[1] - z2_y[0], **rect_style)
    main_ax.add_patch(rect_z1)
    main_ax.add_patch(rect_z2)

    z1_center = (z1_x[0] + (z1_x[1] - z1_x[0]) / 2, z1_y[0] + (z1_y[1] - z1_y[0]) / 2)
    z2_center = (z2_x[0] + (z2_x[1] - z2_x[0]) / 2, z2_y[0] + (z2_y[1] - z2_y[0]) / 2)
    arrow_style = dict(arrowstyle="->", color="#555555", lw=7.0, mutation_scale=35, zorder=10)
    conn_top = patches.ConnectionPatch(
        xyA=z1_center, coordsA=main_ax.transData,
        xyB=(0.0, 0.5), coordsB=zoom_top_ax.transAxes,
        **arrow_style,
    )
    conn_top.set_clip_on(False)
    fig.add_artist(conn_top)

    conn_btm = patches.ConnectionPatch(
        xyA=z2_center, coordsA=main_ax.transData,
        xyB=(0.0, 0.5), coordsB=zoom_btm_ax.transAxes,
        **arrow_style,
    )
    conn_btm.set_clip_on(False)
    fig.add_artist(conn_btm)

    plt.suptitle("Multi-Scale Energy Pareto Analysis", y=1.08, fontsize=48)
    save_figure(fig, PUBLICATION_OUTPUT_DIR / "fig4_multi_scale_energy_pareto.svg", show=False)


def plot_single_day_load_shifting(df_ls_input: pd.DataFrame, date: str, include_price: bool = False, save_prefix: str = "") -> None:
    df_ls = df_ls_input.copy()
    df_ls = df_ls[df_ls["date"] == date]
    df_ls["hour"] = df_ls["datetime"].dt.hour
    fig, ax = plt.subplots(figsize=(14, 8))
    ln1 = ax.plot(df_ls["hour"], df_ls["E_shifted"], label="Shifted Load", color="#2b7bba", linewidth=3.5)
    ln2 = ax.plot(df_ls["hour"], df_ls["EE_total"], label="Original Load", color="#f52121", linewidth=3.5)

    if include_price:
        ax2 = ax.twinx()
        ln3 = ax2.plot(df_ls["hour"], df_ls["price"], label="Price (€)", color="#2ec926", linestyle='--', linewidth=3.5)
        ax2.set_ylabel("Price (€)")
        ax2.set_ylim(bottom=0)
        ax2.grid(False)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Energy (kW)")
    ax.set_title(f"Load Shifting on {date}", pad=20)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    lines = ln1 + ln2 + (ln3 if include_price else [])
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, frameon=False)
    ax.set_xticks(list(range(0, 24, 3)))
    ax.set_xticklabels(list(range(0, 24, 3)))
    ax.set_xlim(-0.5, 23.5)
    plt.tight_layout()
    save_figure(fig, PUBLICATION_OUTPUT_DIR / f"{save_prefix}_single_day_load_shifting_{date}.svg", show=False)


def plot_price_dashboard(
    df_ls_input: pd.DataFrame,
    df_orig_input: pd.DataFrame,
    extreme_high_date=None,
    extreme_midday_date=None,
    *,
    sample_window: bool = False,
) -> None:
    df_ls = df_ls_input.copy()
    df_orig = df_orig_input.copy()
    df_ls["datetime"] = pd.to_datetime(df_ls["datetime"])
    df_orig["datetime"] = pd.to_datetime(df_orig["datetime"])
    if "price_per_kWh" not in df_ls.columns:
        df_ls["price_per_kWh"] = df_ls["price"] / 1000
    if "price_per_kWh" not in df_orig.columns:
        df_orig["price_per_kWh"] = df_orig["price"] / 1000
    df_ls["date"] = df_ls["datetime"].dt.date
    df_ls["hour"] = df_ls["datetime"].dt.hour
    df_orig["date"] = df_orig["datetime"].dt.date
    df_orig["hour"] = df_orig["datetime"].dt.hour

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)

    title_time = "Energy Price vs Time (sample window)" if sample_window else "Annual Energy Price vs Time"
    title_pareto = "Energy Price Pareto Diagram (sample window)" if sample_window else "Energy Price Pareto Diagram"
    title_hour = (
        "Average Energy Price vs Hour (weekdays, sample)"
        if sample_window
        else "Average Energy Price vs Hour (Weekdays)"
    )
    title_two_days = "Energy Price for Two Different Days" if not sample_window else "Energy price — two highlighted days (sample)"

    daily_ls = df_ls.resample("D", on="datetime").mean(numeric_only=True).reset_index()
    ax = axes[0, 0]
    ax.plot(daily_ls["datetime"], daily_ls["price_per_kWh"], color="#2b7bba", linewidth=3.5, label="Energy Price")
    ax.set_title(title_time)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (€/kWh)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    sorted_price = df_ls["price_per_kWh"].sort_values(ascending=False).reset_index(drop=True)
    ax.plot(sorted_price.index, sorted_price, color="#2b7bba", linewidth=3.5, label="Energy Price")
    ax.set_title(title_pareto)
    ax.set_xlabel("Observation Count (Sorted)")
    ax.set_ylabel("Price (€/kWh)")
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    weekdays_ls = df_ls[df_ls["datetime"].dt.weekday < 5]
    hourly_mean_ls = weekdays_ls.groupby("hour")["price_per_kWh"].mean()
    ax.plot(hourly_mean_ls.index, hourly_mean_ls.values, color="#2b7bba", linewidth=3.5, label="Energy Price")
    ax.set_title(title_hour)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Price (€/kWh)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    daily_cost = df_ls.groupby("date")["price_per_kWh"].mean().sort_values(ascending=False)
    detected_high_cost_day = daily_cost.index[0]
    df_mid = df_ls[df_ls["hour"].between(11, 16)]
    midday_mean = df_mid.groupby("date")["price_per_kWh"].mean()
    daily_mean = df_ls.groupby("date")["price_per_kWh"].mean()
    midday_ratio = (midday_mean / daily_mean).dropna().sort_values()
    detected_midday_reduction_day = midday_ratio.index[0] if not midday_ratio.empty else detected_high_cost_day
    if extreme_high_date is not None:
        parsed_high_date = pd.to_datetime(extreme_high_date).date()
        high_cost_day = parsed_high_date if parsed_high_date in daily_cost.index else detected_high_cost_day
    else:
        high_cost_day = detected_high_cost_day
    if extreme_midday_date is not None:
        parsed_midday_date = pd.to_datetime(extreme_midday_date).date()
        reduced_midday_day = parsed_midday_date if parsed_midday_date in daily_cost.index else detected_midday_reduction_day
    else:
        reduced_midday_day = detected_midday_reduction_day
    profile_high = df_ls[df_ls["date"] == high_cost_day].groupby("hour")["price_per_kWh"].mean()
    profile_reduced = df_ls[df_ls["date"] == reduced_midday_day].groupby("hour")["price_per_kWh"].mean()
    ax.plot(profile_high.index, profile_high.values, color="#f52121", linewidth=3.5, label=f"{high_cost_day}")
    ax.plot(profile_reduced.index, profile_reduced.values, color="#2b7bba", linewidth=3.5, label=f"{reduced_midday_day}")
    ax.set_title(title_two_days)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Price (€/kWh)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)

    save_figure(fig, PUBLICATION_OUTPUT_DIR / "fig1_price_dashboard.svg", show=False)


def save_figure(fig: plt.Figure, output_path: Path, show: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_daily_load_profile(df: pd.DataFrame, date_value: str, output_dir: Path, show: bool) -> Optional[Path]:
    day = df[df["datetime"].dt.date == pd.to_datetime(date_value).date()].copy()
    if day.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    ax1.plot(day["datetime"], day["EE_total"], color="black", linewidth=3.5, label="Original Load")
    ax1.plot(day["datetime"], day["E_shifted"], color="#c92a2a", linewidth=3.5, label="Shifted Load")

    if "E_min" in day.columns:
        ax1.plot(day["datetime"], day["E_min"], color="#2f9e44", linestyle="--", linewidth=2.0, label="Min Load")
    if "E_max" in day.columns:
        ax1.plot(day["datetime"], day["E_max"], color="#f08c00", linestyle="--", linewidth=2.0, label="Max Load")

    ax2.plot(day["datetime"], day["price"], color="#3b5bdb", linewidth=2.5, label="Price")

    if "openning_time" in day.columns and day["openning_time"].notna().any():
        opening = int(day["openning_time"].dropna().iloc[0])
        open_dt = pd.to_datetime(f"{date_value} {opening:02d}:00:00")
        ax1.axvline(open_dt, color="#0b7285", linestyle=":", linewidth=2.5, label="Opening Time")

    if "closing_time" in day.columns and day["closing_time"].notna().any():
        closing = int(day["closing_time"].dropna().iloc[0])
        close_dt = pd.to_datetime(f"{date_value} {closing:02d}:00:00")
        ax1.axvline(close_dt, color="#862e9c", linestyle=":", linewidth=2.5, label="Closing Time")

    ax1.set_title(f"Load Shifting Profile - {date_value}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Energy (kWh)")
    ax2.set_ylabel("Price (EUR/MWh)")
    ax2.tick_params(axis="y", labelcolor="#3b5bdb")

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.grid(True, alpha=0.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

    plt.setp(ax1.get_xticklabels(), rotation=45)
    fig.tight_layout()

    out = output_dir / f"daily_profile_{date_value}.svg"
    save_figure(fig, out, show)
    return out


def plot_total_energy_bar(df: pd.DataFrame, output_dir: Path, show: bool) -> Path:
    values = [df["EE_total"].sum(), df["E_shifted"].sum()]
    labels = ["Original Load", "Shifted Load"]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(labels, values, color=["#2b8a3e", "#1c7ed6"], alpha=0.85, edgecolor="black")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.1f}", ha="center", va="bottom")

    ax.set_title("Total Energy Comparison")
    ax.set_ylabel("Energy (kWh)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out = output_dir / "total_energy_comparison.svg"
    save_figure(fig, out, show)
    return out


def plot_total_cost_bar(df: pd.DataFrame, output_dir: Path, show: bool) -> Path:
    values = [df["cost_EE_total"].sum(), df["cost_E_shifted"].sum()]
    labels = ["Original Load", "Shifted Load"]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(labels, values, color=["#f08c00", "#7048e8"], alpha=0.85, edgecolor="black")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"EUR {h:.2f}", ha="center", va="bottom")

    ax.set_title("Total Cost Comparison")
    ax.set_ylabel("Cost (EUR)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out = output_dir / "total_cost_comparison.svg"
    save_figure(fig, out, show)
    return out


def plot_cost_time_series(df: pd.DataFrame, output_dir: Path, show: bool) -> Path:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(df["datetime"], df["cost_EE_total"], color="#c92a2a", linewidth=3.5, label="Original Cost")
    ax.plot(df["datetime"], df["cost_E_shifted"], color="#1971c2", linewidth=3.5, label="Shifted Cost")

    ax.set_title("Hourly Cost Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cost per timestep (EUR)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = output_dir / "cost_timeseries.svg"
    save_figure(fig, out, show)
    return out


def plot_cost_weekly(df: pd.DataFrame, output_dir: Path, show: bool) -> Path:
    weekly = (
        df.set_index("datetime")[["cost_EE_total", "cost_E_shifted"]]
        .resample("W")
        .sum()
        .reset_index()
    )

    x = np.arange(len(weekly))
    width = 0.42

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width / 2.0, weekly["cost_EE_total"], width=width, label="Original Cost", color="#e8590c")
    ax.bar(x + width / 2.0, weekly["cost_E_shifted"], width=width, label="Shifted Cost", color="#364fc7")

    ax.set_title("Weekly Cost Comparison")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cost (EUR)")
    tick_step = max(1, len(weekly) // 12)
    tick_idx = x[::tick_step]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(weekly.loc[tick_idx, "datetime"].dt.strftime("%Y-%m-%d"), rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = output_dir / "cost_weekly.svg"
    save_figure(fig, out, show)
    return out


def plot_cop_regression(df_fit: pd.DataFrame, model: object, degree: int, output_dir: Path, show: bool) -> Path:
    x = df_fit[["RA_EE_total"]].values
    y = df_fit["COP_measured"].values

    x_range = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_pred = model.predict(x_range)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(df_fit["RA_EE_total"], y, s=15, alpha=0.35, color="#495057", label="Observed COP")
    ax.plot(x_range, y_pred, color="#d6336c", linewidth=3.5, label=f"Polynomial Fit (degree {degree})")

    ax.set_title("COP Regression Model")
    ax.set_xlabel("Rolling Mean Electrical Load")
    ax.set_ylabel("COP")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = output_dir / "cop_regression_model.svg"
    save_figure(fig, out, show)
    return out


def plot_cop_pareto(df: pd.DataFrame, output_dir: Path, show: bool) -> Optional[Path]:
    valid = df.dropna(subset=["COP_pred_original", "COP_pred_shifted"]).copy()
    if valid.empty:
        return None

    cop_orig = valid["COP_pred_original"].sort_values(ascending=False).reset_index(drop=True)
    cop_shift = valid["COP_pred_shifted"].sort_values(ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(cop_orig.index, cop_orig.values, color="#c92a2a", linewidth=3.5, label="Original COP")
    ax.plot(cop_shift.index, cop_shift.values, color="#1971c2", linewidth=3.5, label="Shifted COP")

    ax.set_title("COP Pareto Curve")
    ax.set_xlabel("Sorted Observations")
    ax.set_ylabel("Predicted COP")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = output_dir / "cop_pareto_curve.svg"
    save_figure(fig, out, show)
    return out


def build_parser(default_price_csv: Path, default_output: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single publication-ready script for load shifting methodology: "
            "Load_Shifting_Price math, variance metrics, and static figures."
        )
    )
    parser.add_argument("--price-csv", type=Path, default=default_price_csv)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--metrics-json", type=Path, default=default_output / "metrics.json")
    parser.add_argument("--show", action="store_true", help="Display figures interactively in addition to saving")
    parser.add_argument(
        "--public-sample",
        action="store_true",
        help="Publication sample mode: honest chart titles, adaptive Pareto limits, omit monthly block in metrics.",
    )
    parser.add_argument(
        "--dashboard-high-date",
        type=str,
        default=None,
        help="YYYY-MM-DD for dashboard high-price day profile (default: first calendar day in CSV).",
    )
    parser.add_argument(
        "--dashboard-midday-date",
        type=str,
        default=None,
        help="YYYY-MM-DD for dashboard midday-profile day (default: second calendar day, or first if only one).",
    )
    return parser


def main() -> None:
    configure_plot_style()

    project_root = Path(__file__).resolve().parents[1]
    default_price_csv = project_root / "data" / "load-shifting-price.csv"
    default_output = project_root / "figures"

    parser = build_parser(default_price_csv, default_output)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_price_data(args.price_csv)
    df_ls, df_orig, cop_model, cop_cutoff = prepare_data_from_price_file(df)

    span_info = data_span_info(df_ls)
    short_calendar = insufficient_span_for_weekly_cop(span_info)
    sample_mode = bool(args.public_sample) or short_calendar
    use_adaptive_price_pareto = bool(args.public_sample) or short_calendar or (len(df_ls) < 3500)
    use_daily_cop_panel = short_calendar

    total_variance = compute_variance_metrics(df_ls)
    monthly_key = str(df_ls["datetime"].dt.to_period("M").min())
    monthly_variance = calculate_monthly_variance(df_ls, monthly_key)
    dates = sorted(df_ls["datetime"].dt.date.astype(str).unique().tolist())
    daily_variances = {date_value: calculate_daily_variance(df_ls, date_value) for date_value in dates}
    peak_quartiles = calculate_daily_peak_reduction_quartiles(df_ls)

    high_d, mid_d = pick_dashboard_highlight_dates(dates, args.dashboard_high_date, args.dashboard_midday_date)

    plot_adjustments: List[str] = []
    plots_skipped: List[Dict[str, str]] = []
    if use_adaptive_price_pareto:
        plot_adjustments.append("fig7: adaptive Pareto axis limits (short series or public-sample mode)")
    if use_daily_cop_panel:
        plot_adjustments.append("fig6: weekly COP panel replaced by daily mean (calendar span < 7 days)")

    generated_plots: List[str] = []
    plot_price_dashboard(
        df_ls,
        df_orig,
        extreme_high_date=high_d,
        extreme_midday_date=mid_d,
        sample_window=sample_mode,
    )
    plot_single_day_load_shifting(df_ls, dates[0], include_price=True, save_prefix="fig2")
    if len(dates) > 1:
        plot_single_day_load_shifting(df_ls, dates[1], include_price=True, save_prefix="fig3")
    if use_adaptive_price_pareto:
        plot_multi_price_pareto(df_ls, df_orig, zoom_orig_lims=None, zoom1_lims=None, zoom2_lims=None)
    else:
        zo, z1, z2 = default_price_pareto_paper_zooms()
        plot_multi_price_pareto(df_ls, df_orig, zoom_orig_lims=zo, zoom1_lims=z1, zoom2_lims=z2)
    plot_multi_energy_pareto(df_ls, df_orig)
    if cop_model is not None:
        cop_vs_time(df_ls)
        plot_cop_pareto_and_weekly_stack(df_ls, df_orig, use_daily_cop_panel=use_daily_cop_panel)

    generated_plots.extend(
        [
            "price_dashboard",
            f"single_day_{dates[0]}",
            "multi_price_pareto",
            "multi_energy_pareto",
        ]
    )
    if len(dates) > 1:
        generated_plots.append(f"single_day_{dates[1]}")
    if "COP" in df_ls.columns:
        generated_plots.extend(["cop_vs_time", "cop_pareto_and_weekly_stack"])

    variance_block: Dict[str, Any] = {
        "total": total_variance,
        "daily_selected_dates": daily_variances,
        "daily_peak_reduction_quartiles": peak_quartiles,
    }
    if sample_mode:
        variance_block["monthly_first_period_omitted"] = (
            "public_sample_or_short_span: monthly slice not reported for publication sample"
        )
    else:
        variance_block["monthly_first_period"] = {monthly_key: monthly_variance}

    metrics: Dict[str, Any] = {
        "publication_mode": "sample" if sample_mode else "full",
        "data_span": span_info,
        "inputs": {
            "price_csv": str(args.price_csv),
            "records": int(df_ls.shape[0]),
        },
        "variance": variance_block,
        "plots": generated_plots,
        "plot_adjustments": plot_adjustments,
        "plots_skipped": plots_skipped,
    }

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Publication methodology pipeline completed.")
    print(f"Records: {df_ls.shape[0]}")
    print(f"Metrics JSON: {args.metrics_json}")
    print(f"Plots generated: {len(generated_plots)}")
    print("Total variance summary:")
    print(json.dumps(total_variance, indent=2))


if __name__ == "__main__":
    main()
