"""
All Plotly chart builders for the EM Credit Analytics Dashboard.

Each function returns a fully configured go.Figure.
Import apply_chart_style from config.theme to apply the standard layout.

Conventions:
  - Function names start with 'plot_' for charts and 'fig_' for composite figures.
  - All functions accept optional title / height parameters.
  - Colour constants are imported from config.settings.COLORS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config.settings import COLORS
from config.theme import (
    DIVERGING_CORR,
    DIVERGING_RdGn,
    DIVERGING_RdBl,
    SEQUENTIAL_CYAN,
    apply_chart_style,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Correlation heatmap
# ═════════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(
    corr: pd.DataFrame,
    title: str = "Correlation Matrix",
    height: int = 520,
    annot: bool = True,
) -> go.Figure:
    """Static Pearson correlation heatmap."""
    labels = corr.columns.tolist()
    z      = corr.values

    text = [[f"{z[i][j]:.2f}" for j in range(len(labels))] for i in range(len(labels))] if annot else None

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10, "color": COLORS["text"]},
            colorscale=DIVERGING_CORR,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                tickfont=dict(color=COLORS["text_muted"], size=10),
                thickness=12,
            ),
            hoverongaps=False,
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        )
    )
    return apply_chart_style(fig, title=title, height=height)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Rolling correlation / beta line charts
# ═════════════════════════════════════════════════════════════════════════════

def plot_rolling_correlations(
    roll_corr_dict: dict[int, pd.DataFrame],
    target_name: str = "Bond",
    selected_factors: list[str] | None = None,
    height: int = 380,
) -> go.Figure:
    """
    Multi-line chart of rolling correlation vs. selected factors,
    with separate traces per rolling window.
    """
    fig = go.Figure()

    line_styles = {20: "dot", 60: "solid", 120: "dash"}
    color_palette = [
        COLORS["primary"], "#FF6B6B", "#FFE66D",
        "#A8E6CF", "#D4A5FF", "#FFB347",
    ]

    for w_idx, (window, df) in enumerate(roll_corr_dict.items()):
        cols = selected_factors or df.columns.tolist()
        dash = line_styles.get(window, "solid")

        for c_idx, col in enumerate(cols):
            if col not in df.columns:
                continue
            color = color_palette[c_idx % len(color_palette)]
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=f"{col} ({window}d)",
                    line=dict(color=color, dash=dash, width=1.5),
                    opacity=0.85,
                )
            )

    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], line_width=1)

    fig.update_yaxes(range=[-1.05, 1.05])
    return apply_chart_style(
        fig, title=f"Rolling Correlation: {target_name} vs. Macro Factors", height=height
    )


def plot_cross_correlation_bars(
    cross_corr: pd.Series,
    x_name: str,
    y_name: str,
    highlight_lags: list[int] | None = None,
    height: int = 380,
) -> go.Figure:
    """
    Bar chart of cross-correlation at each lag.

    Convention: lag > 0 means x leads y (x moved first).
    Dashed lines show 95% confidence band (±1.96/√n).
    """
    n = cross_corr.notna().sum()
    sig = 1.96 / np.sqrt(max(n, 2))

    lags = cross_corr.index.tolist()
    vals = cross_corr.values

    colours = []
    for i, (lag, val) in enumerate(zip(lags, vals)):
        if highlight_lags and lag in highlight_lags:
            colours.append(COLORS["warning"])
        elif val > 0:
            colours.append(COLORS["primary"])
        else:
            colours.append(COLORS["secondary"])

    fig = go.Figure(
        go.Bar(
            x=lags,
            y=vals,
            marker=dict(color=colours, opacity=0.8),
            hovertemplate="Lag %{x}d<br>Correlation: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_hline(y=0,    line_color=COLORS["neutral"],  line_width=1)
    fig.add_hline(y=sig,  line_dash="dash", line_color=COLORS["positive"], line_width=1,
                  annotation_text="95% CI", annotation_position="top right")
    fig.add_hline(y=-sig, line_dash="dash", line_color=COLORS["positive"], line_width=1)

    fig.update_xaxes(
        title_text=f"Lag (days)  |  + = {x_name} leads {y_name}",
        tickmode="linear", dtick=5,
    )
    fig.update_yaxes(title_text="Pearson Correlation", range=[-1.05, 1.05])
    return apply_chart_style(
        fig,
        title=f"Cross-Correlation: {y_name} vs {x_name}",
        height=height,
    )


def plot_focused_rolling_correlation(
    y: pd.Series,
    x: pd.Series,
    window: int,
    y_name: str,
    x_name: str,
    height: int = 350,
) -> go.Figure:
    """
    Single rolling correlation line between two series, for a chosen window.
    """
    roll = y.rolling(window=window, min_periods=max(5, window // 4)).corr(x)

    fig = go.Figure(
        go.Scatter(
            x=roll.index,
            y=roll.values,
            mode="lines",
            name=f"{window}d rolling corr",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.07)",
            hovertemplate="%{x|%Y-%m-%d}<br>Corr: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0,    line_dash="dot", line_color=COLORS["neutral"], line_width=1)
    fig.add_hline(y=0.5,  line_dash="dot", line_color=COLORS["positive"], line_width=0.8, opacity=0.5)
    fig.add_hline(y=-0.5, line_dash="dot", line_color=COLORS["negative"], line_width=0.8, opacity=0.5)

    fig.update_yaxes(title_text="Correlation", range=[-1.05, 1.05])
    return apply_chart_style(
        fig,
        title=f"Rolling {window}d Correlation: {y_name} vs {x_name}",
        height=height,
    )


def plot_rolling_beta(
    beta_series: pd.Series,
    factor_name: str,
    window: int,
    height: int = 300,
) -> go.Figure:
    """Single rolling beta time series."""
    fig = go.Figure(
        go.Scatter(
            x=beta_series.index,
            y=beta_series.values,
            mode="lines",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.08)",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
    return apply_chart_style(
        fig,
        title=f"Rolling {window}d Beta to {factor_name}",
        height=height,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3. OAS time series
# ═════════════════════════════════════════════════════════════════════════════

def plot_oas_history(
    oas_df: pd.DataFrame,
    bond_ids: list[str],
    title: str = "OAS History (bps)",
    height: int = 380,
) -> go.Figure:
    """Multi-line OAS history chart."""
    fig = go.Figure()
    color_palette = px.colors.qualitative.Plotly

    for i, bid in enumerate(bond_ids):
        if bid not in oas_df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=oas_df.index,
                y=oas_df[bid],
                mode="lines",
                name=bid,
                line=dict(width=1.8, color=color_palette[i % len(color_palette)]),
            )
        )

    fig.update_yaxes(title_text="OAS (bps)")
    return apply_chart_style(fig, title=title, height=height)


# ═════════════════════════════════════════════════════════════════════════════
# 4. RV screener: OAS vs. fitted curve scatter
# ═════════════════════════════════════════════════════════════════════════════

def plot_rv_scatter(
    curve_result: dict,
    country: str,
    height: int = 400,
) -> go.Figure:
    """
    Scatter of actual OAS vs. maturity with the fitted curve overlaid.
    Residuals shown via colour (green=cheap, red=rich).
    """
    if not curve_result:
        return go.Figure()

    mats     = curve_result["maturities"]
    oas_act  = curve_result["oas_actual"]
    oas_fit  = curve_result["oas_fitted"]
    resids   = curve_result["residuals"]
    bond_ids = curve_result["bond_ids"]

    # Generate a smooth fitted curve for display
    x_smooth = np.linspace(mats.min(), mats.max(), 200)
    y_smooth = np.polyval(curve_result["coeffs"], x_smooth)

    # Colour: green if cheap (residual > 0), red if rich
    colours = [COLORS["positive"] if r > 0 else COLORS["negative"] for r in resids]

    fig = go.Figure()

    # Fitted curve
    fig.add_trace(
        go.Scatter(
            x=x_smooth, y=y_smooth,
            mode="lines",
            name="Fitted curve",
            line=dict(color=COLORS["primary"], dash="dash", width=2),
        )
    )

    # Actual points
    fig.add_trace(
        go.Scatter(
            x=mats, y=oas_act,
            mode="markers+text",
            name="Actual OAS",
            marker=dict(size=12, color=colours, line=dict(color="white", width=1)),
            text=[f"{b}<br>Resid: {r:+.0f}bps" for b, r in zip(bond_ids, resids)],
            textposition="top center",
            textfont=dict(size=9, color=COLORS["text_muted"]),
            hovertemplate="<b>%{text}</b><br>Maturity: %{x}Y<br>OAS: %{y:.1f}bps<extra></extra>",
        )
    )

    fig.update_xaxes(title_text="Maturity (Years)")
    fig.update_yaxes(title_text="OAS (bps)")
    return apply_chart_style(fig, title=f"{country} — OAS vs. Fitted Curve", height=height)


def plot_rv_zscore_bar(
    rv_df: pd.DataFrame,
    height: int = 380,
) -> go.Figure:
    """
    Horizontal bar chart of RV z-scores for all bonds.
    Bars are coloured by z-score sign (green=cheap, red=rich).
    """
    sorted_df = rv_df.sort_values("zscore", na_position="last")
    colours   = [
        COLORS["positive"] if z > 0 else COLORS["negative"]
        for z in sorted_df["zscore"].fillna(0)
    ]

    fig = go.Figure(
        go.Bar(
            y=sorted_df["bond_id"],
            x=sorted_df["zscore"],
            orientation="h",
            marker=dict(color=colours, line=dict(color="rgba(0,0,0,0)", width=0)),
            hovertemplate="<b>%{y}</b><br>Z-score: %{x:.2f}<extra></extra>",
        )
    )

    fig.add_vline(x=1.5,  line_dash="dash", line_color=COLORS["positive"], line_width=1)
    fig.add_vline(x=-1.5, line_dash="dash", line_color=COLORS["negative"], line_width=1)
    fig.add_vline(x=0,    line_color=COLORS["neutral"], line_width=0.5)

    fig.update_xaxes(title_text="Z-Score (+ = Cheap, - = Rich)")
    return apply_chart_style(fig, title="RV Z-Scores Across EM Universe", height=height)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Curve trade charts
# ═════════════════════════════════════════════════════════════════════════════

def plot_yield_curve(
    curve_df: pd.DataFrame,
    dates: list | None = None,
    title: str = "EM Credit Curve",
    height: int = 370,
) -> go.Figure:
    """
    Plot one or several snapshots of a country's OAS curve (OAS vs. maturity).

    Parameters
    ----------
    curve_df : DataFrame with columns like '2Y', '5Y', '10Y', '30Y' and date index
    dates    : list of dates to plot; default = latest + 3 historical snapshots
    """
    fig = go.Figure()

    if dates is None:
        # Show latest + roughly 1m and 3m ago
        n = len(curve_df)
        idx_list = [-1]
        if n >= 22:
            idx_list.append(-22)
        if n >= 65:
            idx_list.append(-65)
        dates = [curve_df.index[i] for i in idx_list]

    colour_seq = [COLORS["primary"], COLORS["warning"], COLORS["text_muted"]]
    dash_seq   = ["solid", "dash", "dot"]
    label_seq  = ["Latest", "1M ago", "3M ago"]

    cols = sorted(curve_df.columns, key=lambda c: int(c.replace("Y", "")))
    x    = [int(c.replace("Y", "")) for c in cols]

    for j, dt in enumerate(dates):
        try:
            snap = curve_df.loc[dt, cols]
        except KeyError:
            snap = curve_df.iloc[j][cols] if j < len(curve_df) else None
        if snap is None:
            continue

        fig.add_trace(
            go.Scatter(
                x=x, y=snap.values,
                mode="lines+markers",
                name=label_seq[j] if j < len(label_seq) else str(dt)[:10],
                line=dict(color=colour_seq[j], dash=dash_seq[j], width=2),
                marker=dict(size=7),
            )
        )

    fig.update_xaxes(title_text="Maturity (Years)", tickvals=x)
    fig.update_yaxes(title_text="OAS (bps)")
    return apply_chart_style(fig, title=title, height=height)


def plot_slope_history(
    slope_series: pd.Series,
    zscore_series: pd.Series | None = None,
    metric_name: str = "5Y/10Y Slope",
    height: int = 350,
) -> go.Figure:
    """
    Dual-axis chart: slope (bps, left axis) and z-score (right axis).
    """
    if zscore_series is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    # Slope
    fig.add_trace(
        go.Scatter(
            x=slope_series.index, y=slope_series.values,
            mode="lines", name=metric_name,
            line=dict(color=COLORS["primary"], width=2),
        ),
        **({} if zscore_series is None else {"secondary_y": False}),
    )

    if zscore_series is not None:
        fig.add_trace(
            go.Scatter(
                x=zscore_series.index, y=zscore_series.values,
                mode="lines", name="Z-Score",
                line=dict(color=COLORS["warning"], dash="dot", width=1.5),
                opacity=0.8,
            ),
            secondary_y=True,
        )
        fig.add_hline(y=1.5,  line_dash="dash", line_color=COLORS["positive"], line_width=1)
        fig.add_hline(y=-1.5, line_dash="dash", line_color=COLORS["negative"], line_width=1)
        fig.update_yaxes(title_text="Z-Score", secondary_y=True,
                         tickfont=dict(color=COLORS["warning"]))

    fig.update_yaxes(title_text="Spread (bps)",
                     **({} if zscore_series is None else {"secondary_y": False}))
    return apply_chart_style(fig, title=f"{metric_name} — History & Z-Score", height=height)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Macro ETF overview
# ═════════════════════════════════════════════════════════════════════════════

def plot_macro_performance(
    macro_prices: pd.DataFrame,
    tickers: list[str] | None = None,
    height: int = 380,
) -> go.Figure:
    """
    Normalised total-return index (rebased to 100) for macro ETF proxies.
    """
    tickers = tickers or macro_prices.columns.tolist()
    sub     = macro_prices[[t for t in tickers if t in macro_prices.columns]].copy()
    rebased = sub / sub.iloc[0] * 100

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, col in enumerate(rebased.columns):
        fig.add_trace(
            go.Scatter(
                x=rebased.index, y=rebased[col],
                mode="lines", name=col,
                line=dict(width=1.8, color=colors[i % len(colors)]),
            )
        )

    fig.add_hline(y=100, line_dash="dot", line_color=COLORS["neutral"], line_width=1)
    fig.update_yaxes(title_text="Rebased (100 = start)")
    return apply_chart_style(
        fig, title="Macro Factor Performance (Rebased to 100)", height=height
    )


def plot_top_movers_bar(
    oas_change: pd.Series,
    n: int = 10,
    height: int = 340,
) -> go.Figure:
    """Horizontal bar chart of top OAS movers (largest absolute change)."""
    top = oas_change.abs().nlargest(n).index
    sub = oas_change[top].sort_values()

    colours = [COLORS["positive"] if v < 0 else COLORS["negative"] for v in sub]

    fig = go.Figure(
        go.Bar(
            y=sub.index,
            x=sub.values,
            orientation="h",
            marker=dict(color=colours),
            hovertemplate="<b>%{y}</b><br>OAS chg: %{x:+.1f}bps<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="OAS Change (bps)")
    return apply_chart_style(fig, title="Top OAS Movers", height=height)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Beta summary bar chart
# ═════════════════════════════════════════════════════════════════════════════

def plot_factor_bar_ranked(
    factor_df: pd.DataFrame,
    bond_id: str,
    height: int = 380,
) -> go.Figure:
    """
    Horizontal bar chart ranking macro factors by correlation to a bond.
    Sorted by absolute correlation (highest at top).
    Green = positive correlation, red = negative.
    """
    if factor_df.empty:
        return go.Figure()

    df = factor_df.sort_values("correlation", key=abs, ascending=True)
    colours = [COLORS["positive"] if c > 0 else COLORS["negative"] for c in df["correlation"]]

    fig = go.Figure(
        go.Bar(
            y=df["factor"],
            x=df["correlation"],
            orientation="h",
            marker=dict(color=colours),
            text=[f"{v:+.2f}" for v in df["correlation"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1)
    fig.update_xaxes(title_text="Pearson Correlation", range=[-1.15, 1.15])
    return apply_chart_style(
        fig,
        title=f"Factor Correlation Ranking: {bond_id}",
        height=height,
        show_legend=False,
    )


def plot_rolling_corr_with_stats(
    stats: dict,
    y_name: str,
    x_name: str,
    window: int,
    height: int = 400,
) -> go.Figure:
    """
    Rolling correlation chart with mean / min / max reference lines and
    fill-to-zero area for easy visual assessment of sign and magnitude.
    """
    roll = stats["series"]

    fig = go.Figure(
        go.Scatter(
            x=roll.index,
            y=roll.values,
            mode="lines",
            name=f"{window}d rolling corr",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.07)",
            hovertemplate="%{x|%Y-%m-%d}<br>Corr: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], line_width=1)

    mean_val = stats.get("mean")
    max_val  = stats.get("max")
    min_val  = stats.get("min")

    if mean_val is not None and not np.isnan(mean_val):
        fig.add_hline(
            y=mean_val, line_dash="dash", line_color=COLORS["warning"],
            line_width=1.2, opacity=0.7,
            annotation_text=f"Avg {mean_val:+.2f}",
            annotation_position="top right",
        )
    if max_val is not None and not np.isnan(max_val):
        fig.add_hline(
            y=max_val, line_dash="dot", line_color=COLORS["positive"],
            line_width=1, opacity=0.5,
            annotation_text=f"Max {max_val:+.2f}",
            annotation_position="top right",
        )
    if min_val is not None and not np.isnan(min_val):
        fig.add_hline(
            y=min_val, line_dash="dot", line_color=COLORS["negative"],
            line_width=1, opacity=0.5,
            annotation_text=f"Min {min_val:+.2f}",
            annotation_position="bottom right",
        )

    fig.update_yaxes(title_text="Correlation", range=[-1.05, 1.05])
    return apply_chart_style(
        fig,
        title=f"Rolling {window}d Correlation: {y_name} vs {x_name}",
        height=height,
    )


def plot_bond_macro_matrix(
    matrix: pd.DataFrame,
    title: str = "Bond x Macro Correlation Matrix",
    height: int = 560,
) -> go.Figure:
    """
    Heatmap: rows = bonds (sorted by mean |corr| desc),
    columns = macro factors sorted ascending (most negative left → most positive right).
    Factor labels are placed at the top of the chart.
    """
    df = matrix.copy()

    # Sort rows: bonds with highest average |correlation| first
    df["_avg"] = df.abs().mean(axis=1)
    df = df.sort_values("_avg", ascending=False).drop(columns="_avg")

    # Sort columns: most negative average correlation on the left
    col_means = df.mean(axis=0)
    df = df[col_means.sort_values(ascending=True).index]

    z    = df.values
    text = [[f"{df.iloc[i, j]:.2f}" for j in range(df.shape[1])] for i in range(df.shape[0])]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=df.columns.tolist(),
            y=df.index.tolist(),
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10, "color": COLORS["text"]},
            colorscale=DIVERGING_CORR,
            zmin=-1, zmax=1,
            colorbar=dict(
                tickfont=dict(color=COLORS["text_muted"], size=10),
                thickness=12,
            ),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Corr: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(side="top")
    return apply_chart_style(fig, title=title, height=height)


def plot_bond_bond_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Bond-to-Bond Correlation Matrix",
    height: int = 650,
) -> go.Figure:
    """Square bond-to-bond correlation heatmap."""
    labels = corr_matrix.columns.tolist()
    z      = corr_matrix.values
    text   = [[f"{z[i][j]:.2f}" for j in range(len(labels))] for i in range(len(labels))]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorscale=DIVERGING_CORR,
            zmin=-1, zmax=1,
            colorbar=dict(thickness=12, tickfont=dict(color=COLORS["text_muted"], size=10)),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Corr: %{z:.3f}<extra></extra>",
        )
    )
    return apply_chart_style(fig, title=title, height=height)


def plot_zscore_matrix(
    zscore_matrix: pd.DataFrame,
    title: str = "Spread Z-Score Matrix  (row cheap vs col if Z > 0)",
    height: int = 650,
) -> go.Figure:
    """
    Heatmap of bond-pair spread Z-scores.

    Convention:
        Z > 0  →  row bond is cheap vs column bond  (green)
        Z < 0  →  row bond is rich  vs column bond  (red)
    """
    labels = zscore_matrix.columns.tolist()
    z_raw  = zscore_matrix.values
    z_cap  = np.clip(z_raw, -4, 4)  # cap for colour scale; full value in tooltip

    text = []
    for i in range(len(labels)):
        row = []
        for j in range(len(labels)):
            v = zscore_matrix.iloc[i, j]
            row.append("—" if np.isnan(v) else f"{v:.1f}")
        text.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_cap,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorscale=DIVERGING_RdGn,
            zmin=-4, zmax=4,
            colorbar=dict(
                title=dict(text="Z-Score", font=dict(size=11)),
                thickness=12,
                tickfont=dict(color=COLORS["text_muted"], size=10),
                tickvals=[-4, -2, 0, 2, 4],
                ticktext=["-4 rich", "-2", "0", "+2", "+4 cheap"],
            ),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Z-Score: %{text}<extra></extra>",
        )
    )
    return apply_chart_style(fig, title=title, height=height)


def plot_beta_bars(
    beta_row: pd.Series,
    title: str = "Factor Betas",
    height: int = 320,
) -> go.Figure:
    """Bar chart of estimated betas for a single bond across all factors."""
    beta_cols = {k.replace("beta_", ""): v for k, v in beta_row.items()
                 if k.startswith("beta_") and not pd.isna(v)}
    if not beta_cols:
        return go.Figure()

    factors = list(beta_cols.keys())
    betas   = list(beta_cols.values())
    colours = [COLORS["positive"] if b > 0 else COLORS["negative"] for b in betas]

    fig = go.Figure(
        go.Bar(
            x=factors, y=betas,
            marker=dict(color=colours),
            hovertemplate="<b>%{x}</b><br>Beta: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color=COLORS["neutral"])
    fig.update_yaxes(title_text="Beta")
    return apply_chart_style(fig, title=title, height=height)
