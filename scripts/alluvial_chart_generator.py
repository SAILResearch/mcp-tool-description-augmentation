#!/usr/bin/env python3
# One PDF per model: All tasks -> AS increased -> AE increased -> SR increased
# Uses percentages, larger fonts, and a clean layout suitable for LaTeX subfigures
# Requires: pip install plotly kaleido

import os
import numpy as np
import plotly.graph_objects as go

# ------------ Inputs (percentages) ------------
MODELS = ["GPT-4.1", "Qwen3-Coder", "GLM-4.5"]

# % of all tasks with AS > baseline
PCT_AS_UP = {
    "GPT-4.1": 68.3982684,
    "Qwen3-Coder": 83.9826839,
    "GLM-4.5": 74.2424242,
}

# % of tasks where AE increased when AS increased (conditional)
PCT_AE_UP_GIVEN_AS = {
    "GPT-4.1": 55.06329114,
    "Qwen3-Coder": 42.78350515,
    "GLM-4.5": 41.49659864,
}

# % of tasks where SR increased when AS increased (conditional)
PCT_SR_UP_GIVEN_AS = {
    "GPT-4.1": 20.25316456,
    "Qwen3-Coder": 18.04123711,
    "GLM-4.5": 20.40816327,
}

# ------------ Styles ------------
# Column headers
COL_HEADERS = ["All tasks", "Increasing AS", "Quality ↑", "Success ↑"]

# Colors for nodes and links by stage outcome
COL_ALL = "#BFC7D5"
COL_AS_UP, COL_AS_NOT = "#7DA0C3", "#D8DEE9"
COL_AE_UP, COL_AE_NOT = "#6BB37E", "#CFE8D6"
COL_SR_UP, COL_SR_NOT = "#6C8EBF", "#C7D5EA"

NODE_LABEL_FONT = 16
UI_FONT = 14
FIG_W, FIG_H = 560, 360
MARGINS = dict(l=24, r=24, t=38, b=16)  # small top margin for column headers

OUTDIR = "alluvial_pdfs"
os.makedirs(OUTDIR, exist_ok=True)

def fmt_pct(x):
    return f"{x:.1f}%"

def figure_for_model(model: str) -> go.Figure:
    # Stage math in percentages
    as_up = PCT_AS_UP[model]
    as_not = 100.0 - as_up

    ae_up = as_up * PCT_AE_UP_GIVEN_AS[model] / 100.0
    ae_not = as_up - ae_up

    sr_up = as_up * PCT_SR_UP_GIVEN_AS[model] / 100.0
    sr_not = max(0.0, ae_up - sr_up)

    # -------- Nodes (4 columns = 4 steps) --------
    # 0: All tasks
    # 1: AS up, 2: AS not
    # 3: AE up|AS up, 4: AE not|AS up
    # 5: SR up|AS up, 6: SR not|AS up
    labels = [
        f"{COL_HEADERS[0]}\n100.0%",         # 0
        f"{COL_HEADERS[1]}\n{fmt_pct(as_up)}",     # 1
        f"Not increasing\n{fmt_pct(as_not)}",      # 2
        f"{COL_HEADERS[2]} | AS ↑\n{fmt_pct(ae_up)}",   # 3
        f"Not ↑ | AS ↑\n{fmt_pct(ae_not)}",             # 4
        f"{COL_HEADERS[3]} | AS ↑\n{fmt_pct(sr_up)}",   # 5
        f"Not ↑ | AS ↑\n{fmt_pct(sr_not)}",             # 6
    ]
    node_colors = [COL_ALL, COL_AS_UP, COL_AS_NOT, COL_AE_UP, COL_AE_NOT, COL_SR_UP, COL_SR_NOT]

    # -------- Links (values in percentages) --------
    # All -> AS up / AS not
    # AS up -> AE up / AE not
    # AE up -> SR up / SR not
    sources = [0, 0,  1, 1,  3, 3]
    targets = [1, 2,  3, 4,  5, 6]
    values  = [as_up, as_not, ae_up, ae_not, sr_up, sr_not]
    link_colors = [COL_AS_UP, COL_AS_NOT, COL_AE_UP, COL_AE_NOT, COL_SR_UP, COL_SR_NOT]

    # Column x positions for the 4 steps
    x_all, x_as, x_ae, x_sr = 0.03, 0.33, 0.63, 0.93
    x = [x_all, x_as, x_as, x_ae, x_ae, x_sr, x_sr]

    # Stagger y positions just to separate nodes slightly
    y = np.linspace(0.05, 0.95, len(labels)).tolist()

    sk = go.Sankey(
        arrangement="fixed",
        valueformat=".1f",
        valuesuffix="%",
        node=dict(
            pad=14,
            thickness=18,
            label=labels,
            color=node_colors,
            line=dict(width=0.4, color="rgba(0,0,0,0.25)"),
            x=x, y=y,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate="%{value:.1f}%<extra></extra>",
        ),
        textfont=dict(size=NODE_LABEL_FONT),
        hoverlabel=dict(font_size=NODE_LABEL_FONT),
    )

    fig = go.Figure(sk)

    # Column headers above the columns
    fig.add_annotation(x=x_all, y=1.08, text=COL_HEADERS[0], showarrow=False, font=dict(size=UI_FONT, color="#333"), xref="paper", yref="paper")
    fig.add_annotation(x=x_as,  y=1.08, text=COL_HEADERS[1], showarrow=False, font=dict(size=UI_FONT, color="#333"), xref="paper", yref="paper")
    fig.add_annotation(x=x_ae,  y=1.08, text=COL_HEADERS[2], showarrow=False, font=dict(size=UI_FONT, color="#333"), xref="paper", yref="paper")
    fig.add_annotation(x=x_sr,  y=1.08, text=COL_HEADERS[3], showarrow=False, font=dict(size=UI_FONT, color="#333"), xref="paper", yref="paper")

    fig.update_layout(
        font=dict(size=UI_FONT),
        margin=dict(**MARGINS),
        width=FIG_W,
        height=FIG_H,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig

if __name__ == "__main__":
    for m in MODELS:
        fig = figure_for_model(m)
        fname = m.lower().replace(" ", "-") + "_alluvial_4step.pdf"
        fig.write_image(os.path.join(OUTDIR, fname))
        print("Saved:", os.path.join(OUTDIR, fname))
