import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from vectorbtpro import vbt
    import pandas as pd
    import numpy as np

    return mo, np, pd, vbt


@app.cell
def _(vbt):
    data = vbt.HDFData.pull("data.h5")
    data = data.ohlcv
    return (data,)


@app.cell
def _():
    outer_border = {"border": "1px solid darkgray", "padding": "8px", "border-radius": "8px", "height":"100%", "width":"100%"}
    inner_border = {"border": "1px solid lightgray", "padding": "8px", "height":"100%", "width":"100%"}
    return inner_border, outer_border


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Signal parameters and summary stats
    """)
    return


@app.cell
def _(inner_border, mo):
    signal_descr = mo.vstack([
        mo.md("**Signal Rules**"), 
        mo.md("ENTRY: RSI crosses ABOVE entry_thr (oversold recovery)"), 
        mo.md("EXIT:  RSI crosses BELOW exit_thr  (overbought recovery)"),
    ], align="center").style({**inner_border})
    return (signal_descr,)


@app.cell
def _(inner_border, mo):
    # prepare parameters and corresponding UI controls block
    timeperiod = mo.ui.number(value=6, start=3, stop=60, step=1, label="timeperiod")
    entry_thr = mo.ui.number(value=20, start=5, stop=50, step=1, label="entry_thr")
    exit_thr = mo.ui.number(value=80, start=50, stop=95, step=1, label="exit_thr")
    signal_params = mo.vstack([
        mo.md("**Signal Parameters**"),
        timeperiod, 
        entry_thr, 
        exit_thr
    ], align="center", gap=1).style({**inner_border})
    return entry_thr, exit_thr, signal_params, timeperiod


@app.cell
def _(data, entry_thr, exit_thr, timeperiod, vbt):
    # prepare indicator
    rsi = vbt.talib("RSI").run(
        data.open, # use open prices to avoid look-ahead bias
        timeperiod=timeperiod.value, 
        hide_params=True
    )

    # generate signals
    entries = rsi.rsi.vbt.crossed_above(entry_thr.value)
    exits = rsi.rsi.vbt.crossed_below(exit_thr.value)

    # clean signals
    clean_entries, clean_exits = entries.vbt.signals.clean(exits)

    # run portfolio simulation
    pf = vbt.Portfolio.from_signals(
        data.close, 
        entries=clean_entries, 
        exits=clean_exits
    )
    return clean_entries, clean_exits, entries, pf


@app.cell
def _(inner_border, mo, pd, pf):
    stats_df = pd.DataFrame(columns=["duration", "return"])
    stats_df["duration"] = pf.exit_trades.duration.readable["Value"].describe()
    stats_df["return"] = pf.exit_trades.readable["Return"].describe()
    stats_block = mo.vstack([
        mo.md("**Trade Stats**"),
        mo.ui.table(stats_df, selection=None)
    ], align="center", gap=1).style({**inner_border})
    return (stats_block,)


@app.cell
def _(mo, outer_border, signal_descr, signal_params, stats_block):
    signal_summary_block = mo.hstack([
        mo.vstack([signal_descr, signal_params], align="center"),
        stats_block
    ], widths=[1,2], align="stretch").style({**outer_border})
    return (signal_summary_block,)


@app.cell
def _(signal_summary_block):
    signal_summary_block
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Projections Bands
    """)
    return


@app.cell
def _(clean_entries, data, entries, vbt):
    # calculate entry probability per symbol
    entry_probs = entries[~data.close.isna()].mean() # use only indeces with valid data
    # generate random signals
    rand_entries = vbt.pd_acc.signals.generate_random(
        clean_entries.vbt.wrapper, 
        prob=vbt.to_2d_pc_array(entry_probs.astype(float).values), 
        seed=42
    )
    # keep signals only at indices with valid data
    rand_entries[data.close.isna()] = False
    return (rand_entries,)


@app.cell
def _(mo):
    proj_length = mo.ui.number(value=45, start=5, stop=150, step=1, label="Projection length (bars)")
    return (proj_length,)


@app.cell
def _(clean_entries, clean_exits, data, proj_length, rand_entries):
    entry_ranges = clean_entries.vbt.signals.delta_ranges(proj_length.value, close=data.close)
    entry_ranges = entry_ranges.status_closed
    rand_ranges = rand_entries.vbt.signals.delta_ranges(proj_length.value, close=data.close)
    rand_ranges = rand_ranges.status_closed
    exit_ranges = clean_exits.vbt.signals.delta_ranges(proj_length.value, close=data.close)
    exit_ranges = exit_ranges.status_closed

    entry_projections = entry_ranges.get_projections()
    rand_projections = rand_ranges.get_projections()
    exit_projections = exit_ranges.get_projections()
    return entry_projections, exit_projections, rand_projections


@app.cell
def _(entry_projections, exit_projections, rand_projections):
    proj_bands_fig = entry_projections.vbt.plot_projections(
        plot_projections=False, 
        plot_aux_middle=False, 
        plot_fill=False, 
        lower_trace_kwargs=dict(name="Entry", legendgroup="entry", showlegend=True, line_color="#00cc96"), 
        middle_trace_kwargs=dict(legendgroup="entry", showlegend=False, line_color="#00cc96"), 
        upper_trace_kwargs=dict(legendgroup="entry", showlegend=False, line_color="#00cc96")
    )
    proj_bands_fig = rand_projections.vbt.plot_projections(
        plot_projections=False, 
        plot_aux_middle=False, 
        plot_fill=False, 
        lower_trace_kwargs=dict(name="Random", legendgroup="rand", showlegend=True, line_color="#636efa"), 
        middle_trace_kwargs=dict(legendgroup="rand", showlegend=False, line_color="#636efa"), 
        upper_trace_kwargs=dict(legendgroup="rand", showlegend=False, line_color="#636efa"), 
        fig=proj_bands_fig
    )
    proj_bands_fig = exit_projections.vbt.plot_projections(
        plot_projections=False, 
        plot_aux_middle=False, 
        plot_fill=False, 
        lower_trace_kwargs=dict(name="Exit", legendgroup="exit", showlegend=True, line_color="#ef553b"), 
        middle_trace_kwargs=dict(legendgroup="exit", showlegend=False, line_color="#ef553b"), 
        upper_trace_kwargs=dict(legendgroup="exit", showlegend=False, line_color="#ef553b"), 
        fig=proj_bands_fig
    )
    proj_bands_fig.update_layout(
        template="plotly_white", 
        width=None, 
        autosize=True, 
        legend=dict(
            yanchor="bottom",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    None
    return (proj_bands_fig,)


@app.cell
def _(entry_projections, exit_projections, pd, rand_projections):
    final_stats_df = pd.DataFrame(columns=["entry", "random", "exit"])
    final_stats_df["entry"] = entry_projections.iloc[-1].describe()
    final_stats_df["random"] = rand_projections.iloc[-1].describe()
    final_stats_df["exit"] = exit_projections.iloc[-1].describe()
    return (final_stats_df,)


@app.cell
def _(
    final_stats_df,
    inner_border,
    mo,
    outer_border,
    proj_bands_fig,
    proj_length,
):
    _descr = mo.md("""
    Fixed-window price projections starting from entry, exit, and random signals. 
    Bands show lower/middle/upper quantiles of normalized price paths over the next 
    N bars. A signal with edge should diverge visibly from the random baseline.
    """)

    proj_bands_block = mo.vstack([
        _descr.center().style({**inner_border}),
        proj_length.center().style({**inner_border}), 
        mo.ui.plotly(proj_bands_fig).style({**inner_border}), 
        mo.ui.table(final_stats_df, selection=None).style({**inner_border})
    ]).style({**outer_border})
    return (proj_bands_block,)


@app.cell
def _(proj_bands_block):
    proj_bands_block
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shrink vs Stretch projections
    """)
    return


@app.cell
def _(clean_entries, clean_exits, data):
    # trade ranges: entry->exit and exit->entry
    entry_exit_ranges = clean_entries.vbt.signals.between_ranges(clean_exits, close=data.close).status_closed
    exit_entry_ranges = clean_exits.vbt.signals.between_ranges(clean_entries, close=data.close).status_closed

    # trade projections
    entry_exit_projections = entry_exit_ranges.get_projections()
    exit_entry_projections = exit_entry_ranges.get_projections()
    return (
        entry_exit_projections,
        entry_exit_ranges,
        exit_entry_projections,
        exit_entry_ranges,
    )


@app.cell
def _(inner_border, mo):
    proj_period = mo.ui.number(value=45, start=5, stop=200, step=1, label="proj_period")
    lower_qq = mo.ui.number(value=0.2, start=0.01, stop=0.5, step=0.01, label="lower_qq")
    upper_qq = mo.ui.number(value=0.8, start=0.5, stop=0.99, step=0.01, label="upper_qq")
    controls = mo.hstack([proj_period, lower_qq, upper_qq]).style({**inner_border})
    return controls, lower_qq, proj_period, upper_qq


@app.cell
def _(entry_exit_ranges, exit_entry_ranges, proj_period):
    shrink_entry_proj = entry_exit_ranges.get_projections(proj_period=proj_period.value)
    stretch_entry_proj = entry_exit_ranges.get_projections(proj_period=proj_period.value, extend=True)

    shrink_exit_proj = exit_entry_ranges.get_projections(proj_period=proj_period.value)
    stretch_exit_proj = exit_entry_ranges.get_projections(proj_period=proj_period.value, extend=True)
    return (
        shrink_entry_proj,
        shrink_exit_proj,
        stretch_entry_proj,
        stretch_exit_proj,
    )


@app.cell
def _(mo, vbt):
    def shrink_stretch_plot(shrink_proj, stretch_proj, lower_qq, upper_qq):
        fig = vbt.make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=["Shrink", "Stretch"])
        shrink_proj.vbt.plot_projections(
            plot_lower=f"Q={lower_qq}",
            plot_upper=f"Q={upper_qq}",
            plot_projections=False, 
            add_trace_kwargs=dict(row=1, col=1), 
            fig=fig, 
        )
        stretch_proj.vbt.plot_projections(
            plot_lower=f"Q={lower_qq}",
            plot_upper=f"Q={upper_qq}",
            plot_projections=False,
            add_trace_kwargs=dict(row=1, col=2), 
            fig=fig, 
        )
        fig.update_layout(
            template="plotly_white", 
            width=None, 
            autosize=True, 
            showlegend=False, 
            yaxis_showticklabels=True, 
            yaxis2_showticklabels=True
        )
        return mo.ui.plotly(fig)

    return (shrink_stretch_plot,)


@app.cell
def _(
    inner_border,
    lower_qq,
    mo,
    shrink_entry_proj,
    shrink_exit_proj,
    shrink_stretch_plot,
    stretch_entry_proj,
    stretch_exit_proj,
    upper_qq,
):
    tabs = mo.ui.tabs({
        "entry": shrink_stretch_plot(
            shrink_entry_proj, 
            stretch_entry_proj, 
            lower_qq=lower_qq.value, 
            upper_qq=upper_qq.value
        ),
        "exit": shrink_stretch_plot(
            shrink_exit_proj, 
            stretch_exit_proj, 
            lower_qq=lower_qq.value, 
            upper_qq=upper_qq.value
        )
    }).style({**inner_border})
    return (tabs,)


@app.cell
def _(controls, inner_border, mo, outer_border, tabs):
    _descr = mo.md("""
    Price projections between consecutive entry→exit and exit→entry signals. 
    Shrink trims longer projections to proj_period bars. Stretch extends shorter 
    ones beyond the opposite signal — if the edge disappears after stretching, 
    exit timing is the source of alpha, not entry alone.
    """)

    shrink_stretch_block = mo.vstack([
        _descr.center().style({**inner_border}),
        controls, 
        tabs
    ]).style({**outer_border})
    return (shrink_stretch_block,)


@app.cell
def _(shrink_stretch_block):
    shrink_stretch_block
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Individual Projections Sample
    """)
    return


@app.cell
def _(mo):
    rand_btn = mo.ui.button(label="refresh")
    return (rand_btn,)


@app.cell
def _(mo, np):
    def plot_rand_proj(proj, n=20):
        rand_cols = np.random.choice(proj.shape[1], n, replace=False)
        fig = proj.iloc[:, rand_cols].vbt.plot_projections(plot_bands=False)
        fig.update_layout(
            template="plotly_white", 
            width=None, 
            autosize=True, 
            showlegend=False
        )
        return mo.ui.plotly(fig)

    return (plot_rand_proj,)


@app.cell
def _(
    entry_exit_projections,
    exit_entry_projections,
    inner_border,
    mo,
    outer_border,
    plot_rand_proj,
    rand_btn,
):
    rand_btn
    _fig_en = plot_rand_proj(entry_exit_projections)
    _fig_ex = plot_rand_proj(exit_entry_projections)

    _plots = mo.hstack([
        mo.vstack([mo.md("#### Entry → Exit").center(), _fig_en]).style({**inner_border}),
        mo.vstack([mo.md("#### Exit → Entry").center(), _fig_ex]).style({**inner_border}),
    ], widths="equal")

    _descr = mo.md(
    """
    Random sample of 20 individual projections. Each line is one price path, normalized to start at 1.
    Click refresh to resample — consistent patterns across samples indicate a robust signal.
    """
    )

    rand_block = mo.vstack([
        mo.center(_descr).style({**inner_border}), 
        rand_btn.center().style({**inner_border}), 
        _plots
    ]).style({**outer_border})
    return (rand_block,)


@app.cell
def _(rand_block):
    rand_block
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Portfolio Simulation
    """)
    return


@app.cell
def _(clean_entries, clean_exits, data, vbt):
    pf_en = vbt.Portfolio.from_signals(
        data.close, 
        entries=clean_entries, 
        exits=clean_exits
    )

    pf_ex = vbt.Portfolio.from_signals(
        data.close, 
        entries=clean_exits, 
        exits=clean_entries
    )
    return pf_en, pf_ex


@app.cell
def _(mo, outer_border, pd, pf_en, pf_ex):
    pf_stats_mean = pd.DataFrame(columns=["entry_exit", "exit_entry"])
    pf_stats_mean["entry_exit"] = pf_en.stats()
    pf_stats_mean["exit_entry"] = pf_ex.stats()

    pf_stats_median = pd.DataFrame(columns=["entry_exit", "exit_entry"])
    pf_stats_median["entry_exit"] = pf_en.stats(agg_func=None).median()
    pf_stats_median["exit_entry"] = pf_ex.stats(agg_func=None).median()

    mo.ui.tabs({
        "mean": mo.ui.table(pf_stats_mean, pagination=False, selection=None),
        "median": mo.ui.table(pf_stats_median, pagination=False, selection=None)
    }).style(**outer_border)
    return


if __name__ == "__main__":
    app.run()
