import base64
import io
from copy import deepcopy

import dash_mantine_components as dmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import v_relative, clo_dynamic
from scipy import optimize

from components.drop_down_inline import generate_dropdown_inline
from utils.my_config_file import ElementsIDs, Models, Functionalities
from utils.website_text import TextHome
import matplotlib
from pythermalcomfort.models import adaptive_en
from pythermalcomfort.psychrometrics import t_o, psy_ta_rh

matplotlib.use("Agg")

import plotly.graph_objects as go
from dash import dcc


def chart_selector(selected_model: str, function_selection: str):
    list_charts = deepcopy(Models[selected_model].value.charts)
    if function_selection == Functionalities.Compare.value:
        if selected_model == Models.PMV_ashrae.name:
            list_charts = deepcopy(Models[selected_model].value.charts_compare)

    list_charts = [chart.name for chart in list_charts]
    drop_down_chart_dict = {
        "id": ElementsIDs.chart_selected.value,
        "question": TextHome.chart_selection.value,
        "options": list_charts,
        "multi": False,
        "default": list_charts[0],
    }

    return generate_dropdown_inline(
        drop_down_chart_dict, value=drop_down_chart_dict["default"], clearable=False
    )


def pmv_en_psy_chart(
    inputs: dict = None,
    model="iso",
    function_selection: str = Functionalities.Default,
    use_to: bool = False,
):
    traces = []

    category_3_up = np.linspace(20.5, 27.1, 100)
    category_2_up = np.linspace(21.4, 26.2, 100)
    category_1_up = np.linspace(22.7, 24.7, 100)
    category_3_low = np.array([33.3, 24.2])
    category_2_low = np.array([32, 25.5])
    category_1_low = np.array([30, 27.4])
    category_1_x = np.concatenate((category_1_up, category_1_low))
    category_2_x = np.concatenate((category_2_up, category_2_low))
    category_3_x = np.concatenate((category_3_up, category_3_low))

    # Category III
    category_3_y = []
    for t in category_3_up:
        category_3_y.append(psy_ta_rh(tdb=t, rh=100, p_atm=101325)["hr"] * 1000)
    category_3_y = np.concatenate((category_3_y, [0] * 2))
    traces.append(
        go.Scatter(
            x=category_3_x,
            y=category_3_y,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="toself",
            fillcolor="rgba(0,255,0,0.2)",
            showlegend=False,
            hoverinfo="none",
        )
    )

    # Category II
    category_2_y = []
    for t in category_2_up:
        category_2_y.append(psy_ta_rh(tdb=t, rh=100, p_atm=101325)["hr"] * 1000)
    category_2_y = np.concatenate((category_2_y, [0] * 2))
    traces.append(
        go.Scatter(
            x=category_2_x,
            y=category_2_y,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="toself",
            fillcolor="rgba(0,255,0,0.3)",
            showlegend=False,
            hoverinfo="none",
        )
    )

    # Category I
    category_1_y = []
    for t in category_1_up:
        category_1_y.append(psy_ta_rh(tdb=t, rh=100, p_atm=101325)["hr"] * 1000)
    category_1_y = np.concatenate((category_1_y, [0] * 2))
    traces.append(
        go.Scatter(
            x=category_1_x,
            y=category_1_y,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="toself",
            fillcolor="rgba(0,255,0,0.4)",
            showlegend=False,
            hoverinfo="none",
        )
    )

    rh_list = np.arange(0, 101, 10)
    tdb = np.linspace(10, 36, 500)
    for rh in rh_list:
        hr_list = np.array(
            [psy_ta_rh(tdb=t, rh=rh, p_atm=101325)["hr"] * 1000 for t in tdb]
        )
        trace = go.Scatter(
            x=tdb,
            y=hr_list,
            mode="lines",
            line=dict(color="black", width=1),
            hoverinfo="x+y",
            name=f"{rh}% RH",
            showlegend=False,
        )
        traces.append(trace)

    tdb = inputs[ElementsIDs.t_db_input.value]
    rh = inputs[ElementsIDs.rh_input.value]
    tr = inputs[ElementsIDs.t_r_input.value]

    if use_to:
        x_value = t_o(tdb=tdb, tr=tr, v=inputs[ElementsIDs.v_input.value])
        x_label = "Operative Temperature [°C]"
    else:
        x_value = tdb
        x_label = "Dry-bulb Temperature [°C]"

    red_point = [x_value, psy_ta_rh(tdb, rh, p_atm=101325)["hr"] * 1000]
    traces.append(
        go.Scatter(
            x=[red_point[0]],
            y=[red_point[1]],
            mode="markers",
            marker=dict(
                color="red",
                size=4,
            ),
            showlegend=False,
        )
    )
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = red_point[0] + 0.6 * np.cos(theta)
    circle_y = red_point[1] + 1.2 * np.sin(theta)
    traces.append(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color="red", width=1.5),
            showlegend=False,
        )
    )

    layout = go.Layout(
        xaxis=dict(title=x_label, showgrid=False),
        yaxis=dict(title="Humidity Ratio [g<sub>w</sub>/kg<sub>da</sub>]", showgrid=False),
        showlegend=True,
        plot_bgcolor="white",
        annotations=[
            dict(
                x=14,
                y=28,
                xref="x",
                yref="y",
                text="Hover over the chart for details",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0)",
                font=dict(size=14),
            )
        ],
    )

    fig = go.Figure(data=traces, layout=layout)

    return fig


def generate_adaptive_en_chart(
    inputs: dict = None, model="iso", function_selection: str = Functionalities.Default
):
    traces = []

    tdb = inputs[ElementsIDs.t_db_input.value]
    tr = inputs[ElementsIDs.t_r_input.value]
    v = inputs[ElementsIDs.v_input.value]
    t_running_mean = inputs[ElementsIDs.t_rm_input.value]

    x_values = np.array([10, 30])
    results_min = adaptive_en(tdb=tdb, tr=tr, t_running_mean=x_values[0], v=v)
    results_max = adaptive_en(tdb=tdb, tr=tr, t_running_mean=x_values[1], v=v)

    y_values_cat_iii_up = [
        results_min["tmp_cmf_cat_iii_up"],
        results_max["tmp_cmf_cat_iii_up"],
    ]
    y_values_cat_iii_low = [
        results_min["tmp_cmf_cat_iii_low"],
        results_max["tmp_cmf_cat_iii_low"],
    ]

    y_values_cat_ii_up = [
        results_min["tmp_cmf_cat_ii_up"],
        results_max["tmp_cmf_cat_ii_up"],
    ]
    y_values_cat_ii_low = [
        results_min["tmp_cmf_cat_ii_low"],
        results_max["tmp_cmf_cat_ii_low"],
    ]

    y_values_cat_i_up = [
        results_min["tmp_cmf_cat_i_up"],
        results_max["tmp_cmf_cat_i_up"],
    ]
    y_values_cat_i_low = [
        results_min["tmp_cmf_cat_i_low"],
        results_max["tmp_cmf_cat_i_low"],
    ]

    category_3_x = np.concatenate((x_values, x_values[::-1]))
    category_2_x = np.concatenate((x_values, x_values[::-1]))
    category_1_x = np.concatenate((x_values, x_values[::-1]))

    # traces[0]
    traces.append(
        go.Scatter(
            x=category_3_x,
            y=np.concatenate([y_values_cat_iii_up, y_values_cat_iii_low[::-1]]),
            fill="toself",
            fillcolor="rgba(144, 238, 144, 0.3)",
            line=dict(color="rgba(144, 238, 144, 0)", shape="linear"),
            name="Category III",
            mode="lines",
        )
    )
    # traces[1]
    traces.append(
        go.Scatter(
            x=category_2_x,
            y=np.concatenate([y_values_cat_ii_up, y_values_cat_ii_low[::-1]]),
            fill="toself",
            fillcolor="rgba(34, 139, 34, 0.5)",
            line=dict(color="rgba(34, 139, 34, 0)", shape="linear"),
            name="Category II",
            mode="lines",
        )
    )
    # traces[2]
    traces.append(
        go.Scatter(
            x=category_1_x,
            y=np.concatenate([y_values_cat_i_up, y_values_cat_i_low[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 100, 0, 0.7)",
            line=dict(color="rgba(0, 100, 0, 0)", shape="linear"),
            name="Category I",
            mode="lines",
        )
    )

    # Red point
    x = t_running_mean
    y = t_o(tdb=tdb, tr=tr, v=v)
    red_point = [x, y]
    # traces[3]
    traces.append(
        go.Scatter(
            x=[red_point[0]],
            y=[red_point[1]],
            mode="markers",
            marker=dict(
                color="red",
                size=6,
            ),
            showlegend=False,
        )
    )
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = red_point[0] + 0.5 * np.cos(theta)
    circle_y = red_point[1] + 0.7 * np.sin(theta)
    # traces[4]
    traces.append(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color="red", width=2.5),
            showlegend=False,
        )
    )

    layout = go.Layout(
        title="Adaptive Chart",
        xaxis=dict(
            title="Outdoor Running Mean Temperature [℃]",
            range=[10, 30],
            dtick=2,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1.5,
            ticks="outside",
            ticklen=5,
            showline=True,
            linewidth=1.5,
            linecolor="black",
        ),
        yaxis=dict(
            title="Operative Temperature [℃]",
            range=[14, 36],
            dtick=2,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1.5,
            ticks="outside",
            ticklen=5,
            showline=True,
            linewidth=1.5,
            linecolor="black",
        ),
        legend=dict(x=0.8, y=1),
        showlegend=False,
        plot_bgcolor="white",
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig


def t_rh_pmv(
    inputs: dict = None,
    model: str = "iso",
    function_selection: str = Functionalities.Default,
):
    results = []
    pmv_limits = [-0.5, 0.5]
    # todo determine if the value is IP unit , transfer to SI
    clo_d = clo_dynamic(
        clo=inputs[ElementsIDs.clo_input.value], met=inputs[ElementsIDs.met_input.value]
    )
    vr = v_relative(
        v=inputs[ElementsIDs.v_input.value], met=inputs[ElementsIDs.met_input.value]
    )

    if function_selection == Functionalities.Compare.value:
        try:
            clo_d_compare = clo_dynamic(
                clo=inputs.get(ElementsIDs.clo_input_input2.value),
                met=inputs.get(ElementsIDs.met_input_input2.value),
            )
            vr_compare = v_relative(
                v=inputs.get(ElementsIDs.v_input_input2.value),
                met=inputs.get(ElementsIDs.met_input_input2.value),
            )
        except KeyError as e:
            print(f"KeyError: {e}. Skipping comparison plotting.")
            clo_d_compare, vr_compare = None, None

    def calculate_pmv_results(tr, vr, met, clo):
        results = []
        for pmv_limit in pmv_limits:
            for rh in np.arange(0, 110, 10):

                def function(x):
                    return (
                        pmv(
                            x,
                            tr=tr,
                            vr=vr,
                            rh=rh,
                            met=met,
                            clo=clo,
                            wme=0,
                            standard=model,
                            limit_inputs=False,
                        )
                        - pmv_limit
                    )

                temp = optimize.brentq(function, 10, 100)
                results.append(
                    {
                        "rh": rh,
                        "temp": temp,
                        "pmv_limit": pmv_limit,
                    }
                )
        return pd.DataFrame(results)

    df = calculate_pmv_results(
        tr=inputs[ElementsIDs.t_r_input.value],
        vr=vr,
        met=inputs[ElementsIDs.met_input.value],
        clo=clo_d,
    )

    # Create the Plotly figure
    fig = go.Figure()

    # Add the filled area between PMV limits
    t1 = df[df["pmv_limit"] == pmv_limits[0]]
    t2 = df[df["pmv_limit"] == pmv_limits[1]]
    fig.add_trace(
        go.Scatter(
            x=t1["temp"],
            y=t1["rh"],
            fill=None,
            mode="lines",
            line=dict(color="rgba(59, 189, 237, 0.7)"),
            name=f"{model} Lower Limit",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t2["temp"],
            y=t2["rh"],
            fill="tonexty",
            mode="lines",
            fillcolor="rgba(59, 189, 237, 0.7)",
            line=dict(color="rgba(59, 189, 237, 0.7)"),
            name=f"{model} Upper Limit",
        )
    )

    # Add scatter point for the current input
    fig.add_trace(
        go.Scatter(
            x=[inputs[ElementsIDs.t_db_input.value]],
            y=[inputs[ElementsIDs.rh_input.value]],
            mode="markers",
            marker=dict(color="red", size=8),
            name="Current Input",
            # hoverinfo='skip',
        )
    )

    # Add hover area to allow hover interaction
    # todo: the interaction area should not the whole chart, at least should not include the while area (e.g. blue only)
    x_range = np.linspace(10, 40, 100)
    y_range = np.linspace(0, 100, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    fig.add_trace(
        go.Scatter(
            x=xx.flatten(),
            y=yy.flatten(),
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)"),
            hoverinfo="x+y",
            name="Interactive Hover Area",
        )
    )

    if (
        function_selection == Functionalities.Compare.value
        and clo_d_compare is not None
    ):
        df_compare = calculate_pmv_results(
            tr=inputs[ElementsIDs.t_r_input_input2.value],
            vr=vr_compare,
            met=inputs[ElementsIDs.met_input_input2.value],
            clo=clo_d_compare,
        )
        t1_compare = df_compare[df_compare["pmv_limit"] == pmv_limits[0]]
        t2_compare = df_compare[df_compare["pmv_limit"] == pmv_limits[1]]
        fig.add_trace(
            go.Scatter(
                x=t1_compare["temp"],
                y=t1_compare["rh"],
                fill=None,
                mode="lines",
                line=dict(color="rgba(30,70,100,0.5)"),
                name=f"{model} Compare Lower Limit",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t2_compare["temp"],
                y=t2_compare["rh"],
                fill="tonexty",
                mode="lines",
                fillcolor="rgba(30,70,100,0.5)",
                line=dict(color="rgba(30,70,100,0.5)"),
                name=f"{model} Compare Upper Limit",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[inputs[ElementsIDs.t_db_input_input2.value]],
                y=[inputs[ElementsIDs.rh_input_input2.value]],
                mode="markers",
                marker=dict(color="blue", size=8),
                name="Compare Input",
            )
        )

    annotation_text = (
        f"t<sub>db</sub>: {inputs[ElementsIDs.t_db_input.value]:.1f} °C<br>"
        f"RH: {inputs[ElementsIDs.rh_input.value]:.1f} %<br>"
        f"W<sub>a</sub>: - g/kg<sub>da</sub><br>"
        f"t<sub>wb</sub>: - °C<br>"
        f"t<sub>dp</sub>: - °C<br>"
        f"h: - kJ/kg<br>"
    )

    # top left annotation
    # todo: set it by a more flexible way instead of setup location
    fig.add_annotation(
        # top right
        # x=38,
        # y=89,
        # top left
        x=14,
        y=89,
        xref="x",
        yref="y",
        text=annotation_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
        font=dict(size=14),
    )

    # Update layout
    fig.update_layout(
        yaxis=dict(title="RH (%)", range=[0, 100], dtick=10),
        xaxis=dict(title="Temperature (°C)", range=[10, 40], dtick=2),
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
        hoverdistance=5,
    )

    # Add grid lines and make the spines invisible
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.2)")

    return fig
