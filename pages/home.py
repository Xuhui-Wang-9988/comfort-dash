import dash
import dash_mantine_components as dmc
from dash import html, callback, Output, Input, no_update, State, ctx, dcc

from components.charts import t_rh_pmv, chart_selector
from components.dropdowns import (
    model_selection,
)
from components.functionality_selection import functionality_selection
from components.input_environmental_personal import input_environmental_personal
from components.my_card import my_card
from components.show_results import display_results
from utils.get_inputs import get_inputs,extract_met_clo
from utils.my_config_file import (
    URLS,
    ElementsIDs,
    Dimensions,
    UnitSystem,
    Models,
    Charts,
    ChartsInfo,
    MyStores,

)
import plotly.graph_objects as go

from urllib.parse import parse_qs, urlencode


dash.register_page(__name__, path=URLS.HOME.value)

#
layout = dmc.Stack(
    #ADD ID for the layout
    id=ElementsIDs.HOME_LAYOUT_PAGE.value,
    children=[
        dmc.Grid(
            children=[
                dmc.GridCol(
                    model_selection(),
                    span={"base": 12, "sm": Dimensions.left_container_width.value},
                ),
                dmc.GridCol(
                    functionality_selection(),
                    span={"base": 12, "sm": Dimensions.right_container_width.value},
                ),
            ],
            gutter="xl",
        ),
        dmc.Grid(
            children=[
                my_card(
                    title="Inputs",
                    children=input_environmental_personal(),
                    id=ElementsIDs.INPUT_SECTION.value,
                    span={"base": 12, "sm": Dimensions.left_container_width.value},
                ),
                my_card(
                    title="Results",
                    children=dmc.Stack(
                        [
                            html.Div(
                                id=ElementsIDs.RESULTS_SECTION.value,
                            ),
                            html.Div(
                                id=ElementsIDs.charts_dropdown.value,
                                children=html.Div(id=ElementsIDs.chart_selected.value),
                            ),
                            html.Div(
                                id=ElementsIDs.CHART_CONTAINER.value,
                            ),
                            dmc.Text(id=ElementsIDs.note_model.value),
                            dcc.Location(id=ElementsIDs.URL.value, refresh=False),
                        ],
                    ),
                    span={"base": 12, "sm": Dimensions.right_container_width.value},
                ),
            ],
            gutter="xl",
        ),
    ]
)


@callback(
    Output(MyStores.input_data.value, "data"),
    Output(ElementsIDs.URL.value, "search", allow_duplicate=True),
    Input(ElementsIDs.HOME_LAYOUT_PAGE.value, "n_clicks"),
    Input(ElementsIDs.inputs_form.value, "n_clicks"),
    Input(ElementsIDs.inputs_form.value, "children"),
    Input(ElementsIDs.clo_input.value, "value"),
    Input(ElementsIDs.met_input.value, "value"),
    Input(ElementsIDs.UNIT_TOGGLE.value, "checked"),
    Input(ElementsIDs.chart_selected.value, "value"),
    Input(ElementsIDs.functionality_selection.value, "value"),
    State(ElementsIDs.MODEL_SELECTION.value, "value"),
    State(MyStores.input_data.value, "data"),
    prevent_initial_call=True,
)
def update_store_inputs(
        form_clicks, HOME_LAYOUT_PAGE_clicks, form_content, clo_value, met_value,
        units_selection, chart_selected, functionality_selection, selected_model, current_inputs
):
    if selected_model is None:
        return no_update, no_update

    units = UnitSystem.IP.value if units_selection else UnitSystem.SI.value

    # current if possible
    inputs = current_inputs.copy() if current_inputs else {}

    # update
    new_inputs = get_inputs(selected_model, form_content, units, functionality_selection)
    inputs.update(new_inputs)

    # default
    # default_clo = 1.0
    # default_met = 0.61
    #
    # if ctx.triggered:
    #     triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    #     if triggered_id == ElementsIDs.clo_input.value:
    #         clo_input = Models[selected_model].value.inputs[0]
    #         processed_clo = extract_met_clo(clo_value, default_clo, clo_input.min, clo_input.max)
    #         inputs[ElementsIDs.clo_input.value] = processed_clo if processed_clo is not None else default_clo
    #     elif triggered_id == ElementsIDs.met_input.value:
    #         met_input = Models[selected_model].value.inputs[1]
    #         processed_met = extract_met_clo(met_value, default_met, met_input.min, met_input.max)
    #         inputs[ElementsIDs.met_input.value] = processed_met if processed_met is not None else default_met
    #
    # inputs[ElementsIDs.clo_input.value] = inputs.get(ElementsIDs.clo_input.value, default_clo)
    # inputs[ElementsIDs.met_input.value] = inputs.get(ElementsIDs.met_input.value, default_met)

    inputs[ElementsIDs.UNIT_TOGGLE.value] = units
    inputs[ElementsIDs.MODEL_SELECTION.value] = selected_model
    inputs[ElementsIDs.chart_selected.value] = chart_selected
    inputs[ElementsIDs.functionality_selection.value] = functionality_selection

    for model_input in Models[selected_model].value.inputs:
        if model_input.id not in inputs:
            inputs[model_input.id] = model_input.value

    url_search = f"?{urlencode(inputs)}"
    print(f"url_params: {url_search}")

    return inputs, url_search

@callback(
    Output(ElementsIDs.INPUT_SECTION.value, "children"),
    Input(ElementsIDs.MODEL_SELECTION.value, "value"),
    Input(ElementsIDs.UNIT_TOGGLE.value, "checked"),
    Input(ElementsIDs.functionality_selection.value, "value"),
)
def update_inputs(selected_model, units_selection, function_selection):
    # todo here I should first check if some inputs are already stored in the store
    if selected_model is None:
        return no_update
    units = UnitSystem.IP.value if units_selection else UnitSystem.SI.value
    return input_environmental_personal(selected_model, units, function_selection)


@callback(
    Output(ElementsIDs.note_model.value, "children"),
    Input(ElementsIDs.MODEL_SELECTION.value, "value"),
)
def update_note_model(selected_model):
    if selected_model is None:
        return no_update
    if Models[selected_model].value.note_model:
        return html.Div(
            [
                dmc.Text("Limits of Applicability: ", size="sm", fw=700, span=True),
                dmc.Text(Models[selected_model].value.note_model, size="sm", span=True),
            ]
        )


@callback(
    Output(ElementsIDs.charts_dropdown.value, "children"),
    Input(ElementsIDs.MODEL_SELECTION.value, "value"),
    Input(ElementsIDs.functionality_selection.value, "value"),
)
def update_note_model(selected_model, function_selection):
    if selected_model is None:
        return no_update
    return chart_selector(
        selected_model=selected_model, function_selection=function_selection
    )


@callback(
    Output(ElementsIDs.CHART_CONTAINER.value, "children"),
    Input(MyStores.input_data.value, "data"),
    Input(ElementsIDs.functionality_selection.value, "value"),
)
def update_chart(inputs: dict, function_selection: str):
    selected_model: str = inputs[ElementsIDs.MODEL_SELECTION.value]
    units: str = inputs[ElementsIDs.UNIT_TOGGLE.value]
    chart_selected = inputs[ElementsIDs.chart_selected.value]
    function_selection = inputs[ElementsIDs.functionality_selection.value]

    placeholder = html.Div(
        [
            dmc.Title("Unfortunately this chart has not been implemented yet", order=4),
            dmc.Image(
                src="assets/media/chart_placeholder.png",
            ),
        ]
    )
    image = go.Figure()

    if chart_selected == Charts.t_rh.value.name:
        if selected_model == Models.PMV_EN.name:
            image = t_rh_pmv(
                inputs=inputs, model="iso", function_selection=function_selection
            )
        elif selected_model == Models.PMV_ashrae.name:
            image = t_rh_pmv(
                inputs=inputs, model="ashrae", function_selection=function_selection
            )

    note = ""
    chart: ChartsInfo
    for chart in Models[selected_model].value.charts:
        if chart.name == chart_selected:
            note = chart.note_chart

    graph_component = (
        placeholder
        if not image.data
        else dcc.Graph(
            id=ElementsIDs.GRAPH_HOVER.value,
            figure=image,  # Pass the Plotly figure object here
        )
    )

    return dmc.Stack(
        [
            graph_component,
            html.Div(
                [
                    dmc.Text("Note: ", size="sm", fw=700, span=True),
                    dmc.Text(note, size="sm", span=True),
                ]
            ),
        ]
    )


@callback(
    Output(ElementsIDs.RESULTS_SECTION.value, "children"),
    Input(MyStores.input_data.value, "data"),
)
def update_outputs(inputs: dict):
    return display_results(inputs)
