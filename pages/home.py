import dash
import dash_mantine_components as dmc
from dash import html, callback, Output, Input, no_update, State, ctx, dcc

from components.charts import (
    t_rh_pmv,
    chart_selector,
    SET_outputs_chart,
    pmot_ot_adaptive_ashrae,
)
from components.dropdowns import (
    model_selection,
)
from components.functionality_selection import functionality_selection
from components.input_environmental_personal import input_environmental_personal
from components.my_card import my_card
from components.show_results import display_results
from utils.get_inputs import get_inputs
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

from urllib.parse import parse_qs, urlencode


dash.register_page(__name__, path=URLS.HOME.value)

layout = dmc.Stack(
    [
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
    Input(ElementsIDs.inputs_form.value, "n_clicks"),
    Input(ElementsIDs.inputs_form.value, "children"),
    Input(ElementsIDs.clo_input.value, "value"),
    Input(ElementsIDs.met_input.value, "value"),
    Input(ElementsIDs.UNIT_TOGGLE.value, "checked"),
    Input(ElementsIDs.chart_selected.value, "value"),
    Input(ElementsIDs.functionality_selection.value, "value"),
    State(ElementsIDs.MODEL_SELECTION.value, "value"),
    prevent_initial_call=True,
)
# save the inputs in the store, and update the URL
def update_store_inputs(
    form_clicks: int,
    form_content: dict,
    clo_value: float,
    met_value: float,
    units_selection: str,
    chart_selected: str,
    functionality_selection: str,
    selected_model: str,
):
    if form_clicks is None:
        return no_update, no_update
    units = UnitSystem.IP.value if units_selection else UnitSystem.SI.value
    inputs = get_inputs(selected_model, form_content, units)

    if ctx.triggered:
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == ElementsIDs.clo_input.value:
            inputs[ElementsIDs.clo_input.value] = float(clo_value)
        if triggered_id == ElementsIDs.met_input.value:
            inputs[ElementsIDs.met_input.value] = float(met_value)

    inputs[ElementsIDs.UNIT_TOGGLE.value] = units
    inputs[ElementsIDs.MODEL_SELECTION.value] = selected_model
    inputs[ElementsIDs.chart_selected.value] = chart_selected
    inputs[ElementsIDs.functionality_selection.value] = functionality_selection

    # encode the inputs to be used in the URL
    url_search = f"?{urlencode(inputs)}"
    # print(f"url_params: {url_data}")

    return inputs, url_search


@callback(
    Output(ElementsIDs.MODEL_SELECTION.value, "value"),
    Output(ElementsIDs.INPUT_SECTION.value, "children"),
    Input(ElementsIDs.URL.value, "search"),
    State(MyStores.input_data.value, "data"),
    State(ElementsIDs.UNIT_TOGGLE.value, "checked"),
)
def update_model_and_inputs(url_search, stored_data, units_selection):
    # Parse URL parameters
    url_params = parse_qs(url_search.lstrip("?"))
    url_params = {k: v[0] if len(v) == 1 else v for k, v in url_params.items()}

    # If URL parameters exist, use them; otherwise, fall back to stored data
    params = url_params if url_params else (stored_data or {})

    # Get the selected model from params, or use the default if not found
    selected_model = params.get(
        ElementsIDs.MODEL_SELECTION.value, Models.PMV_ashrae.name
    )

    units = UnitSystem.IP.value if units_selection else UnitSystem.SI.value

    # Convert numeric strings to float
    for key, value in params.items():
        try:
            params[key] = float(value)
        except (ValueError, TypeError):
            pass

    # Ensure that the unit toggle and model selection are always respected
    params[ElementsIDs.UNIT_TOGGLE.value] = units
    params[ElementsIDs.MODEL_SELECTION.value] = selected_model

    # Update the input section
    input_section = input_environmental_personal(
        selected_model, units, url_params=params
    )

    return selected_model, input_section


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
)
def update_note_model(selected_model):
    if selected_model is None:
        return no_update
    return chart_selector(selected_model=selected_model)


@callback(
    Output(ElementsIDs.CHART_CONTAINER.value, "children"),
    Input(MyStores.input_data.value, "data"),
)
def update_chart(
    inputs: dict,
):
    selected_model: str = inputs[ElementsIDs.MODEL_SELECTION.value]
    units: str = inputs[ElementsIDs.UNIT_TOGGLE.value]
    chart_selected = inputs[ElementsIDs.chart_selected.value]

    image = html.Div(
        [
            dmc.Title("Unfortunately this chart has not been implemented yet", order=4),
            dmc.Image(
                src="assets/media/chart_placeholder.png",
            ),
        ]
    )

    if chart_selected == Charts.t_rh.value.name:
        if selected_model == Models.PMV_EN.name:
            image = t_rh_pmv(inputs=inputs, model="iso")
        elif selected_model == Models.PMV_ashrae.name:
            image = t_rh_pmv(inputs=inputs, model="ashrae")
    if chart_selected == Charts.set_outputs.value.name:
        image = SET_outputs_chart(inputs=inputs)
    if chart_selected == Charts.pmot_ot.value.name:
        if selected_model == Models.Adaptive_ASHRAE.name:
            image = pmot_ot_adaptive_ashrae(inputs=inputs, model="ashrae")

    note = ""
    chart: ChartsInfo
    for chart in Models[selected_model].value.charts:
        if chart.name == chart_selected:
            note = chart.note_chart

    return dmc.Stack(
        [
            image,
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
