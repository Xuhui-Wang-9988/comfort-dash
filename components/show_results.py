import dash_mantine_components as dmc
from pythermalcomfort.models import pmv_ppd, adaptive_ashrae
from pythermalcomfort.utilities import v_relative, clo_dynamic, mapping

from utils.get_inputs import get_inputs
from utils.my_config_file import (
    Models,
    UnitSystem,
    UnitConverter,
    ElementsIDs,
    Functionalities,
    CompareInputColor,
)


def display_results(inputs: dict):
    selected_model: str = inputs[ElementsIDs.MODEL_SELECTION.value]
    units: str = inputs[ElementsIDs.UNIT_TOGGLE.value]

    results = []
    # todo add unit detect if IP inputs and conver into SI calculation
    columns: int = 2
    if selected_model == Models.PMV_EN.name or selected_model == Models.PMV_ashrae.name:
        columns = 3
        standard = "ISO"
        if selected_model == Models.PMV_ashrae.name:
            standard = "ashrae"

        r_pmv = pmv_ppd(
            tdb=inputs[ElementsIDs.t_db_input.value],
            tr=inputs[ElementsIDs.t_r_input.value],
            vr=v_relative(
                v=inputs[ElementsIDs.v_input.value],
                met=inputs[ElementsIDs.met_input.value],
            ),
            rh=inputs[ElementsIDs.rh_input.value],
            met=inputs[ElementsIDs.met_input.value],
            clo=clo_dynamic(
                clo=inputs[ElementsIDs.clo_input.value],
                met=inputs[ElementsIDs.met_input.value],
            ),
            wme=0,
            limit_inputs=True,
            standard=standard,
        )

        # Standard Checker for PMV
        # todo: need to add standard for adaptive methods by ensure if the current red point out of area
        if standard == Models.PMV_ashrae.name:
            if -0.5 <= r_pmv["pmv"] <= 0.5:
                compliance_text = "✔  Complies with ASHRAE Standard 55-2023"
                compliance_color = "green"
            else:
                compliance_text = "✘  Does not comply with ASHRAE Standard 55-2023"
                compliance_color = "red"
        else:  # EN
            if -0.7 <= r_pmv["pmv"] <= 0.7:
                compliance_text = "✔  Complies with EN-16798"
                compliance_color = "green"
            else:
                compliance_text = "✘  Does not comply with EN-16798"
                compliance_color = "red"

        standard_checker = dmc.Text(
            compliance_text,
            c=compliance_color,
            ta="center",
            size="md",
            style={"width": "100%"},
        )

        results = [
            standard_checker,
            dmc.SimpleGrid(
                cols=columns,
                spacing="xs",
                verticalSpacing="xs",
                children=[
                    dmc.Center(dmc.Text(f"PMV: {r_pmv['pmv']:.2f}")),
                    dmc.Center(dmc.Text(f"PPD: {r_pmv['ppd']:.1f}%")),
                ],
            ),
        ]

        if selected_model == Models.PMV_ashrae.name:
            comfort_category = mapping(
                r_pmv["pmv"],
                {
                    -2.5: "Cold",
                    -1.5: "Cool",
                    -0.5: "Slightly Cool",
                    0.5: "Neutral",
                    1.5: "Slightly Warm",
                    2.5: "Warm",
                    10: "Hot",
                },
            )
            results[1].children.append(
                dmc.Center(dmc.Text(f"Sensation: {comfort_category}"))
            )
        elif selected_model == Models.PMV_EN.name:
            comfort_category = mapping(
                r_pmv["pmv"],
                {
                    0.2: "I",
                    0.5: "II",
                    0.7: "III",
                    float("inf"): "IV"
                }
            )
            results[1].children.append(
                dmc.Center(dmc.Text(f"Category: {comfort_category}"))
            )

        # todo add unit detect if IP inputs and conver into SI calculation
        if (
            inputs[ElementsIDs.functionality_selection.value]
            == Functionalities.Compare.value
            and selected_model == Models.PMV_ashrae.name
        ):
            r_pmv_input2 = pmv_ppd(
                tdb=inputs[ElementsIDs.t_db_input_input2.value],
                tr=inputs[ElementsIDs.t_r_input_input2.value],
                vr=v_relative(
                    v=inputs[ElementsIDs.v_input_input2.value],
                    met=inputs[ElementsIDs.met_input_input2.value],
                ),
                rh=inputs[ElementsIDs.rh_input_input2.value],
                met=inputs[ElementsIDs.met_input_input2.value],
                clo=clo_dynamic(
                    clo=inputs[ElementsIDs.clo_input_input2.value],
                    met=inputs[ElementsIDs.met_input_input2.value],
                ),
                wme=0,
                limit_inputs=True,
                standard=standard,
            )

            comfort_category = mapping(
                r_pmv_input2["pmv"],
                {
                    -2.5: "Cold",
                    -1.5: "Cool",
                    -0.5: "Slightly Cool",
                    0.5: "Neutral",
                    1.5: "Slightly Warm",
                    2.5: "Warm",
                    10: "Hot",
                },
            )
            results2 = dmc.SimpleGrid(
                cols=columns,
                spacing="xs",
                verticalSpacing="xs",
                children=[
                    dmc.Center(dmc.Text(f"PMV: {r_pmv_input2['pmv']:.2f}")),
                    dmc.Center(dmc.Text(f"PPD: {r_pmv_input2['ppd']:.1f}%")),
                    dmc.Center(dmc.Text(f"Sensation: {comfort_category}"))
                ],
            )
            results.append(results2)

            # Modify the colour
            for i in range(1, len(results)):
                if i == 1 or i == 2:
                    color = CompareInputColor.InputColor1.value if i == 1 else CompareInputColor.InputColor2.value
                    for child in results[i].children:
                        if isinstance(child, dmc.Center) and isinstance(child.children, dmc.Text):
                            child.children.style = {"color": color}

    elif selected_model == Models.Adaptive_ASHRAE.name:
        columns = 1

        adaptive = adaptive_ashrae(
            tdb=inputs[ElementsIDs.t_db_input.value],
            tr=inputs[ElementsIDs.t_r_input.value],
            t_running_mean=inputs[ElementsIDs.t_rm_input.value],
            v=inputs[ElementsIDs.v_input.value],
        )
        if units == UnitSystem.IP.value:
            adaptive.tmp_cmf = round(
                UnitConverter.celsius_to_fahrenheit(adaptive.tmp_cmf), 2
            )
            adaptive.tmp_cmf_80_low = round(
                UnitConverter.celsius_to_fahrenheit(adaptive.tmp_cmf_80_low), 2
            )
            adaptive.tmp_cmf_80_up = round(
                UnitConverter.celsius_to_fahrenheit(adaptive.tmp_cmf_80_up), 2
            )
            adaptive.tmp_cmf_90_low = round(
                UnitConverter.celsius_to_fahrenheit(adaptive.tmp_cmf_90_low), 2
            )
            adaptive.tmp_cmf_90_up = round(
                UnitConverter.celsius_to_fahrenheit(adaptive.tmp_cmf_90_up), 2
            )
        results.append(dmc.Center(dmc.Text(f"Comfort temperature: {adaptive.tmp_cmf}")))
        results.append(
            dmc.Center(
                dmc.Text(
                    f"Comfort range for 80% occupants: {adaptive.tmp_cmf_80_low} - {adaptive.tmp_cmf_80_up}"
                )
            )
        )
        results.append(
            dmc.Center(
                dmc.Text(
                    f"Comfort range for 90% occupants: {adaptive.tmp_cmf_90_low} - {adaptive.tmp_cmf_90_up}"
                )
            )
        )

    return dmc.Stack(
        children=results,
        gap="xs",
        align="stretch",
    )
