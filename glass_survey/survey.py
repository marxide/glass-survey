from functools import partial
import numpy as np
import pymongo
import pandas as pd
import bokeh.plotting
import bokeh.transform
import bokeh.models
import bokeh.layouts
import bokeh.embed
from flask import Blueprint, current_app, request, render_template, json, Response
from glass_uvdata.const import (
    N_VIS_EXPECTED,
    N_CUTS_TARGET,
    N_CYCLES,
    N_CUTS_TARGET_CONFIG,
)
from glass_survey import db

bp = Blueprint("survey", __name__, url_prefix="/survey")


def angle_diff(a, b):
    return 180 - abs(abs(a - b) - 180)


def lmst_to_angle(lmst):
    return 360 * ((lmst % 24) / 24)


def angle_to_lmst(angle):
    return (angle % 360) / 360 * 24


def angle_gaps(angles: pd.Series):
    _angles = angles.values
    _angles_comp = (_angles + 180) % 360
    _angles_all = np.sort(np.concatenate((_angles, _angles_comp)))
    return np.diff(
        _angles_all, prepend=angle_diff(_angles_all.min(), _angles_all.max())
    )


def max_angle_gap(angles: pd.Series):
    return angle_gaps(angles).max()


def min_scans_source(n_scans: pd.Series, sources: pd.DataFrame):
    """Return the source name that has the least number of scans.
    
    Arguments
        n_scans: pd.Series - the scan counts.
        sources: pd.DataFrame - the DataFrame `n_scans` was drawn from. Must contain a `source` column.
    
    Returns
        pd.Series of source names.
    """
    return sources.source.loc[n_scans.idxmin()]


def fill_lmst_gaps(_df: pd.DataFrame, n_target_scans=N_CUTS_TARGET, return_all=True):
    """Iteratively find the largest LMST gap and place a new scan in the middle of the gap
    until the target number of scans is reached.
    
    Arguments:
        _df {pd.DataFrame} -- scans to evaluate for gaps.
    
    Keyword Arguments:
        return_all {bool} -- If true, original scans and proposed scans will be returned,
            otherwise only the proposed scans. (default: {True})
    
    Returns:
        pd.DataFrame -- new proposed scans, and originals if return_all=True. Proposed scans will have the date set to {None}.
    """
    _df = _df.sort_values("lmst_angle")
    while len(_df) < n_target_scans:
        # get LMST gaps
        angles = _df.lmst_angle
        diffs = angles.diff()

        # add diff between first and last scan since LMST wraps
        diffs.iloc[0] = angle_diff(angles.iloc[-1], angles.iloc[0])

        # find largest gap boundaries. this will be where all the conjugate points lie
        conj_gap_end_iidx = diffs.values.argmax()
        conj_gap_start_iidx = conj_gap_end_iidx - 1
        conj_gap_start = _df.iloc[conj_gap_start_iidx]

        # add conjugate point for one of the edge points
        conj_point = conj_gap_start.copy()
        conj_point["lmst"] = (conj_point["lmst"] + 12) % 24
        conj_point["lmst_angle"] = lmst_to_angle(conj_point["lmst"])
        conj_point["datetime"] = None
        conj_point_idx = _df.index.max() + 1
        conj_point.name = conj_point_idx
        _df = _df.append(conj_point)
        _df = _df.sort_values("lmst_angle")

        # redetermine LMST gaps now with the conjugate point included
        angles = _df.lmst_angle
        diffs = angles.diff()
        # add diff between first and last scan since LMST wraps
        diffs.iloc[0] = angle_diff(angles.iloc[-1], angles.iloc[0])

        # find second largest LMST gap, the largest will be the conjugate gap
        diffs.iloc[diffs.values.argmax()] = 0  # remove largest gap
        gap_end_iidx = diffs.values.argmax()
        gap_start_iidx = gap_end_iidx - 1
        gap_start = _df.iloc[gap_start_iidx]
        gap_end = _df.iloc[gap_end_iidx]

        # place scan in gap middle
        # target_lmst_angle = (gap_start.lmst_angle + gap_end.lmst_angle) / 2
        loop = True
        n = 2
        while loop:
            target_lmst_angle = gap_start.lmst_angle + (
                abs(angle_diff(gap_start.lmst_angle, gap_end.lmst_angle)) / n
            )
            proposed_gap = angle_diff(target_lmst_angle, gap_start.lmst_angle)
            # if new gap is larger than the second largest gap in diff (the first largest is the current gap)
            next_largest_gap = diffs.sort_values(ascending=False).iloc[1]
            if proposed_gap > next_largest_gap:
                n += 1
            else:
                loop = False
        target_scan = gap_start.copy()
        target_scan["datetime"] = None
        target_scan["lmst_angle"] = target_lmst_angle
        target_scan["lmst"] = angle_to_lmst(target_lmst_angle)
        target_scan.name = _df.index.max() + 1
        _df = _df.append(target_scan)

        # remove the conjugate
        _df = _df.drop(conj_point_idx).sort_values("lmst_angle")
    if not return_all:
        _df = _df[pd.isna(_df.datetime)]
    return _df


@bp.route("/plots")
def plots() -> Response:
    field = request.args.get("field", "A")
    phase_cal_groups = request.args.get("groupByPhase", False) == "true"
    df = pd.read_sql_query(
        "SELECT datetime, lmst, n_unflagged, frac_vis, array_config, source, ra, `dec`, field, daily_set, `group` FROM scan JOIN pointing ON scan.source_id = pointing.source WHERE pointing.field = %(field)s",
        db.engine,
        params={'field': field}
    )
    df["lmst_angle"] = lmst_to_angle(df.lmst)  # convert LMST values to range 0-360
    df["array_config"] = df.array_config.str.replace(
        r"[A-z]", "km"
    )  # only care about the array config kind (6km, 1.5km)

    # group by source and array_config, sum visibility counts
    grp = df.groupby(["source", "array_config"], as_index=False)
    frac_vis_sources = grp.agg(
        {
            "n_unflagged": "sum",
            "lmst_angle": lambda lmst_angles: max_angle_gap(lmst_angles) / 360 * 24,
            "datetime": "count",
            "ra": "first",
            "dec": "first",
            "field": "first",
            "daily_set": "first",
            "group": "first",
        }
    ).rename(columns={"datetime": "n_cuts", "lmst_angle": "max_lmst_gap"})
    # calculate fraction of unflagged visibilities
    frac_vis_sources["n_cuts_target"] = frac_vis_sources["array_config"].apply(
        N_CUTS_TARGET_CONFIG.get
    )
    frac_vis_sources["n_expected"] = N_VIS_EXPECTED * frac_vis_sources["n_cuts_target"]
    frac_vis_sources["frac_unflagged"] = (
        frac_vis_sources["n_unflagged"] / frac_vis_sources["n_expected"]
    )

    # convert to wide form data and collapse columns that do not change between configs e.g. RA, Dec
    frac_vis_sources = frac_vis_sources.set_index(["source", "array_config"])
    frac_vis_sources = frac_vis_sources.unstack()
    frac_vis_sources.columns = [
        "_".join(col) for col in frac_vis_sources.columns.to_flat_index()
    ]
    for col in ["ra", "dec", "field", "daily_set", "group"]:
        frac_vis_sources = frac_vis_sources.drop(f"{col}_1.5km", axis=1).rename(
            columns={f"{col}_6km": col}
        )

    if phase_cal_groups:
        grp_groups = frac_vis_sources.groupby(['field', 'daily_set', 'group'])
        frac_vis_groups = grp_groups.agg({
            'n_unflagged_6km': 'sum',
            'n_unflagged_1.5km': 'sum',
            'n_expected_6km': 'sum',
            'n_expected_1.5km': 'sum',
            'ra': 'median',
            'dec': 'median',
        })
        frac_vis_groups['frac_unflagged_6km'] = frac_vis_groups['n_unflagged_6km'] / frac_vis_groups['n_expected_6km']
        frac_vis_groups['frac_unflagged_1.5km'] = frac_vis_groups['n_unflagged_1.5km'] / frac_vis_groups['n_expected_1.5km']

        frac_vis_sources = frac_vis_sources.reset_index().set_index(['field', 'daily_set', 'group'])
        cols = (
            'n_unflagged_6km',
            'n_unflagged_1.5km',
            'n_expected_6km',
            'n_expected_1.5km',
            'frac_unflagged_6km',
            'frac_unflagged_1.5km',
        )
        #for col in cols:
        #    frac_vis_sources[col] = 0
        for idx, data in frac_vis_groups.iterrows():
            for col in cols:
                frac_vis_sources.loc[idx, col] = data[col]
        frac_vis_sources = frac_vis_sources.reset_index()
    data = bokeh.models.ColumnDataSource(frac_vis_sources)

    tools = "hover,pan,box_zoom,wheel_zoom,reset,save"
    tooltips = [
        ("source", "@source"),
        ("daily set", "@daily_set"),
        ("group", "@group"),
        ("(RA, Dec)", "(@ra, @dec)"),
        ("Unflagged 6km", "@frac_unflagged_6km"),
        ("Unflagged 1.5km", "@{frac_unflagged_1.5km}"),
    ]

    p1 = bokeh.plotting.figure(
        title="Unflagged fraction 6km",
        x_axis_label="RA",
        y_axis_label="Dec",
        x_range=(data.data["ra"].max()+0.2, data.data["ra"].min()-0.2),
        tools=tools,
        tooltips=tooltips,
    )
    mapper_6km = bokeh.transform.linear_cmap(
        "frac_unflagged_6km", "Viridis256", low=0, high=1
    )
    p1.circle(
        "ra",
        "dec",
        color=mapper_6km,
        radius=0.05,
        source=data,
        hover_line_color="red",
        hover_fill_color=mapper_6km,
    )

    p2 = bokeh.plotting.figure(
        title="Unflagged fraction 1.5km",
        x_axis_label="RA",
        y_axis_label="Dec",
        x_range=p1.x_range,
        y_range=p1.y_range,
        tools=tools,
        tooltips=tooltips,
    )
    mapper_1_5km = bokeh.transform.linear_cmap(
        "frac_unflagged_1.5km", "Viridis256", low=0, high=1
    )
    p2.circle(
        "ra",
        "dec",
        color=mapper_1_5km,
        radius=0.05,
        source=data,
        hover_line_color="red",
        hover_fill_color=mapper_1_5km,
    )

    cbar = bokeh.models.annotations.ColorBar(
        color_mapper=mapper_6km["transform"], location=(0, 0)
    )
    p1.add_layout(cbar, "right")
    p2.add_layout(cbar, "right")

    coord_formatter = bokeh.models.widgets.NumberFormatter(format="0.000")
    frac_formatter = bokeh.models.widgets.NumberFormatter(format="0.0%")
    columns = [
        bokeh.models.widgets.TableColumn(field="source", title="Source"),
        bokeh.models.widgets.TableColumn(field="daily_set", title="Daily Set"),
        bokeh.models.widgets.TableColumn(field="group", title="Phase cal group"),
        bokeh.models.widgets.TableColumn(field="ra", title="RA", formatter=coord_formatter),
        bokeh.models.widgets.TableColumn(field="dec", title="Dec", formatter=coord_formatter),
        bokeh.models.widgets.TableColumn(field="frac_unflagged_6km", title="Unflagged 6km", formatter=frac_formatter),
        bokeh.models.widgets.TableColumn(field="frac_unflagged_1.5km", title="Unflagged 1.5km", formatter=frac_formatter),
    ]
    table = bokeh.models.widgets.DataTable(source=data, columns=columns, height=200)

    #layout = bokeh.layouts.row(p1, p2, sizing_mode="scale_width", margin=(0, 50, 0, 0))
    layout = bokeh.layouts.layout([
        [p1, p2],
        [table],
    ], sizing_mode="scale_width", margin=(0, 50, 0, 0))

    return json.jsonify(bokeh.embed.json_item(layout))


@bp.route("/")
def main():
    return render_template("survey/index.html")

