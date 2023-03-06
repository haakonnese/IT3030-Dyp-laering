import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

data_path = Path("data")

mbas = [
    "NO1",
    "NO2",
    "NO3",
    "NO4",
    "NO5",
]
mba_station_map = {
    "SN18700": "NO1",
    "SN44560": "NO2",
    "SN69100": "NO3",
    "SN90490": "NO4",
    "SN50540": "NO5",
}
frost_source_names = {
    "SN18700": "Oslo (Blindern)",
    "SN44560": "Sola",
    "SN50540": "Bergen",
    "SN69100": "Værnes",
    "SN90490": "Tromsø LH",
}


def _read_frost_data():
    data_file = data_path / "frost_weather.csv"
    return pd.read_csv(
        data_file,
        sep=";",
        parse_dates=["referenceTime"],
    )


def _preprocess_frost_data(weather):
    # Rename columns to snake case and keep only the highest quality measurement
    # per timestep. Drop unnecessary columns.
    weather = (
        weather.rename(
            columns={
                "sourceId": "source_id",
                "referenceTime": "reference_time",
                "elementId": "element_id",
            }
        )
        .sort_values(["source_id", "reference_time", "qualityCode"])
        .groupby(["source_id", "reference_time"])
        .head(1)
        .drop(columns=["unit", "timeOffset", "timeResolution", "qualityCode"])
    )

    # Remove :0 from station names, then rename to something more readable.
    weather["source_id"] = (
        weather["source_id"]
        .replace({id_: id_.replace(":0", "") for id_ in weather["source_id"].unique()})
        .replace(mba_station_map)
    )

    # Pivot and resample (downsample) to 1 hour resolution.
    weather = (
        weather.pivot_table(
            index=["reference_time"], columns=["source_id"], values="value"
        )
        .resample("1H")
        .mean()
    )

    # Fill hole(s) and check that there aren't any long holes.
    hole_cols = weather.isna().any()
    hole_cols = hole_cols[hole_cols].index
    for hole_col in hole_cols:
        len_holes = [
            len(list(g))
            for k, g in itertools.groupby(weather[hole_col], lambda x: np.isnan(x))
            if k
        ]
        len_holes = pd.DataFrame({"holes": len_holes})
        print(f"Holes by length and occurrences in column {hole_col}:")
        print(len_holes.value_counts().head(5))
        interp_lim = 3
        print(f"Filling holes up to length {interp_lim}")
        weather[hole_col] = weather[hole_col].interpolate(limit=interp_lim)
    print(f"Any remaining holes after interpolation? {weather.isna().any().any()}")

    return weather


def _read_consumption_data():
    cons = pd.read_csv(
        data_path / "consumption_tznaive.csv",
        sep=",",
        parse_dates=["timestamp"],
        index_col=0,
    )
    cons.index = cons.index.tz_localize("Europe/Oslo", ambiguous="infer")
    return cons


def _preprocess_consumption_data(cons):
    cons = cons.drop(columns=["metered", "profiled"])
    cons["total"] = -cons["total"]
    cons = cons.reset_index().pivot_table(
        index="timestamp", columns="mba", values="total"
    )
    return cons


def _join_consumption_and_weather(cons, weather):
    df = (
        pd.concat([cons, weather], axis=1, keys=("consumption", "temperature"))
        .swaplevel(axis=1)
        .dropna()
    )
    df = df[sorted(list(df.columns))]
    return df


def read_consumption_and_weather():
    cons_raw = _read_consumption_data()
    cons = _preprocess_consumption_data(cons_raw)

    frost_raw = _read_frost_data()
    weather = _preprocess_frost_data(frost_raw)

    return cons, weather, _join_consumption_and_weather(cons, weather)


def plot_consumption(df):
    # Expects dataframe containing both consumption and weather in multiindex
    # (i.e. as the third DataFrame returned from read_consumption_and_weather.
    cons = df.swaplevel(axis=1)["consumption"].melt(ignore_index=False)
    cons.plot(y="value", color="mba")
    fig = px.line(cons, y="value", facet_row="mba")
    fig.update_layout(
        height=1000,
    )
    fig.show()


def plot_consumption_and_weather(df):
    fig = make_subplots(
        rows=len(df.columns),
        cols=1,
        subplot_titles=[" ".join(cols) for cols in df.columns],
    )
    for j, i in enumerate(df.columns.get_level_values("mba").unique()):
        fig.add_trace(
            go.Scatter({"x": df.index, "y": df[(i, "consumption")]}),
            row=j * 2 + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter({"x": df.index, "y": df[(i, "temperature")]}),
            row=j * 2 + 2,
            col=1,
        )
    fig.update_layout(
        height=1000,
    ).show()


if __name__ == "__main__":
    _, _, df = read_consumption_and_weather()
    df.describe()
