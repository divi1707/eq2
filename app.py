"""Catalogue explorer (Streamlit).

Run from the project folder:
    streamlit run app/app.py

It shows the downloaded USGS catalogue: counts, magnitude types, rough
completeness and an epicentre map. It never creates data. If no catalogue
exists yet it tells you how to get one.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st


def _find_root() -> Path | None:
    """Find the project folder (the one holding src/ and config.yaml).

    Works whether this file is at app/app.py or at the top of the repo,
    so it also runs on Streamlit Community Cloud.
    """
    here = Path(__file__).resolve().parent
    for cand in (here, *here.parents):
        if (cand / "src" / "common.py").exists() and (cand / "config.yaml").exists():
            return cand
    return None


ROOT = _find_root()
if ROOT is None:
    st.error(
        "Cannot find the project files. The folders `src/` and the file `config.yaml` must be "
        "uploaded next to this app. Upload the whole project folder, not only app.py."
    )
    st.stop()
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.common import load_config, resolve_path  # noqa: E402
from src.download_catalogue import CatalogueError, parse_csv_text  # noqa: E402
from src.explore import (  # noqa: E402
    completeness_table,
    decade_type_table,
    events_per_year,
    filter_frame,
    magnitude_histogram,
    prepare_frame,
)

st.set_page_config(page_title="Himalaya catalogue explorer", layout="wide")
cfg = load_config(ROOT / "config.yaml")
saved_csv = resolve_path(cfg, "raw_dir") / "usgs_catalogue_raw.csv"


@st.cache_data(show_spinner="Reading catalogue...")
def read_text(text: str) -> tuple[pd.DataFrame, int]:
    return prepare_frame(parse_csv_text(text, where="catalogue"))


st.title("Central Himalaya earthquake catalogue explorer")
st.caption(
    "Stage 1 view of the USGS catalogue. This is for checking the data. "
    "It shows no forecasts and makes no predictions."
)

# ---------------- data source ----------------
with st.sidebar:
    st.header("Data")
    options = ["Saved catalogue"] if saved_csv.exists() else []
    options.append("Upload a CSV")
    source = st.radio("Source", options)
    uploaded = st.file_uploader("USGS-format CSV", type="csv") if source == "Upload a CSV" else None

text = None
if source == "Saved catalogue":
    text = saved_csv.read_text(encoding="utf-8")
elif uploaded is not None:
    text = uploaded.getvalue().decode("utf-8", errors="replace")

with st.expander("Get the data (download from USGS)", expanded=text is None):
    st.write(
        "This runs the same downloader as the terminal command and uses the area, "
        "dates and magnitude floor in `config.yaml`. It needs internet access to USGS and "
        "can take several minutes."
    )
    if st.button("Download from USGS now"):
        with st.spinner("Downloading..."):
            res = subprocess.run(
                [sys.executable, "-m", "src.download_catalogue", "--config", "config.yaml"],
                cwd=ROOT, capture_output=True, text=True,
            )
        st.code((res.stdout + res.stderr)[-4000:] or "(no output)")
        if res.returncode == 0:
            st.success("Done. Reload the page and choose 'Saved catalogue'.")
        else:
            st.error(f"The download stopped with exit code {res.returncode}. See the log above.")

if text is None:
    st.info("No catalogue loaded yet. Download one above or upload a USGS-format CSV in the sidebar.")
    st.stop()

try:
    frame, n_dropped = read_text(text)
except CatalogueError as exc:
    st.error(f"This file cannot be used: {exc}")
    st.stop()
if frame.empty:
    st.error("The catalogue has no usable rows.")
    st.stop()

# ---------------- filters ----------------
s = cfg["study"]
with st.sidebar:
    st.header("Filters")
    st.caption("The study area is not final. Change it here to explore.")
    la0 = st.number_input("Min latitude", value=float(s["min_lat"]), step=0.25)
    la1 = st.number_input("Max latitude", value=float(s["max_lat"]), step=0.25)
    lo0 = st.number_input("Min longitude", value=float(s["min_lon"]), step=0.25)
    lo1 = st.number_input("Max longitude", value=float(s["max_lon"]), step=0.25)
    y_min, y_max = int(frame["time"].dt.year.min()), int(frame["time"].dt.year.max())
    years = st.slider("Years", y_min, y_max, (y_min, y_max)) if y_min < y_max else (y_min, y_max)
    mag_floor = float(frame["mag"].min()) if frame["mag"].notna().any() else 0.0
    mag_ceil = float(frame["mag"].max()) if frame["mag"].notna().any() else 10.0
    min_mag = st.slider("Minimum magnitude", mag_floor, mag_ceil, mag_floor, 0.1)

if not (la0 < la1 and lo0 < lo1):
    st.error("Check the bounds: minimum must be smaller than maximum.")
    st.stop()

view, n_outside = filter_frame(frame, (la0, la1, lo0, lo1), years, min_mag)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Events in file", f"{len(frame):,}")
c2.metric("Events after filters", f"{len(view):,}")
c3.metric("Outside chosen box", f"{n_outside:,}")
c4.metric("Missing magnitude", f"{int(frame['mag'].isna().sum()):,}")
if n_dropped:
    st.warning(f"{n_dropped} rows were dropped because time or location could not be read.")
if view.empty:
    st.warning("No events match these filters.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Over time", "Magnitudes", "Map", "Table"])

with tab1:
    st.subheader("Events per year")
    st.bar_chart(events_per_year(view))
    st.caption(
        "A jump in yearly counts usually means better detection, not more earthquakes. "
        "Check this before choosing a magnitude threshold."
    )
    st.subheader("Magnitude type by decade")
    st.dataframe(decade_type_table(view), width="stretch")
    st.caption("If the main type changes between decades, magnitudes are not directly comparable.")

with tab2:
    st.subheader("How many events at each magnitude")
    st.bar_chart(magnitude_histogram(view))
    st.subheader("Rough completeness by 5-year window")
    st.dataframe(completeness_table(view), width="stretch", hide_index=True)
    st.caption(
        "Rough_Mc is the most common magnitude in the window (maximum curvature). It is a first "
        "guide and tends to run low. It is not a final threshold. Windows with fewer than 50 events show None."
    )

with tab3:
    st.subheader("Epicentres")
    show = view
    if len(show) > 20000:
        show = show.sample(20000, random_state=0)
        st.caption("Showing a random 20,000 of the filtered events to keep the map fast.")
    pts = show.rename(columns={"lat": "latitude", "lon": "longitude"}).copy()
    pts["size"] = 1500 * pts["mag"].fillna(pts["mag"].min()).clip(lower=1) ** 2
    st.map(pts, latitude="latitude", longitude="longitude", size="size")
    st.caption("Bigger dots are larger magnitudes. Axes are latitude and longitude in degrees.")

with tab4:
    st.dataframe(view.head(1000), width="stretch", hide_index=True)
    st.caption("First 1,000 rows of the filtered events.")
    st.download_button(
        "Download filtered events (CSV)",
        view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_events.csv",
        mime="text/csv",
    )
