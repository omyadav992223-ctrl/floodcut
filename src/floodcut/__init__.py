"""FloodCut: flood-risk and drainage-repair prioritization.

Phase 6: Frontend and Visualization.
  - app.py         : Streamlit interactive frontend.
  - ui/map_view.py : Folium interactive map & offline graph.
  - ui/charts.py   : Matplotlib visualizers for rainfall, flows, & repair gaps.
  - ui/state.py    : Cached state & data accessors.
  - provenance.py  : Data provenance labels and DatasetRecord.
  - rainfall_loader.py : IMD/OpenCity monthly rainfall CSV loading.
  - geo_loader.py  : Geospatial point loading and validation.
"""

__version__ = "0.6.0"
