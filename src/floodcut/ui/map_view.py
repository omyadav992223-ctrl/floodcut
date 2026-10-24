"""Interactive map and graph visualization components for FloodCut.

Distinguishes:
- REAL:IMD_GRID: Official IMD 0.25 deg grid cell centre.
- DERIVED:GEOCODED: Approximate waterlogging hotspot points from civic reports.
- SYNTHETIC: Illustrative drainage network nodes and conduits (not surveyed).
"""

from __future__ import annotations

import folium
import matplotlib.pyplot as plt
import networkx as nx
from branca.element import Element

from floodcut.geo_loader import GeoLayer
from floodcut.models import DrainageNetwork


def build_mumbai_folium_map(
    net: DrainageNetwork,
    imd_grid: GeoLayer,
    hotspots: GeoLayer,
    show_drains: bool = True,
    show_hotspots: bool = True,
    show_imd_grid: bool = True,
) -> folium.Map:
    """Create a Folium map visualizing Mumbai drainage and geospatial layers.

    All features carry explicit provenance tags in popups and labels.
    """
    # Center map on Mumbai drainage area (~19.01 N, 72.84 E)
    m = folium.Map(
        location=[19.015, 72.842],
        zoom_start=12,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # 1. Drainage Conduits (SYNTHETIC)
    if show_drains:
        drain_group = folium.FeatureGroup(name="Drainage Conduits (SYNTHETIC)", show=True)
        for d in net.drains.values():
            j_from = net.junctions.get(d.from_junction)
            j_to = net.junctions.get(d.to_junction)
            if not j_from or not j_to:
                continue

            color = "#d9534f" if d.capacity_factor < 1.0 else "#0275d8"
            dash = "5, 5" if d.capacity_factor < 1.0 else None
            weight = 4 if d.kind in ("open_nallah", "box_drain") else 2.5

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 180px;">
                <b style="color: {color};">Drain: {d.id}</b><br/>
                <b>Kind:</b> {d.kind}<br/>
                <b>Path:</b> {d.from_junction} &rarr; {d.to_junction}<br/>
                <b>Design Cap:</b> {d.design_capacity_m3s:.2f} m³/s<br/>
                <b>Effective Cap:</b> {d.effective_capacity_m3s:.2f} m³/s<br/>
                <b>Condition:</b> {d.capacity_factor * 100:.0f}% healthy<br/>
                <b>Repair Cost:</b> {d.repair_cost:.1f} units<br/>
                <hr style="margin: 4px 0;"/>
                <span style="font-size: 10px; color: #777;">
                    PROVENANCE: <b>SYNTHETIC</b> (Hypothetical capacity & route)
                </span>
            </div>
            """

            folium.PolyLine(
                locations=[[j_from.lat, j_from.lon], [j_to.lat, j_to.lon]],
                color=color,
                weight=weight,
                opacity=0.8,
                dash_array=dash,
                tooltip=f"Drain {d.id} ({d.effective_capacity_m3s:.1f} m³/s) [SYNTHETIC]",
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(drain_group)

        drain_group.add_to(m)

        # Junctions / Inlets / Outfalls (SYNTHETIC)
        node_group = folium.FeatureGroup(name="Network Nodes (SYNTHETIC)", show=True)
        for j in net.junctions.values():
            is_outfall = j.id in net.outfalls
            fill_color = "#5cb85c" if is_outfall else "#337ab7"
            radius = 7 if is_outfall else 5
            icon_name = "tint" if is_outfall else "circle"

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 160px;">
                <b>Node: {j.id}</b> ({j.name})<br/>
                <b>Type:</b> {"Outfall (Tidal Sink)" if is_outfall else "Network Inlet / Junction"}<br/>
                <b>Coordinates:</b> {j.lat:.4f}°N, {j.lon:.4f}°E<br/>
                <hr style="margin: 4px 0;"/>
                <span style="font-size: 10px; color: #777;">
                    PROVENANCE: <b>SYNTHETIC</b> (Illustrative coordinates)
                </span>
            </div>
            """

            folium.CircleMarker(
                location=[j.lat, j.lon],
                radius=radius,
                color="#222",
                weight=1.5,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.9,
                tooltip=f"{'Outfall' if is_outfall else 'Junction'} {j.id}: {j.name}",
                popup=folium.Popup(popup_html, max_width=240),
            ).add_to(node_group)

        node_group.add_to(m)

    # 2. Waterlogging Hotspots (DERIVED:GEOCODED)
    if show_hotspots:
        hotspot_group = folium.FeatureGroup(name="Waterlogging Hotspots (DERIVED:GEOCODED)", show=True)
        for pt in hotspots.points:
            ward = pt.attributes.get("ward", "Unknown")
            notes = pt.attributes.get("notes", "")

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 190px;">
                <b style="color: #e65100;">⚠️ Hotspot: {pt.name}</b><br/>
                <b>Ward:</b> {ward}<br/>
                <b>Coordinates:</b> {pt.lat:.4f}°N, {pt.lon:.4f}°E<br/>
                <b>Notes:</b> {notes}<br/>
                <hr style="margin: 4px 0;"/>
                <span style="font-size: 10px; color: #b71c1c;">
                    PROVENANCE: <b>DERIVED:GEOCODED</b><br/>
                    Approx. location from civic reports (&plusmn;500m).<br/>
                    Not a measured flood depth or extent polygon.
                </span>
            </div>
            """

            folium.Marker(
                location=[pt.lat, pt.lon],
                icon=folium.Icon(color="orange", icon="warning-sign", prefix="glyphicon"),
                tooltip=f"Hotspot: {pt.name} (Ward {ward})",
                popup=folium.Popup(popup_html, max_width=260),
            ).add_to(hotspot_group)

        hotspot_group.add_to(m)

    # 3. IMD Grid Point (REAL:IMD_GRID)
    if show_imd_grid:
        grid_group = folium.FeatureGroup(name="IMD Rainfall Grid (REAL:IMD_GRID)", show=True)
        for pt in imd_grid.points:
            res = pt.attributes.get("grid_resolution_deg", "0.25")
            source = pt.attributes.get("data_source", "IMD via OpenCity")

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 200px;">
                <b style="color: #2e7d32;">🌧️ {pt.name}</b><br/>
                <b>ID:</b> {pt.id}<br/>
                <b>Grid Resolution:</b> {res}° &times; {res}°<br/>
                <b>Coordinates:</b> {pt.lat:.4f}°N, {pt.lon:.4f}°E<br/>
                <b>Source:</b> {source}<br/>
                <hr style="margin: 4px 0;"/>
                <span style="font-size: 10px; color: #1b5e20;">
                    PROVENANCE: <b>REAL:IMD_GRID</b><br/>
                    Cell centre of IMD 0.25° gridded observational dataset.
                </span>
            </div>
            """

            # Draw representative 0.25x0.25 degree boundary box (~27km x 27km)
            dlat = 0.125
            dlon = 0.125
            folium.Rectangle(
                bounds=[[pt.lat - dlat, pt.lon - dlon], [pt.lat + dlat, pt.lon + dlon]],
                color="#2e7d32",
                weight=1.5,
                fill=True,
                fill_color="#81c784",
                fill_opacity=0.15,
                tooltip="IMD 0.25° Grid Cell Area [REAL:IMD_GRID]",
            ).add_to(grid_group)

            folium.Marker(
                location=[pt.lat, pt.lon],
                icon=folium.Icon(color="green", icon="cloud", prefix="glyphicon"),
                tooltip=f"IMD Grid Station: {pt.name} [REAL]",
                popup=folium.Popup(popup_html, max_width=280),
            ).add_to(grid_group)

        grid_group.add_to(m)

    # Add interactive layer controls
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # Add on-map custom HTML legend
    legend_html = """
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
        background-color: white;
        padding: 10px 14px;
        border-radius: 6px;
        box-shadow: 0 0 10px rgba(0,0,0,0.25);
        font-family: Arial, sans-serif;
        font-size: 12px;
        line-height: 1.5;
    ">
        <b style="font-size: 13px;">FloodCut Map Legend</b><br/>
        <span style="color: #2e7d32;">■</span> IMD 0.25° Grid Cell (<b>REAL</b>)<br/>
        <span style="color: #e65100;">▲</span> Waterlogging Hotspot (<b>DERIVED</b>)<br/>
        <span style="color: #0275d8;">—</span> Healthy Drain Conduit (<b>SYNTHETIC</b>)<br/>
        <span style="color: #d9534f;">---</span> Degraded / Blocked Drain (<b>SYNTHETIC</b>)<br/>
        <span style="color: #5cb85c;">●</span> Sea / River Outfall (<b>SYNTHETIC</b>)<br/>
        <span style="color: #337ab7;">●</span> Network Inlet / Junction (<b>SYNTHETIC</b>)
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    return m


def plot_offline_drainage_graph(net: DrainageNetwork) -> plt.Figure:
    """Offline Matplotlib fallback for topological graph inspection."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)

    g = nx.DiGraph()
    pos = {}
    for jid, j in net.junctions.items():
        g.add_node(jid, name=j.name, is_outfall=jid in net.outfalls)
        pos[jid] = (j.lon, j.lat)

    edge_colors = []
    for d in net.drains.values():
        g.add_edge(d.from_junction, d.to_junction, id=d.id, cap=d.effective_capacity_m3s)
        edge_colors.append("#d9534f" if d.capacity_factor < 1.0 else "#0275d8")

    node_colors = [
        "#5cb85c" if g.nodes[n].get("is_outfall") else "#337ab7" for n in g.nodes
    ]

    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=380, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color=edge_colors,
        width=2,
        arrowsize=14,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    ax.set_title(
        "Drainage Network Topology (SYNTHETIC Model Graph)\n"
        "Green: Outfalls | Blue: Junctions | Red: Degraded Drains",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig
