# ============================================================
# 1. Configuration de l'environnement
# ============================================================
import osmnx as ox
import numpy as np
import pyvista as pv
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path
import random
import srtm  # pip install srtm.py
from pyproj import Transformer
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Nombre de cœurs CPU disponibles
NUM_CORES = mp.cpu_count()
print(f"⚡ Multiprocessing activé: {NUM_CORES} cœurs CPU disponibles")


# ============================================================
# 2. Extraction des données OSM
# ============================================================

def extract_osm_data(location, radius=500, download_buildings=True, download_streets=True):
    """Extrait les empreintes de bâtiments et le réseau routier depuis OpenStreetMap."""
    print(f"\n[1/3] Téléchargement des données OSM pour {location}...")

    center_point = ox.geocoder.geocode(location)
    print(f"  ✓ Coordonnées: {center_point[0]:.4f}, {center_point[1]:.4f}")

    buildings = None
    streets = None

    if download_buildings:
        print("  → Téléchargement des bâtiments...", end=" ", flush=True)
        buildings = ox.features_from_point(center_point, tags={'building': True}, dist=radius)
        buildings = buildings.to_crs(epsg=2154)
        print(f"✓ {len(buildings)} bâtiments")
    else:
        print("  → Bâtiments: ignorés (SHOW_BUILDINGS=False)")

    if download_streets:
        print("  → Téléchargement des rues...", end=" ", flush=True)
        streets = ox.graph_from_point(center_point, dist=radius, network_type='all', simplify=False)
        streets = ox.project_graph(streets, to_crs='epsg:2154')
        print(f"✓ {len(streets.edges)} segments")
    else:
        print("  → Rues: ignorées (SHOW_STREETS=False)")

    return buildings, streets, center_point


# ============================================================
# 2b. Données d'élévation du terrain (SRTM) - PARALLÉLISÉ
# ============================================================

def _fetch_elevation_row(args):
    """Worker pour récupérer une ligne d'élévation."""
    row_idx, lat_row, lon_row = args
    elevation_data = srtm.get_data()
    row_elevations = [elevation_data.get_elevation(lat_row[j], lon_row[j]) or 0 for j in range(len(lat_row))]
    return row_idx, row_elevations


def create_terrain_mesh(center_point, radius, resolution=50, z_exaggeration=1.0):
    """Crée un maillage 3D du terrain avec multiprocessing."""
    print(f"\n[2/3] Téléchargement du terrain ({resolution}x{resolution} points) sur {NUM_CORES} cœurs...")

    lat, lon = center_point
    lat_offset = radius / 111000
    lon_offset = radius / (111000 * np.cos(np.radians(lat)))

    lats = np.linspace(lat - lat_offset, lat + lat_offset, resolution)
    lons = np.linspace(lon - lon_offset, lon + lon_offset, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    tasks = [(i, lat_grid[i, :], lon_grid[i, :]) for i in range(resolution)]

    elevations = np.zeros((resolution, resolution))
    completed = 0

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        for row_idx, row_elevations in executor.map(_fetch_elevation_row, tasks):
            elevations[row_idx, :] = row_elevations
            completed += 1
            sys.stdout.write(f"\r  → Élévation: {completed * 100 // resolution}%")
            sys.stdout.flush()

    print(" ✓")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    x_grid, y_grid = transformer.transform(lon_grid, lat_grid)

    elevations_exaggerated = elevations * z_exaggeration
    terrain = pv.StructuredGrid(x_grid, y_grid, elevations_exaggerated)
    terrain['elevation'] = elevations.flatten()

    elevation_data = srtm.get_data()
    print(f"  ✓ Altitude: {elevations.min():.0f}m - {elevations.max():.0f}m (exagération: x{z_exaggeration})")

    return terrain, elevation_data, transformer, z_exaggeration


# ============================================================
# 3. Génération des empreintes de bâtiments avec hauteurs
# ============================================================

def generate_footprints_with_heights(buildings):
    """Convertit les géométries en polygones avec hauteurs réelles."""
    footprints = []
    heights = []
    FLOOR_HEIGHT = 3.0
    DEFAULT_HEIGHT = 10.0

    for _, row in buildings.iterrows():
        geom = row.geometry
        height = None

        if 'height' in buildings.columns and pd.notna(row.get('height')):
            try:
                h = row['height']
                if isinstance(h, str):
                    h = float(h.replace('m', '').replace(' ', ''))
                height = float(h)
            except (ValueError, TypeError):
                pass

        if height is None and 'building:levels' in buildings.columns and pd.notna(row.get('building:levels')):
            try:
                height = float(row['building:levels']) * FLOOR_HEIGHT
            except (ValueError, TypeError):
                pass

        if height is None:
            height = DEFAULT_HEIGHT
        height = max(3.0, min(height, 300.0))

        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                footprints.append(poly)
                heights.append(height)
        elif isinstance(geom, Polygon):
            footprints.append(geom)
            heights.append(height)

    return footprints, heights


# ============================================================
# 4. Génération des bâtiments 3D - ENTIÈREMENT PARALLÉLISÉ
# ============================================================

def _create_building_data(coords, height, ground_elevation):
    """Crée les données brutes d'un bâtiment (vertices et faces)."""
    coords = coords[:-1]
    n = len(coords)

    # Vertices: base puis toit
    base = np.column_stack((coords, np.full(n, ground_elevation)))
    top = np.column_stack((coords, np.full(n, ground_elevation + height)))
    vertices = np.vstack((base, top))

    # Faces
    faces = []
    # Base
    faces.extend([n] + list(range(n)))
    # Toit
    faces.extend([n] + list(range(n, 2 * n)))
    # Murs
    for i in range(n):
        ni = (i + 1) % n
        faces.extend([4, i, ni, n + ni, n + i])

    return vertices, np.array(faces)


def _process_building_batch(args):
    """Worker pour traiter un lot de bâtiments."""
    batch_data, z_exaggeration, building_exaggeration = args

    elevation_data = srtm.get_data()
    transformer_inverse = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    results = []
    for idx, coords, cx, cy, height in batch_data:
        # Élévation du sol
        lon, lat = transformer_inverse.transform(cx, cy)
        elev = elevation_data.get_elevation(lat, lon) or 0
        ground_elevation = elev * z_exaggeration

        # Création du bâtiment
        vertices, faces = _create_building_data(coords, height * building_exaggeration, ground_elevation)
        results.append((idx, vertices, faces, height))

    return results


def extrude_buildings(footprints, heights, elevation_data=None, z_exaggeration=1.0, building_exaggeration=1.0):
    """Extrude les empreintes 2D en bâtiments 3D - entièrement parallélisé."""
    total = len(footprints)

    if total == 0:
        return pv.PolyData(), []

    print(f"  → Extrusion parallèle de {total} bâtiments sur {NUM_CORES} cœurs...")

    # Préparation des données
    building_data = []
    for i, (fp, h) in enumerate(zip(footprints, heights)):
        coords = np.array(fp.exterior.coords)
        centroid = fp.centroid
        building_data.append((i, coords, centroid.x, centroid.y, h))

    # Division en lots pour le multiprocessing
    batch_size = max(1, total // (NUM_CORES * 4))
    batches = []
    for i in range(0, total, batch_size):
        batch = building_data[i:i + batch_size]
        batches.append((batch, z_exaggeration, building_exaggeration))

    # Traitement parallèle
    all_results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = {executor.submit(_process_building_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)
            completed += len(batch_results)
            sys.stdout.write(f"\r  → Extrusion bâtiments: {completed * 100 // total}%")
            sys.stdout.flush()

    print(" ✓")

    # Tri par index
    all_results.sort(key=lambda x: x[0])

    # Fusion rapide: concaténation des vertices et faces
    print("  → Fusion des meshes...", end=" ", flush=True)

    all_vertices = []
    all_faces = []
    all_heights = []
    vertex_offset = 0

    for idx, vertices, faces, height in all_results:
        all_vertices.append(vertices)
        # Décaler les indices des faces
        adjusted_faces = []
        i = 0
        while i < len(faces):
            n_verts = faces[i]
            adjusted_faces.append(n_verts)
            for j in range(1, n_verts + 1):
                adjusted_faces.append(faces[i + j] + vertex_offset)
            i += n_verts + 1
        all_faces.extend(adjusted_faces)
        all_heights.extend([height] * len(vertices))
        vertex_offset += len(vertices)

    # Création du mesh unique
    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.array(all_faces)

    city_mesh = pv.PolyData(combined_vertices, combined_faces)
    city_mesh['height'] = np.array(all_heights)

    print("✓")
    return city_mesh, []


# ============================================================
# 5. Fonction d'export
# ============================================================

def save_to_obj(mesh, output_path):
    """Sauvegarde le modèle 3D au format OBJ."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Sauvegarde du modèle vers {output_path}")
    mesh.save(output_path)
    print("Export terminé")


# ============================================================
# 6. Conversion du réseau routier - ENTIÈREMENT PARALLÉLISÉ
# ============================================================

def _process_street_batch(args):
    """Worker pour traiter un lot de rues."""
    batch_data, z_exaggeration = args

    elevation_data = srtm.get_data()
    transformer_inverse = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    results = []
    for idx, x_coords, y_coords in batch_data:
        elevations = []
        for x, y in zip(x_coords, y_coords):
            lon, lat = transformer_inverse.transform(x, y)
            elev = elevation_data.get_elevation(lat, lon) or 0
            elevations.append(elev * z_exaggeration)

        z_coords = np.array(elevations) + 0.5 * z_exaggeration
        pts = np.column_stack((x_coords, y_coords, z_coords))
        results.append((idx, pts))

    return results


def streetGraph_to_pyvista(st_graph, elevation_data=None, z_exaggeration=1.0):
    """Convertit le graphe des rues en PolyData PyVista - entièrement parallélisé."""
    _, edges = ox.graph_to_gdfs(st_graph)
    total = len(edges)

    if total == 0:
        return pv.PolyData()

    print(f"  → Conversion parallèle de {total} rues sur {NUM_CORES} cœurs...")

    # Préparation des données
    street_data = []
    for idx, geom in enumerate(edges['geometry']):
        x_coords = np.array(geom.xy[0])
        y_coords = np.array(geom.xy[1])
        street_data.append((idx, x_coords, y_coords))

    # Division en lots
    batch_size = max(1, total // (NUM_CORES * 4))
    batches = []
    for i in range(0, total, batch_size):
        batch = street_data[i:i + batch_size]
        batches.append((batch, z_exaggeration))

    # Traitement parallèle
    all_results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = {executor.submit(_process_street_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)
            completed += len(batch_results)
            sys.stdout.write(f"\r  → Conversion rues: {completed * 100 // total}%")
            sys.stdout.flush()

    print(" ✓")

    # Tri par index
    all_results.sort(key=lambda x: x[0])

    # Construction rapide du mesh
    print("  → Fusion des rues...", end=" ", flush=True)

    all_vertices = []
    all_lines = []
    vertex_offset = 0

    for idx, pts in all_results:
        all_vertices.append(pts)
        n = len(pts)
        line = [n] + list(range(vertex_offset, vertex_offset + n))
        all_lines.extend(line)
        vertex_offset += n

    vertices = np.vstack(all_vertices)
    lines = np.array(all_lines)

    print("✓")
    return pv.PolyData(vertices, lines=lines)


# ============================================================
# 7. CONFIGURATION & VISUALISATION
# ============================================================

if __name__ == "__main__":
    # === CONFIGURATION ===
    location = "Montpellier, France"
    radius = 10000

    # Options d'affichage
    SHOW_TERRAIN = False           # Afficher le relief/topographie
    SHOW_BUILDINGS = True         # Afficher les bâtiments
    SHOW_STREETS = True           # Afficher les rues
    COLOR_BY_HEIGHT = True        # Colorer les bâtiments selon leur hauteur
    TERRAIN_RESOLUTION = 1000      # Résolution du terrain
    TERRAIN_EXAGGERATION = 3.0    # Exagération verticale du relief
    BUILDING_EXAGGERATION = 1.0   # Exagération des bâtiments
    STREET_COLOR = 'brown'        # Couleur des rues
    # =====================

    print("=" * 50)
    print(f"  Génération 3D: {location}")
    print(f"  Rayon: {radius}m")
    print("=" * 50)

    # Extraction des données OSM
    buildings, streets, center_point = extract_osm_data(
        location, radius,
        download_buildings=SHOW_BUILDINGS,
        download_streets=SHOW_STREETS
    )

    # Création du terrain
    elevation_data = None
    terrain = None
    z_exag = 1.0
    if SHOW_TERRAIN:
        terrain, elevation_data, _, z_exag = create_terrain_mesh(
            center_point, radius,
            resolution=TERRAIN_RESOLUTION,
            z_exaggeration=TERRAIN_EXAGGERATION
        )
    else:
        print("\n[2/3] Terrain: ignoré (SHOW_TERRAIN=False)")

    # Bâtiments
    print(f"\n[3/3] Génération des meshes 3D...")
    mesh = None
    if SHOW_BUILDINGS and buildings is not None:
        footprints, heights = generate_footprints_with_heights(buildings)
        print(f"  ✓ {len(footprints)} bâtiments ({min(heights):.0f}m - {max(heights):.0f}m)")
        mesh, _ = extrude_buildings(footprints, heights, elevation_data, z_exag, BUILDING_EXAGGERATION)

    # Rues
    street_mesh = None
    if SHOW_STREETS and streets is not None:
        street_mesh = streetGraph_to_pyvista(streets, elevation_data, z_exag)
        print(f"  ✓ {street_mesh.n_points} points de rue")

    # Visualisation
    print("\n✓ Ouverture de la visualisation...")
    p = pv.Plotter(border=False)

    if SHOW_TERRAIN and terrain is not None:
        p.add_mesh(terrain, scalars='elevation', cmap='coolwarm', show_edges=False,
                   opacity=0.8, scalar_bar_args={'title': 'Altitude (m)'})

    if SHOW_BUILDINGS and mesh is not None:
        if COLOR_BY_HEIGHT:
            p.add_mesh(mesh, scalars='height', cmap='coolwarm', show_edges=False,
                       scalar_bar_args={'title': 'Hauteur (m)'})
        else:
            p.add_mesh(mesh, color='gray', show_edges=False)

    if SHOW_STREETS and street_mesh is not None:
        p.add_mesh(street_mesh, color=STREET_COLOR, line_width=2)

    p.show(title='Geospatial Vector Extruction')