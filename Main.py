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

# ============================================================
# 2. Extraction des données OSM
# ============================================================
# Documentation des tags OSM: https://tagfinder.osm.ch

def extract_osm_data(location, radius=500, download_buildings=True, download_streets=True):
    """
    Extrait les empreintes de bâtiments et le réseau routier depuis OpenStreetMap.

    Paramètres:
        location: Nom du lieu (ex: "Montpellier, France")
        radius: Rayon de recherche en mètres
        download_buildings: Télécharger les bâtiments (True/False)
        download_streets: Télécharger les rues (True/False)

    Retourne:
        buildings: GeoDataFrame des bâtiments (ou None)
        streets: Graphe du réseau routier (ou None)
        center_point: Coordonnées du centre (lat, lon)
    """
    print(f"\n[1/3] Téléchargement des données OSM pour {location}...")

    # Récupération des coordonnées lat/lon du lieu
    center_point = ox.geocoder.geocode(location)
    print(f"  ✓ Coordonnées: {center_point[0]:.4f}, {center_point[1]:.4f}")

    buildings = None
    streets = None

    # Téléchargement des empreintes de bâtiments
    if download_buildings:
        print("  → Téléchargement des bâtiments...", end=" ", flush=True)
        buildings = ox.features_from_point(
            center_point,
            tags={'building': True},
            dist=radius
        )
        # Conversion en projection
        buildings = buildings.to_crs(epsg=2154)
        print(f"✓ {len(buildings)} bâtiments")
    else:
        print("  → Bâtiments: ignorés (SHOW_BUILDINGS=False)")

    # Téléchargement du réseau routier (tous types: voiture, piéton, vélo, etc.)
    if download_streets:
        print("  → Téléchargement des rues...", end=" ", flush=True)
        streets = ox.graph_from_point(
            center_point,
            dist=radius,
            network_type='all',  # Inclut toutes les rues
            simplify=False
        )
        # Conversion en projection 
        streets = ox.project_graph(streets, to_crs='epsg:2154')
        print(f"✓ {len(streets.edges)} segments")
    else:
        print("  → Rues: ignorées (SHOW_STREETS=False)")

    return buildings, streets, center_point


# ============================================================
# 2b. Données d'élévation du terrain (SRTM)
# ============================================================

def create_terrain_mesh(center_point, radius, resolution=50, z_exaggeration=1.0):
    """
    Crée un maillage 3D du terrain à partir des données d'élévation SRTM.

    Paramètres:
        center_point: Coordonnées du centre (lat, lon)
        radius: Rayon en mètres
        resolution: Nombre de points par axe (plus = plus détaillé mais plus lent)
        z_exaggeration: Facteur d'exagération verticale (1.0 = réel, 2.0 = 2x plus haut)

    Retourne:
        terrain: Maillage PyVista StructuredGrid
        elevation_data: Objet SRTM pour récupérer l'élévation
        transformer: Transformateur de coordonnées
        z_exaggeration: Facteur d'exagération utilisé
    """
    print(f"\n[2/3] Téléchargement du terrain ({resolution}x{resolution} points)...")

    # Récupération des données d'élévation SRTM
    elevation_data = srtm.get_data()

    # Calcul de la boîte englobante en lat/lon (approximation)
    lat, lon = center_point
    # 1 degré de latitude ≈ 111km, la longitude varie selon la latitude
    lat_offset = radius / 111000
    lon_offset = radius / (111000 * np.cos(np.radians(lat)))

    # Création d'une grille de points lat/lon
    lats = np.linspace(lat - lat_offset, lat + lat_offset, resolution)
    lons = np.linspace(lon - lon_offset, lon + lon_offset, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Récupération de l'élévation pour chaque point avec progression
    total_points = resolution * resolution
    elevations = np.zeros_like(lat_grid)
    for i in range(resolution):
        for j in range(resolution):
            elev = elevation_data.get_elevation(lat_grid[i, j], lon_grid[i, j])
            elevations[i, j] = elev if elev is not None else 0

        # Mise à jour de la progression
        progress = ((i + 1) * resolution) / total_points * 100
        sys.stdout.write(f"\r  → Élévation: {progress:.0f}%")
        sys.stdout.flush()

    print(" ✓")

    # Transformation des coordonnées vers EPSG:2154 
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    x_grid, y_grid = transformer.transform(lon_grid, lat_grid)

    # Application de l'exagération verticale
    elevations_exaggerated = elevations * z_exaggeration

    # Création du maillage PyVista StructuredGrid
    terrain = pv.StructuredGrid(x_grid, y_grid, elevations_exaggerated)
    # Stockage de l'élévation réelle pour la colormap
    terrain['elevation'] = elevations.flatten()

    print(f"  ✓ Altitude: {elevations.min():.0f}m - {elevations.max():.0f}m (exagération: x{z_exaggeration})")

    return terrain, elevation_data, transformer, z_exaggeration


def get_elevation_at_point(x, y, elevation_data, transformer):
    """
    Récupère l'élévation à un point projeté (EPSG:2154).

    Paramètres:
        x, y: Coordonnées
        elevation_data: Objet SRTM
        transformer: Transformateur de coordonnées

    Retourne:
        Élévation en mètres
    """
    # Transformation inverse vers lat/lon
    transformer_inverse = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    lon, lat = transformer_inverse.transform(x, y)
    elev = elevation_data.get_elevation(lat, lon)
    return elev if elev is not None else 0


# ============================================================
# 3. Génération des empreintes de bâtiments avec hauteurs
# ============================================================

def generate_footprints_with_heights(buildings):
    """
    Convertit les géométries GeoDataFrame en polygones Shapely avec hauteurs réelles.
    Utilise les attributs OSM 'height' ou 'building:levels' quand disponibles.

    Paramètres:
        buildings: GeoDataFrame des bâtiments

    Retourne:
        footprints: Liste de polygones Shapely
        heights: Liste des hauteurs correspondantes
    """
    footprints = []
    heights = []

    # Hauteur par défaut par étage (mètres)
    FLOOR_HEIGHT = 3.0
    # Hauteur par défaut si aucune donnée disponible
    DEFAULT_HEIGHT = 10.0

    for idx, row in buildings.iterrows():
        geom = row.geometry

        # Tentative de récupération de la hauteur réelle depuis OSM
        height = None

        # Vérification de l'attribut 'height' (en mètres)
        if 'height' in buildings.columns and pd.notna(row.get('height')):
            try:
                h = row['height']
                # Gestion des chaînes comme "15 m" ou "15"
                if isinstance(h, str):
                    h = float(h.replace('m', '').replace(' ', ''))
                height = float(h)
            except (ValueError, TypeError):
                pass

        # Vérification de l'attribut 'building:levels' (nombre d'étages)
        if height is None and 'building:levels' in buildings.columns and pd.notna(row.get('building:levels')):
            try:
                levels = float(row['building:levels'])
                height = levels * FLOOR_HEIGHT
            except (ValueError, TypeError):
                pass

        # Utilisation de la hauteur par défaut si pas de données
        if height is None:
            height = DEFAULT_HEIGHT

        # Limitation de la hauteur à des valeurs raisonnables
        height = max(3.0, min(height, 300.0))

        # Gestion des MultiPolygon (plusieurs polygones pour un bâtiment)
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                footprints.append(poly)
                heights.append(height)
        elif isinstance(geom, Polygon):
            footprints.append(geom)
            heights.append(height)
        # Les points et autres types de géométrie sont ignorés

    return footprints, heights


# ============================================================
# 4. Génération des bâtiments 3D
# ============================================================

def create_watertight_building(coords, height, ground_elevation=0):
    """
    Crée un maillage 3D étanche (watertight) d'un bâtiment avec base, murs et toit.

    Paramètres:
        coords: Coordonnées du contour du bâtiment
        height: Hauteur du bâtiment
        ground_elevation: Élévation du terrain à cet emplacement

    Retourne:
        points: Tableau numpy des vertices
        faces: Liste des faces (format PyVista)
    """
    # Suppression du dernier point dupliqué des coordonnées
    coords = coords[:-1]
    n_points = len(coords)

    # Création des points pour la base et le sommet (ajout de l'élévation du sol)
    base_points = np.column_stack((coords, np.full(len(coords), ground_elevation)))
    top_points = np.column_stack((coords, np.full(len(coords), ground_elevation + height)))

    # Combinaison de tous les points
    points = np.vstack((base_points, top_points))

    # Création des faces
    faces = []

    # Ajout de la base (triangle fan)
    base_face = [n_points] + list(range(n_points))
    faces.extend(base_face)

    # Ajout du toit (triangle fan)
    roof_indices = list(range(n_points, n_points * 2))
    roof_face = [n_points] + roof_indices
    faces.extend(roof_face)

    # Ajout des murs (quadrilatères)
    for i in range(n_points):
        next_i = (i + 1) % n_points
        wall_face = [4,  # quad (4 sommets)
                     i, next_i,
                     n_points + next_i, n_points + i]  # points du sommet
        faces.extend(wall_face)

    return points, faces


def generate_random_color():
    """
    Génère une couleur RGB aléatoire.

    Retourne:
        Liste [R, G, B] avec valeurs entre 0 et 1
    """
    return [random.random() for _ in range(3)]


def extrude_buildings(footprints, heights, elevation_data=None, z_exaggeration=1.0, building_exaggeration=1.0):
    """
    Extrude les empreintes 2D en bâtiments 3D avec PyVista.
    Colore les bâtiments selon leur hauteur: bleu (bas) à rouge (haut).

    Paramètres:
        footprints: Liste des empreintes de bâtiments (polygones)
        heights: Liste des hauteurs
        elevation_data: Données d'élévation SRTM (optionnel)
        z_exaggeration: Exagération verticale du terrain
        building_exaggeration: Exagération de la hauteur des bâtiments

    Retourne:
        city_mesh: Maillage combiné de tous les bâtiments
        instances_building: Liste des maillages individuels
    """
    # Création d'un maillage PyVista vide pour la ville finale
    city_mesh = pv.PolyData()
    instances_building = []

    # Transformateur inverse pour obtenir lat/lon depuis les coordonnées projetées
    transformer_inverse = None
    if elevation_data is not None:
        transformer_inverse = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    total = len(footprints)
    for i, (footprint, height) in enumerate(zip(footprints, heights)):
        # Récupération des coordonnées de l'empreinte
        coords = np.array(footprint.exterior.coords)

        # Récupération de l'élévation du sol au centroïde du bâtiment
        ground_elevation = 0
        if elevation_data is not None and transformer_inverse is not None:
            centroid = footprint.centroid
            lon, lat = transformer_inverse.transform(centroid.x, centroid.y)
            elev = elevation_data.get_elevation(lat, lon)
            ground_elevation = (elev if elev is not None else 0) * z_exaggeration

        # Création de la géométrie 3D étanche (hauteur avec exagération)
        points, faces = create_watertight_building(coords, height * building_exaggeration, ground_elevation)

        # Création du maillage du bâtiment
        building = pv.PolyData(points, np.array(faces))

        # Attribution de la valeur de hauteur pour la colormap
        building['height'] = np.full(building.n_points, height)

        instances_building.append(building)

        # Ajout au maillage de la ville
        if city_mesh.n_points == 0:
            city_mesh = building
        else:
            city_mesh = city_mesh.merge(building, merge_points=False)

        # Progression
        progress = (i + 1) / total * 100
        sys.stdout.write(f"\r  → Extrusion bâtiments: {progress:.0f}%")
        sys.stdout.flush()

    print(" ✓")
    return city_mesh, instances_building


# ============================================================
# 5. Fonction d'export
# ============================================================

def save_to_obj(mesh, output_path):
    """
    Sauvegarde le modèle 3D au format OBJ.

    Paramètres:
        mesh: Maillage PyVista à exporter
        output_path: Chemin du fichier de sortie
    """
    # Création du répertoire de sortie s'il n'existe pas
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde du maillage en OBJ
    print(f"Sauvegarde du modèle vers {output_path}")
    mesh.save(output_path)
    print("Export terminé")


# ============================================================
# 6. Conversion du réseau routier
# ============================================================

def streetGraph_to_pyvista(st_graph, elevation_data=None, z_exaggeration=1.0):
    """
    Convertit le graphe OSMnx des rues en PolyData PyVista avec lignes.
    Si elevation_data est fourni, les rues suivent l'élévation du terrain.

    Paramètres:
        st_graph: Graphe NetworkX du réseau routier
        elevation_data: Données d'élévation SRTM (optionnel)
        z_exaggeration: Exagération verticale

    Retourne:
        Maillage PyVista PolyData avec lignes
    """
    # Conversion du graphe en DataFrame
    _, edges = ox.graph_to_gdfs(st_graph)

    # Transformateur inverse pour obtenir lat/lon depuis les coordonnées projetées
    transformer_inverse = None
    if elevation_data is not None:
        transformer_inverse = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    def get_elevation_for_coords(x_arr, y_arr):
        """Récupère l'élévation pour un tableau de coordonnées."""
        if elevation_data is None or transformer_inverse is None:
            return np.zeros(len(x_arr))
        elevations = []
        for x, y in zip(x_arr, y_arr):
            lon, lat = transformer_inverse.transform(x, y)
            elev = elevation_data.get_elevation(lat, lon)
            elevations.append((elev if elev is not None else 0) * z_exaggeration)
        return np.array(elevations)

    # Conversion des arêtes en lignes PyVista
    pts_list = []
    total = len(edges)
    for idx, geom in enumerate(edges['geometry']):
        x_coords = np.array(geom.xy[0])
        y_coords = np.array(geom.xy[1])
        # Ajout d'un décalage au-dessus du sol
        z_coords = get_elevation_for_coords(x_coords, y_coords) + 0.5 * z_exaggeration
        pts_list.append(np.column_stack((x_coords, y_coords, z_coords)))

        # Progression
        progress = (idx + 1) / total * 100
        sys.stdout.write(f"\r  → Conversion rues: {progress:.0f}%")
        sys.stdout.flush()

    print(" ✓")

    # Concaténation de tous les vertices
    vertices = np.concatenate(pts_list)

    # Construction des indices de lignes pour PyVista
    lines = []
    j = 0
    for i in range(len(pts_list)):
        pts = pts_list[i]
        vertex_length = len(pts)
        vertex_start = j
        vertex_end = j + vertex_length - 1
        vertex_arr = [vertex_length] + list(range(vertex_start, vertex_end + 1))
        lines.append(vertex_arr)
        j += vertex_length

    return pv.PolyData(vertices, lines=np.hstack(lines))


# ============================================================
# 7. CONFIGURATION & VISUALISATION
# ============================================================

# === CONFIGURATION ===
location = "Chamonix, France"
radius = 2000

# Options d'affichage
SHOW_TERRAIN = True           # Afficher le relief/topographie
SHOW_BUILDINGS = True         # Afficher les bâtiments
SHOW_STREETS = False           # Afficher les rues
COLOR_BY_HEIGHT = True        # Colorer les bâtiments selon leur hauteur (sinon gris)
TERRAIN_RESOLUTION = 200       # Résolution du terrain (plus = plus détaillé mais plus lent)
TERRAIN_EXAGGERATION = 2.0    # Exagération verticale du relief (1.0 = réel)
BUILDING_EXAGGERATION = 1.0   # Exagération de la hauteur des bâtiments (1.0 = réel)
STREET_COLOR = 'white'        # Couleur des rues
# =====================

print("=" * 50)
print(f"  Génération 3D: {location}")
print(f"  Rayon: {radius}m")
print("=" * 50)

# Extraction des données OSM (ignore bâtiments/rues si non nécessaires)
buildings, streets, center_point = extract_osm_data(
    location, radius,
    download_buildings=SHOW_BUILDINGS,
    download_streets=SHOW_STREETS
)

# Création du maillage de terrain (si activé)
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

# Traitement des bâtiments (si activé)
print(f"\n[3/3] Génération des meshes 3D...")
mesh = None
if SHOW_BUILDINGS and buildings is not None:
    footprints, heights = generate_footprints_with_heights(buildings)
    print(f"  ✓ {len(footprints)} bâtiments ({min(heights):.0f}m - {max(heights):.0f}m)")
    mesh, bn_instances = extrude_buildings(footprints, heights, elevation_data, z_exag, BUILDING_EXAGGERATION)

# Conversion du réseau routier (si activé)
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

p.show(title='(c) Florent Labrousse-Lhuissier')
