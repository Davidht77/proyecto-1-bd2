"""Utilidades geograficas para el modulo espacial.

El R-Tree indexa pares (lon, lat) en grados y poda con rectangulos. El filtro
final y las busquedas k-NN usan distancia haversine, de modo que el resultado
sea geograficamente correcto aunque la poda sea aproximada.
"""

from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEGREE_LAT = 111.32


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Distancia de circulo maximo entre dos puntos, en kilometros."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def radius_bbox(lon: float, lat: float, radius_km: float) -> tuple[float, float, float, float]:
    """Rectangulo (min_lon, min_lat, max_lon, max_lat) que contiene el circulo.

    Es una cota conservadora: el filtro haversine posterior elimina las esquinas
    que quedan fuera del radio real.
    """
    radius_km = max(0.0, float(radius_km))
    dlat = radius_km / KM_PER_DEGREE_LAT
    cos_lat = max(0.01, abs(math.cos(math.radians(lat))))
    dlon = radius_km / (KM_PER_DEGREE_LAT * cos_lat)
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def point_in_radius(
    lon: float, lat: float, center_lon: float, center_lat: float, radius_km: float
) -> bool:
    return haversine(lon, lat, center_lon, center_lat) <= radius_km


def min_distance_to_bbox(lon: float, lat: float, mbr: tuple[float, ...]) -> float:
    """Cota inferior de la distancia de un punto a un MBR geografico.

    Se calcula la distancia al punto mas cercano del rectangulo. Para las
    ventanas de ciudad que usa el proyecto es una cota segura; el calculo exacto
    se hace al llegar a las hojas.
    """
    xmin, ymin, xmax, ymax = mbr[:4]
    closest_lon = min(max(lon, xmin), xmax)
    closest_lat = min(max(lat, ymin), ymax)
    return haversine(lon, lat, closest_lon, closest_lat)
