"""Круглый плоский диск в k3d.

Диск ориентирован так, что нормаль его плоскости направлена в начало
координат независимо от положения центра
"""

import numpy as np
import k3d


def make_disk(center, radius, n_segments=32, color=0x4488ff, **kwargs):
    """
    Параметры
    
    center : array_like (3,)
        Положение центра диска. Нормаль плоскости всегда направлена в
        начало координат.
    radius : float
        Радиус диска.
    n_segments : int
        Число сегментов окружности (≥ 3).
    color : int
        Цвет в формате 0xRRGGBB.
    **kwargs
        Дополнительные параметры 

    Возвращает

    k3d.objects.geometry.Mesh

    """
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("center должен быть точкой (x, y, z)")
    if radius <= 0:
        raise ValueError("radius должен быть положительным")
    if n_segments < 3:
        raise ValueError("n_segments должен быть >= 3")

    # Нормаль к плоскости диска направлена в начало координат
    norm = -center / np.linalg.norm(center)

    # Ортонормированный базис в плоскости диска
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(norm, ref)) > 0.99:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(norm, ref)
    u /= np.linalg.norm(u)
    v = np.cross(norm, u)

    # Вершины: центральная + точки на окружности.
    phi = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    rim = center + radius * (np.outer(np.cos(phi), u) + np.outer(np.sin(phi), v))
    vertices = np.vstack([center, rim]).astype(np.float32)

    # Триангуляция 
    ring = np.arange(1, n_segments + 1)
    next_ring = np.roll(ring, -1)
    indices = np.column_stack([np.zeros(n_segments, dtype=np.uint32), ring, next_ring])
    indices = indices.astype(np.uint32)

    return k3d.mesh(vertices, indices, color=color, **kwargs)