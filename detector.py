
from dataclasses import dataclass

import numpy as np

import k3d
from disk import make_disk
from visualize import build_sphere


@dataclass
class PMT:
    """ФЭУ — плоский круг на стенке сферы-детектора.

    position  — координаты центра круга на вспомогательной сфере
    radius  — радиус круга
    direction — единичный вектор нормали, направленный к началу координат
    a_pmt — вероятность выбивания фотоэлектрона 
    """

    position: np.ndarray
    radius: float
    direction: np.ndarray
    a_pmt: float


@dataclass
class Detector:
    """Сфера-детектор радиуса с центром в начале координат."""

    radius: float
    pmts: list[PMT]

    @classmethod
    def fibonacci_sphere(cls, n_pmts: int, radius: float,
                         pmt_radius: float,
                         a_pmt: float) -> "Detector":
        
        if n_pmts < 1:
            raise ValueError("n_pmts должен быть >= 1")
        if radius <= 0:
            raise ValueError("radius должен быть положительным")
        if pmt_radius <= 0:
            raise ValueError("pmt_radius должен быть положительным")
        if pmt_radius >= radius:
            raise ValueError("pmt_radius должен быть меньше radius")
        if not 0.0 <= a_pmt <= 1.0:
            raise ValueError("a_pmt должен быть в диапазоне [0, 1]")

        # Радиус вспомогательной сферы
        aux_radius = np.sqrt(radius ** 2 - pmt_radius ** 2)

        # Равномерное распределение N точек на единичной сфере
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        i = np.arange(n_pmts)
        y = 1.0 - 2.0 * (i + 0.5) / n_pmts        
        r = np.sqrt(1.0 - y * y)                  
        theta = golden_angle * i
        outward = np.stack([
            np.cos(theta) * r,
            y,
            np.sin(theta) * r,
        ], axis=-1)                               
        pmts = [
            PMT(position=u * aux_radius, radius=pmt_radius, direction=-u,
                a_pmt=a_pmt)
            for u in outward
        ]
        return cls(radius=radius, pmts=pmts)

    def draw(self, n_azi: int = 48, n_polar: int = 24) -> "k3d.plot":
        
        plot = k3d.plot()
        plot += build_sphere(radius=self.radius, n_azi=n_azi, n_polar=n_polar)
        for pmt in self.pmts:
            plot += make_disk(pmt.position, pmt.radius, side="double")
        return plot

    def find_pmt_batch(self, points) -> np.ndarray:
        """
        Определить, в области какого ФЭУ лежит каждая точка на стенке сферы.
        """
        u = np.array([pmt.position for pmt in self.pmts], dtype=float)
        u /= np.linalg.norm(u, axis=1, keepdims=True)          
        cos_max = np.sqrt(1 - (self.pmts[0].radius / self.radius) ** 2)
        p = np.asarray(points, dtype=float)
        p = p / np.linalg.norm(p, axis=-1, keepdims=True)      
        inside = (p @ u.T) >= cos_max   
        has = inside.any(axis=-1)
        return np.where(has, inside.argmax(axis=-1), -1)
