
import numpy as np
import k3d


def build_sphere(radius, opacity=0.35, n_azi=48, n_polar=24, color=0x3666AA):
    """Строит UV-сферу радиуса `radius` с центром в начале координат.
    """
   
    theta = np.linspace(0.0, np.pi, n_polar + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, n_azi, endpoint=False)
    theta, phi = np.meshgrid(theta, phi, indexing="ij")
    vertices = np.stack([
        radius * np.sin(theta) * np.cos(phi),
        radius * np.sin(theta) * np.sin(phi),
        radius * np.cos(theta),
    ], axis=-1).reshape(-1, 3).astype(np.float32)
    triangles = []
    for i in range(n_polar):
        for j in range(n_azi):
            a = i * n_azi + j
            b = i * n_azi + (j + 1) % n_azi
            c = (i + 1) * n_azi + j
            d = (i + 1) * n_azi + (j + 1) % n_azi
            triangles.append((a, c, d))
            triangles.append((a, d, b))
    indices = np.array(triangles, dtype=np.uint32).reshape(-1, 3)

    return k3d.mesh(
        vertices, indices,
        color=color,
        opacity=opacity,
        side="double",
        flat_shading=False,
    )
