import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from scipy.sparse.linalg import splu

import time

from scipy.sparse import coo_matrix


from scipy.sparse import (
    lil_matrix,
    coo_matrix
)

REALIZATION = "ORI02"
ORIENTATION_SEED = 20260917


# ============================================================
# STEP 1: 3D Voronoi microstructure for 316L
# ============================================================

# -----------------------------
# 1. Model dimensions
# -----------------------------

Lx = 360.0       # micrometres
Ly = 360.0
Lz = 360.0

# Number of grains
N_grains = 216

# Voxel resolution
# 180 means ~2 um voxel size
Nx = 180
Ny = 180
Nz = 180

dx = Lx / Nx
dy = Ly / Ny
dz = Lz / Nz

print("Voxel size:")
print(f"dx = {dx:.2f} um")
print(f"dy = {dy:.2f} um")
print(f"dz = {dz:.2f} um")


# -----------------------------
# 2. Generate random grain seeds
# -----------------------------

np.random.seed(42)

seeds = np.column_stack([
    np.random.uniform(0, Lx, N_grains),
    np.random.uniform(0, Ly, N_grains),
    np.random.uniform(0, Lz, N_grains)
])

print("\nNumber of grains:", len(seeds))


# -----------------------------
# 3. Generate voxel coordinates
# -----------------------------

x = np.linspace(dx/2, Lx-dx/2, Nx)
y = np.linspace(dy/2, Ly-dy/2, Ny)
z = np.linspace(dz/2, Lz-dz/2, Nz)

X, Y, Z = np.meshgrid(
    x, y, z,
    indexing="ij"
)

points = np.column_stack([
    X.ravel(),
    Y.ravel(),
    Z.ravel()
])


# -----------------------------
# 4. Assign every voxel
#    to nearest grain seed
# -----------------------------

tree = cKDTree(seeds)

distances, grain_ids = tree.query(points)

grain_ids = grain_ids.reshape((Nx, Ny, Nz))

print("\nVoronoi microstructure generated.")
print("Array shape:", grain_ids.shape)


# # ============================================================
# # 5. Generate random crystallographic orientations
# # ============================================================

# # Random rotations
# rotations = R.random(N_grains)

# # Rotation matrices
# orientation_matrices = rotations.as_matrix()

# # Euler angles in degrees
# euler_angles = rotations.as_euler(
#     'ZXZ',
#     degrees=True
# )

# print("\nOrientation matrices shape:",
#       orientation_matrices.shape)

# print("Euler angle array shape:",
#       euler_angles.shape)





# ============================================================
# 5. Generate random crystallographic orientations
# ============================================================


# Use a dedicated seed for crystallographic orientations
rotations = R.random(
    N_grains,
    random_state=ORIENTATION_SEED
)

# Rotation matrices
orientation_matrices = rotations.as_matrix()

# Euler angles in degrees
euler_angles = rotations.as_euler(
    'ZXZ',
    degrees=True
)

print("\nOrientation realization:")
print("REALIZATION =", REALIZATION)
print("ORIENTATION_SEED =", ORIENTATION_SEED)

# ============================================================
# 6. Save microstructure
# ============================================================

np.savez(
    "316L_3D_Voronoi_microstructure.npz",

    grain_ids=grain_ids,

    grain_centers=seeds,

    orientation_matrices=orientation_matrices,

    euler_angles=euler_angles,

    dimensions=np.array([
        Lx, Ly, Lz
    ]),

    voxel_size=np.array([
        dx, dy, dz
    ])
)

print("\nMicrostructure saved as:")
print("316L_3D_Voronoi_microstructure.npz")


# ============================================================
# 7. VISUALIZE GRAIN IDs ON CENTRAL Z SECTION
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

CRITICAL_GRAIN = 115
CRITICAL_ELEMENT = 6333
CRITICAL_POINT = np.array([279.0, 297.0, 243.0])


# ------------------------------------------------------------
# Central z section
# ------------------------------------------------------------

mid_z = Nz // 2

section_xy = grain_ids[:, :, mid_z]


fig, ax = plt.subplots(figsize=(12, 11))

im = ax.imshow(
    section_xy.T,
    origin='lower',
    extent=[0, Lx, 0, Ly],
    interpolation='nearest'
)

ax.set_xlabel("x (μm)", fontsize=18)
ax.set_ylabel("y (μm)", fontsize=18)

ax.set_title(
    f"316L Voronoi Microstructure — z = {z[mid_z]:.1f} μm\n"
    "Grain IDs",
    fontsize=20
)

# ------------------------------------------------------------
# Label each grain visible in this section
# ------------------------------------------------------------

visible_grains = np.unique(section_xy)

for gid in visible_grains:

    # Pixel/voxel positions belonging to this grain
    positions = np.argwhere(section_xy == gid)

    if len(positions) == 0:
        continue

    # Mean position of voxels belonging to this grain
    ix_mean = positions[:, 0].mean()
    iy_mean = positions[:, 1].mean()

    x_pos = x[int(round(ix_mean))]
    y_pos = y[int(round(iy_mean))]

    if gid == CRITICAL_GRAIN:

        ax.text(
            x_pos,
            y_pos,
            str(gid),
            fontsize=16,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='circle,pad=0.25',
                facecolor='white',
                edgecolor='black',
                linewidth=2.5
            )
        )

    else:

        ax.text(
            x_pos,
            y_pos,
            str(gid),
            fontsize=7,
            ha='center',
            va='center'
        )


# ------------------------------------------------------------
# Mark critical grain if it intersects this section
# ------------------------------------------------------------

critical_in_section = np.any(
    section_xy == CRITICAL_GRAIN
)

if critical_in_section:

    positions = np.argwhere(
        section_xy == CRITICAL_GRAIN
    )

    ix = positions[:, 0].mean()
    iy = positions[:, 1].mean()

    ax.scatter(
        x[int(round(ix))],
        y[int(round(iy))],
        s=250,
        facecolors='none',
        edgecolors='black',
        linewidths=3
    )


# ------------------------------------------------------------
# Grid / axes
# ------------------------------------------------------------

ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)

ax.set_aspect('equal')

plt.tight_layout()
plt.show()


# ============================================================
# 8. X-Z SECTION THROUGH CRITICAL ELEMENT 6333
# ============================================================

# Element 6333 center from CPFEM result
xc, yc, zc = CRITICAL_POINT

# Find closest voxel index to the critical point
ix_crit = np.argmin(np.abs(x - xc))
iy_crit = np.argmin(np.abs(y - yc))

section_xz = grain_ids[:, iy_crit, :]


fig, ax = plt.subplots(figsize=(12, 11))

ax.imshow(
    section_xz.T,
    origin='lower',
    extent=[0, Lx, 0, Lz],
    interpolation='nearest'
)

ax.set_xlabel("x (μm)", fontsize=18)
ax.set_ylabel("z (μm)", fontsize=18)

ax.set_title(
    f"316L Voronoi Microstructure — y = {y[iy_crit]:.1f} μm\n"
    f"Through critical Element {CRITICAL_ELEMENT} / Grain {CRITICAL_GRAIN}",
    fontsize=20
)


# ------------------------------------------------------------
# Label visible grains
# ------------------------------------------------------------

visible_grains_xz = np.unique(section_xz)

for gid in visible_grains_xz:

    positions = np.argwhere(section_xz == gid)

    if len(positions) == 0:
        continue

    ix_mean = positions[:, 0].mean()
    iz_mean = positions[:, 1].mean()

    x_pos = x[int(round(ix_mean))]
    z_pos = z[int(round(iz_mean))]

    if gid == CRITICAL_GRAIN:

        ax.text(
            x_pos,
            z_pos,
            str(gid),
            fontsize=16,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='circle,pad=0.25',
                facecolor='white',
                edgecolor='black',
                linewidth=3
            )
        )

    else:

        ax.text(
            x_pos,
            z_pos,
            str(gid),
            fontsize=7,
            ha='center',
            va='center'
        )


# ------------------------------------------------------------
# Mark Element 6333 center
# ------------------------------------------------------------

ax.scatter(
    xc,
    zc,
    s=350,
    marker='x',
    linewidths=4
)

ax.text(
    xc + 8,
    zc + 8,
    f"Element {CRITICAL_ELEMENT}\nGrain {CRITICAL_GRAIN}",
    fontsize=14,
    fontweight='bold',
    ha='left',
    va='bottom',
    bbox=dict(
        boxstyle='round,pad=0.3',
        facecolor='white',
        edgecolor='black',
        linewidth=2
    )
)


ax.set_xlim(0, Lx)
ax.set_ylim(0, Lz)

ax.set_aspect('equal')

plt.tight_layout()
plt.show()


# ============================================================
# 9. Y-Z SECTION THROUGH CRITICAL ELEMENT
# ============================================================

section_yz = grain_ids[ix_crit, :, :]


fig, ax = plt.subplots(figsize=(12, 11))

ax.imshow(
    section_yz.T,
    origin='lower',
    extent=[0, Ly, 0, Lz],
    interpolation='nearest'
)

ax.set_xlabel("y (μm)", fontsize=18)
ax.set_ylabel("z (μm)", fontsize=18)

ax.set_title(
    f"316L Voronoi Microstructure — x = {x[ix_crit]:.1f} μm\n"
    f"Through critical Element {CRITICAL_ELEMENT} / Grain {CRITICAL_GRAIN}",
    fontsize=20
)


visible_grains_yz = np.unique(section_yz)

for gid in visible_grains_yz:

    positions = np.argwhere(section_yz == gid)

    if len(positions) == 0:
        continue

    iy_mean = positions[:, 0].mean()
    iz_mean = positions[:, 1].mean()

    y_pos = y[int(round(iy_mean))]
    z_pos = z[int(round(iz_mean))]

    if gid == CRITICAL_GRAIN:

        ax.text(
            y_pos,
            z_pos,
            str(gid),
            fontsize=16,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='circle,pad=0.25',
                facecolor='white',
                edgecolor='black',
                linewidth=3
            )
        )

    else:

        ax.text(
            y_pos,
            z_pos,
            str(gid),
            fontsize=7,
            ha='center',
            va='center'
        )


# Critical point

ax.scatter(
    yc,
    zc,
    s=350,
    marker='x',
    linewidths=4
)

ax.text(
    yc + 8,
    zc + 8,
    f"Element {CRITICAL_ELEMENT}\nGrain {CRITICAL_GRAIN}",
    fontsize=14,
    fontweight='bold',
    ha='left',
    va='bottom',
    bbox=dict(
        boxstyle='round,pad=0.3',
        facecolor='white',
        edgecolor='black',
        linewidth=2
    )
)

ax.set_xlim(0, Ly)
ax.set_ylim(0, Lz)

ax.set_aspect('equal')

plt.tight_layout()
plt.show()


# ============================================================
# 10. Print basic information
# ============================================================

print("\n==============================")
print("MICROSTRUCTURE SUMMARY")
print("==============================")

print(f"Domain: {Lx} × {Ly} × {Lz} μm³")
print(f"Number of grains: {N_grains}")
print(f"Voxel resolution: {Nx} × {Ny} × {Nz}")
print(f"Voxel size: {dx:.2f} μm")
print("Crystal structure: FCC")
print("Material: 316L stainless steel")
print("Slip systems: 12 FCC {111}<110>")
print("Orientation: random")

# ============================================================
# 11. Calculate grain volume and equivalent diameter
# ============================================================

voxel_volume = dx * dy * dz

grain_volumes = np.zeros(N_grains)

for i in range(N_grains):

    number_of_voxels = np.sum(grain_ids == i)

    grain_volumes[i] = (
        number_of_voxels * voxel_volume
    )

# Equivalent spherical diameter
grain_diameters = (
    6.0 * grain_volumes / np.pi
) ** (1.0 / 3.0)


# ============================================================
# 12. Grain-size statistics
# ============================================================

print("\n==============================")
print("GRAIN SIZE STATISTICS")
print("==============================")

print(
    f"Minimum diameter = "
    f"{grain_diameters.min():.2f} μm"
)

print(
    f"Maximum diameter = "
    f"{grain_diameters.max():.2f} μm"
)

print(
    f"Mean diameter = "
    f"{grain_diameters.mean():.2f} μm"
)

print(
    f"Median diameter = "
    f"{np.median(grain_diameters):.2f} μm"
)

print(
    f"Std. deviation = "
    f"{grain_diameters.std():.2f} μm"
)


# ============================================================
# 13. Grain-size distribution
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    grain_diameters,
    bins=20
)

plt.xlabel(
    "Equivalent grain diameter (μm)",
    fontsize=16
)

plt.ylabel(
    "Number of grains",
    fontsize=16
)

plt.title(
    "Grain Size Distribution",
    fontsize=16
)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D




