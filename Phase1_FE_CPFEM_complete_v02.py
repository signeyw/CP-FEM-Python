import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

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


# ============================================================
# 5. Generate random crystallographic orientations
# ============================================================

# Random rotations
rotations = R.random(N_grains)

# Rotation matrices
orientation_matrices = rotations.as_matrix()

# Euler angles in degrees
euler_angles = rotations.as_euler(
    'ZXZ',
    degrees=True
)

print("\nOrientation matrices shape:",
      orientation_matrices.shape)

print("Euler angle array shape:",
      euler_angles.shape)


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
# 7. Plot central cross-section
# ============================================================

mid_z = Nz // 2

plt.figure(figsize=(8, 7))

plt.imshow(
    grain_ids[:, :, mid_z].T,
    origin='lower',
    extent=[0, Lx, 0, Ly],
    interpolation='nearest'
)

plt.xlabel("x (μm)", fontsize=16)
plt.ylabel("y (μm)", fontsize=16)

plt.title(
    "316L 3D Voronoi Microstructure — Central Section",
    fontsize=16
)

plt.colorbar(label="Grain ID")

plt.tight_layout()
plt.show()


# ============================================================
# 8. Plot central x-z section
# ============================================================

mid_y = Ny // 2

plt.figure(figsize=(8, 7))

plt.imshow(
    grain_ids[:, mid_y, :].T,
    origin='lower',
    extent=[0, Lx, 0, Lz],
    interpolation='nearest'
)

plt.xlabel("x (μm)", fontsize=16)
plt.ylabel("z (μm)", fontsize=16)

plt.title(
    "316L 3D Voronoi Microstructure — x-z Section",
    fontsize=16
)

plt.colorbar(label="Grain ID")

plt.tight_layout()
plt.show()


# ============================================================
# 9. Plot grain seed locations
# ============================================================

fig = plt.figure(figsize=(9, 8))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    seeds[:, 0],
    seeds[:, 1],
    seeds[:, 2],
    s=15
)

ax.set_xlabel("x (μm)")
ax.set_ylabel("y (μm)")
ax.set_zlabel("z (μm)")

ax.set_title(
    "3D Grain Seed Distribution"
)

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

# ============================================================
# MODULE 2
# 3D HEXAHEDRAL FE MESH
# ============================================================

# ------------------------------------------------------------
# 1. Load the microstructure from Module 1
# ------------------------------------------------------------

data = np.load(
    "316L_3D_Voronoi_microstructure.npz"
)

grain_ids = data["grain_ids"]
grain_centers = data["grain_centers"]
orientation_matrices = data["orientation_matrices"]
euler_angles = data["euler_angles"]

Lx, Ly, Lz = data["dimensions"]

print("Microstructure loaded.")

print("Domain:")
print(Lx, Ly, Lz, "μm")

print("Number of grains:",
      len(grain_centers))


# ============================================================
# 2. Define FE mesh resolution
# ============================================================

# ------------------------------------------------------------
# TEST MODEL
# ------------------------------------------------------------

Nx_FE = 20
Ny_FE = 20
Nz_FE = 20

dx_FE = Lx / Nx_FE
dy_FE = Ly / Ny_FE
dz_FE = Lz / Nz_FE

print("\nFE mesh:")
print(
    Nx_FE,
    "×",
    Ny_FE,
    "×",
    Nz_FE
)

print(
    "Element size:",
    dx_FE,
    "μm"
)

print(
    "Number of elements:",
    Nx_FE * Ny_FE * Nz_FE
)


# ============================================================
# 3. Generate FE nodes
# ============================================================

x_nodes = np.linspace(
    0,
    Lx,
    Nx_FE + 1
)

y_nodes = np.linspace(
    0,
    Ly,
    Ny_FE + 1
)

z_nodes = np.linspace(
    0,
    Lz,
    Nz_FE + 1
)


# Create node grid
Xn, Yn, Zn = np.meshgrid(
    x_nodes,
    y_nodes,
    z_nodes,
    indexing="ij"
)


# Node coordinates
nodes = np.column_stack([
    Xn.ravel(),
    Yn.ravel(),
    Zn.ravel()
])


N_nodes = len(nodes)

print(
    "\nNumber of nodes:",
    N_nodes
)


# ============================================================
# 4. Generate hexahedral element connectivity
# ============================================================

def node_id(i, j, k):

    return (
        i * (Ny_FE + 1) * (Nz_FE + 1)
        +
        j * (Nz_FE + 1)
        +
        k
    )


elements = []

for i in range(Nx_FE):

    for j in range(Ny_FE):

        for k in range(Nz_FE):

            n1 = node_id(i,     j,     k)
            n2 = node_id(i + 1, j,     k)
            n3 = node_id(i + 1, j + 1, k)
            n4 = node_id(i,     j + 1, k)

            n5 = node_id(i,     j,     k + 1)
            n6 = node_id(i + 1, j,     k + 1)
            n7 = node_id(i + 1, j + 1, k + 1)
            n8 = node_id(i,     j + 1, k + 1)

            elements.append([
                n1, n2, n3, n4,
                n5, n6, n7, n8
            ])


elements = np.asarray(
    elements,
    dtype=np.int32
)

N_elements = len(elements)

print(
    "Number of elements:",
    N_elements
)


# ============================================================
# 5. Calculate element centers
# ============================================================

element_centers = np.zeros(
    (N_elements, 3)
)

for e in range(N_elements):

    element_nodes = elements[e]

    element_centers[e] = np.mean(
        nodes[element_nodes],
        axis=0
    )


# ============================================================
# 6. Assign each FE element to a grain
# ============================================================

# ------------------------------------------------------------
# Use the original 2 μm microstructure.
# Each FE element center is mapped to the nearest
# microstructure voxel.
# ------------------------------------------------------------

voxel_size = data["voxel_size"]

dx_micro = voxel_size[0]
dy_micro = voxel_size[1]
dz_micro = voxel_size[2]


def position_to_voxel(point):

    x, y, z = point

    ix = int(x / dx_micro)
    iy = int(y / dy_micro)
    iz = int(z / dz_micro)

    # Prevent boundary overflow
    ix = min(max(ix, 0), grain_ids.shape[0] - 1)
    iy = min(max(iy, 0), grain_ids.shape[1] - 1)
    iz = min(max(iz, 0), grain_ids.shape[2] - 1)

    return ix, iy, iz


element_grain = np.zeros(
    N_elements,
    dtype=np.int32
)


for e in range(N_elements):

    ix, iy, iz = position_to_voxel(
        element_centers[e]
    )

    element_grain[e] = grain_ids[
        ix, iy, iz
    ]


print(
    "\nGrain assignment completed."
)


# ============================================================
# 7. Assign crystal orientation to each element
# ============================================================

element_orientation = (
    orientation_matrices[element_grain]
)


element_euler = (
    euler_angles[element_grain]
)


print(
    "Orientation assignment completed."
)

print(
    "Element orientation array:",
    element_orientation.shape
)


# ============================================================
# 8. Check how many grains are represented
# ============================================================

unique_grains = np.unique(
    element_grain
)

print(
    "\nNumber of grains represented in FE mesh:",
    len(unique_grains)
)


# ============================================================
# 9. Check element-grain distribution
# ============================================================

grain_element_count = np.bincount(
    element_grain,
    minlength=len(grain_centers)
)


print("\nFirst 10 grain element counts:")

print(
    grain_element_count[:10]
)


# ============================================================
# 10. Save FE mesh
# ============================================================

np.savez_compressed(
    "316L_FE_mesh_20cube.npz",

    nodes=nodes,

    elements=elements,

    element_centers=element_centers,

    element_grain=element_grain,

    element_orientation=element_orientation,

    element_euler=element_euler,

    grain_centers=grain_centers,

    grain_orientation=orientation_matrices,

    grain_euler=euler_angles,

    dimensions=np.array([
        Lx,
        Ly,
        Lz
    ])
)


print(
    "\nFE mesh saved as:"
)

print(
    "316L_FE_mesh_20cube.npz"
)


# ============================================================
# 11. Visualize FE element centers
# ============================================================

fig = plt.figure(
    figsize=(9, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

scatter = ax.scatter(
    element_centers[:, 0],
    element_centers[:, 1],
    element_centers[:, 2],
    c=element_grain,
    s=8
)

ax.set_xlabel(
    "x (μm)",
    fontsize=14
)

ax.set_ylabel(
    "y (μm)",
    fontsize=14
)

ax.set_zlabel(
    "z (μm)",
    fontsize=14
)

ax.set_title(
    "3D FE Element–Grain Assignment",
    fontsize=16
)

plt.tight_layout()

plt.show()


# ============================================================
# 12. Central cross-section
# ============================================================

# Select elements close to the middle of the cube

z_middle = Lz / 2

tolerance = dz_FE / 2

section = (
    np.abs(
        element_centers[:, 2] - z_middle
    )
    < tolerance
)


section_centers = (
    element_centers[section]
)

section_grains = (
    element_grain[section]
)


plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    section_centers[:, 0],
    section_centers[:, 1],
    c=section_grains,
    s=35,
    marker="s"
)

plt.xlabel(
    "x (μm)",
    fontsize=16
)

plt.ylabel(
    "y (μm)",
    fontsize=16
)

plt.title(
    "FE Mesh Grain Assignment — Central Section",
    fontsize=16
)

plt.axis("equal")

plt.tight_layout()

plt.show()


# ============================================================
# 13. Final summary
# ============================================================

print("\n")
print("==============================")
print("FE MESH SUMMARY")
print("==============================")

print(
    f"Domain: "
    f"{Lx:.1f} × {Ly:.1f} × {Lz:.1f} μm³"
)

print(
    f"Elements: "
    f"{Nx_FE} × {Ny_FE} × {Nz_FE}"
)

print(
    f"Total elements: "
    f"{N_elements}"
)

print(
    f"Total nodes: "
    f"{N_nodes}"
)

print(
    f"Element size: "
    f"{dx_FE:.2f} μm"
)

print(
    f"Represented grains: "
    f"{len(unique_grains)}"
)

print(
    "Element data:"
)

print(
    "  element_grain"
)

print(
    "  element_orientation"
)

print(
    "  element_euler"
)

print(
    "==============================")

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# MODULE 3
# 3D HEXAHEDRAL ELASTIC FINITE ELEMENT SOLVER
# ============================================================


# ------------------------------------------------------------
# 1. Load FE mesh
# ------------------------------------------------------------

data = np.load(
    "316L_FE_mesh_20cube.npz"
)

nodes = data["nodes"]
elements = data["elements"]

N_nodes = len(nodes)
N_elements = len(elements)

print("FE mesh loaded.")

print(
    "Nodes:",
    N_nodes
)

print(
    "Elements:",
    N_elements
)


# ============================================================
# 2. Material properties
# ============================================================

E = 193000.0       # MPa
nu = 0.29

print("\nMaterial:")
print("316L elastic modulus =", E, "MPa")
print("Poisson ratio =", nu)


# ============================================================
# 3. Elastic constitutive matrix
# ============================================================

# Engineering strain convention:
#
# eps =
# [eps_xx,
#  eps_yy,
#  eps_zz,
#  gamma_xy,
#  gamma_yz,
#  gamma_zx]
#
# stress =
# [sigma_xx,
#  sigma_yy,
#  sigma_zz,
#  tau_xy,
#  tau_yz,
#  tau_zx]


C = np.zeros((6, 6))

lam = (
    E * nu
    /
    ((1 + nu) * (1 - 2 * nu))
)

mu = (
    E
    /
    (2 * (1 + nu))
)


C[0, 0] = lam + 2 * mu
C[1, 1] = lam + 2 * mu
C[2, 2] = lam + 2 * mu

C[0, 1] = lam
C[0, 2] = lam
C[1, 0] = lam
C[1, 2] = lam
C[2, 0] = lam
C[2, 1] = lam

C[3, 3] = mu
C[4, 4] = mu
C[5, 5] = mu


print("\nElastic constitutive matrix:")
print(C)


# ============================================================
# 4. Hexahedral shape functions
# ============================================================

def shape_functions_hex8(xi, eta, zeta):

    N = np.zeros(8)

    N[0] = (
        1/8
        * (1-xi)
        * (1-eta)
        * (1-zeta)
    )

    N[1] = (
        1/8
        * (1+xi)
        * (1-eta)
        * (1-zeta)
    )

    N[2] = (
        1/8
        * (1+xi)
        * (1+eta)
        * (1-zeta)
    )

    N[3] = (
        1/8
        * (1-xi)
        * (1+eta)
        * (1-zeta)
    )

    N[4] = (
        1/8
        * (1-xi)
        * (1-eta)
        * (1+zeta)
    )

    N[5] = (
        1/8
        * (1+xi)
        * (1-eta)
        * (1+zeta)
    )

    N[6] = (
        1/8
        * (1+xi)
        * (1+eta)
        * (1+zeta)
    )

    N[7] = (
        1/8
        * (1-xi)
        * (1+eta)
        * (1+zeta)
    )

    return N


# ============================================================
# 5. Derivatives of shape functions
# ============================================================

def shape_function_derivatives(
    xi,
    eta,
    zeta
):

    dN = np.zeros((8, 3))

    # dN/dxi
    dN[0, 0] = -1/8*(1-eta)*(1-zeta)
    dN[1, 0] =  1/8*(1-eta)*(1-zeta)
    dN[2, 0] =  1/8*(1+eta)*(1-zeta)
    dN[3, 0] = -1/8*(1+eta)*(1-zeta)

    dN[4, 0] = -1/8*(1-eta)*(1+zeta)
    dN[5, 0] =  1/8*(1-eta)*(1+zeta)
    dN[6, 0] =  1/8*(1+eta)*(1+zeta)
    dN[7, 0] = -1/8*(1+eta)*(1+zeta)

    # dN/deta
    dN[0, 1] = -1/8*(1-xi)*(1-zeta)
    dN[1, 1] = -1/8*(1+xi)*(1-zeta)
    dN[2, 1] =  1/8*(1+xi)*(1-zeta)
    dN[3, 1] =  1/8*(1-xi)*(1-zeta)

    dN[4, 1] = -1/8*(1-xi)*(1+zeta)
    dN[5, 1] = -1/8*(1+xi)*(1+zeta)
    dN[6, 1] =  1/8*(1+xi)*(1+zeta)
    dN[7, 1] =  1/8*(1-xi)*(1+zeta)

    # dN/dzeta
    dN[0, 2] = -1/8*(1-xi)*(1-eta)
    dN[1, 2] = -1/8*(1+xi)*(1-eta)
    dN[2, 2] = -1/8*(1+xi)*(1+eta)
    dN[3, 2] = -1/8*(1-xi)*(1+eta)

    dN[4, 2] =  1/8*(1-xi)*(1-eta)
    dN[5, 2] =  1/8*(1+xi)*(1-eta)
    dN[6, 2] =  1/8*(1+xi)*(1+eta)
    dN[7, 2] =  1/8*(1-xi)*(1+eta)

    return dN


# ============================================================
# 6. Gauss integration
# ============================================================

g = 1.0 / np.sqrt(3)

gauss_points = [

    (-g, -g, -g),
    ( g, -g, -g),
    ( g,  g, -g),
    (-g,  g, -g),

    (-g, -g,  g),
    ( g, -g,  g),
    ( g,  g,  g),
    (-g,  g,  g)
]

gauss_weights = np.ones(8)


# ============================================================
# 7. B matrix
# ============================================================

def calculate_B_matrix(
    element_coordinates,
    xi,
    eta,
    zeta
):

    dN_dxi = shape_function_derivatives(
        xi,
        eta,
        zeta
    )

    # Jacobian
    J = (
        element_coordinates.T
        @ dN_dxi
    )

    detJ = np.linalg.det(J)

    if detJ <= 0:

        raise ValueError(
            "Negative or zero Jacobian."
        )

    # Convert derivatives to physical coordinates
    dN_dx = (
        dN_dxi
        @ np.linalg.inv(J)
    )

    B = np.zeros((6, 24))

    for i in range(8):

        ix = 3 * i

        dNx = dN_dx[i, 0]
        dNy = dN_dx[i, 1]
        dNz = dN_dx[i, 2]

        B[0, ix]     = dNx
        B[1, ix + 1] = dNy
        B[2, ix + 2] = dNz

        B[3, ix]     = dNy
        B[3, ix + 1] = dNx

        B[4, ix + 1] = dNz
        B[4, ix + 2] = dNy

        B[5, ix]     = dNz
        B[5, ix + 2] = dNx

    return B, detJ


# ============================================================
# 8. Element stiffness matrix
# ============================================================

def element_stiffness(
    element_coordinates
):

    Ke = np.zeros((24, 24))

    for gp, weight in zip(
        gauss_points,
        gauss_weights
    ):

        xi, eta, zeta = gp

        B, detJ = calculate_B_matrix(
            element_coordinates,
            xi,
            eta,
            zeta
        )

        Ke += (
            B.T
            @ C
            @ B
            * detJ
            * weight
        )

    return Ke


# ============================================================
# 9. Test one element
# ============================================================

test_element = elements[0]

test_coordinates = (
    nodes[test_element]
)

Ke_test = element_stiffness(
    test_coordinates
)

print(
    "\nTest element stiffness matrix:"
)

print(
    "Shape:",
    Ke_test.shape
)

print(
    "Symmetry error:",
    np.max(
        np.abs(
            Ke_test - Ke_test.T
        )
    )
)








# ============================================================
# MODULE 3B - CORRECTED
# GLOBAL 3D ELASTIC FE SOLVER
#
# Direct displacement boundary-condition method
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# 1. Load FE mesh
# ============================================================

data = np.load(
    "316L_FE_mesh_20cube.npz"
)

nodes = data["nodes"]
elements = data["elements"]

N_nodes = len(nodes)
N_elements = len(elements)

print("FE mesh loaded.")

print("Nodes:", N_nodes)
print("Elements:", N_elements)


# ============================================================
# 2. Convert geometry μm -> mm
# ============================================================

nodes_mm = nodes / 1000.0

Lx = nodes_mm[:, 0].max()
Ly = nodes_mm[:, 1].max()
Lz = nodes_mm[:, 2].max()

print("\nModel dimensions:")
print(
    f"{Lx:.6f} × "
    f"{Ly:.6f} × "
    f"{Lz:.6f} mm"
)


# ============================================================
# 3. Material properties
# ============================================================

E = 193000.0      # MPa
nu = 0.29

print("\nMaterial:")
print("E  =", E, "MPa")
print("ν  =", nu)


# ============================================================
# 4. Rebuild elastic constitutive matrix
# ============================================================

C = np.zeros((6, 6))

lam = (
    E * nu
    /
    ((1 + nu) * (1 - 2 * nu))
)

mu = (
    E
    /
    (2 * (1 + nu))
)

C[0, 0] = lam + 2*mu
C[1, 1] = lam + 2*mu
C[2, 2] = lam + 2*mu

C[0, 1] = lam
C[0, 2] = lam
C[1, 0] = lam
C[1, 2] = lam
C[2, 0] = lam
C[2, 1] = lam

C[3, 3] = mu
C[4, 4] = mu
C[5, 5] = mu


# ============================================================
# 5. Number of DOFs
# ============================================================

DOF_per_node = 3

N_DOF = (
    N_nodes
    *
    DOF_per_node
)

print(
    "\nTotal DOFs:",
    N_DOF
)


# ============================================================
# 6. Assemble global stiffness matrix
# ============================================================

K = lil_matrix(
    (N_DOF, N_DOF)
)

print(
    "\nAssembling global stiffness matrix..."
)

for e in range(N_elements):

    element_nodes = elements[e]

    element_coordinates = (
        nodes_mm[element_nodes]
    )

    Ke = element_stiffness(
        element_coordinates
    )

    # Global DOFs
    dofs = []

    for n in element_nodes:

        dofs.extend([
            3*n,
            3*n + 1,
            3*n + 2
        ])

    # Assembly
    for a in range(24):

        A = dofs[a]

        for b in range(24):

            B = dofs[b]

            K[A, B] += Ke[a, b]


K = K.tocsr()

print(
    "Global assembly completed."
)


# ============================================================
# 7. Identify bottom and top surfaces
# ============================================================

tol = 1e-12

bottom_nodes = np.where(
    np.abs(
        nodes_mm[:, 2]
    ) < tol
)[0]

top_nodes = np.where(
    np.abs(
        nodes_mm[:, 2] - Lz
    ) < tol
)[0]


print(
    "\nBottom nodes:",
    len(bottom_nodes)
)

print(
    "Top nodes:",
    len(top_nodes)
)


# ============================================================
# 8. Applied displacement
# ============================================================

strain_target = 0.0036

u_top = (
    strain_target
    * Lz
)

print(
    "\nTarget strain:",
    strain_target
)

print(
    "Applied top displacement:",
    u_top,
    "mm"
)


# ============================================================
# 9. Construct prescribed displacement vector
# ============================================================

u_prescribed = np.zeros(
    N_DOF
)

prescribed_dofs = []


# ------------------------------------------------------------
# Bottom surface
# ------------------------------------------------------------

for n in bottom_nodes:

    # Fix z displacement
    dof_z = 3*n + 2

    prescribed_dofs.append(
        dof_z
    )


# ------------------------------------------------------------
# Remove rigid-body x translation
# ------------------------------------------------------------

# Select one bottom node
node_x_fix = bottom_nodes[0]

prescribed_dofs.append(
    3*node_x_fix
)


# ------------------------------------------------------------
# Remove rigid-body y translation
# ------------------------------------------------------------

# Select a different bottom node
node_y_fix = bottom_nodes[-1]

prescribed_dofs.append(
    3*node_y_fix + 1
)


# ------------------------------------------------------------
# Top surface z displacement
# ------------------------------------------------------------

for n in top_nodes:

    dof_z = 3*n + 2

    prescribed_dofs.append(
        dof_z
    )

    u_prescribed[dof_z] = u_top


prescribed_dofs = np.unique(
    prescribed_dofs
)


print(
    "\nNumber of prescribed DOFs:",
    len(prescribed_dofs)
)


# ============================================================
# 10. Free DOFs
# ============================================================

all_dofs = np.arange(
    N_DOF
)

free_dofs = np.setdiff1d(
    all_dofs,
    prescribed_dofs
)


print(
    "Number of free DOFs:",
    len(free_dofs)
)


# ============================================================
# 11. Partition global system
# ============================================================

# Original equation:
#
# K u = F
#
# Partition:
#
# Kff uf + Kfp up = Ff
#
# Therefore:
#
# Kff uf = Ff - Kfp up
#


K_ff = K[
    free_dofs[:, None],
    free_dofs
]

K_fp = K[
    free_dofs[:, None],
    prescribed_dofs
]


F = np.zeros(
    N_DOF
)


F_free = (
    F[free_dofs]
    -
    K_fp @
    u_prescribed[
        prescribed_dofs
    ]
)


# ============================================================
# 12. Solve free DOFs
# ============================================================

print(
    "\nSolving reduced FE system..."
)

u_free = spsolve(
    K_ff,
    F_free
)


# ============================================================
# 13. Construct complete displacement vector
# ============================================================

u = np.zeros(
    N_DOF
)

u[free_dofs] = u_free

u[prescribed_dofs] = (
    u_prescribed[
        prescribed_dofs
    ]
)


displacements = u.reshape(
    (-1, 3)
)


print(
    "FE solution completed."
)


# ============================================================
# 14. Check measured strain
# ============================================================

bottom_average = np.mean(
    displacements[
        bottom_nodes,
        2
    ]
)

top_average = np.mean(
    displacements[
        top_nodes,
        2
    ]
)

measured_strain = (
    top_average
    -
    bottom_average
) / Lz


print("\n==============================")
print("DISPLACEMENT RESULTS")
print("==============================")

print(
    "Target strain:",
    strain_target
)

print(
    "Measured strain:",
    measured_strain
)

print(
    "Bottom displacement:",
    bottom_average,
    "mm"
)

print(
    "Top displacement:",
    top_average,
    "mm"
)


# ============================================================
# 15. Calculate global reaction force
# ============================================================

# Internal force
internal_force = (
    K @ u
)

# Since F = 0 externally for displacement control,
# reactions are internal forces at prescribed DOFs.

reaction = internal_force - F


# ------------------------------------------------------------
# Reaction on top z surface
# ------------------------------------------------------------

top_z_dofs = (
    3*top_nodes + 2
)

top_reaction = np.sum(
    reaction[top_z_dofs]
)


# ============================================================
# 16. Cross-sectional area
# ============================================================

A = Lx * Ly


# ============================================================
# 17. FE axial stress
# ============================================================

sigma_FE = (
    abs(top_reaction)
    /
    A
)


# ============================================================
# 18. Theoretical stress
# ============================================================

sigma_theoretical = (
    E
    *
    strain_target
)


stress_error = (
    abs(
        sigma_FE
        -
        sigma_theoretical
    )
    /
    sigma_theoretical
    *
    100
)


print(
    "\nTheoretical stress:",
    sigma_theoretical,
    "MPa"
)

print(
    "FE stress:",
    sigma_FE,
    "MPa"
)

print(
    "Stress error:",
    stress_error,
    "%"
)


# ============================================================
# 19. Check Poisson contraction
# ============================================================

# Average x and y displacement at top

ux_top = np.mean(
    displacements[
        top_nodes,
        0
    ]
)

uy_top = np.mean(
    displacements[
        top_nodes,
        1
    ]
)


strain_x = (
    ux_top
    /
    Lx
)

strain_y = (
    uy_top
    /
    Ly
)


print(
    "\nLateral strains:"
)

print(
    "epsilon_x =",
    strain_x
)

print(
    "epsilon_y =",
    strain_y
)

print(
    "Expected approximately:",
    -nu * strain_target
)


# ============================================================
# 20. Stress-strain verification
# ============================================================

print("\n==============================")
print("ELASTIC FE VERIFICATION")
print("==============================")

print(
    f"Target strain      = "
    f"{strain_target:.8f}"
)

print(
    f"FE strain          = "
    f"{measured_strain:.8f}"
)

print(
    f"Theoretical stress = "
    f"{sigma_theoretical:.4f} MPa"
)

print(
    f"FE stress          = "
    f"{sigma_FE:.4f} MPa"
)

print(
    f"Stress error       = "
    f"{stress_error:.4f}%"
)

print(
    "=============================="
)

# ============================================================
# MODULE 4A
# FCC CRYSTAL PLASTICITY
#
# Part 1:
# Generate the 12 FCC {111}<110> slip systems
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R


# ============================================================
# 1. Generate FCC slip systems
# ============================================================

def generate_fcc_slip_systems():

    # --------------------------------------------------------
    # FCC slip planes
    #
    # {111}
    # --------------------------------------------------------

    plane_normals = np.array([
        [ 1,  1,  1],
        [ 1,  1, -1],
        [ 1, -1,  1],
        [-1,  1,  1]
    ], dtype=float)


    # --------------------------------------------------------
    # <110> directions
    # --------------------------------------------------------

    directions = np.array([

        [ 1,  1,  0],
        [ 1, -1,  0],
        [ 1,  0,  1],
        [ 1,  0, -1],
        [ 0,  1,  1],
        [ 0,  1, -1]

    ], dtype=float)


    # Normalize plane normals
    plane_normals /= np.linalg.norm(
        plane_normals,
        axis=1
    )[:, None]


    # Normalize directions
    directions /= np.linalg.norm(
        directions,
        axis=1
    )[:, None]


    slip_systems = []


    # --------------------------------------------------------
    # Select only directions lying in the plane
    #
    # n · s = 0
    # --------------------------------------------------------

    for n in plane_normals:

        for s in directions:

            if abs(
                np.dot(n, s)
            ) < 1e-12:

                slip_systems.append(
                    (
                        n.copy(),
                        s.copy()
                    )
                )


    return slip_systems


slip_systems = (
    generate_fcc_slip_systems()
)


# ============================================================
# 2. Check number of slip systems
# ============================================================

print("==============================")
print("FCC SLIP SYSTEM CHECK")
print("==============================")

print(
    "Number of FCC slip systems:",
    len(slip_systems)
)


# ============================================================
# 3. Print slip systems
# ============================================================

for i, (n, s) in enumerate(
    slip_systems
):

    print(
        f"{i+1:2d}: "
        f"n = {n}, "
        f"s = {s}"
    )


# ============================================================
# 4. Mathematical validation
# ============================================================

max_orthogonality_error = 0.0

for n, s in slip_systems:

    error = abs(
        np.dot(n, s)
    )

    max_orthogonality_error = max(
        max_orthogonality_error,
        error
    )


print(
    "\nMaximum n·s error:",
    max_orthogonality_error
)


# ============================================================
# 5. Verify unit vectors
# ============================================================

normal_errors = []
direction_errors = []

for n, s in slip_systems:

    normal_errors.append(
        abs(np.linalg.norm(n) - 1)
    )

    direction_errors.append(
        abs(np.linalg.norm(s) - 1)
    )


print(
    "Maximum normal normalization error:",
    max(normal_errors)
)

print(
    "Maximum direction normalization error:",
    max(direction_errors)
)


# ============================================================
# 6. Schmid tensor for each slip system
# ============================================================

schmid_tensors = []

for n, s in slip_systems:

    P = 0.5 * (
        np.outer(s, n)
        +
        np.outer(n, s)
    )

    schmid_tensors.append(P)


schmid_tensors = np.array(
    schmid_tensors
)


print(
    "\nSchmid tensor array:",
    schmid_tensors.shape
)

# ============================================================
# MODULE 4B
# CRYSTAL ORIENTATION TRANSFORMATION
# ============================================================


# ============================================================
# 1. Load grain orientations
# ============================================================

micro_data = np.load(
    "316L_3D_Voronoi_microstructure.npz"
)

grain_orientation = (
    micro_data[
        "orientation_matrices"
    ]
)

grain_euler = (
    micro_data[
        "euler_angles"
    ]
)


print(
    "Number of grain orientations:",
    len(grain_orientation)
)


# ============================================================
# 2. Select one example grain
# ============================================================

grain_id = 0

R_grain = (
    grain_orientation[grain_id]
)


print(
    "\nSelected grain:",
    grain_id
)

print(
    "Orientation matrix:"
)

print(
    R_grain
)


# ============================================================
# 3. Rotate slip systems
# ============================================================

rotated_slip_systems = []

for n_crystal, s_crystal in slip_systems:

    n_sample = (
        R_grain
        @ n_crystal
    )

    s_sample = (
        R_grain
        @ s_crystal
    )

    # Re-normalize to remove numerical error

    n_sample /= np.linalg.norm(
        n_sample
    )

    s_sample /= np.linalg.norm(
        s_sample
    )

    rotated_slip_systems.append(
        (
            n_sample,
            s_sample
        )
    )


# ============================================================
# 4. Verify transformed systems
# ============================================================

print(
    "\nTransformed FCC slip systems:"
)

for i, (n, s) in enumerate(
    rotated_slip_systems
):

    print(
        f"{i+1:2d}: "
        f"n = {n}, "
        f"s = {s}, "
        f"n·s = {np.dot(n,s):.3e}"
    )

# ============================================================
# MODULE 4C
# SCHMID FACTOR CALCULATION
# ============================================================


sigma_test = 1.0   # arbitrary unit stress


stress_tensor = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, sigma_test]
])


schmid_factors = []


for n, s in rotated_slip_systems:

    tau = (
        s
        @ stress_tensor
        @ n
    )

    schmid_factors.append(
        tau / sigma_test
    )


schmid_factors = np.array(
    schmid_factors
)


print("==============================")
print("SCHMID FACTORS")
print("==============================")


for i, m in enumerate(
    schmid_factors
):

    print(
        f"Slip system {i+1:2d}: "
        f"m = {m:.6f}"
    )


print(
    "\nMaximum absolute Schmid factor:",
    np.max(
        np.abs(schmid_factors)
    )
)


critical_system = np.argmax(
    np.abs(schmid_factors)
)


print(
    "Critical slip system:",
    critical_system + 1
)

# ============================================================
# MODULE 4D
# SINGLE-CRYSTAL CRYSTAL-PLASTICITY MATERIAL POINT
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CP material parameters
# ============================================================

tau0 = 150.0          # MPa
gamma_dot_0 = 0.001   # 1/s
rate_exponent = 20.0

print("==============================")
print("CRYSTAL PLASTICITY PARAMETERS")
print("==============================")

print("Initial CRSS:", tau0, "MPa")
print("Reference slip rate:", gamma_dot_0, "1/s")
print("Rate exponent:", rate_exponent)


# ============================================================
# 2. Use grain 0 orientation
# ============================================================

R_grain = (
    grain_orientation[0]
)


# ============================================================
# 3. Transform slip systems
# ============================================================

rotated_slip_systems = []

for n_crystal, s_crystal in slip_systems:

    n = R_grain @ n_crystal
    s = R_grain @ s_crystal

    n /= np.linalg.norm(n)
    s /= np.linalg.norm(s)

    rotated_slip_systems.append(
        (n, s)
    )


# ============================================================
# 4. Calculate slip rates
# ============================================================

def calculate_slip_rates(
    stress,
    slip_systems,
    resistance,
    gamma_dot_0,
    rate_exponent
):

    slip_rates = np.zeros(
        len(slip_systems)
    )

    resolved_shear = np.zeros(
        len(slip_systems)
    )


    for a, (n, s) in enumerate(
        slip_systems
    ):

        # Resolved shear stress
        tau = (
            s
            @ stress
            @ n
        )

        resolved_shear[a] = tau


        # Power-law slip kinetics

        ratio = (
            abs(tau)
            /
            resistance[a]
        )

        slip_rates[a] = (
            gamma_dot_0
            *
            ratio**rate_exponent
            *
            np.sign(tau)
        )


    return (
        slip_rates,
        resolved_shear
    )


# ============================================================
# 5. Test at several stresses
# ============================================================

test_stresses = np.linspace(
    0,
    800,
    81
)

max_slip_rates = []

max_resolved_shear = []


for sigma in test_stresses:

    stress = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, sigma]
    ])


    resistance = np.full(
        12,
        tau0
    )


    rates, taus = (
        calculate_slip_rates(
            stress,
            rotated_slip_systems,
            resistance,
            gamma_dot_0,
            rate_exponent
        )
    )


    max_slip_rates.append(
        np.max(np.abs(rates))
    )

    max_resolved_shear.append(
        np.max(np.abs(taus))
    )


max_slip_rates = np.array(
    max_slip_rates
)

max_resolved_shear = np.array(
    max_resolved_shear
)


# ============================================================
# 6. Plot slip activity
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.semilogy(
    test_stresses,
    max_slip_rates
)

plt.xlabel(
    "Applied tensile stress (MPa)",
    fontsize=14
)

plt.ylabel(
    "Maximum slip rate (1/s)",
    fontsize=14
)

plt.title(
    "FCC crystal slip activity",
    fontsize=16
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 7. Print selected results
# ============================================================

print("\n==============================")
print("SLIP ACTIVITY CHECK")
print("==============================")

for sigma in [
    100,
    150,
    200,
    300,
    500,
    700
]:

    stress = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, sigma]
    ])


    resistance = np.full(
        12,
        tau0
    )


    rates, taus = (
        calculate_slip_rates(
            stress,
            rotated_slip_systems,
            resistance,
            gamma_dot_0,
            rate_exponent
        )
    )


    print(
        f"\nApplied stress = {sigma} MPa"
    )

    print(
        "Maximum RSS =",
        np.max(np.abs(taus)),
        "MPa"
    )

    print(
        "Maximum slip rate =",
        np.max(np.abs(rates)),
        "1/s"
    )

# ============================================================
# MODULE 4E
# SINGLE-CRYSTAL CRYSTAL PLASTICITY
#
# Incremental small-strain constitutive model
#
# FCC {111}<110>
# Power-law slip
# Isotropic hardening
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Material parameters
# ============================================================

E = 193000.0          # MPa
nu = 0.29             # Poisson ratio

tau0 = 150.0          # Initial CRSS, MPa
tau_sat = 350.0       # Saturation resistance, MPa
h0 = 5000.0           # Hardening modulus, MPa

gamma_dot_0 = 0.001   # Reference slip rate, 1/s
rate_exponent = 20.0

strain_rate = 0.001   # Applied strain rate, 1/s

total_strain = 0.01

n_steps = 1000

dt = (
    total_strain
    /
    strain_rate
    /
    n_steps
)

delta_strain = (
    strain_rate
    * dt
)


print("==============================")
print("SINGLE-CRYSTAL CP PARAMETERS")
print("==============================")

print("E =", E, "MPa")
print("nu =", nu)
print("Initial CRSS =", tau0, "MPa")
print("Saturation CRSS =", tau_sat, "MPa")
print("Hardening modulus =", h0, "MPa")
print("Reference slip rate =", gamma_dot_0, "1/s")
print("Rate exponent =", rate_exponent)
print("Applied strain rate =", strain_rate, "1/s")
print("Total strain =", total_strain)
print("Number of increments =", n_steps)
print("Strain increment =", delta_strain)


# ============================================================
# 2. Elastic stiffness matrix
# ============================================================

C = np.zeros((6, 6))

lam = (
    E * nu
    /
    ((1 + nu) * (1 - 2 * nu))
)

mu = (
    E
    /
    (2 * (1 + nu))
)

C[0, 0] = lam + 2*mu
C[1, 1] = lam + 2*mu
C[2, 2] = lam + 2*mu

C[0, 1] = lam
C[0, 2] = lam
C[1, 0] = lam
C[1, 2] = lam
C[2, 0] = lam
C[2, 1] = lam

C[3, 3] = mu
C[4, 4] = mu
C[5, 5] = mu


# ============================================================
# 3. Select grain 0
# ============================================================

R_grain = (
    grain_orientation[0]
)


# ============================================================
# 4. Rotate slip systems
# ============================================================

rotated_slip_systems = []

for n_crystal, s_crystal in slip_systems:

    n = R_grain @ n_crystal
    s = R_grain @ s_crystal

    n /= np.linalg.norm(n)
    s /= np.linalg.norm(s)

    rotated_slip_systems.append(
        (n, s)
    )


# ============================================================
# 5. Schmid tensors
# ============================================================

P = []

for n, s in rotated_slip_systems:

    P_alpha = 0.5 * (
        np.outer(s, n)
        +
        np.outer(n, s)
    )

    P.append(
        P_alpha
    )

P = np.array(P)


# ============================================================
# 6. Tensor <-> Voigt conversion
# ============================================================

def tensor_to_voigt(A):

    return np.array([
        A[0, 0],
        A[1, 1],
        A[2, 2],
        2*A[0, 1],
        2*A[1, 2],
        2*A[0, 2]
    ])


def voigt_to_tensor(v):

    A = np.zeros((3, 3))

    A[0, 0] = v[0]
    A[1, 1] = v[1]
    A[2, 2] = v[2]

    A[0, 1] = v[3] / 2
    A[1, 0] = v[3] / 2

    A[1, 2] = v[4] / 2
    A[2, 1] = v[4] / 2

    A[0, 2] = v[5] / 2
    A[2, 0] = v[5] / 2

    return A


# ============================================================
# 7. State variables
# ============================================================

plastic_strain = np.zeros(
    (3, 3)
)

total_strain_tensor = np.zeros(
    (3, 3)
)

stress = np.zeros(
    (3, 3)
)

slip_resistance = np.full(
    12,
    tau0
)

accumulated_slip = np.zeros(
    12
)


# ============================================================
# 8. Storage
# ============================================================

strain_history = []
stress_history = []

plastic_strain_history = []
slip_history = []

crss_history = []

rss_history = []


# ============================================================
# 9. Local constitutive calculation
# ============================================================

def calculate_slip_rates_from_stress(
    stress,
    slip_resistance
):

    tau = np.zeros(12)
    gamma_dot = np.zeros(12)

    for a in range(12):

        n = rotated_slip_systems[a][0]
        s = rotated_slip_systems[a][1]

        tau[a] = (
            s
            @ stress
            @ n
        )

        ratio = (
            abs(tau[a])
            /
            slip_resistance[a]
        )

        gamma_dot[a] = (
            gamma_dot_0
            *
            ratio**rate_exponent
            *
            np.sign(tau[a])
        )

    return (
        tau,
        gamma_dot
    )


# ============================================================
# 10. Incremental strain-controlled simulation
# ============================================================

for step in range(
    n_steps
):

    # --------------------------------------------------------
    # Applied total strain increment
    #
    # Uniaxial strain in z
    # --------------------------------------------------------

    total_strain_tensor[2, 2] += (
        delta_strain
    )


    # --------------------------------------------------------
    # Elastic trial stress
    # --------------------------------------------------------

    elastic_strain_trial = (
        total_strain_tensor
        -
        plastic_strain
    )

    stress_trial = voigt_to_tensor(
        C
        @
        tensor_to_voigt(
            elastic_strain_trial
        )
    )


    # --------------------------------------------------------
    # Calculate resolved shear stresses
    # --------------------------------------------------------

    tau_trial = np.zeros(12)

    for a in range(12):

        n = rotated_slip_systems[a][0]
        s = rotated_slip_systems[a][1]

        tau_trial[a] = (
            s
            @ stress_trial
            @ n
        )


    # --------------------------------------------------------
    # Determine slip rates
    # --------------------------------------------------------

    gamma_dot = np.zeros(12)

    for a in range(12):

        ratio = (
            abs(tau_trial[a])
            /
            slip_resistance[a]
        )

        gamma_dot[a] = (
            gamma_dot_0
            *
            ratio**rate_exponent
            *
            np.sign(
                tau_trial[a]
            )
        )


    # --------------------------------------------------------
    # Limit numerical instability
    #
    # This is a prototype material-point model.
    # We limit the maximum slip increment per step.
    # --------------------------------------------------------

    max_gamma_increment = 0.005

    gamma_increment = (
        gamma_dot
        *
        dt
    )

    gamma_increment = np.clip(
        gamma_increment,
        -max_gamma_increment,
        max_gamma_increment
    )


    # --------------------------------------------------------
    # Plastic strain increment
    # --------------------------------------------------------

    plastic_increment = np.zeros(
        (3, 3)
    )

    for a in range(12):

        plastic_increment += (
            gamma_increment[a]
            *
            P[a]
        )


    plastic_strain += (
        plastic_increment
    )


    # --------------------------------------------------------
    # Accumulated slip
    # --------------------------------------------------------

    accumulated_slip += np.abs(
        gamma_increment
    )


    # --------------------------------------------------------
    # Isotropic hardening
    # --------------------------------------------------------

    total_slip_increment = np.sum(
        np.abs(
            gamma_increment
        )
    )


    slip_resistance += (
        h0
        *
        (
            1
            -
            slip_resistance
            /
            tau_sat
        )
        *
        total_slip_increment
    )


    slip_resistance = np.maximum(
        slip_resistance,
        tau0
    )


    # --------------------------------------------------------
    # Recalculate stress
    # --------------------------------------------------------

    elastic_strain = (
        total_strain_tensor
        -
        plastic_strain
    )

    stress = voigt_to_tensor(
        C
        @
        tensor_to_voigt(
            elastic_strain
        )
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    strain_history.append(
        total_strain_tensor[2, 2]
    )

    stress_history.append(
        stress[2, 2]
    )

    plastic_strain_history.append(
        plastic_strain[2, 2]
    )

    slip_history.append(
        np.sum(
            accumulated_slip
        )
    )

    crss_history.append(
        np.mean(
            slip_resistance
        )
    )

    rss_history.append(
        np.max(
            np.abs(
                tau_trial
            )
        )
    )


# Convert arrays

strain_history = np.array(
    strain_history
)

stress_history = np.array(
    stress_history
)

plastic_strain_history = np.array(
    plastic_strain_history
)

slip_history = np.array(
    slip_history
)

crss_history = np.array(
    crss_history
)

rss_history = np.array(
    rss_history
)


# ============================================================
# 11. Results
# ============================================================

print("\n==============================")
print("SINGLE-CRYSTAL CP RESULTS")
print("==============================")

print(
    "Final total strain:",
    strain_history[-1]
)

print(
    "Final stress:",
    stress_history[-1],
    "MPa"
)

print(
    "Final plastic strain:",
    plastic_strain_history[-1]
)

print(
    "Final accumulated slip:",
    slip_history[-1]
)

print(
    "Final mean CRSS:",
    crss_history[-1],
    "MPa"
)

print(
    "Maximum RSS:",
    np.max(rss_history),
    "MPa"
)


# ============================================================
# 12. Stress-strain curve
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    strain_history * 100,
    stress_history,
    linewidth=2
)

plt.xlabel(
    "Engineering strain (%)",
    fontsize=14
)

plt.ylabel(
    "Axial stress (MPa)",
    fontsize=14
)

plt.title(
    "Single-crystal FCC crystal-plasticity response",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 13. Plastic strain evolution
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    strain_history * 100,
    plastic_strain_history,
    linewidth=2
)

plt.xlabel(
    "Engineering strain (%)",
    fontsize=14
)

plt.ylabel(
    "Plastic strain",
    fontsize=14
)

plt.title(
    "Accumulation of crystal plastic strain",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 14. CRSS evolution
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    strain_history * 100,
    crss_history,
    linewidth=2
)

plt.xlabel(
    "Engineering strain (%)",
    fontsize=14
)

plt.ylabel(
    "Mean CRSS (MPa)",
    fontsize=14
)

plt.title(
    "Crystal-plasticity hardening",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()

# ============================================================
# MODULE 4F-CORRECTED
# IMPLICIT SINGLE-CRYSTAL CRYSTAL PLASTICITY
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


# ============================================================
# 1. Material parameters
# ============================================================

E = 193000.0
nu = 0.29

tau0 = 150.0
tau_sat = 350.0
h0 = 5000.0

gamma_dot_0 = 0.001
rate_exponent = 20.0

strain_rate = 0.001

total_strain = 0.01
n_steps = 1000

dt = (
    total_strain
    /
    strain_rate
    /
    n_steps
)

delta_strain = (
    strain_rate * dt
)


print("==============================")
print("CORRECTED IMPLICIT CP")
print("==============================")

print("E =", E, "MPa")
print("nu =", nu)
print("Initial CRSS =", tau0, "MPa")
print("Saturation CRSS =", tau_sat, "MPa")
print("Hardening modulus =", h0, "MPa")
print("Reference slip rate =", gamma_dot_0, "1/s")
print("Rate exponent =", rate_exponent)
print("Strain rate =", strain_rate, "1/s")
print("Total strain =", total_strain)
print("Number of increments =", n_steps)
print("dt =", dt, "s")
print("dε =", delta_strain)


# ============================================================
# 2. Elastic stiffness
# ============================================================

C = np.zeros((6, 6))

lam = (
    E * nu
    /
    ((1 + nu) * (1 - 2 * nu))
)

mu = (
    E
    /
    (2 * (1 + nu))
)

C[0, 0] = lam + 2 * mu
C[1, 1] = lam + 2 * mu
C[2, 2] = lam + 2 * mu

C[0, 1] = lam
C[0, 2] = lam
C[1, 0] = lam
C[1, 2] = lam
C[2, 0] = lam
C[2, 1] = lam

C[3, 3] = mu
C[4, 4] = mu
C[5, 5] = mu


# ============================================================
# 3. Tensor / Voigt conversion
# ============================================================

def tensor_to_voigt(A):

    return np.array([
        A[0, 0],
        A[1, 1],
        A[2, 2],
        2.0 * A[0, 1],
        2.0 * A[1, 2],
        2.0 * A[0, 2]
    ])


def voigt_to_tensor(v):

    A = np.zeros((3, 3))

    A[0, 0] = v[0]
    A[1, 1] = v[1]
    A[2, 2] = v[2]

    A[0, 1] = v[3] / 2
    A[1, 0] = v[3] / 2

    A[1, 2] = v[4] / 2
    A[2, 1] = v[4] / 2

    A[0, 2] = v[5] / 2
    A[2, 0] = v[5] / 2

    return A


# ============================================================
# 4. Grain orientation
# ============================================================

R_grain = grain_orientation[0]


# ============================================================
# 5. Rotate FCC slip systems
# ============================================================

rotated_slip_systems = []

for n_crystal, s_crystal in slip_systems:

    n = R_grain @ n_crystal
    s = R_grain @ s_crystal

    n /= np.linalg.norm(n)
    s /= np.linalg.norm(s)

    rotated_slip_systems.append(
        (n, s)
    )


# ============================================================
# 6. Schmid tensors
# ============================================================

P = np.zeros(
    (12, 3, 3)
)

for a in range(12):

    n = rotated_slip_systems[a][0]
    s = rotated_slip_systems[a][1]

    P[a] = 0.5 * (
        np.outer(s, n)
        +
        np.outer(n, s)
    )


# ============================================================
# 7. Initial state
# ============================================================

plastic_strain = np.zeros(
    (3, 3)
)

slip_resistance = np.full(
    12,
    tau0
)

accumulated_slip = np.zeros(
    12
)


# ============================================================
# 8. Functions
# ============================================================

def calculate_resolved_shear(stress):

    tau = np.zeros(12)

    for a in range(12):

        n = rotated_slip_systems[a][0]
        s = rotated_slip_systems[a][1]

        tau[a] = (
            s
            @ stress
            @ n
        )

    return tau


def calculate_slip_rate(
    tau,
    resistance
):

    ratio = (
        np.abs(tau)
        /
        resistance
    )

    gamma_dot = (
        gamma_dot_0
        *
        ratio**rate_exponent
        *
        np.sign(tau)
    )

    return gamma_dot


# ============================================================
# 9. Local residual
# ============================================================

def local_residual(
    x,
    eps_zz_new,
    plastic_old,
    resistance_old
):

    # --------------------------------------------------------
    # Unknowns
    # --------------------------------------------------------

    eps_xx = x[0]
    eps_yy = x[1]
    eps_xy = x[2]
    eps_yz = x[3]
    eps_xz = x[4]

    dg = x[5:17]


    # --------------------------------------------------------
    # Total strain
    # --------------------------------------------------------

    total_strain = np.array([
        [eps_xx, eps_xy, eps_xz],
        [eps_xy, eps_yy, eps_yz],
        [eps_xz, eps_yz, eps_zz_new]
    ])


    # --------------------------------------------------------
    # Plastic increment
    # --------------------------------------------------------

    plastic_increment = np.zeros(
        (3, 3)
    )

    for a in range(12):

        plastic_increment += (
            dg[a] * P[a]
        )


    plastic_new = (
        plastic_old
        +
        plastic_increment
    )


    # --------------------------------------------------------
    # Elastic strain
    # --------------------------------------------------------

    elastic_strain = (
        total_strain
        -
        plastic_new
    )


    # --------------------------------------------------------
    # Stress
    # --------------------------------------------------------

    stress_new = voigt_to_tensor(
        C
        @
        tensor_to_voigt(
            elastic_strain
        )
    )


    # --------------------------------------------------------
    # RSS
    # --------------------------------------------------------

    tau = calculate_resolved_shear(
        stress_new
    )


    # --------------------------------------------------------
    # Hardening
    # --------------------------------------------------------

    total_slip_increment = np.sum(
        np.abs(dg)
    )

    resistance_new = (
        resistance_old
        +
        h0
        *
        (
            1
            -
            resistance_old
            /
            tau_sat
        )
        *
        total_slip_increment
    )

    resistance_new = np.maximum(
        resistance_new,
        tau0
    )


    # --------------------------------------------------------
    # Slip-rate equation
    # --------------------------------------------------------

    gamma_dot = calculate_slip_rate(
        tau,
        resistance_new
    )

    flow_residual = (
        dg
        -
        dt * gamma_dot
    )


    # --------------------------------------------------------
    # Stress-free conditions
    # --------------------------------------------------------

    stress_residual = np.array([

        stress_new[0, 0],
        stress_new[1, 1],
        stress_new[0, 1],
        stress_new[1, 2],
        stress_new[0, 2]

    ])


    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    stress_scale = E

    slip_scale = max(
        gamma_dot_0 * dt,
        1e-8
    )


    return np.concatenate([

        stress_residual / stress_scale,

        flow_residual / slip_scale

    ])


# ============================================================
# 10. History
# ============================================================

strain_history = []
stress_history = []

plastic_history = []
crss_history = []

slip_history = []
rss_history = []

lateral_x_history = []
lateral_y_history = []

shear_xy_history = []
shear_xz_history = []
shear_yz_history = []

residual_history = []


# ============================================================
# 11. Initial solution
# ============================================================

x_previous = np.zeros(17)

x_previous[0] = 0.0
x_previous[1] = 0.0


# ============================================================
# 12. Increment loop
# ============================================================

converged_steps = 0


for step in range(n_steps):

    eps_zz_new = (
        (step + 1)
        *
        delta_strain
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Use the previous converged solution directly.
    #
    # Do NOT reset lateral strain to elastic Poisson values
    # at every increment.
    # --------------------------------------------------------

    x0 = x_previous.copy()


    # First increment only:
    if step == 0:

        x0[0] = -nu * eps_zz_new
        x0[1] = -nu * eps_zz_new


    # --------------------------------------------------------
    # Nonlinear solve
    # --------------------------------------------------------

    solution = least_squares(
        local_residual,
        x0,
        args=(
            eps_zz_new,
            plastic_strain,
            slip_resistance
        ),
        max_nfev=2000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        verbose=0
    )


    if not solution.success:

        print(
            "\nWARNING: Local solver failed"
        )

        print(
            "Step:",
            step + 1
        )

        print(
            "Strain:",
            eps_zz_new
        )

        print(
            solution.message
        )

        print(
            "Residual norm:",
            np.linalg.norm(
                solution.fun
            )
        )

        break


    # --------------------------------------------------------
    # Save converged solution
    # --------------------------------------------------------

    x = solution.x

    x_previous = x.copy()


    eps_xx = x[0]
    eps_yy = x[1]
    eps_xy = x[2]
    eps_yz = x[3]
    eps_xz = x[4]

    dg = x[5:17]


    # --------------------------------------------------------
    # Total strain tensor
    # --------------------------------------------------------

    total_strain = np.array([
        [eps_xx, eps_xy, eps_xz],
        [eps_xy, eps_yy, eps_yz],
        [eps_xz, eps_yz, eps_zz_new]
    ])


    # --------------------------------------------------------
    # Plastic increment
    # --------------------------------------------------------

    plastic_increment = np.zeros(
        (3, 3)
    )

    for a in range(12):

        plastic_increment += (
            dg[a] * P[a]
        )


    plastic_strain += (
        plastic_increment
    )


    # --------------------------------------------------------
    # Accumulated slip
    # --------------------------------------------------------

    accumulated_slip += np.abs(dg)


    # --------------------------------------------------------
    # Hardening
    # --------------------------------------------------------

    total_slip_increment = np.sum(
        np.abs(dg)
    )

    slip_resistance += (
        h0
        *
        (
            1
            -
            slip_resistance
            /
            tau_sat
        )
        *
        total_slip_increment
    )

    slip_resistance = np.maximum(
        slip_resistance,
        tau0
    )


    # --------------------------------------------------------
    # Final stress
    # --------------------------------------------------------

    elastic_strain = (
        total_strain
        -
        plastic_strain
    )

    stress = voigt_to_tensor(
        C
        @
        tensor_to_voigt(
            elastic_strain
        )
    )


    # --------------------------------------------------------
    # RSS
    # --------------------------------------------------------

    tau = calculate_resolved_shear(
        stress
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    strain_history.append(
        eps_zz_new
    )

    stress_history.append(
        stress[2, 2]
    )

    plastic_history.append(
        plastic_strain[2, 2]
    )

    crss_history.append(
        np.mean(
            slip_resistance
        )
    )

    slip_history.append(
        np.sum(
            accumulated_slip
        )
    )

    rss_history.append(
        np.max(
            np.abs(tau)
        )
    )

    lateral_x_history.append(
        eps_xx
    )

    lateral_y_history.append(
        eps_yy
    )

    shear_xy_history.append(
        eps_xy
    )

    shear_xz_history.append(
        eps_xz
    )

    shear_yz_history.append(
        eps_yz
    )

    residual_history.append(
        np.linalg.norm(
            solution.fun
        )
    )

    converged_steps += 1


# ============================================================
# 13. Convert history
# ============================================================

strain_history = np.array(
    strain_history
)

stress_history = np.array(
    stress_history
)

plastic_history = np.array(
    plastic_history
)

crss_history = np.array(
    crss_history
)

slip_history = np.array(
    slip_history
)

rss_history = np.array(
    rss_history
)

lateral_x_history = np.array(
    lateral_x_history
)

lateral_y_history = np.array(
    lateral_y_history
)

shear_xy_history = np.array(
    shear_xy_history
)

shear_xz_history = np.array(
    shear_xz_history
)

shear_yz_history = np.array(
    shear_yz_history
)

residual_history = np.array(
    residual_history
)


# ============================================================
# 14. Final verification
# ============================================================

print("\n==============================")
print("CORRECTED IMPLICIT CP RESULTS")
print("==============================")

print(
    "Converged increments:",
    converged_steps,
    "/",
    n_steps
)

print(
    "Final strain:",
    strain_history[-1]
)

print(
    "Final axial stress:",
    stress_history[-1],
    "MPa"
)

print(
    "Final plastic strain:",
    plastic_history[-1]
)

print(
    "Final accumulated slip:",
    slip_history[-1]
)

print(
    "Final mean CRSS:",
    crss_history[-1],
    "MPa"
)

print(
    "Maximum RSS:",
    np.max(rss_history),
    "MPa"
)


# ============================================================
# 15. Reconstruct actual final strain tensor
# ============================================================

eps_final = np.array([

    [
        lateral_x_history[-1],
        shear_xy_history[-1],
        shear_xz_history[-1]
    ],

    [
        shear_xy_history[-1],
        lateral_y_history[-1],
        shear_yz_history[-1]
    ],

    [
        shear_xz_history[-1],
        shear_yz_history[-1],
        strain_history[-1]
    ]

])


# ============================================================
# 16. Final stress
# ============================================================

elastic_final = (
    eps_final
    -
    plastic_strain
)

stress_final = voigt_to_tensor(
    C
    @
    tensor_to_voigt(
        elastic_final
    )
)


print("\n==============================")
print("FINAL STRESS STATE")
print("==============================")

print(
    "sigma_xx =",
    stress_final[0, 0],
    "MPa"
)

print(
    "sigma_yy =",
    stress_final[1, 1],
    "MPa"
)

print(
    "sigma_zz =",
    stress_final[2, 2],
    "MPa"
)

print(
    "sigma_xy =",
    stress_final[0, 1],
    "MPa"
)

print(
    "sigma_xz =",
    stress_final[0, 2],
    "MPa"
)

print(
    "sigma_yz =",
    stress_final[1, 2],
    "MPa"
)

print(
    "Final nonlinear residual norm =",
    residual_history[-1]
)


# ============================================================
# 17. Stress-strain curve
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    strain_history * 100,
    stress_history,
    linewidth=2
)

plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Axial stress (MPa)",
    fontsize=14
)

plt.title(
    "Implicit single-crystal CP response",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 18. Lateral strain
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    strain_history * 100,
    lateral_x_history,
    label="epsilon_x"
)

plt.plot(
    strain_history * 100,
    lateral_y_history,
    label="epsilon_y"
)

plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Lateral strain",
    fontsize=14
)

plt.title(
    "Crystal lateral deformation",
    fontsize=15
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 19. CRSS evolution
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    strain_history * 100,
    crss_history,
    linewidth=2
)

plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Mean CRSS (MPa)",
    fontsize=14
)

plt.title(
    "Crystal hardening",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 20. Solver residual
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.semilogy(
    strain_history * 100,
    residual_history,
    linewidth=2
)

plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Nonlinear residual norm",
    fontsize=14
)

plt.title(
    "Local CP solver convergence",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()

# ============================================================
# MODULE 4G
# THREE-GRAIN ORIENTATION VERIFICATION
#
# Purpose:
# Verify that different grain orientations produce
# different crystal-plasticity responses.
#
# Uses:
#   grain_orientation
#   grain_euler
#   slip_systems
#
# from the previously generated microstructure.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CHECK INPUT DATA
# ============================================================

print("==============================")
print("THREE-GRAIN CP VERIFICATION")
print("==============================")

print(
    "Number of available grains:",
    len(grain_orientation)
)

print(
    "Orientation array shape:",
    np.array(grain_orientation).shape
)


# ============================================================
# 2. MATERIAL PARAMETERS
# ============================================================

E = 193000.0
nu = 0.29

tau0 = 150.0
tau_sat = 350.0
h0 = 5000.0

gamma_dot_0 = 0.001
rate_exponent = 20.0

strain_rate = 0.001

total_strain = 0.01
n_steps = 1000

dt = (
    total_strain
    /
    strain_rate
    /
    n_steps
)

delta_strain = (
    strain_rate * dt
)


# ============================================================
# 3. ELASTIC STIFFNESS MATRIX
# ============================================================

C = np.zeros((6, 6))

lam = (
    E * nu
    /
    ((1 + nu) * (1 - 2 * nu))
)

mu = (
    E
    /
    (2 * (1 + nu))
)

C[0, 0] = lam + 2 * mu
C[1, 1] = lam + 2 * mu
C[2, 2] = lam + 2 * mu

C[0, 1] = lam
C[0, 2] = lam
C[1, 0] = lam
C[1, 2] = lam
C[2, 0] = lam
C[2, 1] = lam

C[3, 3] = mu
C[4, 4] = mu
C[5, 5] = mu


# ============================================================
# 4. VOIGT FUNCTIONS
# ============================================================

def tensor_to_voigt(A):

    return np.array([
        A[0, 0],
        A[1, 1],
        A[2, 2],
        2.0 * A[0, 1],
        2.0 * A[1, 2],
        2.0 * A[0, 2]
    ])


def voigt_to_tensor(v):

    A = np.zeros((3, 3))

    A[0, 0] = v[0]
    A[1, 1] = v[1]
    A[2, 2] = v[2]

    A[0, 1] = v[3] / 2
    A[1, 0] = v[3] / 2

    A[1, 2] = v[4] / 2
    A[2, 1] = v[4] / 2

    A[0, 2] = v[5] / 2
    A[2, 0] = v[5] / 2

    return A


# ============================================================
# 5. RUN CP FOR ONE GRAIN
# ============================================================

def run_single_grain_cp(grain_id):

    print("\n--------------------------------")
    print("Running grain", grain_id)
    print("--------------------------------")


    # --------------------------------------------------------
    # Grain orientation
    # --------------------------------------------------------

    R_grain = np.array(
        grain_orientation[grain_id]
    )


    # --------------------------------------------------------
    # Rotate slip systems
    # --------------------------------------------------------

    rotated_slip_systems = []

    for n_crystal, s_crystal in slip_systems:

        n = R_grain @ n_crystal
        s = R_grain @ s_crystal

        n /= np.linalg.norm(n)
        s /= np.linalg.norm(s)

        rotated_slip_systems.append(
            (n, s)
        )


    # --------------------------------------------------------
    # Schmid factors for uniaxial z loading
    # --------------------------------------------------------

    schmid = np.zeros(12)

    loading_direction = np.array([
        0.0,
        0.0,
        1.0
    ])

    for a in range(12):

        n = rotated_slip_systems[a][0]
        s = rotated_slip_systems[a][1]

        schmid[a] = (
            np.dot(
                loading_direction,
                s
            )
            *
            np.dot(
                loading_direction,
                n
            )
        )


    critical_slip_system = np.argmax(
        np.abs(schmid)
    )

    maximum_schmid = np.max(
        np.abs(schmid)
    )


    print(
        "Maximum Schmid factor =",
        maximum_schmid
    )

    print(
        "Critical slip system =",
        critical_slip_system + 1
    )


    # --------------------------------------------------------
    # Schmid tensors
    # --------------------------------------------------------

    P = np.zeros(
        (12, 3, 3)
    )

    for a in range(12):

        n = rotated_slip_systems[a][0]
        s = rotated_slip_systems[a][1]

        P[a] = 0.5 * (
            np.outer(s, n)
            +
            np.outer(n, s)
        )


    # --------------------------------------------------------
    # State variables
    # --------------------------------------------------------

    plastic_strain = np.zeros(
        (3, 3)
    )

    slip_resistance = np.full(
        12,
        tau0
    )

    accumulated_slip = np.zeros(
        12
    )


    # --------------------------------------------------------
    # Functions specific to this grain
    # --------------------------------------------------------

    def calculate_tau(stress):

        tau = np.zeros(12)

        for a in range(12):

            n = rotated_slip_systems[a][0]
            s = rotated_slip_systems[a][1]

            tau[a] = (
                s
                @
                stress
                @
                n
            )

        return tau


    def calculate_slip_rate(
        tau,
        resistance
    ):

        ratio = (
            np.abs(tau)
            /
            resistance
        )

        return (
            gamma_dot_0
            *
            ratio**rate_exponent
            *
            np.sign(tau)
        )


    # --------------------------------------------------------
    # Local residual
    # --------------------------------------------------------

    def residual(
        x,
        eps_zz_new,
        plastic_old,
        resistance_old
    ):

        eps_xx = x[0]
        eps_yy = x[1]
        eps_xy = x[2]
        eps_yz = x[3]
        eps_xz = x[4]

        dg = x[5:17]


        total_eps = np.array([

            [eps_xx, eps_xy, eps_xz],

            [eps_xy, eps_yy, eps_yz],

            [eps_xz, eps_yz, eps_zz_new]

        ])


        plastic_increment = np.zeros(
            (3, 3)
        )

        for a in range(12):

            plastic_increment += (
                dg[a] * P[a]
            )


        plastic_new = (
            plastic_old
            +
            plastic_increment
        )


        elastic_eps = (
            total_eps
            -
            plastic_new
        )


        stress = voigt_to_tensor(
            C
            @
            tensor_to_voigt(
                elastic_eps
            )
        )


        tau = calculate_tau(
            stress
        )


        total_slip_increment = np.sum(
            np.abs(dg)
        )


        resistance_new = (
            resistance_old
            +
            h0
            *
            (
                1.0
                -
                resistance_old
                /
                tau_sat
            )
            *
            total_slip_increment
        )


        resistance_new = np.maximum(
            resistance_new,
            tau0
        )


        gamma_dot = calculate_slip_rate(
            tau,
            resistance_new
        )


        flow_residual = (
            dg
            -
            dt * gamma_dot
        )


        stress_residual = np.array([

            stress[0, 0],
            stress[1, 1],
            stress[0, 1],
            stress[1, 2],
            stress[0, 2]

        ])


        stress_scale = E

        slip_scale = max(
            gamma_dot_0 * dt,
            1e-8
        )


        return np.concatenate([

            stress_residual / stress_scale,

            flow_residual / slip_scale

        ])


    # --------------------------------------------------------
    # Histories
    # --------------------------------------------------------

    strain_hist = []
    stress_hist = []
    plastic_hist = []
    crss_hist = []
    slip_hist = []
    rss_hist = []


    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    x_previous = np.zeros(17)


    # --------------------------------------------------------
    # Increment loop
    # --------------------------------------------------------

    from scipy.optimize import least_squares


    for step in range(n_steps):

        eps_zz_new = (
            (step + 1)
            *
            delta_strain
        )


        x0 = x_previous.copy()


        if step == 0:

            x0[0] = (
                -nu
                *
                eps_zz_new
            )

            x0[1] = (
                -nu
                *
                eps_zz_new
            )


        solution = least_squares(

            residual,
            x0,

            args=(
                eps_zz_new,
                plastic_strain,
                slip_resistance
            ),

            max_nfev=2000,

            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11

        )


        if not solution.success:

            print(
                "Solver failed at step:",
                step + 1
            )

            print(
                solution.message
            )

            break


        x = solution.x

        x_previous = x.copy()


        dg = x[5:17]


        total_eps = np.array([

            [x[0], x[2], x[4]],

            [x[2], x[1], x[3]],

            [x[4], x[3], eps_zz_new]

        ])


        # ----------------------------------------------------
        # Plastic update
        # ----------------------------------------------------

        plastic_increment = np.zeros(
            (3, 3)
        )

        for a in range(12):

            plastic_increment += (
                dg[a] * P[a]
            )


        plastic_strain += (
            plastic_increment
        )


        # ----------------------------------------------------
        # Slip update
        # ----------------------------------------------------

        accumulated_slip += np.abs(dg)


        # ----------------------------------------------------
        # Hardening
        # ----------------------------------------------------

        total_slip_increment = np.sum(
            np.abs(dg)
        )

        slip_resistance += (

            h0
            *
            (
                1.0
                -
                slip_resistance
                /
                tau_sat
            )
            *
            total_slip_increment

        )


        slip_resistance = np.maximum(
            slip_resistance,
            tau0
        )


        # ----------------------------------------------------
        # Stress
        # ----------------------------------------------------

        elastic_eps = (
            total_eps
            -
            plastic_strain
        )


        stress = voigt_to_tensor(

            C
            @
            tensor_to_voigt(
                elastic_eps
            )

        )


        # ----------------------------------------------------
        # RSS
        # ----------------------------------------------------

        tau = calculate_tau(
            stress
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        strain_hist.append(
            eps_zz_new
        )

        stress_hist.append(
            stress[2, 2]
        )

        plastic_hist.append(
            plastic_strain[2, 2]
        )

        crss_hist.append(
            np.mean(
                slip_resistance
            )
        )

        slip_hist.append(
            np.sum(
                accumulated_slip
            )
        )

        rss_hist.append(
            np.max(
                np.abs(tau)
            )
        )


    # --------------------------------------------------------
    # Convert arrays
    # --------------------------------------------------------

    strain_hist = np.array(
        strain_hist
    )

    stress_hist = np.array(
        stress_hist
    )

    plastic_hist = np.array(
        plastic_hist
    )

    crss_hist = np.array(
        crss_hist
    )

    slip_hist = np.array(
        slip_hist
    )

    rss_hist = np.array(
        rss_hist
    )


    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    result = {

        "grain_id":
            grain_id,

        "schmid":
            schmid,

        "max_schmid":
            maximum_schmid,

        "critical_system":
            critical_slip_system,

        "strain":
            strain_hist,

        "stress":
            stress_hist,

        "plastic_strain":
            plastic_hist,

        "crss":
            crss_hist,

        "accumulated_slip":
            slip_hist,

        "rss":
            rss_hist,

        "orientation":
            R_grain

    }


    print(
        "Final stress =",
        stress_hist[-1],
        "MPa"
    )

    print(
        "Final plastic strain =",
        plastic_hist[-1]
    )

    print(
        "Final accumulated slip =",
        slip_hist[-1]
    )

    print(
        "Final mean CRSS =",
        crss_hist[-1],
        "MPa"
    )

    print(
        "Maximum RSS =",
        np.max(rss_hist),
        "MPa"
    )


    return result


# ============================================================
# 6. Select three grains
# ============================================================

test_grains = [
    0,
    1,
    2
]


# ============================================================
# 7. Run the three grains
# ============================================================

results_3grain = []

for grain_id in test_grains:

    results_3grain.append(
        run_single_grain_cp(
            grain_id
        )
    )


# ============================================================
# 8. Summary table
# ============================================================

print("\n")
print("==============================")
print("THREE-GRAIN SUMMARY")
print("==============================")

print(
    "Grain | Max Schmid | Final Stress | "
    "Plastic Strain | Accumulated Slip | Mean CRSS"
)

print("-" * 85)

for r in results_3grain:

    print(
        f"{r['grain_id']:5d} | "
        f"{r['max_schmid']:.6f} | "
        f"{r['stress'][-1]:.3f} MPa | "
        f"{r['plastic_strain'][-1]:.6f} | "
        f"{r['accumulated_slip'][-1]:.6f} | "
        f"{r['crss'][-1]:.3f} MPa"
    )


# ============================================================
# 9. Plot stress-strain responses
# ============================================================

plt.figure(
    figsize=(9, 6)
)

for r in results_3grain:

    plt.plot(

        r["strain"] * 100,

        r["stress"],

        linewidth=2,

        label=(
            "Grain "
            + str(r["grain_id"])
            + "  "
            + "m="
            + f"{r['max_schmid']:.3f}"
        )

    )


plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Axial stress (MPa)",
    fontsize=14
)

plt.title(
    "Orientation-dependent crystal-plasticity response",
    fontsize=15
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 10. Plot plastic strain
# ============================================================

plt.figure(
    figsize=(9, 6)
)

for r in results_3grain:

    plt.plot(

        r["strain"] * 100,

        r["plastic_strain"],

        linewidth=2,

        label=(
            "Grain "
            + str(r["grain_id"])
        )

    )


plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Axial plastic strain",
    fontsize=14
)

plt.title(
    "Orientation-dependent plastic deformation",
    fontsize=15
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 11. Plot accumulated slip
# ============================================================

plt.figure(
    figsize=(9, 6)
)

for r in results_3grain:

    plt.plot(

        r["strain"] * 100,

        r["accumulated_slip"],

        linewidth=2,

        label=(
            "Grain "
            + str(r["grain_id"])
        )

    )


plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Accumulated slip",
    fontsize=14
)

plt.title(
    "Orientation-dependent slip accumulation",
    fontsize=15
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()

# ============================================================
# MODULE 5A-C
# CORRECTED ONE-ELEMENT FE–CPFEM COUPLING
#
# IMPORTANT:
# This uses the SAME 17-variable implicit CP formulation
# that produced the verified grain-0 result:
#
# Final stress ≈ 395.906 MPa
#
# The FE element is used to generate the strain.
# The CP integration algorithm is kept identical.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


print("==============================")
print("MODULE 5A-C")
print("CORRECTED ONE-ELEMENT FE–CPFEM")
print("==============================")


# ============================================================
# 1. MATERIAL PARAMETERS
# ============================================================

E = 193000.0
nu = 0.29

tau0 = 150.0
tau_sat = 350.0
h0 = 5000.0

gamma_dot_0 = 0.001
rate_exponent = 20.0

strain_rate = 0.001

total_strain = 0.01
n_steps = 1000

dt = 0.01
delta_strain = 1e-5


# ============================================================
# 2. ELASTIC MATRIX
# ============================================================

C = np.zeros((6, 6))

lam = (
    E * nu
    /
    ((1 + nu) * (1 - 2 * nu))
)

mu = (
    E
    /
    (2 * (1 + nu))
)

C[0, 0] = lam + 2 * mu
C[1, 1] = lam + 2 * mu
C[2, 2] = lam + 2 * mu

C[0, 1] = lam
C[0, 2] = lam
C[1, 0] = lam
C[1, 2] = lam
C[2, 0] = lam
C[2, 1] = lam

C[3, 3] = mu
C[4, 4] = mu
C[5, 5] = mu


# ============================================================
# 3. VOIGT CONVERSION
# ============================================================

def tensor_to_voigt(A):

    return np.array([
        A[0, 0],
        A[1, 1],
        A[2, 2],
        2*A[0, 1],
        2*A[1, 2],
        2*A[0, 2]
    ])


def voigt_to_tensor(v):

    A = np.zeros((3, 3))

    A[0, 0] = v[0]
    A[1, 1] = v[1]
    A[2, 2] = v[2]

    A[0, 1] = v[3] / 2
    A[1, 0] = v[3] / 2

    A[1, 2] = v[4] / 2
    A[2, 1] = v[4] / 2

    A[0, 2] = v[5] / 2
    A[2, 0] = v[5] / 2

    return A


# ============================================================
# 4. HEX8 ELEMENT
# ============================================================

L = 0.018

nodes = np.array([

    [0.0, 0.0, 0.0],
    [L,   0.0, 0.0],
    [L,   L,   0.0],
    [0.0, L,   0.0],

    [0.0, 0.0, L],
    [L,   0.0, L],
    [L,   L,   L],
    [0.0, L,   L]

])


# ============================================================
# 5. HEX8 SHAPE FUNCTIONS
# ============================================================

def shape_function_derivatives(
    xi,
    eta,
    zeta
):

    dN = np.zeros((8, 3))

    dN[0] = [
        -0.125*(1-eta)*(1-zeta),
        -0.125*(1-xi)*(1-zeta),
        -0.125*(1-xi)*(1-eta)
    ]

    dN[1] = [
         0.125*(1-eta)*(1-zeta),
        -0.125*(1+xi)*(1-zeta),
        -0.125*(1+xi)*(1-eta)
    ]

    dN[2] = [
         0.125*(1+eta)*(1-zeta),
         0.125*(1+xi)*(1-zeta),
        -0.125*(1+xi)*(1+eta)
    ]

    dN[3] = [
        -0.125*(1+eta)*(1-zeta),
         0.125*(1-xi)*(1-zeta),
        -0.125*(1-xi)*(1+eta)
    ]

    dN[4] = [
        -0.125*(1-eta)*(1+zeta),
        -0.125*(1-xi)*(1+zeta),
         0.125*(1-xi)*(1-eta)
    ]

    dN[5] = [
         0.125*(1-eta)*(1+zeta),
        -0.125*(1+xi)*(1+zeta),
         0.125*(1+xi)*(1-eta)
    ]

    dN[6] = [
         0.125*(1+eta)*(1+zeta),
         0.125*(1+xi)*(1+zeta),
         0.125*(1+xi)*(1+eta)
    ]

    dN[7] = [
        -0.125*(1+eta)*(1+zeta),
         0.125*(1-xi)*(1+zeta),
         0.125*(1-xi)*(1+eta)
    ]

    return dN


# ============================================================
# 6. FE B MATRIX
# ============================================================

def calculate_B(node_coordinates):

    dN = shape_function_derivatives(
        0.0,
        0.0,
        0.0
    )

    J = dN.T @ node_coordinates

    detJ = np.linalg.det(J)

    if detJ <= 0:

        raise ValueError(
            "Negative Jacobian."
        )

    dN_dx = dN @ np.linalg.inv(J)

    B = np.zeros((6, 24))

    for a in range(8):

        i = 3*a

        dNx = dN_dx[a, 0]
        dNy = dN_dx[a, 1]
        dNz = dN_dx[a, 2]

        B[0, i] = dNx

        B[1, i+1] = dNy

        B[2, i+2] = dNz

        B[3, i] = dNy
        B[3, i+1] = dNx

        B[4, i+1] = dNz
        B[4, i+2] = dNy

        B[5, i] = dNz
        B[5, i+2] = dNx

    return B, J, detJ


B, J, detJ = calculate_B(nodes)


print("\n==============================")
print("FE ELEMENT CHECK")
print("==============================")

print("B shape =", B.shape)
print("detJ =", detJ)


# ============================================================
# 7. GRAIN 0 ORIENTATION
# ============================================================

grain_id = 0

R_grain = np.array(
    grain_orientation[grain_id]
)


# ============================================================
# 8. ROTATE SLIP SYSTEMS
# ============================================================

rotated_slip_systems = []

for n_crystal, s_crystal in slip_systems:

    n = R_grain @ n_crystal

    s = R_grain @ s_crystal

    n /= np.linalg.norm(n)

    s /= np.linalg.norm(s)

    rotated_slip_systems.append(
        (n, s)
    )


# ============================================================
# 9. SCHMID TENSORS
# ============================================================

P = np.zeros(
    (12, 3, 3)
)

for a in range(12):

    n = rotated_slip_systems[a][0]
    s = rotated_slip_systems[a][1]

    P[a] = 0.5 * (

        np.outer(s, n)
        +
        np.outer(n, s)

    )


# ============================================================
# 10. RSS
# ============================================================

def calculate_tau(stress):

    tau = np.zeros(12)

    for a in range(12):

        n = rotated_slip_systems[a][0]
        s = rotated_slip_systems[a][1]

        tau[a] = s @ stress @ n

    return tau


# ============================================================
# 11. SLIP RATE
# ============================================================

def calculate_slip_rate(
    tau,
    resistance
):

    ratio = (
        np.abs(tau)
        /
        resistance
    )

    return (

        gamma_dot_0
        *
        ratio**rate_exponent
        *
        np.sign(tau)

    )


# ============================================================
# 12. STATE VARIABLES
# ============================================================

plastic_strain = np.zeros(
    (3, 3)
)

slip_resistance = np.full(
    12,
    tau0
)

accumulated_slip = np.zeros(
    12
)


# ============================================================
# 13. HISTORY
# ============================================================

strain_history = []

stress_history = []

plastic_history = []

slip_history = []

crss_history = []

rss_history = []

residual_history = []


# ============================================================
# 14. PREVIOUS SOLUTION VECTOR
#
# THIS IS IMPORTANT.
#
# We retain the complete previous nonlinear solution as the
# initial guess for the next increment.
# ============================================================

x_previous = np.zeros(17)


# ============================================================
# 15. LOAD LOOP
# ============================================================

for step in range(n_steps):


    eps_zz = (
        (step + 1)
        *
        delta_strain
    )


    # --------------------------------------------------------
    # Initial guess from previous increment
    # --------------------------------------------------------

    x0 = x_previous.copy()


    if step == 0:

        x0[0] = -nu * eps_zz

        x0[1] = -nu * eps_zz


    # ========================================================
    # LOCAL RESIDUAL
    # ========================================================

    def local_residual(x):

        eps_xx = x[0]
        eps_yy = x[1]

        eps_xy = x[2]
        eps_yz = x[3]
        eps_xz = x[4]

        dg = x[5:17]


        # ----------------------------------------------------
        # Total strain tensor
        # ----------------------------------------------------

        total_eps = np.array([

            [
                eps_xx,
                eps_xy,
                eps_xz
            ],

            [
                eps_xy,
                eps_yy,
                eps_yz
            ],

            [
                eps_xz,
                eps_yz,
                eps_zz
            ]

        ])


        # ----------------------------------------------------
        # Plastic increment
        # ----------------------------------------------------

        plastic_increment = np.zeros(
            (3, 3)
        )

        for a in range(12):

            plastic_increment += (
                dg[a] * P[a]
            )


        plastic_new = (
            plastic_strain
            +
            plastic_increment
        )


        # ----------------------------------------------------
        # Elastic strain
        # ----------------------------------------------------

        elastic_eps = (
            total_eps
            -
            plastic_new
        )


        # ----------------------------------------------------
        # Stress
        # ----------------------------------------------------

        stress = voigt_to_tensor(

            C
            @
            tensor_to_voigt(
                elastic_eps
            )

        )


        # ----------------------------------------------------
        # RSS
        # ----------------------------------------------------

        tau = calculate_tau(
            stress
        )


        # ----------------------------------------------------
        # Hardening
        # ----------------------------------------------------

        total_slip_increment = np.sum(
            np.abs(dg)
        )


        resistance_new = (

            slip_resistance

            +

            h0
            *
            (
                1.0
                -
                slip_resistance
                /
                tau_sat
            )
            *
            total_slip_increment

        )


        resistance_new = np.maximum(
            resistance_new,
            tau0
        )


        # ----------------------------------------------------
        # Slip rate
        # ----------------------------------------------------

        gamma_dot = calculate_slip_rate(

            tau,
            resistance_new

        )


        # ----------------------------------------------------
        # Flow residual
        # ----------------------------------------------------

        flow_residual = (

            dg
            -
            dt * gamma_dot

        )


        # ----------------------------------------------------
        # Stress-free lateral/shear conditions
        # ----------------------------------------------------

        stress_residual = np.array([

            stress[0, 0],
            stress[1, 1],
            stress[0, 1],
            stress[1, 2],
            stress[0, 2]

        ])


        # ----------------------------------------------------
        # Scaling
        # ----------------------------------------------------

        stress_scale = E

        slip_scale = max(
            gamma_dot_0 * dt,
            1e-8
        )


        return np.concatenate([

            stress_residual
            /
            stress_scale,

            flow_residual
            /
            slip_scale

        ])


    # ========================================================
    # SOLVE LOCAL CP PROBLEM
    # ========================================================

    solution = least_squares(

        local_residual,

        x0,

        max_nfev=2000,

        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11

    )


    if not solution.success:

        raise RuntimeError(

            "Local CP solver failed at "
            f"step {step+1}: "
            +
            solution.message

        )


    # --------------------------------------------------------
    # Save complete solution
    # --------------------------------------------------------

    x_previous = solution.x.copy()


    # --------------------------------------------------------
    # Extract slip increments
    # --------------------------------------------------------

    dg = solution.x[5:17]


    # --------------------------------------------------------
    # Reconstruct total strain
    # --------------------------------------------------------

    total_eps = np.array([

        [
            solution.x[0],
            solution.x[2],
            solution.x[4]
        ],

        [
            solution.x[2],
            solution.x[1],
            solution.x[3]
        ],

        [
            solution.x[4],
            solution.x[3],
            eps_zz
        ]

    ])


    # --------------------------------------------------------
    # Plastic increment
    # --------------------------------------------------------

    plastic_increment = np.zeros(
        (3, 3)
    )

    for a in range(12):

        plastic_increment += (
            dg[a] * P[a]
        )


    # --------------------------------------------------------
    # Update plastic strain
    # --------------------------------------------------------

    plastic_strain += (
        plastic_increment
    )


    # --------------------------------------------------------
    # Update accumulated slip
    # --------------------------------------------------------

    accumulated_slip += np.abs(dg)


    # --------------------------------------------------------
    # Update CRSS
    # --------------------------------------------------------

    total_slip_increment = np.sum(
        np.abs(dg)
    )


    slip_resistance += (

        h0
        *
        (
            1.0
            -
            slip_resistance
            /
            tau_sat
        )
        *
        total_slip_increment

    )


    slip_resistance = np.maximum(
        slip_resistance,
        tau0
    )


    # --------------------------------------------------------
    # Calculate final stress
    # --------------------------------------------------------

    elastic_eps = (
        total_eps
        -
        plastic_strain
    )


    stress = voigt_to_tensor(

        C
        @
        tensor_to_voigt(
            elastic_eps
        )

    )


    tau = calculate_tau(
        stress
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    strain_history.append(
        eps_zz
    )

    stress_history.append(
        stress[2, 2]
    )

    plastic_history.append(
        plastic_strain[2, 2]
    )

    slip_history.append(
        np.sum(
            accumulated_slip
        )
    )

    crss_history.append(
        np.mean(
            slip_resistance
        )
    )

    rss_history.append(
        np.max(
            np.abs(tau)
        )
    )

    residual_history.append(
        np.linalg.norm(
            solution.fun
        )
    )


# ============================================================
# 16. CONVERT ARRAYS
# ============================================================

strain_history = np.array(
    strain_history
)

stress_history = np.array(
    stress_history
)

plastic_history = np.array(
    plastic_history
)

slip_history = np.array(
    slip_history
)

crss_history = np.array(
    crss_history
)

rss_history = np.array(
    rss_history
)

residual_history = np.array(
    residual_history
)


# ============================================================
# 17. FINAL RESULT
# ============================================================

print("\n==============================")
print("CORRECTED FE–CP RESULTS")
print("==============================")

print(
    "Converged increments:",
    len(strain_history),
    "/",
    n_steps
)

print(
    "Final strain:",
    strain_history[-1]
)

print(
    "Final axial stress:",
    stress_history[-1],
    "MPa"
)

print(
    "Final plastic strain:",
    plastic_history[-1]
)

print(
    "Final accumulated slip:",
    slip_history[-1]
)

print(
    "Final mean CRSS:",
    crss_history[-1],
    "MPa"
)

print(
    "Maximum RSS:",
    np.max(rss_history),
    "MPa"
)

print(
    "Maximum residual:",
    np.max(residual_history)
)


# ============================================================
# 18. FINAL STRESS STATE
# ============================================================

print("\n==============================")
print("FINAL STRESS STATE")
print("==============================")

print(
    "sigma_xx =",
    stress[0, 0],
    "MPa"
)

print(
    "sigma_yy =",
    stress[1, 1],
    "MPa"
)

print(
    "sigma_zz =",
    stress[2, 2],
    "MPa"
)

print(
    "sigma_xy =",
    stress[0, 1],
    "MPa"
)

print(
    "sigma_xz =",
    stress[0, 2],
    "MPa"
)

print(
    "sigma_yz =",
    stress[1, 2],
    "MPa"
)


# ============================================================
# 19. COMPARE AGAINST VERIFIED RESULT
# ============================================================

reference_stress = 395.90632217011205

difference = (
    stress_history[-1]
    -
    reference_stress
)

error = (
    abs(difference)
    /
    reference_stress
    *
    100
)

print("\n==============================")
print("VERIFICATION")
print("==============================")

print(
    "Reference grain-0 CP stress =",
    reference_stress,
    "MPa"
)

print(
    "FE–CP stress =",
    stress_history[-1],
    "MPa"
)

print(
    "Difference =",
    difference,
    "MPa"
)

print(
    "Relative difference =",
    error,
    "%"
)


# ============================================================
# 20. STRESS-STRAIN CURVE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(

    strain_history * 100,

    stress_history,

    linewidth=2,

    label="Corrected FE–CP"

)

plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Axial stress (MPa)",
    fontsize=14
)

plt.title(
    "One-element FE–CPFEM verification",
    fontsize=15
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 21. CONVERGENCE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.semilogy(

    strain_history * 100,

    residual_history,

    linewidth=2

)

plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Local CP residual norm",
    fontsize=14
)

plt.title(
    "Local CP convergence",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()

# ================================================================
# MODULE 5B-OPTIMIZED
# 3D FE–CPFEM SOLVER FOR 316L
#
# Main optimizations:
#   1. Precompute element DOFs
#   2. Precompute free/prescribed DOFs
#   3. Factorize K_ff only once
#   4. Use grain-level Schmid tensors
#   5. Replace scipy least_squares with local Newton solver
#   6. Avoid dictionaries inside the hot element loop
#   7. Store trial CP states in numerical arrays
#   8. Vectorize the 12 slip-system calculations
#
# Constitutive formulation is kept consistent with v01.
# ================================================================

import numpy as np
import time

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


# ================================================================
# 0. USER SETTINGS
# ================================================================

E = 193000.0
NU = 0.29

TAU0 = 150.0
TAU_SAT = 350.0
HARDENING_MODULUS = 5000.0

GAMMA0 = 1.0e-3
RATE_EXPONENT = 20.0
STRAIN_RATE = 1.0e-3

TOTAL_STRAIN = 0.01

INITIAL_STRAIN_STEP = 5.0e-4
MIN_STRAIN_STEP = 5.0e-5
MAX_STRAIN_STEP = 1.0e-3

MAX_GLOBAL_ITER = 30
GLOBAL_TOL = 1.0e-8

GLOBAL_DAMPING = 1.0

# Local Newton parameters
CP_MAX_ITER = 30
CP_TOL = 1.0e-9
CP_LINESEARCH_MIN = 1.0e-4

# Debug checks can be turned off during production
DEBUG_CP = False
DEBUG_GLOBAL = True


# ================================================================
# 1. LOAD FE MESH
# ================================================================

print()
print("=" * 70)
print("MODULE 5B-OPTIMIZED")
print("=" * 70)

data = np.load(
    "316L_FE_mesh_20cube.npz"
)

nodes = data["nodes"]
elements = data["elements"]

element_grain = data["element_grain"]
grain_orientation = data["grain_orientation"]

Lx, Ly, Lz = data["dimensions"]

n_nodes = len(nodes)
n_elements = len(elements)

NDOF = 3 * n_nodes

print("Nodes    =", n_nodes)
print("Elements =", n_elements)
print("DOFs     =", NDOF)
print("Grains   =", len(grain_orientation))


# ================================================================
# 2. ELASTIC CONSTITUTIVE MATRIX
# ================================================================

lam = (
    E * NU
    /
    ((1.0 + NU) * (1.0 - 2.0 * NU))
)

mu = (
    E
    /
    (2.0 * (1.0 + NU))
)

C_ELASTIC = np.zeros((6, 6))

C_ELASTIC[0, 0] = lam + 2.0 * mu
C_ELASTIC[1, 1] = lam + 2.0 * mu
C_ELASTIC[2, 2] = lam + 2.0 * mu

C_ELASTIC[0, 1] = lam
C_ELASTIC[0, 2] = lam
C_ELASTIC[1, 0] = lam
C_ELASTIC[1, 2] = lam
C_ELASTIC[2, 0] = lam
C_ELASTIC[2, 1] = lam

C_ELASTIC[3, 3] = mu
C_ELASTIC[4, 4] = mu
C_ELASTIC[5, 5] = mu


# ================================================================
# 3. VOIGT / TENSOR CONVERSION
#
# Convention:
#
# [xx, yy, zz, xy, yz, xz]
#
# Engineering shear strain is used in the FE formulation.
# ================================================================

def tensor_to_voigt(A):

    return np.array([
        A[0, 0],
        A[1, 1],
        A[2, 2],
        2.0 * A[0, 1],
        2.0 * A[1, 2],
        2.0 * A[0, 2]
    ])


def voigt_to_tensor(v):

    A = np.zeros((3, 3))

    A[0, 0] = v[0]
    A[1, 1] = v[1]
    A[2, 2] = v[2]

    A[0, 1] = 0.5 * v[3]
    A[1, 0] = 0.5 * v[3]

    A[1, 2] = 0.5 * v[4]
    A[2, 1] = 0.5 * v[4]

    A[0, 2] = 0.5 * v[5]
    A[2, 0] = 0.5 * v[5]

    return A


# ================================================================
# 4. ELEMENT DOF CONNECTIVITY
#
# Compute ONCE.
# ================================================================

print()
print("Precomputing element DOFs...")

t0 = time.time()

element_dofs = np.empty(
    (n_elements, 24),
    dtype=np.int32
)

for e in range(n_elements):

    conn = elements[e]

    element_dofs[e, 0::3] = 3 * conn
    element_dofs[e, 1::3] = 3 * conn + 1
    element_dofs[e, 2::3] = 3 * conn + 2


print(
    f"Element DOF construction = "
    f"{time.time() - t0:.3f} s"
)


# ================================================================
# 5. BUILD HEX8 B MATRICES
#
# IMPORTANT:
# This uses the same one-B-per-element approach as your v01.
#
# For the current regular mesh all elements have identical geometry.
# Therefore B can be computed once and reused.
# ================================================================

# If B_all already exists from your previous Module 3/5 section,
# keep it.
#
# Otherwise the code below assumes B_all and element_volume have
# already been constructed.


if "B_all" not in globals():

    raise RuntimeError(
        "B_all is not available. "
        "Keep the FE B-matrix construction from Module 3/5A "
        "before running Module 5B-OPTIMIZED."
    )


# All elements are geometrically identical.
B0 = np.asarray(B_all[0], dtype=np.float64)

if not np.allclose(
    B_all,
    B0[None, :, :],
    rtol=1.0e-10,
    atol=1.0e-12
):
    print(
        "WARNING: B matrices are not identical. "
        "Keeping B_all."
    )
else:
    print(
        "All FE elements have identical B matrix."
    )

    B_all = np.broadcast_to(
        B0,
        (n_elements, 6, 24)
    )


# ================================================================
# 6. FCC SLIP SYSTEMS
#
# Use the same crystal slip-system construction from Module 4.
#
# We expect:
#
# crystal_slip_systems.shape = (12,3,3)
#
# Each matrix is the symmetric Schmid tensor.
# ================================================================

if "crystal_slip_systems" in globals():

    crystal_S = np.asarray(
        crystal_slip_systems,
        dtype=np.float64
    )

elif "slip_systems" in globals():

    crystal_S = np.asarray(
        slip_systems,
        dtype=np.float64
    )

else:

    raise RuntimeError(
        "FCC crystal slip systems are not available."
    )


if crystal_S.shape != (12, 3, 3):

    raise RuntimeError(
        f"Expected slip systems shape (12,3,3), "
        f"got {crystal_S.shape}"
    )


# ================================================================
# 7. BUILD GRAIN-LEVEL SCHMID TENSORS
#
# v01 rotates the 12 systems for all 8000 elements.
#
# However, elements belonging to the same grain have the same
# crystallographic orientation.
#
# Therefore calculate:
#
#       216 grains × 12 systems
#
# instead of:
#
#       8000 elements × 12 systems
#
# ================================================================

print()
print("Building grain-level Schmid tensors...")

t0 = time.time()

n_grains = len(grain_orientation)

grain_S = np.empty(
    (n_grains, 12, 3, 3),
    dtype=np.float64
)

for g in range(n_grains):

    Q = grain_orientation[g]

    for a in range(12):

        # Transform crystal tensor to sample frame
        S = Q @ crystal_S[a] @ Q.T

        # Re-normalize symmetric Schmid tensor
        S = 0.5 * (S + S.T)

        grain_S[g, a] = S


# Element -> grain mapping
element_S = grain_S[element_grain]

print(
    f"Schmid tensor construction = "
    f"{time.time() - t0:.3f} s"
)

print(
    "element_S shape =",
    element_S.shape
)


# ================================================================
# 8. BOUNDARY CONDITIONS
# ================================================================

tol_geom = 1.0e-10

z_min = np.min(nodes[:, 2])
z_max = np.max(nodes[:, 2])

bottom_nodes = np.where(
    np.abs(nodes[:, 2] - z_min) < tol_geom
)[0]

top_nodes = np.where(
    np.abs(nodes[:, 2] - z_max) < tol_geom
)[0]


top_z_dofs = (
    3 * top_nodes + 2
).astype(np.int32)


bottom_fixed = (
    3 * bottom_nodes + 2
).tolist()


reference_x_node = bottom_nodes[0]

reference_y_node = bottom_nodes[
    np.argmax(nodes[bottom_nodes, 0])
]


bottom_fixed.append(
    3 * reference_x_node
)

bottom_fixed.append(
    3 * reference_y_node + 1
)


bottom_fixed = np.array(
    sorted(set(bottom_fixed)),
    dtype=np.int32
)


prescribed_dofs = np.unique(
    np.concatenate([
        bottom_fixed,
        top_z_dofs
    ])
)


free_mask = np.ones(
    NDOF,
    dtype=bool
)

free_mask[prescribed_dofs] = False

free_dofs = np.flatnonzero(
    free_mask
).astype(np.int32)


print()
print("Bottom nodes =", len(bottom_nodes))
print("Top nodes    =", len(top_nodes))
print("Fixed DOFs   =", len(bottom_fixed))
print("Prescribed   =", len(prescribed_dofs))
print("Free DOFs    =", len(free_dofs))


# ================================================================
# 9. GLOBAL ELASTIC STIFFNESS
#
# If K_global already exists, reuse it.
# Otherwise construct it.
# ================================================================

if "K_global" not in globals():

    from scipy.sparse import lil_matrix

    print()
    print("Assembling global stiffness...")

    t0 = time.time()

    K_global = lil_matrix(
        (NDOF, NDOF)
    )

    for e in range(n_elements):

        B = B_all[e]

        ke = (
            B.T
            @ C_ELASTIC
            @ B
            * element_volume
        )

        dofs = element_dofs[e]

        K_global[
            np.ix_(dofs, dofs)
        ] += ke

    K_global = K_global.tocsr()

    print(
        f"K assembly = "
        f"{time.time() - t0:.3f} s"
    )


# ================================================================
# 10. REDUCED GLOBAL MATRIX
#
# THIS IS A MAJOR SPEEDUP.
#
# v01 repeatedly performs:
#
#     K_global[np.ix_(free, free)]
#
# followed by spsolve().
#
# Here we do it ONCE.
# ================================================================

print()
print("Preparing global linear solver...")

t0 = time.time()

K_ff = K_global[
    np.ix_(
        free_dofs,
        free_dofs
    )
].tocsc()


K_solver = splu(
    K_ff
)


print(
    f"K_ff factorization = "
    f"{time.time() - t0:.3f} s"
)


# ================================================================
# 11. CP STATE ARRAYS
# ================================================================

U = np.zeros(
    NDOF,
    dtype=np.float64
)


plastic_strain_committed = np.zeros(
    (n_elements, 3, 3),
    dtype=np.float64
)


resistance_committed = np.full(
    (n_elements, 12),
    TAU0,
    dtype=np.float64
)


accumulated_slip_committed = np.zeros(
    n_elements,
    dtype=np.float64
)


cp_x_committed = np.zeros(
    (n_elements, 12),
    dtype=np.float64
)


# ================================================================
# 12. LOCAL CP STATE
#
# Vectorized replacement for the nested calculate_state()
# in v01.
# ================================================================

def cp_state(
    total_eps,
    plastic_old,
    resistance_old,
    systems,
    dgamma
):

    # ------------------------------------------------------------
    # Plastic strain increment
    #
    # Δεp = sum_a Δγ_a S_a
    # ------------------------------------------------------------

    plastic_increment = np.einsum(
        "a,aij->ij",
        dgamma,
        systems
    )

    plastic_new = (
        plastic_old
        +
        plastic_increment
    )


    # ------------------------------------------------------------
    # Elastic strain
    # ------------------------------------------------------------

    elastic_new = (
        total_eps
        -
        plastic_new
    )


    elastic_voigt = np.array([
        elastic_new[0, 0],
        elastic_new[1, 1],
        elastic_new[2, 2],
        2.0 * elastic_new[0, 1],
        2.0 * elastic_new[1, 2],
        2.0 * elastic_new[0, 2]
    ])


    # ------------------------------------------------------------
    # Stress
    # ------------------------------------------------------------

    stress_voigt = (
        C_ELASTIC
        @ elastic_voigt
    )


    stress = np.array([
        [stress_voigt[0],
         stress_voigt[3],
         stress_voigt[5]],

        [stress_voigt[3],
         stress_voigt[1],
         stress_voigt[4]],

        [stress_voigt[5],
         stress_voigt[4],
         stress_voigt[2]]
    ])


    # ------------------------------------------------------------
    # RSS
    #
    # RSS_a = sigma : S_a
    # ------------------------------------------------------------

    rss = np.einsum(
        "ij,aij->a",
        stress,
        systems
    )


    # ------------------------------------------------------------
    # Hardening
    # ------------------------------------------------------------

    abs_dgamma = np.abs(
        dgamma
    )

    hardening_factor = np.maximum(
        0.0,
        1.0 - resistance_old / TAU_SAT
    )

    resistance_new = (
        resistance_old
        +
        HARDENING_MODULUS
        *
        hardening_factor
        *
        abs_dgamma
    )

    resistance_new = np.minimum(
        resistance_new,
        TAU_SAT
    )


    # ------------------------------------------------------------
    # Slip rate
    # ------------------------------------------------------------

    resistance_safe = np.maximum(
        resistance_new,
        TAU0
    )

    ratio = (
        np.abs(rss)
        /
        resistance_safe
    )

    ratio = np.minimum(
        ratio,
        1.0e3
    )


    rates = (
        GAMMA0
        *
        ratio ** RATE_EXPONENT
    )


    rates = np.where(
        np.abs(rss) > 0.0,
        np.sign(rss) * rates,
        0.0
    )


    return (
        plastic_new,
        stress,
        stress_voigt,
        resistance_new,
        rss,
        rates
    )


# ================================================================
# 13. LOCAL CP RESIDUAL
# ================================================================

def cp_residual(
    dgamma,
    total_eps,
    plastic_old,
    resistance_old,
    systems,
    dt
):

    (
        plastic_new,
        stress,
        stress_voigt,
        resistance_new,
        rss,
        rates
    ) = cp_state(
        total_eps,
        plastic_old,
        resistance_old,
        systems,
        dgamma
    )


    residual = (
        dgamma / dt
        -
        rates
    )


    return residual / GAMMA0


# ================================================================
# 14. NUMERICAL JACOBIAN
#
# This is still numerical, but only 12 × 12.
#
# The important difference is that we control the Newton iteration
# directly instead of invoking a general-purpose least_squares
# optimizer for every element.
# ================================================================

def cp_jacobian(
    x,
    total_eps,
    plastic_old,
    resistance_old,
    systems,
    dt,
    r0
):

    n = 12

    J = np.empty(
        (n, n),
        dtype=np.float64
    )


    # Relative perturbation
    h_base = 1.0e-7


    for j in range(n):

        h = h_base * max(
            1.0,
            abs(x[j])
        )

        xp = x.copy()
        xm = x.copy()

        xp[j] += h
        xm[j] -= h

        rp = cp_residual(
            xp,
            total_eps,
            plastic_old,
            resistance_old,
            systems,
            dt
        )

        rm = cp_residual(
            xm,
            total_eps,
            plastic_old,
            resistance_old,
            systems,
            dt
        )

        J[:, j] = (
            rp - rm
        ) / (
            2.0 * h
        )


    return J


# ================================================================
# 15. LOCAL CP NEWTON SOLVER
# ================================================================

def solve_cp_local(
    total_eps,
    plastic_old,
    resistance_old,
    systems,
    x_initial,
    dt
):

    x = np.asarray(
        x_initial,
        dtype=np.float64
    ).copy()


    if not np.all(
        np.isfinite(x)
    ):
        x.fill(0.0)


    # ------------------------------------------------------------
    # Newton iterations
    # ------------------------------------------------------------

    for iteration in range(
        CP_MAX_ITER
    ):

        r = cp_residual(
            x,
            total_eps,
            plastic_old,
            resistance_old,
            systems,
            dt
        )


        rnorm = np.linalg.norm(
            r
        )


        if rnorm < CP_TOL:

            (
                plastic_new,
                stress,
                stress_voigt,
                resistance_new,
                rss,
                rates
            ) = cp_state(
                total_eps,
                plastic_old,
                resistance_old,
                systems,
                x
            )

            return (
                x,
                plastic_new,
                stress,
                stress_voigt,
                resistance_new,
                rss,
                rnorm,
                iteration + 1,
                True
            )


        # --------------------------------------------------------
        # Numerical Jacobian
        # --------------------------------------------------------

        J = cp_jacobian(
            x,
            total_eps,
            plastic_old,
            resistance_old,
            systems,
            dt,
            r
        )


        # --------------------------------------------------------
        # Newton correction
        # --------------------------------------------------------

        try:

            dx = np.linalg.solve(
                J,
                -r
            )

        except np.linalg.LinAlgError:

            return (
                x,
                None,
                None,
                None,
                None,
                None,
                rnorm,
                iteration + 1,
                False
            )


        if not np.all(
            np.isfinite(dx)
        ):

            return (
                x,
                None,
                None,
                None,
                None,
                None,
                rnorm,
                iteration + 1,
                False
            )


        # --------------------------------------------------------
        # Backtracking line search
        # --------------------------------------------------------

        alpha = 1.0

        accepted = False

        while (
            alpha
            >=
            CP_LINESEARCH_MIN
        ):

            x_trial = (
                x
                +
                alpha * dx
            )


            r_trial = cp_residual(
                x_trial,
                total_eps,
                plastic_old,
                resistance_old,
                systems,
                dt
            )


            r_trial_norm = np.linalg.norm(
                r_trial
            )


            if (
                np.isfinite(r_trial_norm)
                and
                r_trial_norm < rnorm
            ):

                x = x_trial
                accepted = True
                break


            alpha *= 0.5


        if not accepted:

            return (
                x,
                None,
                None,
                None,
                None,
                None,
                rnorm,
                iteration + 1,
                False
            )


    # ------------------------------------------------------------
    # Maximum iterations reached
    # ------------------------------------------------------------

    r = cp_residual(
        x,
        total_eps,
        plastic_old,
        resistance_old,
        systems,
        dt
    )

    rnorm = np.linalg.norm(
        r
    )


    if rnorm < CP_TOL:

        (
            plastic_new,
            stress,
            stress_voigt,
            resistance_new,
            rss,
            rates
        ) = cp_state(
            total_eps,
            plastic_old,
            resistance_old,
            systems,
            x
        )

        return (
            x,
            plastic_new,
            stress,
            stress_voigt,
            resistance_new,
            rss,
            rnorm,
            CP_MAX_ITER,
            True
        )


    return (
        x,
        None,
        None,
        None,
        None,
        None,
        rnorm,
        CP_MAX_ITER,
        False
    )


# ================================================================
# 16. CP MATERIAL-POINT TEST
# ================================================================

print()
print("=" * 70)
print("CP MATERIAL-POINT TEST")
print("=" * 70)


test_element = 0

test_strain = np.zeros(
    (3, 3)
)

test_strain[2, 2] = 1.0e-4

test_dt = (
    1.0e-4
    /
    STRAIN_RATE
)


(
    test_x,
    test_plastic,
    test_stress,
    test_stress_voigt,
    test_resistance,
    test_rss,
    test_residual,
    test_niter,
    test_success
) = solve_cp_local(
    test_strain,
    plastic_strain_committed[test_element],
    resistance_committed[test_element],
    element_S[test_element],
    cp_x_committed[test_element],
    test_dt
)


print(
    "Success     =",
    test_success
)

print(
    "Iterations  =",
    test_niter
)

print(
    "Residual    =",
    test_residual
)

if not test_success:

    raise RuntimeError(
        "Optimized CP material-point test failed."
    )

print(
    "Stress zz   =",
    test_stress[2, 2],
    "MPa"
)


# ================================================================
# 17. GLOBAL SOLVER STATE
# ================================================================

current_strain = 0.0

strain_step = (
    INITIAL_STRAIN_STEP
)

step = 0


global_strain_history = []
global_stress_history = []

step_history = []

cp_state_history = []


# ================================================================
# 18. PREALLOCATED TRIAL ARRAYS
#
# These replace the list of 8000 dictionaries used in v01.
# ================================================================

trial_stress = np.zeros(
    (n_elements, 3, 3),
    dtype=np.float64
)

trial_stress_voigt = np.zeros(
    (n_elements, 6),
    dtype=np.float64
)

trial_plastic = np.zeros(
    (n_elements, 3, 3),
    dtype=np.float64
)

trial_resistance = np.zeros(
    (n_elements, 12),
    dtype=np.float64
)

trial_accumulated = np.zeros(
    n_elements,
    dtype=np.float64
)

trial_slip = np.zeros(
    (n_elements, 12),
    dtype=np.float64
)

trial_rss = np.zeros(
    (n_elements, 12),
    dtype=np.float64
)

trial_cp_residual = np.zeros(
    n_elements,
    dtype=np.float64
)

trial_cp_iterations = np.zeros(
    n_elements,
    dtype=np.int32
)


# ================================================================
# 19. INTERNAL FORCE ARRAY
# ================================================================

F_internal = np.zeros(
    NDOF,
    dtype=np.float64
)


# ================================================================
# 20. GLOBAL FE–CPFEM SOLVER
# ================================================================

total_start = time.time()


while (
    current_strain
    <
    TOTAL_STRAIN - 1.0e-14
):

    step += 1


    target_strain = min(
        current_strain
        +
        strain_step,
        TOTAL_STRAIN
    )


    actual_increment = (
        target_strain
        -
        current_strain
    )


    target_displacement = (
        target_strain
        *
        Lz
    )


    cp_dt = (
        actual_increment
        /
        STRAIN_RATE
    )


    if DEBUG_GLOBAL:

        print()
        print("=" * 70)
        print(
            f"LOAD STEP {step}"
        )
        print("=" * 70)

        print(
            f"Current strain = "
            f"{current_strain:.8e}"
        )

        print(
            f"Target strain  = "
            f"{target_strain:.8e}"
        )

        print(
            f"Increment      = "
            f"{actual_increment:.8e}"
        )


    # ============================================================
    # BACKUP COMMITTED STATE
    # ============================================================

    U_backup = U.copy()

    plastic_backup = (
        plastic_strain_committed.copy()
    )

    resistance_backup = (
        resistance_committed.copy()
    )

    accumulated_backup = (
        accumulated_slip_committed.copy()
    )

    cp_x_backup = (
        cp_x_committed.copy()
    )


    # ============================================================
    # INITIAL GLOBAL DISPLACEMENT
    # ============================================================

    U_iter = U.copy()


    U_iter[
        top_z_dofs
    ] = target_displacement


    U_iter[
        bottom_fixed
    ] = 0.0


    converged = False


    # ============================================================
    # GLOBAL NEWTON LOOP
    # ============================================================

    for global_iter in range(
        1,
        MAX_GLOBAL_ITER + 1
    ):


        iteration_start = time.time()


        F_internal.fill(
            0.0
        )


        cp_failed = False

        failed_element = -1


        # ========================================================
        # ELEMENT LOOP
        # ========================================================

        for e in range(
            n_elements
        ):


            dofs = element_dofs[e]


            Ue = U_iter[
                dofs
            ]


            B = B_all[e]


            # ----------------------------------------------------
            # FE strain
            # ----------------------------------------------------

            strain_voigt = (
                B
                @
                Ue
            )


            strain_tensor = np.array([
                [strain_voigt[0],
                 0.5 * strain_voigt[3],
                 0.5 * strain_voigt[5]],

                [0.5 * strain_voigt[3],
                 strain_voigt[1],
                 0.5 * strain_voigt[4]],

                [0.5 * strain_voigt[5],
                 0.5 * strain_voigt[4],
                 strain_voigt[2]]
            ])


            # ----------------------------------------------------
            # Local CP solve
            # ----------------------------------------------------

            (
                dgamma,
                plastic_new,
                stress,
                stress_voigt,
                resistance_new,
                rss,
                cp_res,
                cp_niter,
                cp_success
            ) = solve_cp_local(

                strain_tensor,

                plastic_backup[e],

                resistance_backup[e],

                element_S[e],

                cp_x_backup[e],

                cp_dt
            )


            if not cp_success:

                cp_failed = True

                failed_element = e

                break


            # ----------------------------------------------------
            # Store numerical state
            # ----------------------------------------------------

            trial_slip[e] = dgamma

            trial_plastic[e] = (
                plastic_new
            )

            trial_stress[e] = (
                stress
            )

            trial_stress_voigt[e] = (
                stress_voigt
            )

            trial_resistance[e] = (
                resistance_new
            )

            trial_rss[e] = (
                rss
            )

            trial_cp_residual[e] = (
                cp_res
            )

            trial_cp_iterations[e] = (
                cp_niter
            )

            trial_accumulated[e] = (
                accumulated_backup[e]
                +
                np.sum(
                    np.abs(dgamma)
                )
            )


            # ----------------------------------------------------
            # Element internal force
            # ----------------------------------------------------

            fint_e = (
                B.T
                @
                stress_voigt
                *
                element_volume
            )


            F_internal[
                dofs
            ] += fint_e


        # ========================================================
        # CP FAILURE
        # ========================================================

        if cp_failed:

            print()
            print(
                "CP LOCAL SOLVER FAILED"
            )

            print(
                "Element =",
                failed_element
            )

            print(
                "Grain =",
                element_grain[
                    failed_element
                ]
            )

            converged = False

            break


        # ========================================================
        # GLOBAL RESIDUAL
        # ========================================================

        residual = (
            F_internal[
                free_dofs
            ]
        )


        residual_norm = np.linalg.norm(
            residual
        )


        force_scale = max(
            np.linalg.norm(
                F_internal
            ),
            1.0
        )


        relative_residual = (
            residual_norm
            /
            force_scale
        )


        if DEBUG_GLOBAL:

            print(
                f"Global iter {global_iter:2d}: "
                f"residual = "
                f"{relative_residual:.6e} "
                f"CP time = "
                f"{time.time()-iteration_start:.2f} s"
            )


        # ========================================================
        # GLOBAL CONVERGENCE
        # ========================================================

        if (
            relative_residual
            <
            GLOBAL_TOL
        ):

            converged = True

            break


        # ========================================================
        # GLOBAL DISPLACEMENT CORRECTION
        #
        # IMPORTANT:
        #
        # K_solver was factorized only once.
        # ========================================================

        try:

            delta_U_free = (
                K_solver.solve(
                    -residual
                )
            )

        except Exception as exc:

            print(
                "Global linear solve failed:"
            )

            print(exc)

            converged = False

            break


        if not np.all(
            np.isfinite(
                delta_U_free
            )
        ):

            print(
                "Non-finite global displacement correction."
            )

            converged = False

            break


        delta_U_free *= (
            GLOBAL_DAMPING
        )


        U_iter[
            free_dofs
        ] += delta_U_free


        # Re-enforce BCs

        U_iter[
            top_z_dofs
        ] = target_displacement

        U_iter[
            bottom_fixed
        ] = 0.0


    # ============================================================
    # LOAD STEP FAILURE
    # ============================================================

    if not converged:

        print()
        print("=" * 70)
        print(
            "LOAD STEP FAILED"
        )
        print("=" * 70)

        print(
            f"Step = {step}"
        )

        print(
            f"Target strain = "
            f"{target_strain:.8e}"
        )

        print(
            f"Last global iteration = "
            f"{global_iter}"
        )

        print(
            f"Last relative residual = "
            f"{relative_residual:.6e}"
        )


        # --------------------------------------------------------
        # Restore state
        # --------------------------------------------------------

        U = U_backup.copy()

        plastic_strain_committed = (
            plastic_backup.copy()
        )

        resistance_committed = (
            resistance_backup.copy()
        )

        accumulated_slip_committed = (
            accumulated_backup.copy()
        )

        cp_x_committed = (
            cp_x_backup.copy()
        )


        # --------------------------------------------------------
        # Reduce increment
        # --------------------------------------------------------

        strain_step *= 0.5


        print(
            "Reducing strain step to:",
            strain_step
        )


        if (
            strain_step
            <
            MIN_STRAIN_STEP
        ):

            raise RuntimeError(
                "FE–CPFEM failed repeatedly. "
                "Minimum strain step reached."
            )


        step -= 1

        continue


    # ============================================================
    # COMMIT GLOBAL STATE
    # ============================================================

    U = U_iter.copy()


    plastic_strain_committed = (
        trial_plastic.copy()
    )


    resistance_committed = (
        trial_resistance.copy()
    )


    accumulated_slip_committed = (
        trial_accumulated.copy()
    )


    cp_x_committed = (
        trial_slip.copy()
    )


    current_strain = (
        target_strain
    )


    # ============================================================
    # MACROSCOPIC RESPONSE
    #
    # All elements have the same volume.
    # Therefore the volume average is simply the arithmetic mean.
    # ============================================================

    sigma_tensor_average = np.mean(
        trial_stress,
        axis=0
    )


    sigma_macro = (
        sigma_tensor_average[2, 2]
    )


    max_rss = np.max(
        np.abs(
            trial_rss
        )
    )


    mean_crss = np.mean(
        trial_resistance
    )


    mean_plastic_strain = np.mean(
        np.linalg.norm(
            trial_plastic,
            axis=(1, 2)
        )
    )


    mean_accumulated_slip = np.mean(
        trial_accumulated
    )


    # ============================================================
    # HISTORY
    # ============================================================

    global_strain_history.append(
        current_strain
    )

    global_stress_history.append(
        sigma_macro
    )

    step_history.append(
        step
    )


    cp_state_history.append({

        "strain":
            current_strain,

        "stress":
            sigma_macro,

        "mean_crss":
            mean_crss,

        "max_rss":
            max_rss,

        "mean_plastic_strain":
            mean_plastic_strain,

        "mean_accumulated_slip":
            mean_accumulated_slip,

        "global_iterations":
            global_iter,

        "relative_residual":
            relative_residual

    })


    # ============================================================
    # STEP OUTPUT
    # ============================================================

    print()
    print("=" * 70)

    print(
        f"LOAD STEP {step} CONVERGED"
    )

    print("=" * 70)

    print(
        f"Strain             = "
        f"{current_strain:.8e}"
    )

    print(
        f"Stress             = "
        f"{sigma_macro:.6f} MPa"
    )

    print(
        f"Mean CRSS          = "
        f"{mean_crss:.6f} MPa"
    )

    print(
        f"Maximum RSS        = "
        f"{max_rss:.6f} MPa"
    )

    print(
        f"Mean plastic strain = "
        f"{mean_plastic_strain:.8e}"
    )

    print(
        f"Mean accumulated slip = "
        f"{mean_accumulated_slip:.8e}"
    )

    print(
        f"Global iterations  = "
        f"{global_iter}"
    )

    print(
        f"Relative residual  = "
        f"{relative_residual:.6e}"
    )


    # ============================================================
    # ADAPTIVE LOAD STEP
    # ============================================================

    if global_iter <= 4:

        strain_step = min(
            strain_step * 1.50,
            MAX_STRAIN_STEP
        )

    elif global_iter <= 8:

        strain_step = min(
            strain_step * 1.25,
            MAX_STRAIN_STEP
        )

    elif global_iter >= 15:

        strain_step = max(
            strain_step * 0.75,
            MIN_STRAIN_STEP
        )


# ================================================================
# 21. FINAL RESULTS
# ================================================================

total_time = (
    time.time()
    -
    total_start
)


print()
print()
print("=" * 70)
print("OPTIMIZED FE–CPFEM FINAL RESULTS")
print("=" * 70)


print(
    f"Final strain = "
    f"{current_strain:.8e}"
)


print(
    f"Final stress = "
    f"{global_stress_history[-1]:.6f} MPa"
)


print(
    f"Final mean CRSS = "
    f"{cp_state_history[-1]['mean_crss']:.6f} MPa"
)


print(
    f"Final maximum RSS = "
    f"{cp_state_history[-1]['max_rss']:.6f} MPa"
)


print(
    f"Final mean plastic strain = "
    f"{cp_state_history[-1]['mean_plastic_strain']:.8e}"
)


print(
    f"Final mean accumulated slip = "
    f"{cp_state_history[-1]['mean_accumulated_slip']:.8e}"
)


print(
    f"TOTAL SOLVER TIME = "
    f"{total_time:.3f} s"
)


# ================================================================
# 22. FINAL ELEMENT STATISTICS
# ================================================================

final_plastic = np.linalg.norm(
    plastic_strain_committed,
    axis=(1, 2)
)

final_crss = np.mean(
    resistance_committed,
    axis=1
)

final_accumulated = (
    accumulated_slip_committed
)

final_rss = np.max(
    np.abs(
        trial_rss
    ),
    axis=1
)

final_stress_zz = (
    trial_stress[:, 2, 2]
)


# ================================================================
# 23. SAVE RESULTS
# ================================================================

np.savez_compressed(

    "316L_FE_CPFEM_optimized_results.npz",

    nodes=nodes,

    elements=elements,

    element_grain=element_grain,

    element_dofs=element_dofs,

    U=U,

    global_strain_history=np.asarray(
        global_strain_history
    ),

    global_stress_history=np.asarray(
        global_stress_history
    ),

    step_history=np.asarray(
        step_history
    ),

    plastic_strain=
        plastic_strain_committed,

    resistance=
        resistance_committed,

    accumulated_slip=
        accumulated_slip_committed,

    slip_increment=
        cp_x_committed,

    final_plastic=
        final_plastic,

    final_crss=
        final_crss,

    final_accumulated=
        final_accumulated,

    final_rss=
        final_rss,

    final_stress_zz=
        final_stress_zz,

    sigma_tensor_average=
        sigma_tensor_average

)


print()
print(
    "Results saved as:"
)

print(
    "316L_FE_CPFEM_optimized_results.npz"
)

print("=" * 70)