
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np

def plot_trajectory_on_mesh(mesh, viaPoints, offset=None):
    """
    Visualize a 3D trajectory mapped onto a triangular mesh.

    The offset is applied only for visualization, along the
    approximate surface normals. The original via-points remain unchanged.
    """

    viaPoints = np.asarray(viaPoints, dtype=float)

    if viaPoints.ndim != 2 or viaPoints.shape[1] != 3:
        raise ValueError("viaPoints must have shape (N, 3).")

    if len(viaPoints) < 2:
        raise ValueError("At least two via-points are required.")

    # Set offset relative to mesh size if not specified
    if offset is None:
        offset = 0.005 * mesh.scale

    # Find nearest mesh vertices and their normals
    tree = cKDTree(mesh.vertices)
    _, vertex_indices = tree.query(viaPoints)

    normals = mesh.vertex_normals[vertex_indices]

    # Normalize normals
    normals /= np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-12
    )

    # Offset trajectory for clearer visualization
    display_points = viaPoints + offset * normals

    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Original mesh
    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        color="#A9C4D0",
        alpha=0.35,
        edgecolor="#526B78",
        linewidth=0.15,
        antialiased=True
    )

    # Mapped trajectory
    ax.plot(
        display_points[:, 0],
        display_points[:, 1],
        display_points[:, 2],
        color="#E63946",
        linewidth=2.5,
        label="Mapped trajectory",
        zorder=5
    )

    # Via-points
    ax.scatter(
        display_points[:, 0],
        display_points[:, 1],
        display_points[:, 2],
        color="#E63946",
        s=9,
        depthshade=False,
        label="Via-points"
    )

    # Start and end points
    ax.scatter(
        *display_points[0],
        color="#2A9D8F",
        s=65,
        edgecolors="black",
        label="Start",
        depthshade=False
    )

    ax.scatter(
        *display_points[-1],
        color="#F4A261",
        s=65,
        edgecolors="black",
        label="End",
        depthshade=False
    )

    # Labels and appearance
    ax.set_title("3D Trajectory Mapping", fontsize=15, pad=18)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend(loc="upper right")
    ax.set_box_aspect((
        np.ptp(mesh.vertices[:, 0]),
        np.ptp(mesh.vertices[:, 1]),
        np.ptp(mesh.vertices[:, 2])
    ))

    ax.view_init(elev=25, azim=45)
    ax.grid(False)

    plt.tight_layout()
    plt.show()
