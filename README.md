# Parameterization-Based 2D-to-3D Trajectory Mapping

A Python implementation for transferring predefined 2D path patterns onto 3D triangular mesh surfaces using **As-Rigid-As-Possible (ARAP) parameterization**.

## Overview

The project maps trajectories defined in a normalized 2D unit square onto non-planar 3D surfaces.

The input mesh is first parameterized into a 2D domain using ARAP. A trajectory generated in the unit square is then transferred to the parameterized surface through boundary correspondence and Coons patch interpolation. Finally, barycentric coordinates are used to reconstruct the mapped trajectory on the original 3D mesh.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0f9c5201-2cbb-4f23-be4a-4bf78e00cda5" width="100%" alt="2D-to-3D trajectory mapping"/>
</p>

## Method

1. **Mesh preprocessing:** Prepare the triangular mesh for parameterization.
2. **ARAP parameterization:** Flatten the 3D mesh into a 2D UV domain using harmonic initialization and ARAP optimization.
3. **Pattern generation:** Generate a predefined or user-defined trajectory inside the unit square.
4. **Boundary correspondence:** Divide the boundaries of the unit square and parameterized surface into four corresponding segments.
5. **Pattern transfer:** Map the trajectory using Coons patch interpolation.
6. **3D reconstruction:** Recover the mapped points on the original mesh using barycentric coordinates.

## Project Structure

```text
user_defined_Path_Mapping/
│
├── meshes/                  # Input 3D mesh models
│
├── src/
│   ├── __init__.py
│   ├── main.py              # Main execution script
│   ├── meshHandler.py       # Mesh loading and preprocessing
│   ├── boundaryDetection.py # Mesh boundary detection
│   ├── contourSegments.py   # Boundary segmentation
│   ├── coonPatch.py         # Coons patch interpolation
│   ├── interpolation.py     # Geometric interpolation
│   ├── mapping.py           # 2D-to-3D pattern mapping
│   └── patterns.py          # 2D trajectory generation
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Uses

Python · NumPy · SciPy · libigl · Trimesh · Shapely · Vedo · VTK

## Reference

This implementation is based on the following published work:

M. Mahrukh, G. Palli, M. Gambazza, and C. Melchiorri, "Robotic Path Projection from 2D Square to 3D Single Patch Surfaces Using ARAP Parameterization," *2025 25th International Conference on Control, Automation and Systems (ICCAS)*, pp. 321–326, 2025.

[https://doi.org/10.23919/ICCAS66577.2025.11301316](https://doi.org/10.23919/ICCAS66577.2025.11301316)
