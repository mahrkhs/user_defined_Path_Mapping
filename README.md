A Python implementation of geometry-aware trajectory generation by mapping predefined 2D patterns onto 3D triangular mesh surfaces using As-Rigid-As-Possible (ARAP) parameterization.

This project implements a framework for transferring predefined 2D trajectory patterns onto non-planar 3D surfaces.

The approach first parameterizes a 3D triangular mesh into a 2D domain using ARAP. A desired trajectory is generated inside a unit square and mapped onto the parameterized surface through boundary correspondence and Coons patch interpolation. The resulting 2D points are then reconstructed onto the original 3D surface using barycentric coordinates. The framework enables geometry-aware path generation for robotic applications involving surface treatment, robotic drawing, inspection, painting and polishing.

<img width="1790" height="568" alt="image" src="https://github.com/user-attachments/assets/0f9c5201-2cbb-4f23-be4a-4bf78e00cda5" />



The framework consists of the following stages:

Mesh preprocessing: Clean the input triangular mesh, handle problematic faces and prepare its topology for parameterization.

ARAP parameterization: Flatten the 3D mesh into a 2D UV domain using harmonic initialization followed by ARAP optimization.

2D pattern generation: Generate predefined or user-defined trajectory patterns within a normalized unit square.

Boundary correspondence: Extract and segment the boundaries of the unit square and parameterized surface into four corresponding segments.

Pattern mapping: Transfer the 2D pattern onto the parameterized surface using boundary correspondence and Coons patch interpolation.

3D reconstruction: Identify the corresponding UV triangles and reconstruct the mapped points on the original 3D mesh using barycentric coordinates.

Visualization: Display the generated trajectory over the original 3D surface.
