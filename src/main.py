import trimesh
import numpy as np
from mapping import TrajectoryGeneration

mesh=trimesh.load('test_meshes/lilium.obj')
pathType='MapSquareWave'
initialize=TrajectoryGeneration(mesh,pathType,parameters=None)
final=initialize.computeTrajectory()
viaPoints,mapped2DPoints,outerBoundary,cornerPoints=final
viaPoints=np.array(viaPoints)



