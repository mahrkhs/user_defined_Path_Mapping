import trimesh
import numpy as np
from mapping import TrajectoryGeneration
from plot import plot_trajectory_on_mesh

mesh=trimesh.load('meshes/lilium.obj')
pathType='MapSquareWave' #'MapCircularSpiral'
paramet=[]
initialize=TrajectoryGeneration(mesh,pathType,parameters=paramet)
final=initialize.computeTrajectory()
viaPoints,mapped2DPoints,outerBoundary,cornerPoints=final
viaPoints=np.array(viaPoints)
plot_trajectory_on_mesh(mesh, viaPoints,offset=0.25)