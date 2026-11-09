from shapely import LineString, Polygon
import numpy as np
from meshHandler import MeshFlattener
from boundaryDetection import boundaryDectection3DPoints, unitSquare, extractPoints
from coonPatch import mapping2DUsingDistanceIntpl
from patterns import PatternsForUnitSquare
from contourSegments import contourSegmentDetection
from interpolation import triangularInterpolation 
class TrajectoryGeneration():
    def __init__(self,mesh,pathType,parameters):
        self.mesh=mesh #trimesh.load(mesh)
        self.mesh.vertices=self.mesh.vertices
        self.pathType=pathType
        self.parameters=parameters
        self.mainBoundary = None
        self.boundary3Dloop= None
        self.points_2d = None
        self.corners = None
        self.pointSpacing3D=None
        self.pointSpacing2D = None
        self.innerBoundary=None
        self.outerBoundary=None
        self.viaPoints=None
        self.viaPoints2D=None
        self.distortionRatio = []

    def parameterizeAndExtractBoundary(self):
        flattener = MeshFlattener(self.mesh)
        self.mesh = flattener.cleanMesh()
        uv=flattener.arapParameterization()
        uvPolygon=Polygon(uv)
        if uvPolygon.area > 0:
            print ("polygon area >0")
        else:
            print("area not valid")
        #uv=flattener.lscm_parameterization()
        max_attempts = 5
        attempts = 0
        while (uv.size == 0 or not flattener.is_uv_valid(uv)) and attempts < max_attempts:
            print(f"Attempt {attempts + 1}: UV not valid, trying to clean mesh.")
            self.mesh = flattener.cleanMesh()
            flattener = MeshFlattener(self.mesh)
            uv = flattener.arapParameterization()  
            #uv = flattener.lscm_parameterization()
            attempts += 1
        if uv.size == 0 or not flattener.is_uv_valid(uv):
            print("Max attempts reached, UV is still not valid via ARAP.")
            uv=flattener.lscm_parameterization()
        else:
            print(f"UV is valid after {attempts} attempt(s): {uv}")
        self.points_2d=uv
        # self.mesh.vertices=self.normalize_vertices(self.mesh.vertices,target_range=[0,100])
        # self.points_2d=self.normalize_vertices(self.points_2d,target_range=[0,100])
        self.outerBoundary,self.innerBoundary,self.boundary3Dloop=boundaryDectection3DPoints(self.mesh,self.points_2d)

        print('parameterization done.Boundary')    

    def mappingFromSquare(self):
            numOfSamples = 100  #samples used for each boundary segment in mapping process.
            xmin, xmax, ymin, ymax = 0, 1, 0, 1
            _, squareSegments = unitSquare(xmin, xmax, ymin, ymax)
            squareSegments = [extractPoints(S, (LineString(S).length) / numOfSamples) for S in squareSegments]
            patterns = PatternsForUnitSquare(xmin, xmax, ymin, ymax)
            #d_boundary = extractPoints(self.d_boundary, 0.02)
            canalyzer = contourSegmentDetection(self.points_2d,self.outerBoundary, num_corners=4, curvature_threshold=15, min_distance=0.05)
            corners, segments = canalyzer.run()
            self.corners = corners
            tContour, rContour, bContour, lContour = canalyzer.classifySegments()
            # segments=[segments['top'],segments['right'],segments['bottom'],segments['left']]
            # self.corners=np.array([tContour[0],rContour[0],bContour[0],lContour[0]])
            contours = [tContour, rContour, bContour, lContour]
            segments=[tContour, np.flipud(rContour), np.flipud(bContour), lContour]
            # segments = [tContour, rContour, bContour, lContour]
            segments=[extractPoints(s, (LineString(s).length) / numOfSamples) for s in segments]
            contours = [extractPoints(C, (LineString(C).length) / numOfSamples) for C in contours]
            return patterns, squareSegments, contours, self.corners,segments
    
    def computeTrajectory(self):
        # path_points = []
        # spiral_path = []

        if self.pathType == "MapCircularSpiral":
                self.parameterizeAndExtractBoundary()
                print("Executing MapCircularSpiral")
                numOfTurns= self.parameters[0] if len(self.parameters) > 0 else 3
                self.pointSpacing2D = self.parameters[1] if len(self.parameters) > 1 else 0.05
                numOfPoints=100
                maxRadius=0.45
                center=(0.5, 0.5)
                patterns, squareSegments, contours, self.corners, segments = self.mappingFromSquare()
                originalPoints = patterns.spiralPattern(center, maxRadius, numOfPoints, numOfTurns)
                # originalPoints = np.vstack([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
                mapped_points = mapping2DUsingDistanceIntpl(originalPoints, squareSegments, contours).mappedPoints
                self.viaPoints2D=np.array(mapped_points)
                # mapped2DPoints=extractPoints(mapped_points,self.pointSpacing2D)
                interpolator = triangularInterpolation(self.mesh.faces, self.mesh.vertices, self.points_2d, mapped_points)
                self.viaPoints3D= interpolator.path3D()
                viaPoints = self.viaPoints3D
                return viaPoints,mapped_points,self.outerBoundary,self.corners

        elif self.pathType == "MapSquareWave": 
                self.parameterizeAndExtractBoundary()
                print("Executing MapSquareWave")
                #self.pointSpacing=self.parameters[0] if len(self.parameters) > 0 else 0.08
                self.numOfPoints=self.parameters[0] if len(self.parameters) > 0 else 100
                axis = 'y' if len(self.parameters) > 1 and self.parameters[1]==0  else 'x'
                step_size = int(self.parameters[2]) if len(self.parameters) > 2 else 0.1
                center=(0.5, 0.5)
                patterns, squareSegments, contours, self.corners, segments = self.mappingFromSquare()
                # Example usage

                originalPoints = patterns.squareWaveGaiotto(period=2, num_points=69, 
                        amplitude_u=0.7, amplitude_v=0.7, 
                        offset_u=0.15, offset_v=0.15, 
                        axis='y')
                mapped2DPoints = mapping2DUsingDistanceIntpl(originalPoints, squareSegments, contours).mappedPoints
                self.viaPoints2D=np.array(mapped2DPoints)
                self.viaPoints2D=extractPoints(self.viaPoints2D,(LineString(self.viaPoints2D).length)/200)
                interpolator = triangularInterpolation(self.mesh.faces, self.mesh.vertices, self.points_2d, mapped2DPoints)
                viaPoints= interpolator.path3D()
                self.viaPoints3D= np.array(viaPoints)
                return self.viaPoints3D,mapped2DPoints,self.outerBoundary,self.corners