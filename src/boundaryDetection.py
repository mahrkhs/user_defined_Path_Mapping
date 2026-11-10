from shapely.geometry import LineString,Point,Polygon,LinearRing,MultiPolygon
from collections import defaultdict
import numpy as np
from scipy.spatial import ConvexHull, KDTree
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay


class triangulation2D:
    """For triangulation of 2D points.
    -points are 2D points.
    -smoothing factor is for 2d boundary smoothing.if set low,will detect small holes also if set highthen it will detect only large holes.
    """
    def __init__(self, points,smoothingFactor=0.01):
        self.points = np.array(points)
        self.tri = Delaunay(self.points)
        self.tri_vertex_indices = self.tri.simplices
        self.smoothingFactor=smoothingFactor
        self.computeBoundaryEdges()
        assert self.points.shape[0] > 3, "Need at least four points"
    def addEdge(self, edges, i, j):
        if (i, j) in edges or (j, i) in edges:
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            edges.remove((j, i))
            return
        edges.add((i, j))
    def computeBoundaryEdges(self):
        edges = set()
        for ia, ib, ic in self.tri_vertex_indices:
            pa = self.points[ia];pb = self.points[ib];pc = self.points[ic]
            a = np.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
            b = np.sqrt((pb[0] - pc[0])**2 + (pb[1] - pc[1])**2)
            c = np.sqrt((pc[0] - pa[0])**2 + (pc[1] - pa[1])**2)
            s = (a + b + c) / 2.0
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
            if circum_r < self.smoothingFactor:  
                self.addEdge(edges, ia, ib);self.addEdge(edges, ib, ic);self.addEdge(edges, ic, ia)
        self.boundary_edges = list(edges)

    def get_boundaryEdges(self):
        return self.boundary_edges
    
    def get_edges(self):
        return self.edge

    def tri_Vertex_indices(self):
        return self.tri_vertex_indices

    def get_2Dtriangle_vertices(self):
        triangle_vertices=[]
        for ia, ib, ic in self.tri_vertex_indices:
            triangle_vertices.append([self.points[ia],self.points[ib],self.points[ic]])
        return triangle_vertices
    
    def get_center(self):
        return [float(sum(l))/len(l) for l in zip(*self.points)]

    def getBoundaryVertices(self):
        boundary_points = []
        
        # thresholdDistance=0.01
        for i, j in self.boundary_edges:
            if tuple(self.points[i]) not in map(tuple, boundary_points):
                boundary_points.append(self.points[i])
            if tuple(self.points[j]) not in map(tuple, boundary_points):
                boundary_points.append(self.points[j])

        if len(boundary_points) != 0:
            ordered_points = [boundary_points[0]]
            remaining_points = list(boundary_points[1:])
            while remaining_points:
                last_point = ordered_points[-1]
                distances = np.linalg.norm(np.array(remaining_points) - last_point, axis=1)
                nearest_index = np.argmin(distances)
                ordered_points.append(remaining_points.pop(nearest_index))
                boundary=np.array(ordered_points)       

        return boundary

    def separateBoundary(self,boundary): #,thresholdDistance=0.055
        switchingFound=False
        boundary1=[];boundary2=[]
        innerBoundary=np.array([]);outerBoundary=np.array([])
        maxDist=0  
        for j in range(len(boundary)-1):
             currentPoint=np.array(boundary[j]) 
             nextPoint=np.array(boundary[j+1])
             distance = np.linalg.norm(currentPoint - nextPoint)
             if distance > maxDist:
                 maxDist = distance   
        for i in range(len(boundary)-1):
            currentPoint=np.array(boundary[i])
            if not switchingFound:
                nextPoint=np.array(boundary[i+1])
                distance = np.linalg.norm(currentPoint - nextPoint)
                if distance<(maxDist-(0.05*maxDist)):  # thresholdDistance
                    boundary1.append(currentPoint)
                else:
                    switchingFound=True
                    boundary1.append(currentPoint)
                    boundary2.extend(boundary[i+1:])
        # return boundary1,boundary2
        def close_loop(boundary):
            if len(boundary) > 0 and not np.array_equal(boundary[0], boundary[-1]):
                boundary.append(boundary[0])  # Make sure the boundary forms a closed loop
            return boundary
        boundary1 = close_loop(boundary1)
        boundary2 = close_loop(boundary2)        
        if Polygon(boundary1).contains(Polygon(boundary2)):
             outerBoundary=np.array(boundary1)
             innerBoundary=np.array(boundary2)
        elif Polygon(boundary2).contains(Polygon(boundary1)):
             outerBoundary=np.array(boundary2)
             innerBoundary=np.array(boundary1)
        else:
            print("no differentiable boundary,May be it does not have a hole or adjust smoothing factor in concavehull")
        
        @staticmethod
        def ensure_clockwise(points):
            # Calculate the signed area using the shoelace formula
            area = 0.5 * np.sum(points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1])
            if area > 0:  # If the area is positive, points are counterclockwise
                points = points[::-1]  # Reverse to make it clockwise
            return points

        innerBoundary = ensure_clockwise(innerBoundary)
        outerBoundary = ensure_clockwise(outerBoundary)

        return np.array(innerBoundary),np.array(outerBoundary)
    
def extractPoints(cntr,pointSpacing,reverseOrder=False):
        if type(cntr)!= LineString:
            line = LineString(cntr) 
        else:
            line=cntr
        num_points = max(int(line.length / pointSpacing), 2)
        distances = np.linspace(0, line.length, num_points)
        points = np.array([line.interpolate(distance).coords[0] for distance in distances])
        return points[::-1] if reverseOrder else points

def boundaryDectection3DPoints(mesh,uv_coords, visualize=False):
    edges = mesh.edges_unique
    faces = mesh.faces
    edge_count = {}
    Hole=False
    for face in faces:
        face_edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        
        for edge in face_edges:
            if edge in edge_count:
                edge_count[edge] += 1
            else:
                edge_count[edge] = 1

    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    boundary_edges = np.array(boundary_edges)

    if len(boundary_edges) == 0:
        print("No boundary edges found.")
        return None, None

    print(f"Found {len(boundary_edges)} boundary edge(s).")

    adjacency_list = defaultdict(list)
    for edge in boundary_edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])

    def find_loop(start_vertex, adjacency_list):
        current_vertex = start_vertex
        loop = [current_vertex]
        while True:
            neighbors = adjacency_list[current_vertex]
            for neighbor in neighbors:
                if len(loop) < 2 or neighbor != loop[-2]:
                    loop.append(neighbor)
                    current_vertex = neighbor
                    break
            if loop[0] == loop[-1]:
                break
        return loop

    boundary_loops = []
    visited = set()
    for edge in boundary_edges:
        for vertex in edge:
            if vertex not in visited:
                loop = find_loop(vertex, adjacency_list)
                boundary_loops.append(loop)
                visited.update(loop)

    if len(boundary_loops) == 0:
        print("No boundary loops found.")
        return None, None
    print(f"Found {len(boundary_loops)} boundary loop(s).")
    boundary_uv_loops = []
    for loop in boundary_loops:
        loop_uv = list([uv_coords[v] for v in loop])
        boundary_uv_loops.append(loop_uv)

    outer_boundary_uv = np.array([loop for loop in boundary_uv_loops[0]])
    print(f"Outer boundary shape: {outer_boundary_uv.shape}")
    #outerBoundary = ensureClockwise(outer_boundary_uv)
    outerBoundary = extractPoints(outer_boundary_uv, ((LineString(outer_boundary_uv).length)/100))
    outerBoundary=np.vstack([outerBoundary,outerBoundary[0]])

    if visualize:
        plt.figure()
        plt.triplot(uv_coords[:, 0], uv_coords[:, 1], mesh.faces, color='gray', alpha=0.5)
        plt.plot(outer_boundary_uv[:, 0], outer_boundary_uv[:, 1], 'r-', label='Outer Boundary')
        # for i, inner_boundary in enumerate(inner_boundary_uvs):
        # plt.plot(inner_boundary_uvs[:, 0], inner_boundary_uvs[:, 1], 'b-', label='Inner Boundary')
        plt.legend()
        plt.title("Parameterized Mesh with Detected Boundaries")
        plt.show()
    if len(boundary_uv_loops) <2:
        return outerBoundary,boundary_loops

def findOptimalBoundary(points2D, smoothing_range=np.arange(0.05, 10, 0.05), holes=False):
    optimal_smoothing_factor = 0.1
    mainBoundary = None

    # if not holes:
    data2d = triangulation2D(points2D, smoothingFactor=optimal_smoothing_factor)
    mainBoundary = data2d.getBoundaryVertices()
    polygon = Polygon(mainBoundary)
    print("setusing")
    if polygon.is_valid and polygon.is_simple:
        print("passingsetvalue")
        return mainBoundary
    else:
        for factor in smoothing_range:
            print("checking SFactor in range")
            data2d = triangulation2D(points2D, smoothingFactor=factor)
            mainBoundary = data2d.getBoundaryVertices()
            polygon = Polygon(mainBoundary)
            if polygon.is_valid and polygon.is_simple:
                return mainBoundary





def unitSquare(xmin,xmax,ymin,ymax):     #spacing=0.1
    cornerPoints=np.array(([xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]))
    x, y = zip(*cornerPoints)
    sq_bl=np.min(x),np.min(y);sq_br=np.max(x),np.min(y);sq_tl=np.min(x),np.max(y);sq_tr=np.max(x),np.max(y)
    sq_corners=[sq_tl,sq_bl,sq_br,sq_tr];sbottom=[sq_br,sq_bl];sright=[sq_tr,sq_br]
    sleft=[sq_bl,sq_tl];stop=[sq_tl,sq_tr];sq_segments=stop,sright,sbottom,sleft
    return sq_corners,sq_segments

