from shapely.geometry import LineString,Point,Polygon,LinearRing,MultiPolygon
from collections import defaultdict
import numpy as np
from scipy.spatial import ConvexHull, KDTree
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


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

    def calculate_area(points):
        if len(points) < 3:
            return 0
        hull = ConvexHull(points)
        return hull.volume
    
    def ensureClockwise(points):
        area = 0.5 * np.sum(points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1])
        if area > 0:  # If the area is positive, points are counterclockwise
            points = points[::-1]  # Reverse to make it clockwise
        return points

    def denseClusterFilter(points, eps=0.01, min_samples=10):
        # DBSCAN clustering to identify dense regions
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        filtered_points = points[labels != -1]
        return filtered_points

    def rollBP(inner_boundary, outer_boundary):
        outer_tree = KDTree(outer_boundary)
        nearest_distance, nearest_outer_index = outer_tree.query(inner_boundary[0])
        inner_boundary = np.roll(inner_boundary, -0, axis=0)  # Inner boundary starts at index 0 already
        outer_boundary = np.roll(outer_boundary, -nearest_outer_index, axis=0)

        return inner_boundary, outer_boundary
    if len(boundary_uv_loops)>=2:
        boundary_areas = [calculate_area(loop_uv) for loop_uv in boundary_uv_loops]
        outer_boundary_index = np.argmax(boundary_areas)
        inner_boundary_indices = [i for i in range(len(boundary_areas)) if i != outer_boundary_index]
        outer_boundary_uv = np.squeeze(np.array(boundary_uv_loops[outer_boundary_index]))
        inner_boundary_uvs = [np.squeeze(np.array(boundary_uv_loops[i])) for i in inner_boundary_indices]
        inner_boundary_uvs=np.concatenate(inner_boundary_uvs)
        #inner_boundary_uvs = denseClusterFilter(inner_boundary_uvs)
        print(f"Outer boundary shape: {outer_boundary_uv.shape}")
        print(f"Inner boundary shape: {inner_boundary_uvs.shape}")
        innerBoundary = ensureClockwise(inner_boundary_uvs)
        outerBoundary = ensureClockwise(outer_boundary_uv)    
        innerBoundary=extractPoints(innerBoundary,((LineString(innerBoundary).length)/100))
        outerBoundary=extractPoints(outerBoundary,((LineString(outerBoundary).length)/100))
        outerBoundary=np.vstack([outerBoundary,outerBoundary[0]])
        innerBoundary=np.vstack([innerBoundary,innerBoundary[0]])
        innerBoundary, outerBoundary = rollBP(innerBoundary, outerBoundary)
        Hole=True
    else:
        #boundary_areas = [calculate_area(boundary_uv_loops)]
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
        plt.plot(inner_boundary_uvs[:, 0], inner_boundary_uvs[:, 1], 'b-', label='Inner Boundary')
        plt.legend()
        plt.title("Parameterized Mesh with Detected Boundaries")
        plt.show()
    if len(boundary_uv_loops) <2:
        return Hole,outerBoundary,None,boundary_loops
    else:
        return Hole,outerBoundary,innerBoundary,boundary_loops

def findOptimalBoundary(points2D, smoothing_range=np.arange(0.05, 10, 0.05), holes=False):
    optimal_smoothing_factor = 0.1
    mainBoundary = None

    if not holes:
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

    else:
        data2d = triangulation2D(points2D, smoothingFactor=optimal_smoothing_factor)
        mainBoundary = data2d.getBoundaryVertices()
        innerBoundary,outerBoundary= data2d.separateBoundary(mainBoundary)
        if (innerBoundary.size or outerBoundary.size)==0:
            data2d = triangulation2D(points2D, smoothingFactor=(optimal_smoothing_factor/10))
            mainBoundary = data2d.getBoundaryVertices()
        outerBoundary = np.vstack([outerBoundary, outerBoundary[0]])
        innerBoundary = np.vstack([innerBoundary, innerBoundary[0]])
        if not LinearRing(outerBoundary).is_ccw:
            outerBoundary = np.flipud(outerBoundary)
        if LinearRing(innerBoundary).is_ccw:
            innerBoundary = np.flipud(innerBoundary)
        outerPolygon = Polygon(outerBoundary)
        outerPolygon = outerPolygon.buffer(0)
        innerPolygon=Polygon(innerBoundary)
        innerPolygon=innerPolygon.buffer(0)
        if outerPolygon.is_valid and outerPolygon.is_simple:
            print("outer boundary is validated")
            if innerPolygon.is_valid and innerPolygon.is_simple and outerPolygon.contains(innerPolygon) : 
                print("Inner boundary is validated")
                return innerBoundary,outerBoundary,mainBoundary
            
        else:
            for factor in smoothing_range:
                data2d = triangulation2D(points2D, smoothingFactor=factor)
                mainBoundary = data2d.getBoundaryVertices()
                innerBoundary,outerBoundary= data2d.separateBoundary(mainBoundary)
                if outerBoundary.size == 0 or innerBoundary.size == 0:
                    print(f"No differentiable boundary detected with smoothing factor {factor}.")
                    continue
                outerBoundary = np.vstack([outerBoundary, outerBoundary[0]]) if outerBoundary.size > 0 else outerBoundary
                innerBoundary = np.vstack([innerBoundary, innerBoundary[0]]) if innerBoundary.size > 0 else innerBoundary

                if outerBoundary.size > 0 and not LinearRing(outerBoundary).is_ccw:
                    outerBoundary = np.flipud(outerBoundary)
                if innerBoundary.size > 0 and LinearRing(innerBoundary).is_ccw:
                    innerBoundary = np.flipud(innerBoundary)
                outerPolygon = Polygon(outerBoundary)
                innerPolygon = Polygon(innerBoundary)
                outerPolygon = outerPolygon.buffer(0)
                innerPolygon=innerPolygon.buffer(0)
                if outerPolygon.is_valid and outerPolygon.is_simple and outerPolygon.contains(innerPolygon):
                    return innerBoundary,outerBoundary, mainBoundary





def unitSquare(xmin,xmax,ymin,ymax):     #spacing=0.1
    cornerPoints=np.array(([xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]))
    x, y = zip(*cornerPoints)
    sq_bl=np.min(x),np.min(y);sq_br=np.max(x),np.min(y);sq_tl=np.min(x),np.max(y);sq_tr=np.max(x),np.max(y)
    sq_corners=[sq_tl,sq_bl,sq_br,sq_tr];sbottom=[sq_br,sq_bl];sright=[sq_tr,sq_br]
    sleft=[sq_bl,sq_tl];stop=[sq_tl,sq_tr];sq_segments=stop,sright,sbottom,sleft
    return sq_corners,sq_segments