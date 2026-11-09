from shapely import LineString, Polygon
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
class contourSegmentDetection:
    def __init__(self, uv,points, num_corners=4, curvature_threshold=15, min_distance=0.05):
        self.uv=uv
        self.points = np.array(points)
        self.num_corners = num_corners
        self.curvature_threshold = curvature_threshold
        self.min_distance = min_distance    #min is the distance between corners.
        self.detected_corners = None
        self.curvature = None
        self.sorted_corners = None
        self.edges = []
        self.contour_segments = []
        self.segments = {}
        self.centroid = None
    # def divide_boundary_equally(self):
    #     """
    #     Divide a set of boundary points into equal-length segments and extract equally spaced indices.
        
    #     Returns:
    #         list: Indices of equally spaced points along the boundary.
    #     """
    #     # Convert points into a LineString to compute cumulative distance
    #     boundary_line = LineString(self.points)
    #     total_length = boundary_line.length  # Total perimeter length
        
    #     # Compute the target segment lengths
    #     segment_length = total_length / self.num_corners
        
    #     # Generate equally spaced target points along the boundary
    #     equally_spaced_points = [boundary_line.interpolate(i * segment_length) for i in range(self.num_corners)]
    #     equally_spaced_points = [(p.x, p.y) for p in equally_spaced_points]
        
    #     # Use KDTree to find closest existing points in self.points
    #     tree = cKDTree(self.points)  # Build a KDTree for fast lookup
    #     distances, indices = tree.query(equally_spaced_points)  # Find closest indices

    #     return np.array(indices)
    def divide_boundary_equally(self):
        
        polygon = Polygon(self.points)
        min_rot_rect = polygon.minimum_rotated_rectangle
        centroid = polygon.centroid.coords[0]
        min_rot_rect_coords = list(min_rot_rect.exterior.coords)[:-1]  # Drop the closing point which is a repeat

        # Use KDTree to find closest original points to the corners of the minimum rotated rectangle
        tree = cKDTree(self.points)
        optimal_indices = []

        for corner in min_rot_rect_coords:
            # Find the k closest points to each corner
            _, indices = tree.query(corner, k=3)

            # Determine which of these k points is farthest from the centroid
            farthest_index = max(indices, key=lambda index: np.linalg.norm(np.array(self.points[index]) - np.array(centroid)))
            optimal_indices.append(farthest_index)
        return np.array(optimal_indices)
    
    def computeCurvature(self):
        n = len(self.points)
        curvature = np.zeros(n)
        for i in range(1, n-1):
            p_prev = self.points[i-1]
            p_curr = self.points[i]
            p_next = self.points[i+1]

            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angle = np.abs(np.degrees(angle))

            if angle > 180:
                angle = 360 - angle

            curvature[i] = angle
        self.curvature = curvature

    def detectCorners(self):
        self.computeCurvature()
        corners = np.where(self.curvature > self.curvature_threshold)[0]
        return corners, self.curvature

    def highCurvatureCornerSelection(self, corners, num_corners, min_distance):
        sorted_corners = sorted(corners, key=lambda c: self.curvature[c], reverse=True)
        selected_corners = []
        for c in sorted_corners:
            if len(selected_corners) >= num_corners:
                break
            if all(np.linalg.norm(self.points[c] - self.points[sc]) > min_distance for sc in selected_corners):
                selected_corners.append(c)
        if len(selected_corners) < 4:
            print("Less than 4 corners have been detected (", len(selected_corners), "corners detected )")
        return selected_corners

    def sortCorners(self):
        cntrlPnts = self.divide_boundary_equally()
        corners_array=self.points[cntrlPnts]
        #[75, 54, 26, 97] for turtle best corners
        # corners_array = self.points[self.detected_corners] [48, 28, 99, 74] 2d point 
        # 3d points indices 
        # corners_arraymain=self.uv[np.array([5015, 1320, 6381, 6416])] #kittenface cornerindices
        # corners_arraymain=self.uv[np.array([ 948,  989,  927, 2968])] #turtle
        # corners_arraymain=self.uv[np.array([  11,    7, 1027, 1120])] #foursided star
        # corners_arraymain=self.uv[np.array([  21,    3529, 4596, 800])] #maskholesfilled
        # tree = cKDTree(self.points)                                                                  
        # dists, nearest_indices = tree.query(corners_arraymain)
        # corners_array = self.points[nearest_indices]
        center = np.mean(corners_array, axis=0)
        angles = np.arctan2(corners_array[:, 1] - center[1], corners_array[:, 0] - center[0])
        self.sorted_corners = corners_array[np.argsort(angles)]
        

    def directionConsistence(self):
        total_angle = 0
        for i in range(1, len(self.points) - 1):
            p1 = self.points[i - 1]
            p2 = self.points[i]
            p3 = self.points[i + 1]
            v1 = p2 - p1
            v2 = p3 - p2
            angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
            total_angle += angle
        direction = np.sign(total_angle)
        if direction == -1:
            self.sorted_corners = self.sorted_corners[::-1]

    def findSegment(self, points, corner1, corner2):
        start_idx = np.where((points == corner1).all(axis=1))[0][0]
        end_idx = np.where((points == corner2).all(axis=1))[0][0]
        if start_idx < end_idx:
            return points[start_idx:end_idx + 1]
        else:
            return np.vstack((points[start_idx:], points[:end_idx + 1]))

    def classifySegments(self):
        centroid = np.mean(self.points, axis=0)
        self.centroid = centroid

        sorted_corners = self.sorted_corners
        self.edges = [(sorted_corners[i], sorted_corners[(i + 1) % len(sorted_corners)]) for i in range(len(sorted_corners))]

        for edge in self.edges:
            seg = self.findSegment(self.points, tuple(edge[0]), tuple(edge[1]))
            self.contour_segments.append(seg)

        right_segment = None
        max_intersect_x = -np.inf

        for segment in self.contour_segments:
            segment_points = np.array(segment)
            for i in range(len(segment_points) - 1):
                p1 = segment_points[i]
                p2 = segment_points[i + 1]
                if (p1[1] <= centroid[1] < p2[1]) or (p2[1] <= centroid[1] < p1[1]):
                    intersect_x = p1[0] + (centroid[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                    if intersect_x > max_intersect_x:
                        max_intersect_x = intersect_x
                        right_segment = segment

        if right_segment is not None:
            current_segment = right_segment
            self.segments["right"] = current_segment
            angle1 = np.arctan2(current_segment[ 0, 1]-self.centroid[1], current_segment[ 0, 0]-self.centroid[0])
            angle2 = np.arctan2(current_segment[-1, 1]-self.centroid[1], current_segment[-1, 0]-self.centroid[0])
            if angle1 > angle2:
                order = ["bottom", "left", "top"]
                clockwise = True
            else:
                order = ["top", "left", "bottom"]
                clockwise = False
            
            for name in order:
                last_point = current_segment[-1]
                next_segment = None
                for segment in self.contour_segments:
                    if np.array_equal(segment[0], last_point):
                        next_segment = segment
                        break
                if next_segment is not None:
                    self.segments[name] = next_segment
                    current_segment = next_segment
            
            if clockwise :
                self.segments['right'] = np.flipud(self.segments['right'])
                self.segments['bottom'] = np.flipud(self.segments['bottom'])
            else:
                self.segments['left'] = np.flipud(self.segments['left'])
                self.segments['top'] = np.flipud(self.segments['top'])

        return self.segments['top'], self.segments['right'], self.segments['bottom'], self.segments['left']

    def ensure_counterclockwise(self, points):
        s = 0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            s += (x2 - x1) * (y2 + y1)
        if s > 0:
            return points[::-1]
        return points

    def ensure_consistent_directions(self):
        for key in self.segments:
            self.segments[key] = self.ensure_counterclockwise(self.segments[key])

    def plot_results(self):
        colors = {'top': 'blue', 'left': 'orange', 'bottom': 'green', 'right': 'red'}
        for name, segment in self.segments.items():
            x, y = zip(*segment)
            plt.plot(x, y, '-', label=f'{name.capitalize()} Segment', color=colors[name])
        for i, corner in enumerate(self.sorted_corners):
            plt.scatter(*corner, s=100, label=f'Corner {i + 1}')
        plt.scatter(*self.centroid, s=100, label='Centroid', color='cyan')
        plt.title("Contour Segments")
        plt.axis('equal')
        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.grid(True)
        plt.legend()
        plt.show()


    
    def run(self):
        corners, _ = self.detectCorners()
        self.detected_corners = self.highCurvatureCornerSelection(corners, self.num_corners, self.min_distance)
        self.sortCorners()
        self.directionConsistence()
        self.classifySegments()
        self.ensure_consistent_directions()
       # self.plot_results()
        return self.sorted_corners, self.segments