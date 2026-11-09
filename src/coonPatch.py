import numpy as np
from shapely import LineString, Polygon
from scipy.spatial import cKDTree, KDTree
from shapely.geometry import Point



class mapping2DUsingDistanceIntpl():
    """Given points to be mapped,source square segments,destination segments(Order of segments is: top,left,bottom,right)
    Returns:mapped points in irregular shape(Having Four sides)"""
    def __init__ (self,pointsinSquare,sourceSegments,destSegments):
        self.pointsinSquare = pointsinSquare
        self.sourceSegments = sourceSegments
        self.destSegments = destSegments
        top,right,bottom,left=self.destSegments
        top=self.ExtractPoints(top)
        right=self.ExtractPoints(right)
        bottom=self.ExtractPoints(bottom)
        left=self.ExtractPoints(left)
        self.destSegments=top,right,bottom,left
        self.destBoundary = np.vstack([top, right, bottom, left])
        self.sourceBoundary = np.vstack([self.sourceSegments[0],self.sourceSegments[1] , self.sourceSegments[2], self.sourceSegments[3]])
        # # self.mappedPoints = np.array([
        #     self.mapPointToIrregularShapeMVC(p) for p in self.pointsinSquare
        # ])
        self.mappedPoints = [self.mapPointToIrregularShapeCoonsPatch(point) for point in pointsinSquare]
        # self.mappedPoints = [self.mapPointToIrregularShape(point) for point in pointsinSquare]
    
    def calculatePointDistances(self,point):
        Distances = []  #;subSegments=[]
        for segment in self.sourceSegments:
            seg_tree = KDTree(segment)
            distances,_ = seg_tree.query(point, k=2)
            Distances.append(distances[0])
        return Distances  #,subSegments
            # line = LineString(np.array(segment))
            # point=Point(point);d=point.distance(line);
            #subseg=[indices[0],indices[1]]
            #subSegments.append(subseg)

    @staticmethod
    def interpolateShapes(points_A, points_B, alpha):
        return (1 - alpha) * points_A + alpha * points_B  
    def ExtractPoints(self,cntr, reverseOrder=False):
        line = LineString(cntr)
        #num_points = max(int(line.length / self.pointSpacing), 2)
        num_points=100
        distances = np.linspace(0, line.length, num_points)
        points = [line.interpolate(distance).coords[0] for distance in distances]
        points=np.array(points)
        return points[::-1] if reverseOrder else points
    
    
    def mapPointToIrregularShapeCoonsPatch(self, point):
        distances= self.calculatePointDistances(point)
        _, _, db, dl = distances    # Distance of point from bottom and left borders of unit square
        top, right, bottom, left = self.destSegments
        
        assert np.all(bottom[0]  == left[0])  , "Border segments ordered incorrectly"
        assert np.all(bottom[-1] == right[0]) , "Border segments ordered incorrectly"
        assert np.all(top[0]     == left[-1]) , "Border segments ordered incorrectly"
        assert np.all(top[-1]    == right[-1]), "Border segments ordered incorrectly"
        
        c0 = LineString(bottom)
        c1 = LineString(top)
        d0 = LineString(left)
        d1 = LineString(right)
        # c0 = LineString(left)
        # c1 = LineString(right)
        # d0 = LineString(top)
        # d1 = LineString(bottom)
        
        mappedPoint = mapping2DUsingDistanceIntpl.bilinearCoonsPatch(dl, db, c0, c1, d0, d1)
        return mappedPoint
    
    def mvc_weights(self, p, boundary):
        n = len(boundary)
        w = np.zeros(n)
        for i in range(n):
            pi = boundary[i]
            pi1 = boundary[(i + 1) % n]
            pi_1 = boundary[(i - 1) % n]

            vi = pi - p
            vi1 = pi1 - p
            vi_1 = pi_1 - p

            norm_vi = np.linalg.norm(vi)
            norm_vi1 = np.linalg.norm(vi1)

            alpha_i = np.arccos(
                np.clip(np.dot(vi, vi1) / (norm_vi * norm_vi1 + 1e-10), -1, 1)
            )

            tan_alpha_half = np.tan(alpha_i / 2.0)
            w[i] = tan_alpha_half / (norm_vi + 1e-10)

        w_sum = np.sum(w)
        w /= (w_sum + 1e-10)
        return w

    def mapPointToIrregularShapeMVC(self, point):
        # Compute MVC weights for point w.r.t. destination boundary
        weights = self.mvc_weights(point, self.sourceBoundary)

        # Interpolate mapped point in destination shape
        mapped_p = np.dot(weights, self.destBoundary)
        return mapped_p
    
    @staticmethod
    def bilinearCoonsPatch(s:float, t:float, c0:LineString, c1:LineString, d0:LineString, d1:LineString):
        c0_0 = np.array(c0.line_interpolate_point(0, normalized=True).coords[0])
        c0_s = np.array(c0.line_interpolate_point(s, normalized=True).coords[0])
        c0_1 = np.array(c0.line_interpolate_point(1, normalized=True).coords[0])
        
        c1_0 = np.array(c1.line_interpolate_point(0, normalized=True).coords[0])
        c1_s = np.array(c1.line_interpolate_point(s, normalized=True).coords[0])
        c1_1 = np.array(c1.line_interpolate_point(1, normalized=True).coords[0])
        
        d0_t = np.array(d0.line_interpolate_point(t, normalized=True).coords[0])
        
        d1_t = np.array(d1.line_interpolate_point(t, normalized=True).coords[0])
        
        Lc = (1-t)*c0_s + t*c1_s
        Ld = (1-s)*d0_t + s*d1_t
        B = c0_0*(1-s)*(1-t) + c0_1*s*(1-t) + c1_0*(1-s)*t + c1_1*s*t
        C = Lc + Ld - B
        
        return (C[0], C[1])