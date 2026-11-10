import numpy as np
class triangularInterpolation:
    """ Given information of 3D geometry(v,vt,f)according to obj file and samplepoints,
    it gives 2d and 3d scan points suitable for that specific geometry.faces should contain both information of 3d and 2d faces"""
    def __init__(self,faces,vertices,vt,samplePoints):
        self.faces = faces
        self.vt = vt
        self.vertices = vertices
        self.samplePoints = samplePoints
        self.interpolated_points_3d =self.path3D() #, self.scan2D

    @staticmethod    
    def barycentricCoordinates2D(point, triangle_vertices_2d):
        x,y = (point)
        x1,y1 = triangle_vertices_2d[0]
        (x2, y2) = triangle_vertices_2d[1]
        (x3, y3) = triangle_vertices_2d[2]
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if denom !=0:
            u = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
            v = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
            w = 1 - u - v
            return (u, v, w)
        else:
            return None

    @staticmethod
    def isPointInside2DTriangle(bary_coords):
        u, v, w = bary_coords
        return (u >= 0) and (v >= 0) and (w >= 0) and (u + v + w <= 1)

    def findAssociated2Dtriangle(self, point):
        try:
            for face_index, face in enumerate(self.faces):
                vt_indices = [vt for _, vt in face if vt is not None]
                if len(vt_indices) == 3:
                    triangle_vts = [self.vt[vt_index] for vt_index in vt_indices]
                    bary_coords = self.barycentricCoordinates2D(point, triangle_vts)
                if bary_coords is not None and self.isPointInside2DTriangle(bary_coords):
                        return face_index, bary_coords  
            return None
        except:      # for using original mesh information incase flattening is done within gui using python command
            for face_index, face in enumerate(self.faces):
                if len(face) == 3:
                    triangle_vts = [self.vt[idx] for idx in face]  
                    bary_coords = self.barycentricCoordinates2D(point, triangle_vts)
                    if bary_coords is not None and self.isPointInside2DTriangle(bary_coords):
                        return face_index, bary_coords
            
    
    def get3dVertices(self, face_index):
        face = self.faces[face_index]
        try:
            vertex_indices = [v for v, _ in face]
            triangle_vertices_3d = [self.vertices[idx] for idx in vertex_indices]
        except:
            triangle_vertices_3d = [self.vertices[idx] for idx in face]
        return np.array(triangle_vertices_3d)
    
    def interpolate3D(self, bary_coords, triangle_vertices_3d):
        point_3d = np.dot(bary_coords, triangle_vertices_3d)  
        return point_3d
    def path3D(self):
        #scan2D=[]

        interpolatedPoints3D = []
        for index, point in enumerate(self.samplePoints):
            triangle2d = self.findAssociated2Dtriangle(point)
            if triangle2d is not None:
                #scan2D.append(point)
                face_index, bary_coords = triangle2d
                triangle_vertices_3d = self.get3dVertices(face_index)
                point_3d = self.interpolate3D(bary_coords, triangle_vertices_3d)
                interpolatedPoints3D.append(point_3d)
            else:
                print(f'no triangle found for point: {point}')
                pass
        
        return interpolatedPoints3D #, scan2D 
