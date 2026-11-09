import igl
import numpy as np
import vedo
import vtk
import trimesh


class MeshCleaner:
    def __init__(self, mesh):
        self.mesh = mesh

    def fix_normals(self):
        polydata = self.mesh.polydata()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()  
        normals.ComputeCellNormalsOn()   
        normals.SplittingOff()           
        normals.ConsistencyOn()          
        normals.AutoOrientNormalsOn()    
        normals.Update()

        self.mesh._update(normals.GetOutput())
        return self.mesh

    def non_manifold_faces(self, remove_all=True, remove_boundary=False, collapse_edges=False):
        if int(remove_all) + int(remove_boundary) + int(collapse_edges) > 1:
            raise ValueError("Select only one option")
        self.mesh.add_ids()  # Mark original point and cell ids
        nme = self.mesh.boundaries(boundary_edges=False, non_manifold_edges=True)
        nme_pids = nme.pointdata["PointID"]

        bnd_cids = []
        if remove_boundary:
            bnd_cids = self.mesh.boundaries(return_cell_ids=True)

        all_nf = []
        toremove = []
        faces = self.mesh.cells
        points = self.mesh.vertices
        new_points = np.array(points)
        
        line_list = nme.lines

        for e in vedo.utils.progressbar(line_list, delay=5, title="parsing faces"):
            ie0, ie1 = e
            ip0, ip1 = nme_pids[ie0], nme_pids[ie1]
            for i, f in enumerate(faces):
                if ip0 in f and ip1 in f:
                    all_nf.append(i)
                    if remove_all:
                        toremove.append(i)
                    elif remove_boundary:
                        if i in bnd_cids:
                            toremove.append(i)
                    elif collapse_edges:
                        m = (points[ip0] + points[ip1]) / 2
                        new_points[ip0] = m
                        new_points[ip1] = m

        if collapse_edges:
            self.mesh.vertices = new_points  
            self.mesh.clean()
        elif toremove:
            self.mesh.delete_cells(toremove)
        else:
            mark = np.zeros(self.mesh.ncells, dtype=np.uint8)
            mark[all_nf] = 1
            self.mesh.celldata["NonManifoldCell"] = mark

        return self.mesh

    def cleanMesh(self):
        print("Is the mesh watertight?", self.mesh.is_watertight)
        if self.mesh is not None:
            print("Mesh is loaded initially.")
            vedo_mesh = vedo.Mesh([self.mesh.vertices, self.mesh.faces])
            init = MeshCleaner(vedo_mesh)
            vedo_mesh = init.non_manifold_faces(remove_all=True)
            vedo_mesh.fill_holes()
            print("Is the cleaned mesh manifold?", vedo_mesh.is_manifold())
            vedo_mesh = init.fix_normals()
            cleaned_vertices = vedo_mesh.vertices  
            cleaned_faces = vedo_mesh.cells 
            self.mesh = trimesh.Trimesh(cleaned_vertices, cleaned_faces)

        return self.mesh
    
class MeshFlattener:
    def __init__(self, inputMesh):
        self.mesh = inputMesh
        self.V = np.array(self.mesh.vertices)
        self.F = np.array(self.mesh.faces, dtype=np.int32)
        self.bnd = igl.boundary_loop(self.F)


    def harmonic_parameterization(self):
        print("Performing Harmonic Parameterization...")
        bnd_uv = igl.map_vertices_to_circle(self.V, self.bnd)
        uv = igl.harmonic(self.V, self.F, self.bnd, bnd_uv, 1)
        return uv
    def arapParameterization(self): 
        print("performing ARAP")
        bnd_uv = igl.map_vertices_to_circle(self.V, self.bnd)
        uv = igl.harmonic(self.V, self.F, self.bnd, bnd_uv, 1)
        arap = igl.ARAP(self.V, self.F, 2, np.zeros(0))  # 2 for 2D, np.zeros(0) for no initial guess
        arap_uv = arap.solve(np.zeros((0, 0)), uv)  # Solve ARAP using harmonic parameterization as the initial guess
        return arap_uv
    def lscm_parameterization(self):
        print("Performing LSCM Parameterization...")
        bnd_points = np.array([self.bnd[0], self.bnd[len(self.bnd) // 2]], dtype=np.int32)
        uv_fixed = np.array([[0.0, 0.0], [1.0, 0.0]])
        uv = igl.lscm(self.V, self.F, bnd_points, uv_fixed)
        print("lscm complete")
        print(f"{uv}")
        return uv[1]
    

    def cleanMesh(self):
        print("Is the mesh watertight?", (self.mesh).is_watertight)
        if self.mesh is not None:
            cleaner = MeshCleaner(self.mesh)
            self.mesh = cleaner.cleanMesh()
        return self.mesh
    
    @staticmethod
    def is_uv_valid(uv, threshold=1e-2, min_nonzero_ratio=0.1):
        uv_magnitude = np.linalg.norm(uv, axis=1)
        nonzero_count = np.sum(uv_magnitude > threshold)
        nonzero_ratio = nonzero_count / len(uv_magnitude)
        return nonzero_ratio >= min_nonzero_ratio
    
    # def plot_mesh_2d(self, uv, F, title):
    #     fig, ax = plt.subplots()
    #     ax.triplot(uv[:, 0], uv[:, 1], F, 'go-', lw=1.0)
    #     ax.set_aspect('equal')
    #     ax.set_title(title)
    #     ax.set_xlabel('U')
    #     ax.set_ylabel('V')
    #     plt.show()