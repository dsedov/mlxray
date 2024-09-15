from pxr import UsdGeom, Usd, UsdShade
from pxr import Gf
from core.geo import Geo
from usd.loader import UsdLoader
import mlx.core as mx
import numpy as np
class UsdGeo:
    def load_geos(usd_loader: UsdLoader, materials: list):
        geos = []
        norms = []
        mats = []
        for geo_prim in usd_loader.find_geos():
            print(f"\nLoaded Geo: {geo_prim.GetPath()}")
            xform = UsdGeom.Xformable(geo_prim).ComputeLocalToWorldTransform(time=Usd.TimeCode.Default())
            mesh_prim = UsdGeom.Mesh(geo_prim)
            bound_material = UsdShade.MaterialBindingAPI(geo_prim).ComputeBoundMaterial()
            if bound_material:
                material_name = bound_material[0].GetPrim().GetPath()
                print(f"Material name: {material_name}")
            else:
                material_name = "ERROR"

            material_id = -1 
            for i, material in enumerate(materials):
                if material.name == material_name:
                    material_id = i
                    break
            if material_id == -1:
                raise Exception(f"Material not found: {material_name}")

            pointsData = mesh_prim.GetPointsAttr().Get()
            pointsData = mesh_prim.GetPointsAttr().Get()
            faceVertexCounts = mesh_prim.GetFaceVertexCountsAttr().Get()
            faceVertexIndices = mesh_prim.GetFaceVertexIndicesAttr().Get()
            normalsAttr = mesh_prim.GetNormalsAttr()
            normalsData = normalsAttr.Get() if normalsAttr else None
            if normalsData is None:
                print("No normals found, computing normals")

            vertices = np.empty((0, 3), dtype=np.float32)
            for point in pointsData:
                transformedPoint = xform.Transform(point)
                vertices = np.vstack((vertices, np.array([transformedPoint[0], transformedPoint[1], transformedPoint[2]])))

            normals = np.empty((0, 3), dtype=np.float32)
            normalXForm = xform.GetInverse().GetTranspose()
            if normalsData is not None:
                for normal in normalsData:
                    transformedNormal = normalXForm.TransformDir(normal)
                    normal = np.array([transformedNormal[0], transformedNormal[1], transformedNormal[2]])
                    normal = normal / np.linalg.norm(normal)
                    normals = np.vstack((normals, normal))

            index = 0
            triangles = np.empty((0, 3), dtype=np.float32)
            vnormals   = np.empty((0, 3), dtype=np.float32)
            mat_indices = np.empty((0, 1), dtype=np.int32)
            for faceVertexCount in faceVertexCounts:
                if(faceVertexCount < 3):
                    print(f"Skipping face with {faceVertexCount} vertices")
                    index += faceVertexCount
                    continue
                
                for i in range(1, faceVertexCount - 1):
                    v0 = faceVertexIndices[index]
                    v1 = faceVertexIndices[index + i]
                    v2 = faceVertexIndices[index + i + 1]

                    if  v0 < 0 or v0 >= vertices.shape[0] or v1 < 0 or v1 >= vertices.shape[0] or v2 < 0 or v2 >= vertices.shape[0]:
                        print(f"Skipping face with invalid indices: {v0}, {v1}, {v2}")
                        continue
                    
                    if index < normals.shape[0] and index + i < normals.shape[0] and index + i + 1 < normals.shape[0]:
                        triangles = np.vstack((triangles, np.array([ vertices[v0], vertices[v1], vertices[v2]])))
                        vnormals = np.vstack((vnormals, np.array([normals[v0], normals[v1], normals[v2]])))
                    else:
                        face_normal = np.cross(vertices[v1] - vertices[v0], vertices[v2] - vertices[v0])
                        face_normal = face_normal / np.linalg.norm(face_normal)
                        triangles = np.vstack((triangles, np.array([ vertices[v0], vertices[v1], vertices[v2]])))
                        vnormals = np.vstack((vnormals, np.array([face_normal, face_normal, face_normal])))
                    mat_indices = np.vstack((mat_indices, np.array([material_id])))  
                index += faceVertexCount
            geos.append(triangles)
            norms.append(vnormals)
            mats.append(mat_indices)
        return geos, norms, mats