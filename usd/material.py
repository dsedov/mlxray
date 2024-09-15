
from usd.loader import UsdLoader
from pxr import Usd, UsdShade
import mlx.core as mx
from core.material import Material
class UsdMaterial:
    def find_materials_recursively(prim: Usd.Prim, materials: list = []):
        for child in prim.GetChildren():
            if child.IsA(UsdShade.Shader):
                materials.append(UsdShade.Shader(child))
            UsdMaterial.find_materials_recursively(child, materials)
        return materials

    def load_materials(usd_loader: UsdLoader):
        materials = []

        usd_materials = UsdMaterial.find_materials_recursively(usd_loader.stage.GetPseudoRoot())

        for usd_material in usd_materials:
            parent = usd_material.GetPrim().GetParent()
            print(f"\nLoaded material: {parent.GetPath()}")
            base_weight = 1.0
            base_color = mx.array([1.0, 1.0, 1.0])
            metalness = 0.0
            transmission = 0.0
            specular = 0.0
            specular_roughness = 0.0
            ior = 1.0

            shader = usd_material
            if shader:
                print(f"Shader: {shader.GetPath()}")
                base_weight_attribute = shader.GetInput("base")
                if base_weight_attribute:
                    base_weight = base_weight_attribute.Get()
                    print(f"Base weight: {base_weight}")
                
                base_color_attribute = shader.GetInput("base_color")
                if base_color_attribute:
                    base_color = mx.array(base_color_attribute.Get())
                    print(f"Base color: {base_color}")

                metalness_attribute = shader.GetInput("metalness")
                if metalness_attribute:
                    metalness = metalness_attribute.Get()
                    print(f"Metalness: {metalness}")

                transmission_attribute = shader.GetInput("transmission")
                if transmission_attribute:
                    transmission = transmission_attribute.Get()
                    print(f"Transmission: {transmission}")  

                specular_attribute = shader.GetInput("specular")
                if specular_attribute:
                    specular = specular_attribute.Get()
                    print(f"Specular: {specular}")

                specular_roughness_attribute = shader.GetInput("specular_roughness")
                if specular_roughness_attribute:
                    specular_roughness = specular_roughness_attribute.Get()
                    print(f"Specular roughness: {specular_roughness}")

                ior_attribute = shader.GetInput("ior")
                if ior_attribute:
                    ior = ior_attribute.Get()
                    print(f"IOR: {ior}")
            else:
                print(f"No shader found for {usd_material.GetPath()}")
            material = Material(
                name=parent.GetPath(),
                base_weight=base_weight, 
                base_color=base_color, 
                metalness=metalness, 
                transmission=transmission, 
                specular=specular, 
                specular_roughness=specular_roughness, 
                ior=ior)
            materials.append(material)
        return materials
        