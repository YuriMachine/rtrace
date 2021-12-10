
extern crate nalgebra_glm as glm;

#[allow(dead_code)]
mod components {
    use nalgebra_glm::vec3;

    const INVALID: i32 = -1;
    const RAY_EPS: f32 = 1e-4;

    struct Ray {
        origin: glm::Vec3,
        direction: glm::Vec3,
        tmin: f32,
        tmax: f32
    }

    impl Default for Ray {
        fn default() -> Self {
            Ray {
                origin: glm::vec3(0.0, 0.0, 0.0),
                direction: glm::vec3(0.0, 0.0, 1.0),
                tmin: RAY_EPS,
                tmax: f32::MAX
            }
         }
    }
    

    struct Camera {
        frame: glm::Mat3x4,
        orthographic: bool,
        lens: f32,
        film: f32,
        aspect: f32,
        focus: f32,
        aperture: f32
    }

    impl Default for Camera {
        fn default() -> Self {
            Camera {
                frame: glm::mat3x4(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                orthographic: false,
                lens: 0.050,
                film: 0.036,
                aspect: 1.5,
                focus: 10000.0,
                aperture: 0.0
            }
         }
    }

    enum MaterialType {
        Matte,
        Glossy,
        Reflective,
        Transparent,
        Refractive
    }

    struct Material {
        material: MaterialType,
        emission: glm::Vec3,
        color: glm::Vec3,
        roughness: f32,
        metallic: f32,
        ior: f32,
        scattering: glm::Vec3,
        scanisotropy: f32,
        trdepth: f32,
        opacity: f32,
        // textures
        emission_tex: i32,
        color_tex: i32,
        roughness_tex: i32,
        scattering_tex: i32,
        normal_tex: i32
    }

    impl Default for Material {
        fn default() -> Self {
            Material {
                material: MaterialType::Matte,
                emission: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
                roughness: 0.0,
                metallic: 0.0,
                ior: 1.5,
                scattering: glm::vec3(0.0, 0.0, 0.0),
                scanisotropy: 0.0,
                trdepth: 0.01,
                opacity: 1.0,
                // textures
                emission_tex: INVALID,
                color_tex: INVALID,
                roughness_tex: INVALID,
                scattering_tex: INVALID,
                normal_tex: INVALID
            }
         }
    }

    #[derive(Default)]
        struct Texture {
        width: i32,
        height: i32,
        linear: bool,
        pixelsf: Vec<glm::Vec4>,
        pixelsb: Vec<glm::BVec4>
    }
    
    struct Instance {
        frame: glm::Mat3x4,
        shape: i32,
        material: i32
    }

    impl Default for Instance {
        fn default() -> Self {
            Instance {
                frame: glm::mat3x4(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                shape: INVALID,
                material: INVALID
            }
         }
    }
    
    struct Environment {
        frame: glm::Mat3x4,
        emission: glm::Vec3,
        emission_tex: i32
    }

    impl Default for Environment {
        fn default() -> Self {
            Environment {
                frame: glm::mat3x4(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                emission: glm::vec3(0.0, 0.0, 0.0),
                emission_tex: INVALID
            }
        }
    }

    #[derive(Default)]
    struct Shape {
        // element data
        points: Vec<i32>,
        lines: Vec<glm::IVec2>,
        triangles: Vec<glm::IVec3>,
        quads: Vec<glm::IVec4>,
        // vertex data
        positions: Vec<glm::Vec3>,
        normals: Vec<glm::Vec3>,
        texcoords: Vec<glm::Vec2>,
        colors: Vec<glm::Vec4>,
        radius: Vec<f32>,
        tangents: Vec<glm::Vec4>
    }

    #[derive(Default)]
    struct Scene {
        cameras: Vec<Camera>,
        instances: Vec<Instance>,
        environments: Vec<Environment>,
        shapes: Vec<Shape>,
        textures: Vec<Texture>,
        materials: Vec<Material>
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
