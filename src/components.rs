use nalgebra_glm::vec3;

const INVALID: i32 = -1;
const RAY_EPS: f32 = 1e-4;

pub struct Ray {
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

pub struct Camera {
    pub frame: glm::Mat3x4,
    pub orthographic: bool,
    pub lens: f32,
    pub film: f32,
    pub aspect: f32,
    pub focus: f32,
    pub aperture: f32
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

pub struct Material {
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
pub struct Texture {
    width: i32,
    height: i32,
    linear: bool,
    pixelsf: Vec<glm::Vec4>,
    pixelsb: Vec<glm::BVec4>
}

pub struct Instance {
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

pub struct Environment {
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
pub struct Shape {
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
pub struct Scene {
    pub cameras: Vec<Camera>,
    pub instances: Vec<Instance>,
    pub environments: Vec<Environment>,
    pub shapes: Vec<Shape>,
    pub textures: Vec<Texture>,
    pub materials: Vec<Material>
}

impl Scene {
    fn make_cornellbox() -> Scene {
        let mut cameras = Vec::new();
        let mut shapes = Vec::new();
        let mut materials = Vec::new();
        let mut instances = Vec::new();

        let camera = Camera {
            frame: glm::mat3x4(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 3.9),
            orthographic: false,
            lens: 0.035,
            film: 0.024,
            aspect: 1.0,
            focus: 3.9,
            aperture: 0.0
        };
        cameras.push(camera);
        // floor
        let floor_shape = Shape {
            positions: vec![vec3(-1.0, 0.0, 1.0), vec3(1.0, 0.0, 1.0), vec3(1.0, 0.0, -1.0), vec3(-1.0, 0.0, -1.0)],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0)],
            ..Default::default()
        };
        shapes.push(floor_shape);
        let floor_material = Material {
            color: vec3(0.725, 0.71, 0.68),
            ..Default::default()
        };
        materials.push(floor_material);
        let floor_instance = Instance {
            shape: shapes.len() as i32 - 1,
            material: materials.len() as i32 - 1,
            ..Default::default()
        };
        instances.push(floor_instance);
        // ceiling
        let ceiling_shape = Shape {
            positions: vec![vec3(-1.0, 2.0, 1.0), vec3(-1.0, 2.0, -1.0), vec3(1.0, 2.0, -1.0), vec3(1.0, 2.0, 1.0)],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0)],
            ..Default::default()
        };
        shapes.push(ceiling_shape);
        let ceiling_material = Material {
            color: vec3(0.725, 0.71, 0.68),
            ..Default::default()
        };
        materials.push(ceiling_material);
        let ceiling_instance = Instance {
            shape: shapes.len() as i32 - 1,
            material: materials.len() as i32 - 1,
            ..Default::default()
        };
        instances.push(ceiling_instance);
        // backwall
        let backwall_shape = Shape {
            positions: vec![vec3(-1.0, 0.0, -1.0), vec3(1.0, 0.0, -1.0), vec3(1.0, 2.0, -1.0), vec3(-1.0, 2.0, -1.0)],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0)],
            ..Default::default()
        };
        shapes.push(backwall_shape);
        let backwall_material = Material {
            color: vec3(0.725, 0.71, 0.68),
            ..Default::default()
        };
        materials.push(backwall_material);
        let backwall_instance = Instance {
            shape: shapes.len() as i32 - 1,
            material: materials.len() as i32 - 1,
            ..Default::default()
        };
        instances.push(backwall_instance);
        // rightwall
        let rightwall_shape = Shape {
            positions: vec![vec3(1.0, 0.0, -1.0), vec3(1.0, 0.0, 1.0), vec3(1.0, 2.0, 1.0), vec3(1.0, 2.0, -1.0)],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0)],
            ..Default::default()
        };
        shapes.push(rightwall_shape);
        let rightwall_material = Material {
            color: vec3(0.14, 0.45, 0.091),
            ..Default::default()
        };
        materials.push(rightwall_material);
        let rightwall_instance = Instance {
            shape: shapes.len() as i32 - 1,
            material: materials.len() as i32 - 1,
            ..Default::default()
        };
        instances.push(rightwall_instance);
        // leftwall
        let leftwall_shape = Shape {
            positions: vec![vec3(-1.0, 0.0, 1.0), vec3(-1.0, 0.0, -1.0), vec3(-1.0, 2.0, -1.0), vec3(-1.0, 2.0, 1.0)],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0)],
            ..Default::default()
        };
        shapes.push(leftwall_shape);
        let leftwall_material = Material {
            color: vec3(0.63, 0.065, 0.05),
            ..Default::default()
        };
        materials.push(leftwall_material);
        let leftwall_instance = Instance {
            shape: shapes.len() as i32 - 1,
            material: materials.len() as i32 - 1,
            ..Default::default()
        };
        instances.push(leftwall_instance);
        // shortbox
        let shortbox_shape = Shape {
            positions: vec![vec3(0.53, 0.6, 0.75), vec3(0.7, 0.6, 0.17), vec3(0.13, 0.6, 0.0),
                            vec3(-0.05, 0.6, 0.57), vec3(-0.05, 0.0, 0.57), vec3(-0.05, 0.6, 0.57),
                            vec3(0.13, 0.6, 0.0), vec3(0.13, 0.0, 0.0), vec3(0.53, 0.0, 0.75),
                            vec3(0.53, 0.6, 0.75), vec3(-0.05, 0.6, 0.57), vec3(-0.05, 0.0, 0.57),
                            vec3(0.7, 0.0, 0.17), vec3(0.7, 0.6, 0.17), vec3(0.53, 0.6, 0.75),
                            vec3(0.53, 0.0, 0.75), vec3(0.13, 0.0, 0.0), vec3(0.13, 0.6, 0.0),
                            vec3(0.7, 0.6, 0.17), vec3(0.7, 0.0, 0.17), vec3(0.53, 0.0, 0.75),
                            vec3(0.7, 0.0, 0.17), vec3(0.13, 0.0, 0.0), vec3(-0.05, 0.0, 0.57)],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0), vec3(4, 5, 6),
                            vec3(6, 7, 4), vec3(8, 9, 10), vec3(10, 11, 8),
                            vec3(12, 13, 14), vec3(14, 15, 12), vec3(16, 17, 18),
                            vec3(18, 19, 16), vec3(20, 21, 22), vec3(22, 23, 20)],
            ..Default::default()
        };
        shapes.push(shortbox_shape);
        let shortbox_material = Material {
            color: vec3(0.725, 0.71, 0.68),
            ..Default::default()
        };
        materials.push(shortbox_material);
        let shortbox_instance = Instance {
            shape: shapes.len() as i32 - 1,
            material: materials.len() as i32 - 1,
            ..Default::default()
        };
        instances.push(shortbox_instance);
        // tallbox
        let tallbox_shape = Shape {
            positions: vec![vec3(-0.53, 1.2, 0.09), vec3(0.04, 1.2, -0.09), vec3(-0.14, 1.2, -0.67),
                            vec3(-0.71, 1.2, -0.49), vec3(-0.53, 0.0, 0.09), vec3(-0.53, 1.2, 0.09),
                            vec3(-0.71, 1.2, -0.49), vec3(-0.71, 0.0, -0.49), vec3(-0.71, 0.0, -0.49),
                            vec3(-0.71, 1.2, -0.49), vec3(-0.14, 1.2, -0.67), vec3(-0.14, 0.0, -0.67),
                            vec3(-0.14, 0.0, -0.67), vec3(-0.14, 1.2, -0.67), vec3(0.04, 1.2, -0.09), 
                            vec3(0.04, 0.0, -0.09), vec3(0.04, 0.0, -0.09), vec3(0.04, 1.2, -0.09),
                            vec3(-0.53, 1.2, 0.09), vec3(-0.53, 0.0, 0.09), vec3(-0.53, 0.0, 0.09),
                            vec3(0.04, 0.0, -0.09), vec3(-0.14, 0.0, -0.67), vec3(-0.71, 0.0, -0.49)],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0), vec3(4, 5, 6),
                            vec3(6, 7, 4), vec3(8, 9, 10), vec3(10, 11, 8),
                            vec3(12, 13, 14), vec3(14, 15, 12), vec3(16, 17, 18),
                            vec3(18, 19, 16), vec3(20, 21, 22), vec3(22, 23, 20)],
            ..Default::default()
        };
        shapes.push(tallbox_shape);
        let tallbox_material = Material {
            color: vec3(0.725, 0.71, 0.68),
            ..Default::default()
        };
        materials.push(tallbox_material);
        let tallbox_instance = Instance {
            shape: shapes.len() as i32 - 1,
            material: materials.len() as i32 - 1,
            ..Default::default()
        };
        instances.push(tallbox_instance);
        // light
        let light_shape = Shape {
            positions: vec![vec3(-0.25, 1.99, 0.25), vec3(-0.25, 1.99, -0.25),
                            vec3(0.25, 1.99, -0.25), vec3(0.25, 1.99, 0.25)],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0)],
            ..Default::default()
        };
        shapes.push(light_shape);
        let light_material = Material {
            emission: vec3(17.0, 12.0, 4.0),
            ..Default::default()
        };
        materials.push(light_material);
        let light_instance = Instance {
            shape: shapes.len() as i32 - 1,
            material: materials.len() as i32 - 1,
            ..Default::default()
        };
        instances.push(light_instance);
        Scene { 
            cameras, 
            instances, 
            shapes, 
            materials,
            ..Default::default()
        }
    }
}
