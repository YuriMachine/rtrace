use rand::{SeedableRng, Rng};
use rand::rngs::SmallRng;
use nalgebra_glm::{Vec2, vec2, Mat3x4, Vec3, normalize};
use crate::components;

pub struct RaytraceParams {
    pub camera: usize,
    pub resolution: usize,
    //raytrace_shader_type shader     = raytrace_shader_type::raytrace;
    pub samples: i32,
    pub bounces: i32,
    pub noparallel: bool,
    pub pratio: i32,
    pub exposure: f32,
    pub filmic: bool
}

impl Default for RaytraceParams {
    fn default() -> Self {
        RaytraceParams {
            camera: 0,
            resolution: 720,
            //raytrace_shader_type shader     = raytrace_shader_type::raytrace;
            samples: 512,
            bounces: 4,
            noparallel: false,
            pratio: 8,
            exposure: 0.0,
            filmic: false,
        }
    }
}

#[derive(Default)]
pub struct RaytraceState {
    pub width: usize,
    pub height: usize,
    pub samples: i32,
    pub image: Vec<glm::Vec4>,
    pub hits: Vec<i32>,
    pub rngs: Vec<SmallRng>
}

impl RaytraceState {
    fn from_scene(scene: components::Scene, params: RaytraceParams) -> Self {
        let camera = &scene.cameras[params.camera];
        let (width, height) = if camera.aspect >= 1.0 {
            (params.resolution, (params.resolution as f32 / camera.aspect).round() as usize)
        } else {
            (params.resolution, (params.resolution as f32 * camera.aspect).round() as usize)
        };
        let samples = 0;
        let image = Vec::with_capacity(width * height);
        let hits = Vec::with_capacity(width * height);

        let rngs: Vec<SmallRng> = (0..width * height)
            .map(|_| SmallRng::seed_from_u64(961748941))
            .collect();

            RaytraceState {
            width,
            height,
            samples,
            image,
            hits,
            rngs
        }
    }
}

pub fn rand2(rng: &mut SmallRng) -> Vec2 {
    let (rng_x, rng_y) = rng.gen::<(f32, f32)>();
    vec2(rng_x, rng_y)
}

pub fn transform_point(a: &Mat3x4, b: &Vec3) -> Vec3 {
    a.column(0) * b.x + a.column(1) * b.y + a.column(2) * b.z + a.column(3)
}

pub fn transform_direction(a: &Mat3x4, b: &Vec3) -> Vec3 {
    normalize(&(a.column(0) * b.x + a.column(1) * b.y + a.column(2) * b.z))
}