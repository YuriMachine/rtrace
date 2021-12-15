use crate::{bvh::BvhData, components, components::Ray, components::Scene, trace};
use nalgebra_glm::{normalize, vec2, vec4, Mat3x4, Vec2, Vec3, Vec4};
use parking_lot::Mutex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub struct RaytraceParams {
    pub camera: usize,
    pub resolution: usize,
    pub shader: fn(&Scene, &BvhData, &Ray, i32, &Mutex<SmallRng>, &RaytraceParams) -> Vec4,
    pub samples: i32,
    pub bounces: i32,
    pub noparallel: bool,
    pub pratio: i32,
    pub exposure: f32,
    pub filmic: bool,
}

impl Default for RaytraceParams {
    fn default() -> Self {
        RaytraceParams {
            camera: 0,
            resolution: 1280,
            shader: trace::shade_color,
            samples: 8,
            bounces: 4,
            noparallel: false,
            pratio: 8,
            exposure: 0.0,
            filmic: false,
        }
    }
}

#[derive(Debug, Default)]
pub struct RaytraceState {
    pub width: usize,
    pub height: usize,
    pub samples: i32,
    pub image: Vec<Vec4>,
    pub rngs: Vec<Mutex<SmallRng>>,
}

impl RaytraceState {
    pub fn from_scene(scene: &components::Scene, params: &RaytraceParams) -> Self {
        let camera = &scene.cameras[params.camera];
        let (width, height) = if camera.aspect >= 1.0 {
            (
                params.resolution,
                (params.resolution as f32 / camera.aspect).round() as usize,
            )
        } else {
            (
                params.resolution,
                (params.resolution as f32 * camera.aspect).round() as usize,
            )
        };
        let samples = 0;
        let image = vec![vec4(0.0, 0.0, 0.0, 0.0); width * height];

        let rngs: Vec<Mutex<SmallRng>> = (0..width * height)
            .map(|_| Mutex::new(SmallRng::from_entropy()))
            .collect();

        RaytraceState {
            width,
            height,
            samples,
            image,
            rngs,
        }
    }
}

pub fn rand2(rng: &Mutex<SmallRng>) -> Vec2 {
    let (rng_x, rng_y) = rng.lock().gen::<(f32, f32)>();
    vec2(rng_x, rng_y)
}

pub fn transform_point(a: &Mat3x4, b: &Vec3) -> Vec3 {
    a.column(0) * b.x + a.column(1) * b.y + a.column(2) * b.z + a.column(3)
}

pub fn transform_direction(a: &Mat3x4, b: &Vec3) -> Vec3 {
    normalize(&(a.column(0) * b.x + a.column(1) * b.y + a.column(2) * b.z))
}
