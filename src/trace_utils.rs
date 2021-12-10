use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::components;

struct RaytraceParams {
    camera: usize,
    resolution: usize,
    //raytrace_shader_type shader     = raytrace_shader_type::raytrace;
    samples: i32,
    bounces: i32,
    noparallel: bool,
    pratio: i32,
    exposure: f32,
    filmic: bool
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
struct RaytraceState {
    width: usize,
    height: usize,
    samples: i32,
    image: Vec<glm::Vec4>,
    hits: Vec<i32>,
    rngs: Vec<SmallRng>
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

