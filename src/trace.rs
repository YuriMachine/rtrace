use glm::vec2;
use glm::{Vec4, vec4};
use rand::prelude::SmallRng;

use crate::{utils::{RaytraceState, RaytraceParams, rand2}, components::{Scene, Ray}};

fn raytrace_samples(state: &mut RaytraceState, params: &RaytraceParams, scene: &Scene) {
    for j in 0..state.height {
        for i in 0..state.width {
            let idx = state.width * j + i;
            let rng = &mut state.rngs[idx];
            let puv = rand2(rng);
            let uv = vec2((i as f32 + puv.x) / state.width as f32, (j as f32 + puv.y) / state.height as f32);
            //let ray = eval_camera(scene.cameras[params.camera], uv);
            let ray = Ray::default();
            let radiance = shade_raytrace(scene, ray, params.bounces, rng, params);
            if radiance != vec4(0.0, 0.0, 0.0, 0.0) {
                state.hits[idx] += 1;
            }
            state.image[idx] += radiance;
        }
    }
    state.samples += 1;
}

fn shade_raytrace(scene: &Scene, ray: Ray, bounce: i32, rng: &mut SmallRng, params: &RaytraceParams) -> Vec4 {
    vec4(0.0, 0.0, 0.0, 0.0)
}