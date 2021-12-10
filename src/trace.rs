use glm::vec2;
use glm::{Vec4, vec4};
use glm::is_null;
use glm::epsilon;
use rand::prelude::SmallRng;

use crate::{utils::{RaytraceState, RaytraceParams, rand2}, components::{Scene, Ray}};

fn raytrace_samples(state: &mut RaytraceState, params: &RaytraceParams, scene: &Scene) {
    for idx in 0..state.width * state.height {
        let (i, j) = (idx % state.width, idx / state.width);
        let camera = &scene.cameras[params.camera];
        let rng = &mut state.rngs[idx];
        let puv = rand2(rng);
        let uv = vec2((i as f32 + puv.x) / state.width as f32, (j as f32 + puv.y) / state.height as f32);
        let ray = camera.eval(uv, rand2(rng));
        let radiance = shade_raytrace(scene, ray, params.bounces, rng, params);
        if !is_null(&radiance, epsilon()) {
            state.hits[idx] += 1;
        }
        state.image[idx] += radiance;
    }
    state.samples += 1;
}

fn shade_raytrace(scene: &Scene, ray: Ray, bounce: i32, rng: &mut SmallRng, params: &RaytraceParams) -> Vec4 {
    vec4(0.0, 0.0, 0.0, 0.0)
}