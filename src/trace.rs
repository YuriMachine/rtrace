use parking_lot::Mutex;

use glm::vec2;
use glm::{vec4, Vec4};
use nalgebra_glm::vec3_to_vec4;
use rand::prelude::SmallRng;
use rayon::prelude::*;

use crate::bvh::BvhData;
use crate::{
    components::{Ray, Scene},
    utils::{rand2, RaytraceParams, RaytraceState},
};

pub fn raytrace_samples(
    state: &mut RaytraceState,
    params: &RaytraceParams,
    scene: &Scene,
    bvh: &BvhData,
) {
    let camera = &scene.cameras[params.camera];
    let image_ptr: usize = state.image.as_mut_ptr() as _;
    (0..state.width * state.height)
        .into_par_iter()
        .for_each(|idx| {
            let (i, j) = (idx % state.width, idx / state.width);
            let rng = &state.rngs[idx];
            let puv = rand2(rng);
            let uv = vec2(
                (i as f32 + puv.x) / state.width as f32,
                (j as f32 + puv.y) / state.height as f32,
            );
            let ray = camera.eval(uv, rand2(rng));
            let radiance = (params.shader)(scene, bvh, &ray, params.bounces, rng, params);
            unsafe {
                *(image_ptr as *mut Vec4).add(idx) += radiance;
            }
        });
    state.samples += 1;
}

pub fn shade_color(
    scene: &Scene,
    bvh: &BvhData,
    ray: &Ray,
    _bounce: i32,
    _rng: &Mutex<SmallRng>,
    _params: &RaytraceParams,
) -> Vec4 {
    let intersection = bvh.intersect(ray);
    if !intersection.hit {
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
    let instance = &scene.instances[intersection.instance];
    //let element = intersection.element;
    //let uv = intersection.uv;
    let material = &scene.materials[instance.material];
    /*
    let colorTo = vec4(1.0, 1.0, 1.0, 1.0);
    let shape = scene.shapes[instance.shape];
    if !shape.triangles.is_empty() {
        let t = shape.triangles[element];
        //interpolate_triangle(shape.colors[t.x], shape.colors[t.y], shape.colors[t.z], uv);
    }
    */
    vec3_to_vec4(&material.color) //.component_mul(&colorTo)
}

pub fn shade_raytrace(
    scene: &Scene,
    bvh: &BvhData,
    ray: &Ray,
    bounce: i32,
    rng: &Mutex<SmallRng>,
    params: &RaytraceParams,
) -> Vec4 {
    let intersection = bvh.intersect(ray);
    if !intersection.hit {
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
    let instance = &scene.instances[intersection.instance];
    //let element = intersection.element;
    //let uv = intersection.uv;
    let material = &scene.materials[instance.material];

    vec3_to_vec4(&material.color) //.component_mul(&colorTo)
}
