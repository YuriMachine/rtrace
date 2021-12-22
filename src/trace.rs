use crate::bvh::BvhData;
use crate::utils::{rand1, sample_disk};
use crate::{one3, scene::*, utils::*, vec_comp_mul, zero3, zero4};
use glm::{epsilon, is_null, min2_scalar, vec2, vec3, vec3_to_vec4, vec4};
use glm::{Vec3, Vec4};
use parking_lot::Mutex;
use rand::prelude::SmallRng;
use rayon::prelude::*;

const RAY_EPS: f32 = 1e-4;

#[derive(Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub tmin: f32,
    pub tmax: f32,
}

impl Default for Ray {
    fn default() -> Self {
        Ray {
            origin: zero3!(),
            direction: vec3(0.0, 0.0, 1.0),
            tmin: RAY_EPS,
            tmax: f32::MAX,
        }
    }
}

pub fn raytrace_samples(
    state: &mut RaytraceState,
    params: &RaytraceParams,
    scene: &Scene,
    bvh: &BvhData<'_>,
) {
    if state.samples >= params.samples {
        return;
    }
    state.samples += 1;
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
            let mut ray = camera.eval(uv, sample_disk(rand2(rng)));
            let mut radiance = (params.shader)(scene, bvh, &mut ray, rng, params);
            if radiance.max() > params.clamp {
                radiance = radiance * (params.clamp / radiance.max());
            }
            unsafe {
                *(image_ptr as *mut Vec4).add(idx) += radiance;
            }
        });
}

pub fn shade_color(
    scene: &Scene,
    bvh: &BvhData<'_>,
    ray: &mut Ray,
    _rng: &Mutex<SmallRng>,
    _params: &RaytraceParams,
) -> Vec4 {
    let intersection = bvh.intersect(ray);
    if !intersection.hit {
        return zero4!();
    }
    let material = scene.eval_material(&intersection);
    vec3_to_vec4(&material.color)
}

pub fn shade_normals(
    scene: &Scene,
    bvh: &BvhData<'_>,
    ray: &mut Ray,
    _rng: &Mutex<SmallRng>,
    _params: &RaytraceParams,
) -> Vec4 {
    let intersection = bvh.intersect(ray);
    if !intersection.hit {
        return zero4!();
    }
    let outgoing = -ray.direction;
    let mut normal = scene.eval_shading_normal(&intersection, &outgoing);
    normal = (normal * 0.5).add_scalar(0.5);
    vec3_to_vec4(&normal)
}

pub fn shade_position(
    scene: &Scene,
    bvh: &BvhData<'_>,
    ray: &mut Ray,
    _rng: &Mutex<SmallRng>,
    _params: &RaytraceParams,
) -> Vec4 {
    let intersection = bvh.intersect(ray);
    if !intersection.hit {
        return zero4!();
    }
    let mut position = scene.eval_shading_position(&intersection);
    position = (position * 0.5).add_scalar(0.5);
    vec3_to_vec4(&position)
}

pub fn shade_eyelight(
    scene: &Scene,
    bvh: &BvhData<'_>,
    ray: &mut Ray,
    _rng: &Mutex<SmallRng>,
    _params: &RaytraceParams,
) -> Vec4 {
    let mut radiance = zero3!();
    let intersection = bvh.intersect(ray);
    if !intersection.hit {
        return vec3_to_vec4(&radiance);
    }

    // prepare shading point
    let outgoing = -ray.direction;
    let normal = scene.eval_shading_normal(&intersection, &outgoing);
    let material = scene.eval_material(&intersection);

    // accumulate emission
    radiance += material.eval_emission(&normal, &outgoing);

    let incoming = outgoing;
    radiance += material.eval_bsdfcos(&normal, &outgoing, &incoming);
    vec3_to_vec4(&radiance)
}

pub fn shade_naive(
    scene: &Scene,
    bvh: &BvhData<'_>,
    ray: &mut Ray,
    rng: &Mutex<SmallRng>,
    params: &RaytraceParams,
) -> Vec4 {
    let mut radiance = zero3!();
    let mut weight = one3!();
    let mut bounce = 0;
    let mut hit_alpha = 0.0;
    while bounce < params.bounces {
        let intersection = bvh.intersect(ray);
        if !intersection.hit {
            radiance += vec_comp_mul!(weight, &scene.eval_environment(ray.direction));
            break;
        }

        // prepare shading point
        let outgoing = -ray.direction;
        let position = scene.eval_shading_position(&intersection);
        let normal = scene.eval_shading_normal(&intersection, &outgoing);
        let material = scene.eval_material(&intersection);

        // handle opacity
        if material.opacity < 1.0 && rand1(rng) >= material.opacity {
            ray.origin = position + ray.direction * 1e-2;
            bounce -= 1;
            continue;
        }
        if bounce == 0 {
            hit_alpha = 1.0;
        }

        // accumulate emission
        radiance += vec_comp_mul!(weight, &material.eval_emission(&normal, &outgoing));

        // next direction
        let incoming;
        if material.roughness != 0.0 {
            incoming = material.sample_bsdfcos(&normal, &outgoing, rand1(rng), &rand2(rng));
            if is_null(&incoming, epsilon()) {
                break;
            }
            let eval_bsdfcos = material.eval_bsdfcos(&normal, &outgoing, &incoming)
                / material.sample_bsdfcos_pdf(&normal, &outgoing, &incoming);
            weight = vec_comp_mul!(weight, &eval_bsdfcos);
        } else {
            incoming = material.sample_delta(&normal, &outgoing, rand1(rng));
            if is_null(&incoming, epsilon()) {
                break;
            }
            let eval_delta = material.eval_delta(&normal, &outgoing, &incoming)
                / material.sample_delta_pdf(&normal, &outgoing, &incoming);
            weight = vec_comp_mul!(weight, &eval_delta);
        }

        // check weight
        if is_null(&weight, epsilon()) || !is_finite(&weight) {
            break;
        }

        // russian roulette
        if bounce > 3 {
            let rr_prob = min2_scalar(weight.max(), 0.99);
            if rand1(rng) >= rr_prob {
                break;
            }
            weight *= 1.0 / rr_prob;
        }

        // setup next iteration
        bounce += 1;
        ray.origin = position;
        ray.direction = incoming;
    }
    vec4(radiance.x, radiance.y, radiance.z, hit_alpha)
}

pub fn shade_raytrace(
    scene: &Scene,
    bvh: &BvhData<'_>,
    ray: &mut Ray,
    rng: &Mutex<SmallRng>,
    params: &RaytraceParams,
) -> Vec4 {
    let mut radiance = zero3!();
    let mut weight = one3!();
    let mut bounce = 0;
    let mut hit_alpha = 0.0;
    while bounce < params.bounces {
        let intersection = bvh.intersect(ray);
        if !intersection.hit {
            radiance += vec_comp_mul!(weight, &scene.eval_environment(ray.direction));
            break;
        }

        // prepare shading point
        let outgoing = -ray.direction;
        let position = scene.eval_shading_position(&intersection);
        let normal = scene.eval_shading_normal(&intersection, &outgoing);
        let material = scene.eval_material(&intersection);

        // handle opacity
        if material.opacity < 1.0 && rand1(rng) >= material.opacity {
            ray.origin = position + ray.direction * 1e-2;
            bounce -= 1;
            continue;
        }
        if bounce == 0 {
            hit_alpha = 1.0;
        }

        // accumulate emission
        radiance += vec_comp_mul!(weight, &material.eval_emission(&normal, &outgoing));

        // next direction
        let incoming;
        if material.roughness != 0.0 {
            incoming = material.sample_bsdfcos(&normal, &outgoing, rand1(rng), &rand2(rng));
            if is_null(&incoming, epsilon()) {
                break;
            }
            let eval_bsdfcos = material.eval_bsdfcos(&normal, &outgoing, &incoming)
                / material.sample_bsdfcos_pdf(&normal, &outgoing, &incoming);
            weight = vec_comp_mul!(weight, &eval_bsdfcos);
        } else {
            incoming = material.sample_delta(&normal, &outgoing, rand1(rng));
            if is_null(&incoming, epsilon()) {
                break;
            }
            let eval_delta = material.eval_delta(&normal, &outgoing, &incoming)
                / material.sample_delta_pdf(&normal, &outgoing, &incoming);
            weight = vec_comp_mul!(weight, &eval_delta);
        }

        // check weight
        if is_null(&weight, epsilon()) || !is_finite(&weight) {
            break;
        }

        // russian roulette
        if bounce > 3 {
            let rr_prob = min2_scalar(weight.max(), 0.99);
            if rand1(rng) >= rr_prob {
                break;
            }
            weight *= 1.0 / rr_prob;
        }

        // setup next iteration
        bounce += 1;
        ray.origin = position;
        ray.direction = incoming;
    }
    vec4(radiance.x, radiance.y, radiance.z, hit_alpha)
}
