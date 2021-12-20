use std::f32::consts::PI;

use crate::{bvh::BvhData, scene::*, trace, trace::Ray};
use glm::{inverse, make_mat3, make_mat3x4, mat3x3, normalize, transpose, vec2, vec3, vec4, Mat3};
use glm::{Mat3x4, Vec2, Vec3, Vec4};
use parking_lot::Mutex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub struct RaytraceParams {
    pub camera: usize,
    pub resolution: usize,
    pub shader: fn(&Scene, &BvhData<'_>, &mut Ray, &Mutex<SmallRng>, &RaytraceParams) -> Vec4,
    pub samples: i32,
    pub bounces: i32,
    pub noparallel: bool,
    pub pratio: i32,
    pub exposure: f32,
    pub filmic: bool,
    pub clamp: f32,
}

impl Default for RaytraceParams {
    fn default() -> Self {
        RaytraceParams {
            camera: 0,
            resolution: 720,
            shader: trace::shade_raytrace,
            samples: 64,
            bounces: 8,
            noparallel: false,
            pratio: 8,
            exposure: 0.0,
            filmic: false,
            clamp: 10.0,
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
    pub fn from_scene(scene: &Scene, params: &RaytraceParams) -> Self {
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

pub fn rand1(rng: &Mutex<SmallRng>) -> f32 {
    rng.lock().gen::<f32>()
}

pub fn rand2(rng: &Mutex<SmallRng>) -> Vec2 {
    let (rng_x, rng_y) = rng.lock().gen::<(f32, f32)>();
    vec2(rng_x, rng_y)
}

pub fn transform_point(a: &Mat3x4, b: &Vec3) -> Vec3 {
    a.column(0) * b.x + a.column(1) * b.y + a.column(2) * b.z + a.column(3)
}

pub fn transform_vector_frame(a: &Mat3x4, b: &Vec3) -> Vec3 {
    a.column(0) * b.x + a.column(1) * b.y + a.column(2) * b.z
}

pub fn transform_direction_frame(a: &Mat3x4, b: &Vec3) -> Vec3 {
    normalize(&transform_vector_frame(a, b))
}

pub fn transform_normal_frame(a: &Mat3x4, b: &Vec3, non_rigid: bool) -> Vec3 {
    if non_rigid {
        let a_rotation = &transpose(&inverse(&mat3x3(
            a.m11, a.m12, a.m13, a.m21, a.m22, a.m23, a.m31, a.m32, a.m33,
        )));

        normalize(&transform_vector_mat(a_rotation, b))
    } else {
        normalize(&transform_vector_frame(a, b))
    }
}

pub fn transform_vector_mat(a: &Mat3, b: &Vec3) -> Vec3 {
    a * b
}

pub fn transform_direction_mat(a: &Mat3, b: &Vec3) -> Vec3 {
    normalize(&transform_vector_mat(a, b))
}

pub fn interpolate_line<'a, T: 'a>(p0: &'a T, p1: &'a T, u: f32) -> T
where
    &'a T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>,
    T: std::ops::Add<Output = T>,
{
    p0 * (1.0 - u) + p1 * u
}

pub fn interpolate_triangle<'a, T: 'a>(p0: &'a T, p1: &'a T, p2: &'a T, uv: &Vec2) -> T
where
    &'a T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>,
    T: std::ops::Add<Output = T>,
{
    p0 * (1.0 - uv.x - uv.y) + p1 * uv.x + p2 * uv.y
}

pub fn interpolate_quad<'a, T: 'a>(p0: &'a T, p1: &'a T, p2: &'a T, p3: &'a T, uv: &Vec2) -> T
where
    &'a T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>,
    T: std::ops::Add<Output = T>,
{
    if uv.x + uv.y <= 1.0 {
        interpolate_triangle(p0, p1, p3, uv)
    } else {
        interpolate_triangle(p2, p3, p1, &vec2(1.0 - uv.x, 1.0 - uv.y))
    }
}

pub fn orthonormalize(a: &Vec3, b: &Vec3) -> Vec3 {
    ((a - b) * a.dot(b)).normalize()
}

pub fn inverse_frame(frame: &Mat3x4, non_rigid: bool) -> Mat3x4 {
    let rotation = make_mat3(
        &[
            frame.column(0).as_slice(),
            frame.column(1).as_slice(),
            frame.column(2).as_slice(),
        ]
        .concat(),
    );
    if non_rigid {
        let minv = inverse(&rotation);
        make_mat3x4(
            &[
                minv.column(0).as_slice(),
                minv.column(1).as_slice(),
                minv.column(2).as_slice(),
                (-(minv * frame.column(3))).as_slice(),
            ]
            .concat(),
        )
    } else {
        let minv = transpose(&rotation);
        make_mat3x4(
            &[
                minv.column(0).as_slice(),
                minv.column(1).as_slice(),
                minv.column(2).as_slice(),
                (-(minv * frame.column(3))).as_slice(),
            ]
            .concat(),
        )
    }
}

pub fn basis_fromz(normal: &Vec3) -> Mat3 {
    let z = normalize(normal);
    let sign = copysignf(1.0, z.z);
    let a = -1.0 / (sign + z.z);
    let b = z.x * z.y * a;
    let x = vec3(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x);
    let y = vec3(b, sign + z.y * z.y * a, -z.y);

    make_mat3(&[x.as_slice(), y.as_slice(), z.as_slice()].concat())
}

pub fn copysignf(magnitude: f32, sign: f32) -> f32 {
    let mut ux = magnitude.to_bits();
    let uy = sign.to_bits();
    ux &= 0x7fffffff;
    ux |= uy & 0x80000000;
    f32::from_bits(ux)
}

pub fn sample_disk(ruv: Vec2) -> Vec2 {
    let r = f32::sqrt(ruv.y);
    let phi = 2.0 * PI * ruv.x;
    vec2(f32::cos(phi) * r, f32::sin(phi) * r)
}

pub fn to_srgb(component: f32, gamma: f32) -> u8 {
    (component.max(0.0).min(1.0).powf(1.0 / gamma) * 255.0) as u8
}

pub fn srgb_to_rgb(color: Vec4) -> Vec4 {
    let compute_srgb = |srgb: f32| -> f32 {
        if srgb <= 0.04045 {
            srgb / 12.92
        } else {
            ((srgb + 0.055) / (1.0 + 0.055)).powf(2.4)
        }
    };
    vec4(
        compute_srgb(color.x),
        compute_srgb(color.y),
        compute_srgb(color.z),
        compute_srgb(color.w),
    )
}
