use crate::scene_components::MaterialType;
use crate::shading::MaterialPoint;
use crate::{bvh::BvhData, scene::*, trace, trace::Ray};
use glm::{dot, inverse, make_mat3, make_mat3x4, mat3x3, normalize, transpose, vec2, vec3, vec4};
use glm::{Mat3, Mat3x4, Vec2, Vec3, Vec4};
use parking_lot::Mutex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::f32::consts::PI;

#[macro_export]
macro_rules! zero2 {
    () => {
        Vec2::zeros()
    };
}

#[macro_export]
macro_rules! zero3 {
    () => {
        Vec3::zeros()
    };
}

#[macro_export]
macro_rules! zero4 {
    () => {
        Vec4::zeros()
    };
}

#[macro_export]
macro_rules! one3 {
    () => {
        vec3(1.0, 1.0, 1.0)
    };
}

#[macro_export]
macro_rules! one4 {
    () => {
        vec4(1.0, 1.0, 1.0, 1.0)
    };
}

#[macro_export]
macro_rules! vec_comp_mul {
    ($a:expr, $b:expr) => {
        $a.component_mul($b)
    };
}

#[macro_export]
macro_rules! vec_comp_div {
    ($a:expr, $b:expr) => {
        $a.component_div($b)
    };
}

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

impl RaytraceParams {
    pub fn from_args(args: &clap::ArgMatches<'_>) -> Self {
        let shader_name = args.value_of("shader").unwrap();
        let shader = match shader_name {
            "color" => trace::shade_color,
            "eyelight" => trace::shade_eyelight,
            "normal" => trace::shade_normals,
            "position" => trace::shade_position,
            "naive" => trace::shade_naive,
            "raytrace" => trace::shade_raytrace,
            _ => trace::shade_raytrace,
        };
        RaytraceParams {
            resolution: clap::value_t!(args.value_of("resolution"), usize).unwrap(),
            samples: clap::value_t!(args.value_of("samples"), i32).unwrap(),
            bounces: clap::value_t!(args.value_of("bounces"), i32).unwrap(),
            clamp: clap::value_t!(args.value_of("clamp"), f32).unwrap(),
            noparallel: clap::value_t!(args.value_of("noparallel"), bool).unwrap(),
            shader,
            ..Default::default()
        }
    }
}

impl Default for RaytraceParams {
    fn default() -> Self {
        RaytraceParams {
            camera: 0,
            resolution: 720,
            shader: trace::shade_raytrace,
            samples: 256,
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
                (params.resolution as f32 * camera.aspect).round() as usize,
                params.resolution,
            )
        };
        let samples = 0;
        let image = vec![zero4!(); width * height];

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

#[inline(always)]
pub fn rand1(rng: &Mutex<SmallRng>) -> f32 {
    rng.lock().gen::<f32>()
}

#[inline(always)]
pub fn rand2(rng: &Mutex<SmallRng>) -> Vec2 {
    let (rng_x, rng_y) = rng.lock().gen::<(f32, f32)>();
    vec2(rng_x, rng_y)
}

#[inline(always)]
pub fn transform_point(a: &Mat3x4, b: &Vec3) -> Vec3 {
    a.column(0) * b.x + a.column(1) * b.y + a.column(2) * b.z + a.column(3)
}

#[inline(always)]
pub fn transform_vector_frame(a: &Mat3x4, b: &Vec3) -> Vec3 {
    a.column(0) * b.x + a.column(1) * b.y + a.column(2) * b.z
}

#[inline(always)]
pub fn transform_direction_frame(a: &Mat3x4, b: &Vec3) -> Vec3 {
    normalize(&transform_vector_frame(a, b))
}

#[inline(always)]
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

#[inline(always)]
pub fn transform_vector_mat(a: &Mat3, b: &Vec3) -> Vec3 {
    a * b
}

#[inline(always)]
pub fn transform_direction_mat(a: &Mat3, b: &Vec3) -> Vec3 {
    normalize(&transform_vector_mat(a, b))
}

#[inline(always)]
pub fn interpolate_line<'a, T: 'a>(p0: &'a T, p1: &'a T, u: f32) -> T
where
    &'a T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>,
    T: std::ops::Add<Output = T>,
{
    p0 * (1.0 - u) + p1 * u
}

#[inline(always)]
pub fn interpolate_triangle<'a, T: 'a>(p0: &'a T, p1: &'a T, p2: &'a T, uv: &Vec2) -> T
where
    &'a T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>,
    T: std::ops::Add<Output = T>,
{
    p0 * (1.0 - uv.x - uv.y) + p1 * uv.x + p2 * uv.y
}

#[inline(always)]
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

#[inline(always)]
pub fn triangle_area(p0: &Vec3, p1: &Vec3, p2: &Vec3) -> f32 {
    glm::length(&glm::cross(&(p1 - p0), &(p2 - p0))) / 2.0
}

#[inline(always)]
pub fn quad_area(p0: &Vec3, p1: &Vec3, p2: &Vec3, p3: &Vec3) -> f32 {
    triangle_area(p0, p1, p3) + triangle_area(p2, p3, p1)
}

#[inline(always)]
pub fn line_tangent(p0: &Vec3, p1: &Vec3) -> Vec3 {
    normalize(&(p1 - p0))
}

#[inline(always)]
pub fn triangle_normal(p0: &Vec3, p1: &Vec3, p2: &Vec3) -> Vec3 {
    normalize(&glm::cross(&(p1 - p0), &(p2 - p0)))
}

#[inline(always)]
pub fn quad_normal(p0: &Vec3, p1: &Vec3, p2: &Vec3, p3: &Vec3) -> Vec3 {
    normalize(&(triangle_normal(p0, p1, p3) + triangle_normal(p2, p3, p1)))
}

pub fn triangle_tangents_fromuv(
    p0: &Vec3,
    p1: &Vec3,
    p2: &Vec3,
    uv0: &Vec2,
    uv1: &Vec2,
    uv2: &Vec2,
) -> (Vec3, Vec3) {
    // Follows the definition in http://www.terathon.com/code/tangent.html and
    // https://gist.github.com/aras-p/2843984
    // normal points up from texture space
    let p = p1 - p0;
    let q = p2 - p0;
    let s = vec2(uv1.x - uv0.x, uv2.x - uv0.x);
    let t = vec2(uv1.y - uv0.y, uv2.y - uv0.y);
    let div = s.x * t.y - s.y * t.x;

    if div != 0.0 {
        let tu = vec3(
            t.y * p.x - t.x * q.x,
            t.y * p.y - t.x * q.y,
            t.y * p.z - t.x * q.z,
        ) / div;
        let tv = vec3(
            s.x * q.x - s.y * p.x,
            s.x * q.y - s.y * p.y,
            s.x * q.z - s.y * p.z,
        ) / div;
        (tu, tv)
    } else {
        (vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0))
    }
}

pub fn quad_tangents_fromuv(
    p0: &Vec3,
    p1: &Vec3,
    p2: &Vec3,
    p3: &Vec3,
    uv0: &Vec2,
    uv1: &Vec2,
    uv2: &Vec2,
    uv3: &Vec2,
    current_uv: &Vec2,
) -> (Vec3, Vec3) {
    if current_uv.x + current_uv.y <= 1.0 {
        triangle_tangents_fromuv(p0, p1, p3, uv0, uv1, uv3)
    } else {
        triangle_tangents_fromuv(p2, p3, p1, uv2, uv3, uv1)
    }
}

#[inline(always)]
pub fn orthonormalize(a: &Vec3, b: &Vec3) -> Vec3 {
    normalize(&(a - b * dot(a, b)))
}

#[inline(always)]
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

#[inline(always)]
pub fn basis_fromz(normal: &Vec3) -> Mat3 {
    let z = normalize(normal);
    let sign = 1.0_f32.copysign(z.z);
    let a = -1.0 / (sign + z.z);
    let b = z.x * z.y * a;
    let x = vec3(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x);
    let y = vec3(b, sign + z.y * z.y * a, -z.y);

    make_mat3(&[x.as_slice(), y.as_slice(), z.as_slice()].concat())
}

#[inline(always)]
pub fn sample_disk(ruv: Vec2) -> Vec2 {
    let r = f32::sqrt(ruv.y);
    let phi = 2.0 * PI * ruv.x;
    vec2(f32::cos(phi) * r, f32::sin(phi) * r)
}

#[inline(always)]
pub fn to_srgb(component: f32, gamma: f32) -> u8 {
    (component.max(0.0).min(1.0).powf(1.0 / gamma) * 255.0) as u8
}

#[inline(always)]
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

#[inline(always)]
pub fn is_finite(weight: &Vec3) -> bool {
    f32::is_finite(weight.x) && f32::is_finite(weight.y) && f32::is_finite(weight.z)
}

#[inline(always)]
pub fn is_delta(material: &MaterialPoint) -> bool {
    (material.m_type == MaterialType::Reflective && material.roughness == 0.0)
        || (material.m_type == MaterialType::Refractive && material.roughness == 0.0)
        || (material.m_type == MaterialType::Transparent && material.roughness == 0.0)
        || (material.m_type == MaterialType::Volumetric)
}

#[inline(always)]
pub fn is_volumetric(material: &MaterialPoint) -> bool {
    material.m_type == MaterialType::Refractive
        || material.m_type == MaterialType::Volumetric
        || material.m_type == MaterialType::Subsurface
}

pub fn mean3(vec: &Vec3) -> f32 {
    glm::comp_add(&vec) / 3.0
}
