use crate::bvh::BvhIntersection;
use crate::trace::Ray;
use crate::utils::*;
use glm::{mat3x4, normalize, triangle_normal, vec2, vec3, vec4};
use glm::{Mat3x4, TVec2, TVec3, TVec4, Vec2, Vec3, Vec4};
use serde::Deserialize;
const INVALID: usize = usize::MAX;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Camera {
    pub frame: Mat3x4,
    pub orthographic: bool,
    pub lens: f32,
    pub film: f32,
    pub aspect: f32,
    pub focus: f32,
    pub aperture: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            frame: mat3x4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            orthographic: false,
            lens: 0.050,
            film: 0.036,
            aspect: 1.5,
            focus: 10000.0,
            aperture: 0.0,
        }
    }
}

impl Camera {
    pub fn eval(&self, image_uv: Vec2, lens_uv: Vec2) -> Ray {
        let film = if self.aspect >= 1.0 {
            vec2(self.film, self.film / self.aspect)
        } else {
            vec2(self.film * self.aspect, self.film)
        };
        if !self.orthographic {
            let q = vec3(
                film.x * (0.5 - image_uv.x),
                film.y * (image_uv.y - 0.5),
                self.lens,
            );
            // ray direction through the lens center
            let dc = -normalize(&q);
            // point on the lens
            let e = vec3(
                lens_uv.x * self.aperture / 2.0,
                lens_uv.y * self.aperture / 2.0,
                0.0,
            );
            // point on the focus plane
            let p = dc * self.focus / dc.z.abs();
            // correct ray direction to account for camera focusing
            let d = normalize(&(p - e));
            Ray {
                origin: transform_point(&self.frame, &e),
                direction: transform_direction_frame(&self.frame, &d),
                ..Default::default()
            }
        } else {
            let scale = 1.0 / self.lens;
            let q = vec3(
                film.x * (0.5 - image_uv.x) * scale,
                film.y * (image_uv.y - 0.5) * scale,
                self.lens,
            );
            // point on the lens
            let e = vec3(-q.x, -q.y, 0.0)
                + vec3(
                    lens_uv.x * self.aperture / 2.0,
                    lens_uv.y * self.aperture / 2.0,
                    0.0,
                );
            let p = vec3(-q.x, -q.y, -self.focus);
            // correct ray direction to account for camera focusing
            let d = normalize(&(p - e));
            Ray {
                origin: transform_point(&self.frame, &e),
                direction: transform_direction_frame(&self.frame, &d),
                ..Default::default()
            }
        }
    }
}

#[derive(PartialEq, Copy, Clone, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaterialType {
    Matte,
    Glossy,
    Reflective,
    Transparent,
    Refractive,
    Volumetric,
    Subsurface,
    Gltfpbr,
}

#[derive(Deserialize, Debug)]
#[serde(default)]
pub struct Material {
    #[serde(rename = "type")]
    pub m_type: MaterialType,
    pub emission: Vec3,
    pub color: Vec3,
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,
    pub scattering: Vec3,
    pub scanisotropy: f32,
    pub trdepth: f32,
    pub opacity: f32,
    // textures
    pub emission_tex: usize,
    pub color_tex: usize,
    pub roughness_tex: usize,
    pub scattering_tex: usize,
    pub normal_tex: usize,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            m_type: MaterialType::Matte,
            emission: Vec3::zeros(),
            color: Vec3::zeros(),
            roughness: 0.0,
            metallic: 0.0,
            ior: 1.5,
            scattering: Vec3::zeros(),
            scanisotropy: 0.0,
            trdepth: 0.01,
            opacity: 1.0,
            // textures
            emission_tex: INVALID,
            color_tex: INVALID,
            roughness_tex: INVALID,
            scattering_tex: INVALID,
            normal_tex: INVALID,
        }
    }
}

#[derive(Default, Deserialize, Debug)]
#[serde(default)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub linear: bool,
    #[serde(skip)]
    pub hdr: Vec<image::Rgb<f32>>,
    #[serde(skip)]
    pub bytes: image::RgbaImage,
    pub uri: String,
}

impl Texture {
    pub fn lookup(&self, i: u32, j: u32, as_linear: bool) -> Vec4 {
        let color = if !self.hdr.is_empty() {
            // handle hdr
            let color = self.hdr[(j * self.width + i) as usize].0;
            vec4(color[0], color[1], color[2], 1.0)
        } else if !self.bytes.is_empty() {
            // handle bytes
            let color = self.bytes.get_pixel(i, j);
            vec4(
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
                color[3] as f32 / 255.0,
            )
        } else {
            vec4(1.0, 1.0, 1.0, 1.0)
        };
        if as_linear && !self.linear {
            srgb_to_rgb(color)
        } else {
            color
        }
    }
}

#[derive(Deserialize, Debug)]
#[serde(default)]
pub struct Instance {
    pub frame: Mat3x4,
    pub shape: usize,
    pub material: usize,
}

impl Default for Instance {
    fn default() -> Self {
        Instance {
            frame: mat3x4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            shape: INVALID,
            material: INVALID,
        }
    }
}

#[derive(Deserialize, Debug)]
#[serde(default)]
pub struct Environment {
    pub frame: Mat3x4,
    pub emission: Vec3,
    pub emission_tex: usize,
}

impl Default for Environment {
    fn default() -> Self {
        Environment {
            frame: mat3x4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            emission: Vec3::zeros(),
            emission_tex: INVALID,
        }
    }
}

#[derive(Default, Debug, Deserialize)]
#[serde(default)]
pub struct Shape {
    // element data
    // this should be usize but you need to handle embree
    pub points: Vec<i32>,
    pub lines: Vec<TVec2<i32>>,
    pub triangles: Vec<TVec3<i32>>,
    pub quads: Vec<TVec4<i32>>,
    // vertex data
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub texcoords: Vec<Vec2>,
    pub colors: Vec<Vec4>,
    pub radius: Vec<f32>,
    pub tangents: Vec<Vec4>,
    pub uri: String,
}

impl Shape {
    pub fn eval_position(&self, intersection: &BvhIntersection) -> Vec3 {
        let element = intersection.element;
        if !self.triangles.is_empty() {
            let triangle = &self.triangles[element];
            interpolate_triangle(
                &self.positions[triangle.x as usize],
                &self.positions[triangle.y as usize],
                &self.positions[triangle.z as usize],
                &intersection.uv,
            )
        } else if !self.quads.is_empty() {
            let quad = &self.quads[element];
            interpolate_quad(
                &self.positions[quad.x as usize],
                &self.positions[quad.y as usize],
                &self.positions[quad.z as usize],
                &self.positions[quad.w as usize],
                &intersection.uv,
            )
        } else if !self.lines.is_empty() {
            let line = &self.lines[element];
            interpolate_line(
                &self.positions[line.x as usize],
                &self.positions[line.y as usize],
                intersection.uv.x,
            )
        } else if !self.points.is_empty() {
            let point = self.points[element];
            self.positions[point as usize]
        } else {
            Vec3::zeros()
        }
    }

    pub fn eval_normal(&self, instance: &Instance, intersection: &BvhIntersection) -> Vec3 {
        let element = intersection.element;
        if !self.triangles.is_empty() {
            let triangle = self.triangles[element];
            transform_normal_frame(
                &instance.frame,
                &triangle_normal(
                    &self.positions[triangle.x as usize],
                    &self.positions[triangle.y as usize],
                    &self.positions[triangle.z as usize],
                ),
                false,
            )
        } else if !self.quads.is_empty() {
            let quad = self.quads[element];
            let quad_normal = (triangle_normal(
                &self.positions[quad.x as usize],
                &self.positions[quad.y as usize],
                &self.positions[quad.w as usize],
            ) + triangle_normal(
                &self.positions[quad.z as usize],
                &self.positions[quad.w as usize],
                &self.positions[quad.y as usize],
            ))
            .normalize();
            transform_normal_frame(&instance.frame, &quad_normal, false)
        } else if !self.lines.is_empty() {
            let line = self.lines[element];
            transform_normal_frame(
                &instance.frame,
                &(self.positions[line.y as usize] - (self.positions[line.x as usize]).normalize()),
                false,
            )
        } else if self.points.is_empty() {
            vec3(0.0, 0.0, 1.0)
        } else {
            Vec3::zeros()
        }
    }
}
