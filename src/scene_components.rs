use crate::bvh::BvhIntersection;
use crate::trace::Ray;
use crate::utils::*;
use glm::{dot, mat3x4, normalize, triangle_normal, vec2, vec3, vec4};
use glm::{Mat3x4, TVec2, TVec3, TVec4, Vec2, Vec3, Vec4};
use serde::Deserialize;
use std::f32::consts::PI;
const INVALID: usize = usize::MAX;
const MIN_ROUGHNESS: f32 = 0.03 * 0.03;

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

pub struct MaterialPoint {
    pub m_type: MaterialType,
    pub emission: Vec3,
    pub color: Vec3,
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,
    pub density: Vec3,
    pub scattering: Vec3,
    pub scanisotropy: f32,
    pub trdepth: f32,
    pub opacity: f32,
}

impl Default for MaterialPoint {
    fn default() -> Self {
        MaterialPoint {
            m_type: MaterialType::Matte,
            emission: Vec3::zeros(),
            color: Vec3::zeros(),
            roughness: 0.0,
            metallic: 0.0,
            ior: 1.0,
            density: Vec3::zeros(),
            scattering: Vec3::zeros(),
            scanisotropy: 0.0,
            trdepth: 0.01,
            opacity: 1.0,
        }
    }
}

impl MaterialPoint {
    pub fn eval_emission(&self, normal: &Vec3, outgoing: &Vec3) -> Vec3 {
        if dot(normal, outgoing) >= 0.0 {
            self.emission
        } else {
            Vec3::zeros()
        }
    }

    pub fn sample_bsdfcos(&self, normal: &Vec3, outgoing: &Vec3, rnl: f32, rn: &Vec2) -> Vec3 {
        match self.m_type {
            MaterialType::Matte => self.sample_matte(normal, outgoing, rn),
            MaterialType::Glossy => self.sample_glossy(normal, outgoing, rnl, rn),
            _ => Vec3::zeros(),
        }
    }

    pub fn eval_bsdfcos(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if self.roughness == 0.0 {
            return Vec3::zeros();
        }
        match self.m_type {
            MaterialType::Matte => self.eval_matte(normal, outgoing, incoming),
            MaterialType::Glossy => self.eval_glossy(normal, outgoing, incoming),
            _ => Vec3::zeros(),
        }
    }

    pub fn sample_bsdfcos_pdf(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        if self.roughness == 0.0 {
            return 0.0;
        }
        match self.m_type {
            MaterialType::Matte => self.sample_matte_pdf(normal, outgoing, incoming),
            MaterialType::Glossy => self.sample_glossy_pdf(normal, outgoing, incoming),
            _ => 0.0,
        }
    }

    fn sample_matte(&self, normal: &Vec3, outgoing: &Vec3, rn: &Vec2) -> Vec3 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        Self::sample_hemisphere_cos(&up_normal, rn)
    }

    fn eval_matte(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return Vec3::zeros();
        }
        self.color / PI * dot(normal, incoming).abs()
    }

    fn sample_matte_pdf(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return 0.0;
        }
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        Self::sample_hemisphere_cos_pdf(&up_normal, incoming)
    }

    fn sample_glossy(&self, normal: &Vec3, outgoing: &Vec3, rnl: f32, rn: &Vec2) -> Vec3 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        if rnl < Self::fresnel_dielectric(self.ior, &up_normal, outgoing) {
            let halfway = Self::sample_microfacet(self.roughness, &up_normal, rn, true);
            let incoming = glm::reflect_vec(&(-outgoing), &halfway);
            if !Self::same_hemisphere(&up_normal, outgoing, &incoming) {
                Vec3::zeros()
            } else {
                incoming
            }
        } else {
            Self::sample_hemisphere_cos(&up_normal, rn)
        }
    }

    fn eval_glossy(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return Vec3::zeros();
        }
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let F1 = Self::fresnel_dielectric(self.ior, &up_normal, outgoing);
        let halfway = normalize(&(incoming + outgoing));
        let F = Self::fresnel_dielectric(self.ior, &halfway, incoming);
        let D = Self::microfacet_distribution(self.roughness, &up_normal, &halfway, true);
        let G =
            Self::microfacet_shadowing(self.roughness, &up_normal, &halfway, outgoing, incoming);
        self.color * (1.0 - F1) / PI * (dot(&up_normal, incoming).abs())
            + vec3(1.0, 1.0, 1.0) * F * D * G
                / (4.0 * dot(&up_normal, outgoing) * dot(&up_normal, incoming))
                * (dot(&up_normal, incoming).abs())
    }

    fn sample_glossy_pdf(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let halfway = normalize(&(outgoing + incoming));
        let F = Self::fresnel_dielectric(self.ior, &up_normal, outgoing);
        F * Self::sample_microfacet_pdf(self.roughness, &up_normal, &halfway)
            / (4.0 * (dot(outgoing, &halfway).abs()))
            + (1.0 - F) * Self::sample_hemisphere_cos_pdf(&up_normal, incoming)
    }

    pub fn sample_hemisphere_cos(normal: &Vec3, rn: &Vec2) -> Vec3 {
        let z = f32::sqrt(rn.y);
        let r = f32::sqrt(1.0 - z * z);
        let phi = 2.0 * PI * rn.x;
        let local_direction = vec3(r * f32::cos(phi), r * f32::sin(phi), z);
        transform_direction_mat(&basis_fromz(normal), &local_direction)
    }

    fn sample_hemisphere_cos_pdf(normal: &Vec3, incoming: &Vec3) -> f32 {
        let cosw = dot(normal, incoming);
        if cosw <= 0.0 {
            0.0
        } else {
            cosw / PI
        }
    }

    fn fresnel_dielectric(eta: f32, normal: &Vec3, outgoing: &Vec3) -> f32 {
        // Implementation from
        // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
        let cosw = dot(normal, outgoing).abs();

        let sin2 = 1.0 - cosw * cosw;
        let eta2 = eta * eta;

        let cos2t = 1.0 - sin2 / eta2;
        if cos2t < 0.0 {
            return 1.0;
        } // tir

        let t0 = f32::sqrt(cos2t);
        let t1 = eta * t0;
        let t2 = eta * cosw;

        let rs = (cosw - t1) / (cosw + t1);
        let rp = (t0 - t2) / (t0 + t2);

        (rs * rs + rp * rp) / 2.0
    }

    fn microfacet_shadowing(
        roughness: f32,
        normal: &Vec3,
        halfway: &Vec3,
        outgoing: &Vec3,
        incoming: &Vec3,
    ) -> f32 {
        Self::microfacet_shadowing1(roughness, normal, halfway, outgoing, true)
            * Self::microfacet_shadowing1(roughness, normal, halfway, incoming, true)
    }

    fn microfacet_shadowing1(
        roughness: f32,
        normal: &Vec3,
        halfway: &Vec3,
        direction: &Vec3,
        ggx: bool,
    ) -> f32 {
        // https://google.github.io/filament/Filament.html#materialsystem/specularbrdf
        // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
        // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#appendix-b-brdf-implementation
        let cosine = dot(normal, direction);
        let cosineh = dot(halfway, direction);
        if cosine * cosineh <= 0.0 {
            return 0.0;
        }
        let roughness2 = roughness * roughness;
        let cosine2 = cosine * cosine;
        if ggx {
            return 2.0 * cosine.abs()
                / (cosine.abs() + f32::sqrt(cosine2 - roughness2 * cosine2 + roughness2));
        } else {
            let ci = cosine.abs() / (roughness * f32::sqrt(1.0 - cosine2));
            if ci < 1.6 {
                return (3.535 * ci + 2.181 * ci * ci) / (1.0 + 2.276 * ci + 2.577 * ci * ci);
            } else {
                return 1.0;
            }
        }
    }

    fn microfacet_distribution(roughness: f32, normal: &Vec3, halfway: &Vec3, ggx: bool) -> f32 {
        // https://google.github.io/filament/Filament.html#materialsystem/specularbrdf
        // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
        let cosine = dot(normal, halfway);
        if cosine <= 0.0 {
            return 0.0;
        }
        let roughness2 = roughness * roughness;
        let cosine2 = cosine * cosine;
        if ggx {
            return roughness2
                / (PI
                    * (cosine2 * roughness2 + 1.0 - cosine2)
                    * (cosine2 * roughness2 + 1.0 - cosine2));
        } else {
            return f32::exp((cosine2 - 1.0) / (roughness2 * cosine2))
                / (PI * roughness2 * cosine2 * cosine2);
        }
    }

    fn same_hemisphere(normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> bool {
        dot(normal, outgoing) * dot(normal, incoming) >= 0.0
    }

    fn sample_microfacet(roughness: f32, normal: &Vec3, rn: &Vec2, ggx: bool) -> Vec3 {
        let phi = 2.0 * PI * rn.x;
        let roughness2 = roughness * roughness;
        let theta = if ggx {
            f32::atan(roughness * f32::sqrt(rn.y / (1.0 - rn.y)))
        } else {
            f32::atan(f32::sqrt(-roughness2 * f32::ln(1.0 - rn.y)))
        };
        let local_half_vector = vec3(
            f32::cos(phi) * f32::sin(theta),
            f32::sin(phi) * f32::sin(theta),
            f32::cos(theta),
        );
        transform_direction_mat(&basis_fromz(&normal), &local_half_vector)
    }

    fn sample_microfacet_pdf(roughness: f32, normal: &Vec3, halfway: &Vec3) -> f32 {
        let cosine = dot(normal, halfway);
        if cosine < 0.0 {
            0.0
        } else {
            Self::microfacet_distribution(roughness, normal, halfway, true) * cosine
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
