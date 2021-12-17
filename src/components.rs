use crate::bvh::BvhIntersection;
use crate::utils::*;
use glm::{clamp, dot, log, mat3x4, normalize, triangle_normal, vec2, vec3, vec4};
use glm::{BVec4, Mat3x4, TVec2, TVec3, TVec4, Vec2, Vec3, Vec4};
use std::f32::consts::PI;
const INVALID: usize = usize::MAX;
const RAY_EPS: f32 = 1e-4;
const MIN_ROUGHNESS: f32 = 0.03 * 0.03;

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
            origin: Vec3::zeros(),
            direction: vec3(0.0, 0.0, 1.0),
            tmin: RAY_EPS,
            tmax: f32::MAX,
        }
    }
}

#[derive(Debug)]
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

#[derive(PartialEq, Copy, Clone)]
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

pub struct Material {
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
            _ => Vec3::zeros(),
        }
    }

    pub fn eval_bsdfcos(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if self.roughness == 0.0 {
            return Vec3::zeros();
        }
        match self.m_type {
            MaterialType::Matte => self.eval_matte(normal, outgoing, incoming),
            _ => Vec3::zeros(),
        }
    }

    pub fn sample_bsdfcos_pdf(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        if self.roughness == 0.0 {
            return 0.0;
        }
        match self.m_type {
            MaterialType::Matte => self.sample_matte_pdf(normal, outgoing, incoming),
            _ => 0.0,
        }
    }

    fn sample_matte(&self, normal: &Vec3, outgoing: &Vec3, rn: &Vec2) -> Vec3 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        MaterialPoint::sample_hemisphere_cos(&up_normal, rn)
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
        MaterialPoint::sample_hemisphere_cos_pdf(&up_normal, incoming)
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
}

#[derive(Default)]
pub struct Texture {
    pub width: i32,
    pub height: i32,
    pub linear: bool,
    pub pixelsf: Vec<Vec4>,
    pub pixelsb: Vec<BVec4>,
}

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

#[derive(Default)]
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
}

impl Shape {
    fn eval_position(&self, intersection: &BvhIntersection) -> Vec3 {
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

    fn eval_normal(&self, instance: &Instance, intersection: &BvhIntersection) -> Vec3 {
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

#[derive(Default)]
pub struct Scene {
    pub cameras: Vec<Camera>,
    pub instances: Vec<Instance>,
    pub environments: Vec<Environment>,
    pub shapes: Vec<Shape>,
    pub textures: Vec<Texture>,
    pub materials: Vec<Material>,
}

impl Scene {
    pub fn eval_shading_position(&self, intersection: &BvhIntersection) -> Vec3 {
        let instance = &self.instances[intersection.instance];
        let shape = &self.shapes[instance.shape];
        if !shape.triangles.is_empty() || !shape.quads.is_empty() {
            self.eval_position(instance, intersection)
        } else if !shape.lines.is_empty() {
            self.eval_position(instance, intersection)
        } else if !shape.points.is_empty() {
            shape.eval_position(intersection)
        } else {
            Vec3::zeros()
        }
    }

    fn eval_position(&self, instance: &Instance, intersection: &BvhIntersection) -> Vec3 {
        let shape = &self.shapes[instance.shape];
        let element = intersection.element;
        if !shape.triangles.is_empty() {
            let triangle = &shape.triangles[element];
            transform_point(
                &instance.frame,
                &interpolate_triangle(
                    &shape.positions[triangle.x as usize],
                    &shape.positions[triangle.y as usize],
                    &shape.positions[triangle.z as usize],
                    &intersection.uv,
                ),
            )
        } else if !shape.quads.is_empty() {
            let quad = &shape.quads[element];
            transform_point(
                &instance.frame,
                &interpolate_quad(
                    &shape.positions[quad.x as usize],
                    &shape.positions[quad.y as usize],
                    &shape.positions[quad.z as usize],
                    &shape.positions[quad.w as usize],
                    &intersection.uv,
                ),
            )
        } else if !shape.lines.is_empty() {
            let line = &shape.lines[element];
            transform_point(
                &instance.frame,
                &interpolate_line(
                    &shape.positions[line.x as usize],
                    &shape.positions[line.y as usize],
                    intersection.uv.x,
                ),
            )
        } else if !shape.points.is_empty() {
            let point = shape.points[element];
            transform_point(&instance.frame, &shape.positions[point as usize])
        } else {
            Vec3::zeros()
        }
    }

    pub fn eval_shading_normal(&self, intersection: &BvhIntersection, outgoing: &Vec3) -> Vec3 {
        let instance = &self.instances[intersection.instance];
        let shape = &self.shapes[instance.shape];
        let material = &self.materials[instance.material];
        let uv = intersection.uv;
        if !shape.triangles.is_empty() || !shape.quads.is_empty() {
            let normal = if material.normal_tex == INVALID {
                self.eval_normal(instance, intersection)
            } else {
                self.eval_normalmap(instance, intersection)
            };
            if dot(&normal, outgoing) >= 0.0 || material.m_type == MaterialType::Refractive {
                normal
            } else {
                -normal
            }
        } else if !shape.lines.is_empty() {
            let normal = self.eval_normal(instance, intersection);
            orthonormalize(outgoing, &normal)
        } else if !shape.points.is_empty() {
            // HACK: sphere
            if true {
                transform_direction_frame(
                    &instance.frame,
                    &vec3(
                        f32::cos(2.0 * PI * uv.x) * f32::sin(PI * uv.y),
                        f32::sin(2.0 * PI * uv.x) * f32::sin(PI * uv.y),
                        f32::cos(PI * uv.y),
                    ),
                )
            } else {
                outgoing.clone()
            }
        } else {
            Vec3::zeros()
        }
    }

    fn eval_normal(&self, instance: &Instance, intersection: &BvhIntersection) -> Vec3 {
        let shape = &self.shapes[instance.shape];
        let element = intersection.element;
        if shape.normals.is_empty() {
            return shape.eval_normal(instance, intersection);
        }
        if !shape.triangles.is_empty() {
            let triangle = shape.triangles[element];
            transform_normal_frame(
                &instance.frame,
                &interpolate_triangle(
                    &shape.normals[triangle.x as usize],
                    &shape.normals[triangle.y as usize],
                    &shape.normals[triangle.z as usize],
                    &intersection.uv,
                )
                .normalize(),
                false,
            )
        } else if !shape.quads.is_empty() {
            let quad = shape.quads[element];
            transform_normal_frame(
                &instance.frame,
                &interpolate_quad(
                    &shape.normals[quad.x as usize],
                    &shape.normals[quad.y as usize],
                    &shape.normals[quad.z as usize],
                    &shape.normals[quad.w as usize],
                    &intersection.uv,
                )
                .normalize(),
                false,
            )
        } else if !shape.lines.is_empty() {
            let line = shape.lines[element];
            transform_normal_frame(
                &instance.frame,
                &interpolate_line(
                    &shape.normals[line.x as usize],
                    &shape.normals[line.y as usize],
                    intersection.uv.x,
                )
                .normalize(),
                false,
            )
        } else if !shape.points.is_empty() {
            transform_normal_frame(
                &instance.frame,
                &shape.normals[shape.points[element as usize] as usize].normalize(),
                false,
            )
        } else {
            Vec3::zeros()
        }
    }

    fn eval_normalmap(&self, instance: &Instance, intersection: &BvhIntersection) -> Vec3 {
        Vec3::zeros()
    }

    fn eval_color(&self, instance: &Instance, intersection: &BvhIntersection) -> Vec4 {
        let shape = &self.shapes[instance.shape];
        let element = intersection.element;
        if shape.colors.is_empty() {
            return vec4(1.0, 1.0, 1.0, 1.0);
        }
        if !shape.triangles.is_empty() {
            let t = shape.triangles[element];
            interpolate_triangle(
                &shape.colors[t.x as usize],
                &shape.colors[t.y as usize],
                &shape.colors[t.z as usize],
                &intersection.uv,
            )
        } else if !shape.quads.is_empty() {
            let q = shape.quads[element];
            interpolate_quad(
                &shape.colors[q.x as usize],
                &shape.colors[q.y as usize],
                &shape.colors[q.z as usize],
                &shape.colors[q.w as usize],
                &intersection.uv,
            )
        } else if !shape.lines.is_empty() {
            let l = shape.lines[element];
            interpolate_line(
                &shape.colors[l.x as usize],
                &shape.colors[l.y as usize],
                intersection.uv.x,
            )
        } else if !shape.points.is_empty() {
            shape.colors[shape.points[element as usize] as usize]
        } else {
            Vec4::zeros()
        }
    }

    pub fn eval_material(&self, intersection: &BvhIntersection) -> MaterialPoint {
        let instance = &self.instances[intersection.instance];
        let material = &self.materials[instance.material];
        //let texcoord = self.eval_texcoord(instance, intersection);

        // evaluate textures
        /*
        let emission_tex = eval_texture(
            scene, material.emission_tex, texcoord, true);
        let color_tex     = eval_texture(scene, material.color_tex, texcoord, true);
        let roughness_tex = eval_texture(
            scene, material.roughness_tex, texcoord, false);
        let scattering_tex = eval_texture(
            scene, material.scattering_tex, texcoord, true);
        */
        let color_shp = self.eval_color(instance, intersection);

        // material point
        let m_type = material.m_type;
        let emission = material.emission; // * xyz(emission_tex);
        let color = material.color.component_mul(&color_shp.xyz()); // * xyz(color_tex);
        let opacity = material.opacity; // * color_tex.w * color_shp.w;
        let metallic = material.metallic; // * roughness_tex.z;
        let mut roughness = material.roughness; // * roughness_tex.y;
        roughness *= roughness;
        let ior = material.ior;
        let scattering = material.scattering; //* xyz(scattering_tex);
        let scanisotropy = material.scanisotropy;
        let trdepth = material.trdepth;

        // volume density
        let density = if m_type == MaterialType::Refractive
            || m_type == MaterialType::Volumetric
            || m_type == MaterialType::Subsurface
        {
            -log(&clamp(&color, 0.0001, 1.0)) / trdepth
        } else {
            Vec3::zeros()
        };

        // fix roughness
        if m_type == MaterialType::Matte
            || m_type == MaterialType::Gltfpbr
            || m_type == MaterialType::Glossy
        {
            roughness = roughness.clamp(MIN_ROUGHNESS, 1.0);
        } else if m_type == MaterialType::Volumetric {
            roughness = 0.0;
        } else {
            if roughness < MIN_ROUGHNESS {
                roughness = 0.0;
            }
        }

        MaterialPoint {
            m_type,
            emission,
            color,
            roughness,
            metallic,
            ior,
            density,
            scattering,
            scanisotropy,
            trdepth,
            opacity,
        }
    }

    fn eval_texcoord(&self, instance: &Instance, intersection: &BvhIntersection) {
        todo!()
    }

    pub fn make_cornellbox() -> Scene {
        let mut cameras = Vec::new();
        let mut shapes = Vec::new();
        let mut materials = Vec::new();
        let mut instances = Vec::new();

        let camera = Camera {
            frame: mat3x4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 3.9),
            orthographic: false,
            lens: 0.035,
            film: 0.024,
            aspect: 1.0,
            focus: 3.9,
            aperture: 0.0,
        };
        cameras.push(camera);
        // floor
        let floor_shape = Shape {
            positions: vec![
                vec3(-1.0, 0.0, 1.0),
                vec3(1.0, 0.0, 1.0),
                vec3(1.0, 0.0, -1.0),
                vec3(-1.0, 0.0, -1.0),
            ],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0)],
            ..Default::default()
        };
        shapes.push(floor_shape);
        let floor_material = Material {
            color: vec3(0.725, 0.71, 0.68),
            ..Default::default()
        };
        materials.push(floor_material);
        let floor_instance = Instance {
            shape: shapes.len() - 1,
            material: materials.len() - 1,
            ..Default::default()
        };
        instances.push(floor_instance);
        // ceiling
        let ceiling_shape = Shape {
            positions: vec![
                vec3(-1.0, 2.0, 1.0),
                vec3(-1.0, 2.0, -1.0),
                vec3(1.0, 2.0, -1.0),
                vec3(1.0, 2.0, 1.0),
            ],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0)],
            ..Default::default()
        };
        shapes.push(ceiling_shape);
        let ceiling_material = Material {
            color: vec3(0.725, 0.71, 0.68),
            ..Default::default()
        };
        materials.push(ceiling_material);
        let ceiling_instance = Instance {
            shape: shapes.len() - 1,
            material: materials.len() - 1,
            ..Default::default()
        };
        instances.push(ceiling_instance);
        // backwall
        let backwall_shape = Shape {
            positions: vec![
                vec3(-1.0, 0.0, -1.0),
                vec3(1.0, 0.0, -1.0),
                vec3(1.0, 2.0, -1.0),
                vec3(-1.0, 2.0, -1.0),
            ],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0)],
            ..Default::default()
        };
        shapes.push(backwall_shape);
        let backwall_material = Material {
            color: vec3(0.725, 0.71, 0.68),
            ..Default::default()
        };
        materials.push(backwall_material);
        let backwall_instance = Instance {
            shape: shapes.len() - 1,
            material: materials.len() - 1,
            ..Default::default()
        };
        instances.push(backwall_instance);
        // rightwall
        let rightwall_shape = Shape {
            positions: vec![
                vec3(1.0, 0.0, -1.0),
                vec3(1.0, 0.0, 1.0),
                vec3(1.0, 2.0, 1.0),
                vec3(1.0, 2.0, -1.0),
            ],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0)],
            ..Default::default()
        };
        shapes.push(rightwall_shape);
        let rightwall_material = Material {
            color: vec3(0.14, 0.45, 0.091),
            ..Default::default()
        };
        materials.push(rightwall_material);
        let rightwall_instance = Instance {
            shape: shapes.len() - 1,
            material: materials.len() - 1,
            ..Default::default()
        };
        instances.push(rightwall_instance);
        // leftwall
        let leftwall_shape = Shape {
            positions: vec![
                vec3(-1.0, 0.0, 1.0),
                vec3(-1.0, 0.0, -1.0),
                vec3(-1.0, 2.0, -1.0),
                vec3(-1.0, 2.0, 1.0),
            ],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0)],
            ..Default::default()
        };
        shapes.push(leftwall_shape);
        let leftwall_material = Material {
            color: vec3(0.63, 0.065, 0.05),
            ..Default::default()
        };
        materials.push(leftwall_material);
        let leftwall_instance = Instance {
            shape: shapes.len() - 1,
            material: materials.len() - 1,
            ..Default::default()
        };
        instances.push(leftwall_instance);
        // shortbox
        let shortbox_shape = Shape {
            positions: vec![
                vec3(0.53, 0.6, 0.75),
                vec3(0.7, 0.6, 0.17),
                vec3(0.13, 0.6, 0.0),
                vec3(-0.05, 0.6, 0.57),
                vec3(-0.05, 0.0, 0.57),
                vec3(-0.05, 0.6, 0.57),
                vec3(0.13, 0.6, 0.0),
                vec3(0.13, 0.0, 0.0),
                vec3(0.53, 0.0, 0.75),
                vec3(0.53, 0.6, 0.75),
                vec3(-0.05, 0.6, 0.57),
                vec3(-0.05, 0.0, 0.57),
                vec3(0.7, 0.0, 0.17),
                vec3(0.7, 0.6, 0.17),
                vec3(0.53, 0.6, 0.75),
                vec3(0.53, 0.0, 0.75),
                vec3(0.13, 0.0, 0.0),
                vec3(0.13, 0.6, 0.0),
                vec3(0.7, 0.6, 0.17),
                vec3(0.7, 0.0, 0.17),
                vec3(0.53, 0.0, 0.75),
                vec3(0.7, 0.0, 0.17),
                vec3(0.13, 0.0, 0.0),
                vec3(-0.05, 0.0, 0.57),
            ],
            triangles: vec![
                vec3(0, 1, 2),
                vec3(2, 3, 0),
                vec3(4, 5, 6),
                vec3(6, 7, 4),
                vec3(8, 9, 10),
                vec3(10, 11, 8),
                vec3(12, 13, 14),
                vec3(14, 15, 12),
                vec3(16, 17, 18),
                vec3(18, 19, 16),
                vec3(20, 21, 22),
                vec3(22, 23, 20),
            ],
            ..Default::default()
        };
        shapes.push(shortbox_shape);
        let shortbox_material = Material {
            color: vec3(0.725, 0.71, 0.68),
            ..Default::default()
        };
        materials.push(shortbox_material);
        let shortbox_instance = Instance {
            shape: shapes.len() - 1,
            material: materials.len() - 1,
            ..Default::default()
        };
        instances.push(shortbox_instance);
        // tallbox
        let tallbox_shape = Shape {
            positions: vec![
                vec3(-0.53, 1.2, 0.09),
                vec3(0.04, 1.2, -0.09),
                vec3(-0.14, 1.2, -0.67),
                vec3(-0.71, 1.2, -0.49),
                vec3(-0.53, 0.0, 0.09),
                vec3(-0.53, 1.2, 0.09),
                vec3(-0.71, 1.2, -0.49),
                vec3(-0.71, 0.0, -0.49),
                vec3(-0.71, 0.0, -0.49),
                vec3(-0.71, 1.2, -0.49),
                vec3(-0.14, 1.2, -0.67),
                vec3(-0.14, 0.0, -0.67),
                vec3(-0.14, 0.0, -0.67),
                vec3(-0.14, 1.2, -0.67),
                vec3(0.04, 1.2, -0.09),
                vec3(0.04, 0.0, -0.09),
                vec3(0.04, 0.0, -0.09),
                vec3(0.04, 1.2, -0.09),
                vec3(-0.53, 1.2, 0.09),
                vec3(-0.53, 0.0, 0.09),
                vec3(-0.53, 0.0, 0.09),
                vec3(0.04, 0.0, -0.09),
                vec3(-0.14, 0.0, -0.67),
                vec3(-0.71, 0.0, -0.49),
            ],
            triangles: vec![
                vec3(0, 1, 2),
                vec3(2, 3, 0),
                vec3(4, 5, 6),
                vec3(6, 7, 4),
                vec3(8, 9, 10),
                vec3(10, 11, 8),
                vec3(12, 13, 14),
                vec3(14, 15, 12),
                vec3(16, 17, 18),
                vec3(18, 19, 16),
                vec3(20, 21, 22),
                vec3(22, 23, 20),
            ],
            ..Default::default()
        };
        shapes.push(tallbox_shape);
        let tallbox_material = Material {
            color: vec3(0.725, 0.71, 0.68),
            ..Default::default()
        };
        materials.push(tallbox_material);
        let tallbox_instance = Instance {
            shape: shapes.len() - 1,
            material: materials.len() - 1,
            ..Default::default()
        };
        instances.push(tallbox_instance);
        // light
        let light_shape = Shape {
            positions: vec![
                vec3(-0.25, 1.99, 0.25),
                vec3(-0.25, 1.99, -0.25),
                vec3(0.25, 1.99, -0.25),
                vec3(0.25, 1.99, 0.25),
            ],
            triangles: vec![vec3(0, 1, 2), vec3(2, 3, 0)],
            ..Default::default()
        };
        shapes.push(light_shape);
        let light_material = Material {
            emission: vec3(17.0, 12.0, 4.0),
            ..Default::default()
        };
        materials.push(light_material);
        let light_instance = Instance {
            shape: shapes.len() - 1,
            material: materials.len() - 1,
            ..Default::default()
        };
        instances.push(light_instance);
        Scene {
            cameras,
            instances,
            shapes,
            materials,
            ..Default::default()
        }
    }
}
