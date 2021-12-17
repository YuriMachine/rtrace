use crate::bvh::BvhIntersection;
use crate::utils::*;
use glm::{clamp, dot, log, mat3x4, vec3, vec4};
use glm::{Vec3, Vec4};
use std::f32::consts::PI;
const INVALID: usize = usize::MAX;
const RAY_EPS: f32 = 1e-4;
const MIN_ROUGHNESS: f32 = 0.03 * 0.03;
use crate::scene_components::*;

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
