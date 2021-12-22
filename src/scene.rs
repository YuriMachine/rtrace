use crate::bvh::BvhIntersection;
use crate::shading::MaterialPoint;
use crate::{one4, scene_components::*, zero3, zero4};
use crate::{utils::*, zero2};
use glm::{clamp, dot, log, mat3x4, vec2, vec3, vec4, TVec2, Vec2};
use glm::{Vec3, Vec4};
use linked_hash_map::LinkedHashMap;
use ply::ply::Property;
use ply_rs as ply;
use serde::Deserialize;
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

const INVALID: usize = usize::MAX;
const MIN_ROUGHNESS: f32 = 0.03 * 0.03;

#[derive(Default, Debug, Deserialize)]
#[serde(default)]
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
            zero3!()
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
            zero3!()
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
            zero3!()
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
            zero3!()
        }
    }

    fn eval_normalmap(&self, instance: &Instance, intersection: &BvhIntersection) -> Vec3 {
        let shape = &self.shapes[instance.shape];
        let material = &self.materials[instance.material];
        let element = intersection.element;
        // apply normal mapping
        let normal = self.eval_normal(instance, intersection);
        let texcoord = self.eval_texcoord(instance, intersection);
        if material.normal_tex != INVALID
            && (!shape.triangles.is_empty() || !shape.quads.is_empty())
        {
            /*
            let normalmap  = -1.0 + 2.0 * self.eval_texture(material.normal_tex, &texcoord, false, false, false).xyz();
            let (tu, tv)    = eval_element_tangents(scene, instance, element);
            let frame       = frame3f{tu, tv, normal, {0, 0, 0}};
            frame.x          = orthonormalize(frame.x, frame.z);
            frame.y          = normalize(cross(frame.z, frame.x));
            let flip_v      = dot(frame.y, tv) < 0;
            normalmap.y *= flip_v ? 1 : -1;  // flip vertical axis
            normal = transform_normal(frame, normalmap);
            */
        }
        todo!()
    }

    fn eval_color(&self, instance: &Instance, intersection: &BvhIntersection) -> Vec4 {
        let shape = &self.shapes[instance.shape];
        let element = intersection.element;
        if shape.colors.is_empty() {
            return one4!();
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
            zero4!()
        }
    }

    pub fn eval_environment(&self, direction: Vec3) -> Vec3 {
        let mut emission = zero3!();
        for environment in &self.environments {
            let wl =
                transform_direction_frame(&inverse_frame(&environment.frame, false), &direction);
            let mut texcoord = vec2(
                f32::atan2(wl.z, wl.x) / (2.0 * PI),
                f32::acos(wl.y.clamp(-1.0, 1.0)) / PI,
            );
            if texcoord.x < 0.0 {
                texcoord.x += 1.0;
            }

            let texture = self
                .eval_texture(environment.emission_tex, &texcoord, false, false, false)
                .xyz();
            emission += environment.emission.component_mul(&texture);
        }
        emission
    }

    pub fn eval_texture(
        &self,
        texture_idx: usize,
        uv: &Vec2,
        as_linear: bool,
        no_interpolation: bool,
        clamp_to_edge: bool,
    ) -> Vec4 {
        if texture_idx == INVALID {
            return one4!();
        }
        let texture = &self.textures[texture_idx];
        if texture.width == 0 || texture.height == 0 {
            return zero4!();
        }
        // get coordinates normalized for tiling
        let (s, t) = if clamp_to_edge {
            (
                uv.x.clamp(0.0, 1.0) * texture.width as f32,
                uv.y.clamp(0.0, 1.0) * texture.height as f32,
            )
        } else {
            (
                if ((uv.x % 1.0) * texture.width as f32) < 0.0 {
                    (uv.x % 1.0) * texture.width as f32 + texture.width as f32
                } else {
                    (uv.x % 1.0) * texture.width as f32
                },
                if ((uv.y % 1.0) * texture.height as f32) < 0.0 {
                    (uv.y % 1.0) * texture.height as f32 + texture.height as f32
                } else {
                    (uv.y % 1.0) * texture.height as f32
                },
            )
        };

        // get image coordinates and residuals
        let i = (s as u32).clamp(0, texture.width - 1);
        let j = (t as u32).clamp(0, texture.height - 1);
        let ii = (i + 1) % texture.width;
        let jj = (j + 1) % texture.height;
        let u = s - i as f32;
        let v = t - j as f32;

        // handle interpolation
        if no_interpolation {
            texture.lookup(i, j, as_linear)
        } else {
            texture.lookup(i, j, as_linear) * (1.0 - u) * (1.0 - v)
                + texture.lookup(i, jj, as_linear) * (1.0 - u) * v
                + texture.lookup(ii, j, as_linear) * u * (1.0 - v)
                + texture.lookup(ii, jj, as_linear) * u * v
        }
    }

    pub fn eval_material(&self, intersection: &BvhIntersection) -> MaterialPoint {
        let instance = &self.instances[intersection.instance];
        let material = &self.materials[instance.material];
        let texcoord = &self.eval_texcoord(instance, intersection);

        // evaluate textures
        let emission_tex = self.eval_texture(material.emission_tex, texcoord, true, false, false);
        let color_tex = self.eval_texture(material.color_tex, texcoord, true, false, false);
        let roughness_tex =
            self.eval_texture(material.roughness_tex, texcoord, false, false, false);
        let scattering_tex =
            self.eval_texture(material.scattering_tex, texcoord, true, false, false);
        let color_shp = self.eval_color(instance, intersection);

        // material point
        let m_type = material.m_type;
        let emission = material.emission.component_mul(&emission_tex.xyz());
        let color = material
            .color
            .component_mul(&color_shp.xyz())
            .component_mul(&color_tex.xyz());
        let opacity = material.opacity * color_tex.w * color_shp.w;
        let metallic = material.metallic * roughness_tex.z;
        let mut roughness = material.roughness * roughness_tex.y;
        roughness *= roughness;
        let ior = material.ior;
        let scattering = material.scattering.component_mul(&(scattering_tex).xyz());
        let scanisotropy = material.scanisotropy;
        let trdepth = material.trdepth;

        // volume density
        let density = if m_type == MaterialType::Refractive
            || m_type == MaterialType::Volumetric
            || m_type == MaterialType::Subsurface
        {
            -log(&clamp(&color, 0.0001, 1.0)) / trdepth
        } else {
            zero3!()
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

    fn eval_texcoord(&self, instance: &Instance, intersection: &BvhIntersection) -> Vec2 {
        let shape = &self.shapes[instance.shape];
        let element = intersection.element;
        let uv = intersection.uv;
        if shape.texcoords.is_empty() {
            return uv;
        }
        if !shape.triangles.is_empty() {
            let t = shape.triangles[element];
            return interpolate_triangle(
                &shape.texcoords[t.x as usize],
                &shape.texcoords[t.y as usize],
                &shape.texcoords[t.z as usize],
                &uv,
            );
        } else if !shape.quads.is_empty() {
            let q = shape.quads[element];
            return interpolate_quad(
                &shape.texcoords[q.x as usize],
                &shape.texcoords[q.y as usize],
                &shape.texcoords[q.z as usize],
                &shape.texcoords[q.w as usize],
                &uv,
            );
        } else if !shape.lines.is_empty() {
            let l = shape.lines[element];
            return interpolate_line(
                &shape.texcoords[l.x as usize],
                &shape.texcoords[l.y as usize],
                uv.x,
            );
        } else if !shape.points.is_empty() {
            return shape.texcoords[shape.points[element as usize] as usize];
        } else {
            return zero2!();
        }
    }

    pub fn from_json<P: AsRef<Path> + Copy>(path: P) -> Scene {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let mut scene: Scene = serde_json::from_reader(reader).expect("unable to parse JSON");

        for shape in &mut scene.shapes {
            if !shape.uri.is_empty() {
                let mut ply_file =
                    File::open(path.as_ref().parent().unwrap().join(&shape.uri)).unwrap();
                let parser = ply::parser::Parser::<ply::ply::DefaultElement>::new();
                let ply = parser.read_ply(&mut ply_file).unwrap();

                if ply.payload.get("vertex").is_some() {
                    for vertex in &ply.payload["vertex"] {
                        ply_get_positions(vertex, &mut shape.positions);
                        ply_get_normals(vertex, &mut shape.normals);
                        ply_get_texcoords(vertex, &mut shape.texcoords);
                        ply_get_colors(vertex, &mut shape.colors);
                        ply_get_radius(vertex, &mut shape.radius);
                    }
                }
                if ply.payload.get("face").is_some() {
                    for face in &ply.payload["face"] {
                        ply_get_faces(face, shape);
                    }
                }
                if ply.payload.get("line").is_some() {
                    for line in &ply.payload["line"] {
                        ply_get_lines(line, &mut shape.lines);
                    }
                }
                if ply.payload.get("point").is_some() {
                    for point in &ply.payload["point"] {
                        ply_get_points(point, &mut shape.points);
                    }
                }
            }
        }

        for texture in &mut scene.textures {
            if !texture.uri.is_empty() {
                let path = path.as_ref().parent().unwrap().join(&texture.uri);
                let extension = match path.extension() {
                    None => "",
                    Some(os_str) => {
                        if os_str == "png" {
                            "png"
                        } else if os_str == "hdr" {
                            "hdr"
                        } else {
                            ""
                        }
                    }
                };
                if extension == "png" {
                    let rgba = image::open(path).unwrap().into_rgba8();
                    texture.height = rgba.height();
                    texture.width = rgba.width();
                    texture.linear = false;
                    texture.bytes = rgba;
                } else if extension == "hdr" {
                    let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
                    let decoder = image::codecs::hdr::HdrDecoder::new(file).unwrap();
                    let metadata = decoder.metadata();
                    texture.height = metadata.height;
                    texture.width = metadata.width;
                    texture.linear = true;
                    texture.hdr = decoder.read_image_hdr().unwrap();
                }
            }
        }
        scene
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

fn ply_get_positions(vertex: &LinkedHashMap<String, Property>, positions: &mut Vec<Vec3>) {
    if vertex.get("x").is_none() || vertex.get("y").is_none() || vertex.get("z").is_none() {
        return;
    }
    let x = match vertex["x"] {
        Property::Float(c) => c,
        _ => unreachable!(),
    };
    let y = match vertex["y"] {
        Property::Float(c) => c,
        _ => unreachable!(),
    };
    let z = match vertex["z"] {
        Property::Float(c) => c,
        _ => unreachable!(),
    };
    positions.push(vec3(x, y, z));
}

fn ply_get_normals(vertex: &LinkedHashMap<String, Property>, normals: &mut Vec<Vec3>) {
    if vertex.get("nx").is_none() || vertex.get("ny").is_none() || vertex.get("nz").is_none() {
        return;
    }
    let nx = match vertex["nx"] {
        Property::Float(c) => c,
        _ => unreachable!(),
    };
    let ny = match vertex["ny"] {
        Property::Float(c) => c,
        _ => unreachable!(),
    };
    let nz = match vertex["nz"] {
        Property::Float(c) => c,
        _ => unreachable!(),
    };
    normals.push(vec3(nx, ny, nz));
}

fn ply_get_texcoords(vertex: &LinkedHashMap<String, Property>, texcoords: &mut Vec<Vec2>) {
    if vertex.get("u").is_none() || vertex.get("v").is_none() {
        return;
    }
    let u = match vertex["u"] {
        Property::Float(c) => c,
        _ => unreachable!(),
    };
    let v = match vertex["v"] {
        Property::Float(c) => c,
        _ => unreachable!(),
    };
    // flipping
    texcoords.push(vec2(u, 1.0 - v));
}

fn ply_get_radius(vertex: &LinkedHashMap<String, Property>, radii: &mut Vec<f32>) {
    if vertex.get("radius").is_none() {
        return;
    }
    let radius = match vertex["radius"] {
        Property::Float(c) => c,
        _ => unreachable!(),
    };
    radii.push(radius);
}

fn ply_get_colors(vertex: &LinkedHashMap<String, Property>, colors: &mut Vec<Vec4>) {
    if vertex.get("red").is_none() || vertex.get("green").is_none() || vertex.get("blue").is_none()
    {
        return;
    }
    let red = match vertex["red"] {
        Property::Float(c) => c,
        _ => unreachable!(),
    };
    let green = match vertex["green"] {
        Property::Float(c) => c,
        _ => unreachable!(),
    };
    let blue = match vertex["blue"] {
        Property::Float(c) => c,
        _ => unreachable!(),
    };
    if vertex.get("alpha").is_some() {
        let alpha = match vertex["alpha"] {
            Property::Float(c) => c,
            _ => unreachable!(),
        };
        colors.push(vec4(red, green, blue, alpha));
    } else {
        colors.push(vec4(red, green, blue, 1.0));
    }
}

fn ply_get_faces(face: &LinkedHashMap<String, Property>, shape: &mut Shape) {
    if face.get("vertex_indices").is_none() {
        return;
    }
    let vertex_indices = match &face["vertex_indices"] {
        Property::ListInt(c) => c,
        _ => unreachable!(),
    };
    if vertex_indices.len() == 3 {
        shape.triangles.push(vec3(
            vertex_indices[0],
            vertex_indices[1],
            vertex_indices[2],
        ));
    } else if vertex_indices.len() == 4 {
        shape.quads.push(vec4(
            vertex_indices[0],
            vertex_indices[1],
            vertex_indices[2],
            vertex_indices[3],
        ));
    }
}

fn ply_get_points(point: &LinkedHashMap<String, Property>, points: &mut Vec<i32>) {
    if point.get("vertex_indices").is_none() {
        return;
    }
    let vertex_indices = match point["vertex_indices"] {
        Property::Int(c) => c,
        _ => unreachable!(),
    };
    points.push(vertex_indices);
}

fn ply_get_lines(line: &LinkedHashMap<String, Property>, lines: &mut Vec<TVec2<i32>>) {
    if line.get("vertex_indices").is_none() {
        return;
    }
    let vertex_indices = match &line["vertex_indices"] {
        Property::ListInt(c) => c,
        _ => unreachable!(),
    };
    lines.push(vec2(vertex_indices[0], vertex_indices[1]))
}
