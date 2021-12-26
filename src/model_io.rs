pub mod ply {
    use crate::scene_components::*;
    use crate::*;
    use glm::{vec2, vec3, vec4};
    use glm::{TVec2, Vec2, Vec3, Vec4};
    use linked_hash_map::LinkedHashMap;
    use ply_rs::ply::Property;

    #[inline(always)]
    pub fn get_positions(vertex: &LinkedHashMap<String, Property>, positions: &mut Vec<Vec3>) {
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

    #[inline(always)]
    pub fn get_normals(vertex: &LinkedHashMap<String, Property>, normals: &mut Vec<Vec3>) {
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

    #[inline(always)]
    pub fn get_texcoords(vertex: &LinkedHashMap<String, Property>, texcoords: &mut Vec<Vec2>) {
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

    #[inline(always)]
    pub fn get_radius(vertex: &LinkedHashMap<String, Property>, radii: &mut Vec<f32>) {
        if vertex.get("radius").is_none() {
            return;
        }
        let radius = match vertex["radius"] {
            Property::Float(c) => c,
            _ => unreachable!(),
        };
        radii.push(radius);
    }

    #[inline(always)]
    pub fn get_colors(vertex: &LinkedHashMap<String, Property>, colors: &mut Vec<Vec4>) {
        if vertex.get("red").is_none()
            || vertex.get("green").is_none()
            || vertex.get("blue").is_none()
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

    #[inline(always)]
    pub fn get_faces(face: &LinkedHashMap<String, Property>, shape: &mut Shape) {
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

    #[inline(always)]
    pub fn get_points(point: &LinkedHashMap<String, Property>, points: &mut Vec<i32>) {
        if point.get("vertex_indices").is_none() {
            return;
        }
        let vertex_indices = match point["vertex_indices"] {
            Property::Int(c) => c,
            _ => unreachable!(),
        };
        points.push(vertex_indices);
    }

    #[inline(always)]
    pub fn get_lines(line: &LinkedHashMap<String, Property>, lines: &mut Vec<TVec2<i32>>) {
        if line.get("vertex_indices").is_none() {
            return;
        }
        let vertex_indices = match &line["vertex_indices"] {
            Property::ListInt(c) => c,
            _ => unreachable!(),
        };
        lines.push(vec2(vertex_indices[0], vertex_indices[1]))
    }
}
