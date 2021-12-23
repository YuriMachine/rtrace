use crate::utils::inverse_frame;
use crate::{scene::Scene, trace::Ray, zero2};
use glm::{vec2, vec4};
use glm::{Vec2, Vec4};

#[derive(Debug)]
pub struct BvhIntersection {
    pub instance: usize,
    pub element: usize,
    pub uv: Vec2,
    pub distance: f32,
    pub hit: bool,
}

impl Default for BvhIntersection {
    fn default() -> Self {
        BvhIntersection {
            instance: usize::MAX,
            element: usize::MAX,
            uv: zero2!(),
            distance: 0.0,
            hit: false,
        }
    }
}

pub struct BvhData<'a> {
    pub scene: embree::Scene<'a>,
    pub shapes: Vec<embree::Scene<'a>>,
}

impl BvhData<'_> {
    pub fn from_scene<'a>(
        device: &'a embree::Device,
        scene: &Scene,
        highquality: bool,
    ) -> BvhData<'a> {
        let mut bvh_shapes = Vec::with_capacity(scene.shapes.len());

        for shape in &scene.shapes {
            let escene = embree::Scene::new(device);
            unsafe {
                if highquality {
                    embree::sys::rtcSetSceneBuildQuality(
                        escene.handle(),
                        embree::BuildQuality::HIGH,
                    );
                } else {
                    embree::sys::rtcSetSceneFlags(escene.handle(), embree::SceneFlags::COMPACT);
                }
            }
            if !shape.lines.is_empty() {
                let mut elines: Vec<i32> = Vec::new();
                let mut epositions: Vec<Vec4> = Vec::new();
                let mut last_index = -1;
                for l in &shape.lines {
                    if last_index == l.x {
                        elines.push(epositions.len() as i32 - 1);
                        let posy = &shape.positions[l.y as usize];
                        let rady = shape.radius[l.y as usize];
                        epositions.push(vec4(posy.x, posy.y, posy.z, rady));
                    } else {
                        elines.push(epositions.len() as i32);
                        let posx = &shape.positions[l.x as usize];
                        let radx = shape.radius[l.x as usize];
                        epositions.push(vec4(posx.x, posx.y, posx.z, radx));
                        let posy = &shape.positions[l.y as usize];
                        let rady = shape.radius[l.y as usize];
                        epositions.push(vec4(posy.x, posy.y, posy.z, rady));
                    }
                    last_index = l.y;
                }
                unsafe {
                    use embree::sys::*;
                    use embree::*;
                    let egeometry = rtcNewGeometry(device.handle, GeometryType::FLAT_LINEAR_CURVE);
                    rtcSetGeometryVertexAttributeCount(egeometry, 1);

                    let embree_positions = rtcSetNewGeometryBuffer(
                        egeometry,
                        BufferType::VERTEX,
                        0,
                        Format::FLOAT4,
                        4 * 4,
                        epositions.len(),
                    );
                    let embree_lines = rtcSetNewGeometryBuffer(
                        egeometry,
                        BufferType::INDEX,
                        0,
                        Format::UINT,
                        4,
                        elines.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        epositions.as_ptr() as *mut std::ffi::c_void,
                        embree_positions,
                        epositions.len() * 16,
                    );
                    std::ptr::copy_nonoverlapping(
                        elines.as_ptr() as *mut std::ffi::c_void,
                        embree_lines,
                        elines.len() * 4,
                    );
                    rtcCommitGeometry(egeometry);
                    rtcAttachGeometryByID(escene.handle(), egeometry, 0);
                    rtcReleaseGeometry(egeometry);
                }
            } else if !shape.triangles.is_empty() {
                unsafe {
                    use embree::sys::*;
                    use embree::*;
                    let egeometry = rtcNewGeometry(device.handle, GeometryType::TRIANGLE);
                    rtcSetGeometryVertexAttributeCount(egeometry, 1);

                    let embree_positions = rtcSetNewGeometryBuffer(
                        egeometry,
                        BufferType::VERTEX,
                        0,
                        Format::FLOAT3,
                        3 * 4,
                        shape.positions.len(),
                    );
                    let embree_triangles = rtcSetNewGeometryBuffer(
                        egeometry,
                        BufferType::INDEX,
                        0,
                        Format::UINT3,
                        3 * 4,
                        shape.triangles.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        shape.positions.as_ptr() as *mut std::ffi::c_void,
                        embree_positions,
                        shape.positions.len() * 12,
                    );
                    std::ptr::copy_nonoverlapping(
                        shape.triangles.as_ptr() as *mut std::ffi::c_void,
                        embree_triangles,
                        shape.triangles.len() * 12,
                    );
                    rtcCommitGeometry(egeometry);
                    rtcAttachGeometryByID(escene.handle(), egeometry, 0);
                    rtcReleaseGeometry(egeometry);
                }
            } else if !shape.quads.is_empty() {
                unsafe {
                    use embree::sys::*;
                    use embree::*;
                    let egeometry = rtcNewGeometry(device.handle, GeometryType::QUAD);
                    rtcSetGeometryVertexAttributeCount(egeometry, 1);

                    let embree_positions = rtcSetNewGeometryBuffer(
                        egeometry,
                        BufferType::VERTEX,
                        0,
                        Format::FLOAT3,
                        3 * 4,
                        shape.positions.len(),
                    );
                    let embree_quads = rtcSetNewGeometryBuffer(
                        egeometry,
                        BufferType::INDEX,
                        0,
                        Format::UINT4,
                        4 * 4,
                        shape.quads.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        shape.positions.as_ptr() as *mut std::ffi::c_void,
                        embree_positions,
                        shape.positions.len() * 12,
                    );
                    std::ptr::copy_nonoverlapping(
                        shape.quads.as_ptr() as *mut std::ffi::c_void,
                        embree_quads,
                        shape.quads.len() * 16,
                    );
                    rtcCommitGeometry(egeometry);
                    rtcAttachGeometryByID(escene.handle(), egeometry, 0);
                    rtcReleaseGeometry(egeometry);
                }
            } else {
                // handle errors
            }
            escene.commit();
            bvh_shapes.push(escene);
        }

        let mut escene = embree::Scene::new(device);
        unsafe {
            if highquality {
                embree::sys::rtcSetSceneBuildQuality(escene.handle(), embree::BuildQuality::HIGH);
            } else {
                embree::sys::rtcSetSceneFlags(escene.handle(), embree::SceneFlags::COMPACT);
            }
        }

        for instance_id in 0..scene.instances.len() {
            let scene_instance = &scene.instances[instance_id];
            let scene_bvh = &bvh_shapes[scene_instance.shape];
            let mut instance = embree::Instance::unanimated(device, scene_bvh);
            instance.set_transform3x4(&scene_instance.frame);
            let mut geometry = embree::Geometry::Instance(instance);
            geometry.commit();
            escene.attach_geometry(geometry);
        }
        escene.commit();

        BvhData {
            scene: escene,
            shapes: bvh_shapes,
        }
    }

    pub fn intersect(&self, ray: &Ray) -> BvhIntersection {
        let embree_ray = embree::Ray::new(ray.origin, ray.direction, ray.tmin, ray.tmax);
        let mut ray_hit = embree::RayHit::new(embree_ray);
        let mut intersection_ctx = embree::IntersectContext::incoherent();
        self.scene.intersect(&mut intersection_ctx, &mut ray_hit);
        if ray_hit.hit.hit() {
            BvhIntersection {
                instance: ray_hit.hit.instID[0] as usize,
                element: ray_hit.hit.primID as usize,
                uv: vec2(ray_hit.hit.u, ray_hit.hit.v),
                distance: ray_hit.ray.tfar,
                hit: true,
            }
        } else {
            BvhIntersection::default()
        }
    }

    pub fn intersect_instance(
        &self,
        scene: &Scene,
        instance_idx: usize,
        direction: Ray,
    ) -> BvhIntersection {
        let instance = &scene.instances[instance_idx];
        let inv_ray = direction.transform(&inverse_frame(&instance.frame, true));
        let embree_ray = embree::Ray::new(
            inv_ray.origin,
            inv_ray.direction,
            inv_ray.tmin,
            inv_ray.tmax,
        );
        let mut ray_hit = embree::RayHit::new(embree_ray);
        let mut intersection_ctx = embree::IntersectContext::incoherent();
        self.shapes[instance.shape].intersect(&mut intersection_ctx, &mut ray_hit);
        if ray_hit.hit.hit() {
            BvhIntersection {
                instance: instance_idx,
                element: ray_hit.hit.primID as usize,
                uv: vec2(ray_hit.hit.u, ray_hit.hit.v),
                distance: ray_hit.ray.tfar,
                hit: true,
            }
        } else {
            BvhIntersection::default()
        }
    }
}
