use rtrace::bvh::*;
use rtrace::components::*;
use rtrace::trace::*;
use rtrace::utils::*;

pub fn main() {
    let scene = Scene::make_cornellbox();
    let device = embree::Device::new();
    let bvh = BvhData::from_scene(&device, &scene, true);
    let params = RaytraceParams::default();
    if params.noparallel {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();
    }
    let mut state = RaytraceState::from_scene(&scene, &params);
    for _ in 0..params.samples {
        raytrace_samples(&mut state, &params, &scene, &bvh);
    }

    let mut image_bytes = Vec::with_capacity(state.width * state.height * 3);
    for pixel in state.image.chunks(state.width) {
        for rgb in pixel {
            let wrapped = image::Rgb([rgb.x, rgb.y, rgb.z]);
            let rgb8 = image::codecs::hdr::to_rgbe8(wrapped)
                .to_ldr_scale_gamma(1.0 / params.samples as f32, 1.0);
            image_bytes.push(rgb8[0]);
            image_bytes.push(rgb8[1]);
            image_bytes.push(rgb8[2]);
        }
    }
    image::save_buffer(
        "output.png",
        &image_bytes,
        state.width as u32,
        state.height as u32,
        image::ColorType::Rgb8,
    )
    .expect("Failed to save output image");
}
