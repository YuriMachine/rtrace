use image::ImageBuffer;
use rtrace::bvh::*;
use rtrace::scene::*;
use rtrace::trace::*;
use rtrace::utils::*;

pub fn main() {
    //let scene = Scene::make_cornellbox();
    //let scene = Scene::from_json("C:\\Users\\porcu\\Documents\\University\\computer graphics\\02_pathtrace_out\\tests\\01_cornellbox\\cornellbox.json");
    let scene = Scene::from_json("C:\\Users\\porcu\\Documents\\University\\computer graphics\\02_pathtrace_out\\tests\\02_matte\\matte.json");
    //println!("{:#?}", scene);
    let device = embree::Device::new();
    let bvh = BvhData::from_scene(&device, &scene, false);
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
            let scaled = rgb / state.samples as f32;
            image_bytes.push(to_srgb(scaled.x, 2.2));
            image_bytes.push(to_srgb(scaled.y, 2.2));
            image_bytes.push(to_srgb(scaled.z, 2.2));
            /*
            let wrapped = image::Rgb([rgb.x, rgb.y, rgb.z]);
            let rgb8 = image::codecs::hdr::to_rgbe8(wrapped);
                .to_ldr_scale_gamma(1.0 / state.samples as f32, 1.0);
            image_bytes.push(rgb8[0]);
            image_bytes.push(rgb8[1]);
            image_bytes.push(rgb8[2]);
            */
        }
    }
    let img: image::RgbImage =
        ImageBuffer::from_raw(state.width as u32, state.height as u32, image_bytes)
            .expect("Image buffer has incorrect size");
    img.save("output.png").expect("Failed to save image");
}
