use clap::{App, Arg};
use image::ImageBuffer;
use indicatif::ProgressBar;
use rtrace::bvh::*;
use rtrace::scene::*;
use rtrace::trace::*;
use rtrace::utils::*;

pub fn main() {
    let matches = App::new("rtrace")
        .version("0.1.0")
        .author("Vincenzo Guarino")
        .about("Rust porting of Yocto/GL")
        .arg(
            Arg::with_name("scene")
                .long("--scene")
                .takes_value(true)
                .required(true)
                .help("JSON scene file"),
        )
        .arg(
            Arg::with_name("output")
                .short("--out")
                .long("--output")
                .takes_value(true)
                .default_value("out.png")
                .help("Output path"),
        )
        .arg(
            Arg::with_name("resolution")
                .short("--res")
                .long("--resolution")
                .takes_value(true)
                .default_value("1280")
                .help("Image resolution"),
        )
        .arg(
            Arg::with_name("shader")
                .long("--shader")
                .takes_value(true)
                .default_value("raytrace")
                .help("shader type"),
        )
        .arg(
            Arg::with_name("samples")
                .long("--samples")
                .takes_value(true)
                .default_value("256")
                .help("number of samples"),
        )
        .arg(
            Arg::with_name("bounces")
                .long("--bounces")
                .takes_value(true)
                .default_value("8")
                .help("number of bounces"),
        )
        .arg(
            Arg::with_name("clamp")
                .long("--clamp")
                .takes_value(true)
                .default_value("10.0")
                .help("clamp value"),
        )
        .arg(
            Arg::with_name("noparallel")
                .long("--noparallel")
                .takes_value(true)
                .default_value("false")
                .help("disable threading"),
        )
        .get_matches();

    let scene_path = matches.value_of("scene").unwrap();
    let output_path = matches.value_of("output").unwrap();
    println!("Loading scene...");
    let scene_bar = ProgressBar::new(1);
    scene_bar.inc(0);
    let scene = Scene::from_json(scene_path);
    scene_bar.finish();
    let device = embree::Device::new();
    let bvh = BvhData::from_scene(&device, &scene, false);
    let params = RaytraceParams::from_args(&matches);
    if params.noparallel {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();
    }
    let mut state = RaytraceState::from_scene(&scene, &params);
    println!("Rendering...");
    let samples_bar = ProgressBar::new(params.samples as u64);
    for _ in 0..params.samples {
        raytrace_samples(&mut state, &params, &scene, &bvh);
        samples_bar.inc(1);
    }
    samples_bar.finish();

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
    img.save(output_path).expect("Failed to save image");
}
