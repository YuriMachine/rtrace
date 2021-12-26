use indicatif::ProgressBar;
use rtrace::bvh::*;
use rtrace::scene::*;
use rtrace::trace;
use rtrace::utils::*;

pub fn main() {
    // load params
    let args = RaytraceParams::get_args();
    let params = RaytraceParams::from_args(&args);
    let scene_path = args.value_of("scene").unwrap();

    // scene progress bar
    println!("Loading scene...");
    let scene_bar = ProgressBar::new(1);
    scene_bar.inc(0);

    // load scene
    let scene = Scene::from_json(scene_path);
    let device = embree::Device::new();
    let bvh = BvhData::from_scene(&device, &scene, false);
    let mut state = RaytraceState::from_scene(&scene, &params);
    scene_bar.finish();

    // rendering progress bar
    println!("Rendering...");
    let samples_bar = ProgressBar::new(params.samples as u64);
    let style = indicatif::ProgressStyle::default_bar()
        .template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
        )
        .progress_chars("#>-");
    samples_bar.set_style(style);

    // start rendering
    for _ in 0..params.samples {
        trace::raytrace_samples(&mut state, &params, &scene, &bvh);
        samples_bar.inc(1);
    }
    samples_bar.finish();

    // output final image
    let output_path = args.value_of("output").unwrap();
    state.save_image(output_path);
}
