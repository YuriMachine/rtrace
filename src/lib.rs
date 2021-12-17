#![warn(clippy::all)]
#![warn(rust_2018_idioms)]

extern crate nalgebra_glm as glm;
#[allow(dead_code)]
pub mod bvh;
#[allow(dead_code)]
pub mod scene;
#[allow(dead_code)]
pub mod scene_components;
#[allow(dead_code)]
pub mod trace;
#[allow(dead_code)]
pub mod utils;
