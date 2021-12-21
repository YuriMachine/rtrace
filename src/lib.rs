#![warn(clippy::all)]
#![warn(rust_2018_idioms)]

extern crate nalgebra_glm as glm;

pub mod bvh;
pub mod scene;
pub mod scene_components;
pub mod shading;
pub mod trace;
pub mod utils;
