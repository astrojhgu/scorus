#![feature(vec_resize_default)]
extern crate num_traits;
extern crate rand;
extern crate rayon;
extern crate special;
//use num_traits::float::Float;

pub mod mcmc;
pub mod utils;
pub mod integration;
pub mod opt;
pub mod polynomial;
pub mod coordinates;
pub mod healpix;
pub mod map_proj;
