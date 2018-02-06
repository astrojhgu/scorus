#![feature(vec_resize_default)]
extern crate num_traits;
extern crate rand;
extern crate scoped_threadpool;
extern crate special;
extern crate rayon;
//use num_traits::float::Float;

pub mod mcmc;
pub mod utils;
pub mod integration;
pub mod opt;
pub mod polynomial;
