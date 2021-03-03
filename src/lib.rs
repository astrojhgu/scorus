//#![feature(with_options)]
//! # Scientific Computing for Rust-lang

extern crate num_traits;
extern crate rand;
extern crate rand_distr;
extern crate rayon;
extern crate special;
//extern crate pdqselect;
//use num_traits::float::Float;

/// Coordinate-system related computings
pub mod coordinates;

/// Healpix utilities, including pixel indexing and interpolation
pub mod healpix;

/// Numerical integration, currently only simposons method is implemented
pub mod integration;

/// Interpolation methods
pub mod interpolation;

/// map projection
pub mod map_proj;

/// Monte-Carlo Markov Chain-based statistical inferring
pub mod mcmc;

/// Numerical optimization
pub mod opt;

/// Evaluation of ordinary and legendre polynomials
pub mod polynomial;

/// Generate random vectors
pub mod rand_vec;

/// Search nearest points on a sphere
pub mod space_search;

/// Spherical tessellation
pub mod sph_tessellation;

/// Some utilities used by other modules
pub mod utils;

///some basic math functions
pub mod basic;

//Linear space
pub mod linear_space;

pub mod kmeans;

pub mod autodiff;
