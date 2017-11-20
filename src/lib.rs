#![feature(vec_resize_default)]
extern crate num_traits;
extern crate rand;
extern crate scoped_threadpool;

//use num_traits::float::Float;

pub mod utils;
pub mod ensemble_sample;
pub mod ptsample;
pub mod init_ensemble;
pub mod mcmc_errors;
pub use utils::HasLength;
pub use utils::Resizeable;
pub use utils::ItemSwapable;
pub use utils::shuffle;
pub use init_ensemble::get_one_init_realization;


impl<T> HasLength for Vec<T> {
    fn length(&self) -> usize {
        (*self).len()
    }
}


impl<T: std::default::Default> Resizeable for Vec<T> {
    fn resize(&mut self, x: usize) {
        //(*self).resize_default(x)
        Vec::resize_default(self, x);
    }
}

impl<T> ItemSwapable for Vec<T> {
    fn swap_items(&mut self, i: usize, j: usize) {
        (*self).swap(i, j);
    }
}

pub fn print_length<T: HasLength>(x: &T) -> usize {
    x.length()
}
