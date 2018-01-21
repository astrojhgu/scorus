use std::default::Default;

pub mod utils;
pub mod ensemble_sample;
pub mod ptsample;
pub mod init_ensemble;
pub mod mcmc_errors;
pub mod arms;
pub mod functions;
pub mod graph;

pub use super::utils::{HasLength, ItemSwapable, Resizeable};
pub use self::utils::shuffle;
pub use self::init_ensemble::get_one_init_realization;

impl<T> HasLength for Vec<T> {
    fn length(&self) -> usize {
        (*self).len()
    }
}

impl<T: Default> Resizeable for Vec<T> {
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
