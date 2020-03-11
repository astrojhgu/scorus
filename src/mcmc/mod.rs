pub mod arms;
pub mod ensemble_sample;
pub mod functions;
pub mod graph;
pub mod init_ensemble;
pub mod mcmc_errors;
//pub mod mcmc_vec;
pub mod hmc;
pub mod nuts;
pub mod twalk;
pub mod utils;

pub use super::utils::{HasElement, HasLen, InitFromLen, ItemSwapable, Resizeable};
//pub use self::utils::shuffle;
pub use self::init_ensemble::get_one_init_realization;

impl<T> HasLen for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl<T: Default + Clone> InitFromLen for Vec<T> {
    fn init(len: usize) -> Vec<T> {
        vec![T::default(); len]
    }
}

impl<T> HasElement for Vec<T> {
    type ElmType = T;
}

impl<T: Clone> Resizeable for Vec<T> {
    fn resize(&mut self, s: usize, x: T) {
        //(*self).resize_default(x)
        //Vec::resize_default(self, x);
        //(*self).resize(s, x);
        Vec::resize(self, s, x);
    }
}

impl<T> ItemSwapable for Vec<T> {
    fn swap_items(&mut self, i: usize, j: usize) {
        (*self).swap(i, j);
    }
}

pub fn print_length<T: HasLen>(x: &T) -> usize {
    x.len()
}
