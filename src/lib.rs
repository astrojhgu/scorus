#![feature(vec_resize_default)]
extern crate num_traits;
extern crate scoped_threadpool;

extern crate rand;
mod utils;
mod ensemble_sample;
use num_traits::float::Float;
pub use utils::HasLength;
pub use utils::Resizeable;
pub use utils::ItemSwapable;
pub use utils::shuffle;
pub use utils::draw_z;
pub use ensemble_sample::sample;

impl<T> HasLength for Vec<T> {
    fn length(&self) -> usize {
        (*self).len()
    }
}


impl<T: std::default::Default> Resizeable for Vec<T> {
    fn resize(&mut self, x: usize) {
        (*self).resize_default(x)
    }
}

impl<T> ItemSwapable for Vec<T> {
    fn swap_items(&mut self, i: usize, j: usize) {
        (*self).swap(i, j);
    }
}

pub fn bar<T:Float+rand::Rand+std::cmp::PartialOrd+rand::distributions::range::SampleRange, U: rand::Rng>(rng: &mut U, x1:T, x2:T)-> T{
    let xx = vec![1, 2, 3];
    xx.length();
    rng.gen_range(x2 - x1, x2 + x1)
}

pub fn print_length<T: HasLength>(x: &T) -> usize {
    x.length()
}
