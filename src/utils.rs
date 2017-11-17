extern crate std;
extern crate rand;
use num_traits::float::Float;
use num_traits::identities::one;
use num_traits::identities::zero;
pub trait HasLength {
    fn length(&self) -> usize;
}

pub trait Resizeable {
    fn resize(&mut self, usize);
}

pub trait ItemSwapable {
    fn swap_items(&mut self, i: usize, j: usize);
}

pub fn shuffle<T: HasLength + Clone + ItemSwapable, U: rand::Rng>(arr: &T, rng: &mut U) -> T {
    let mut x = arr.clone();
    let l = arr.length();
    for i in (1..l).rev() {
        let i1 = rng.gen_range(0, i + 1);
        x.swap_items(i, i1);
    }
    x
}

pub fn draw_z<
    T: Float
        + rand::Rand
        + std::cmp::PartialOrd
        + rand::distributions::range::SampleRange,
    U: rand::Rng,
>(
    rng: &mut U,
    a: T,
) -> T {
    let sqrt_a: T = a.sqrt();
    let unit: T = one();
    let two = unit + unit;
    let p: T = rng.gen_range(zero(), two * (sqrt_a - unit / sqrt_a));
    let y: T = unit / sqrt_a + p / (two);
    y * y
}
