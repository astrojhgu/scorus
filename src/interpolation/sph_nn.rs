use num_traits::float::Float;
use num_traits::Bounded;
use rand::Rng;
use rand::seq::SliceRandom;
use std::fmt::Debug;
use std::marker::Copy;

use super::super::coordinates::sphcoord::SphCoord;
use super::super::space_search::{MetricPoint, VpTree};
use std::ops::{Add, Div, Mul};

impl<T> MetricPoint for SphCoord<T>
where
    T: Float + Copy + Bounded + Debug,
{
    type Distance = T;
    fn distance_to(&self, other: &Self) -> T {
        let d = self.angle_between(*other);
        //println!("d={:?}", d);
        d
    }
}

pub struct NNInterpBuilder<T, V>
where
    T: Float + Copy + Bounded + Debug,
    V: Copy + Default + Debug,
{
    points: Vec<(SphCoord<T>, V)>,
}

pub struct Interpolator<T, V>
where
    T: Float + Copy + Bounded + Debug,
    V: Copy + Default + Debug,
{
    tree: VpTree<SphCoord<T>, V>,
}

impl<
        T: 'static + Float + Copy + Bounded + Debug,
        V: 'static
            + Copy
            + Default
            + Add<V, Output = V>
            + Mul<T, Output = V>
            + Div<T, Output = V>
            + Debug,
    > NNInterpBuilder<T, V>
{
    pub fn add_point(mut self, p: SphCoord<T>, v: V) -> NNInterpBuilder<T, V> {
        self.points.push((p, v));
        self
    }

    pub fn shuffle<U: Rng>(mut self, rng: &mut U) -> NNInterpBuilder<T, V> {
        //rng.shuffle(&mut (self.points));
        self.points.shuffle(rng);
        self
    }

    pub fn done(self) -> Box<Fn(&SphCoord<T>, usize) -> V> {
        let ipt = Interpolator {
            tree: VpTree::new(self.points),
        };
        Box::new(move |x: &SphCoord<T>, n: usize| ipt.interp(x, n))
    }
}

#[allow(non_snake_case)]
impl<
        T: Float + Copy + Bounded + Debug,
        V: Copy + Default + Add<V, Output = V> + Mul<T, Output = V> + Div<T, Output = V> + Debug,
    > Interpolator<T, V>
{
    pub fn new() -> NNInterpBuilder<T, V> {
        NNInterpBuilder { points: Vec::new() }
    }

    pub fn interp(&self, p: &SphCoord<T>, n: usize) -> V {
        let mut points = self.tree.search(p, n);
        let mut result = V::default();
        let mut wgt = T::zero();
        let ONE = T::one();
        //println!("{:?} {:?}", p.pol, p.az);
        //println!("{}", points.len());
        while let Some(x) = points.pop() {
            //  println!("{:?} {:?}", x.dist, self.values[x.idx]);
            if x.dist == T::zero() {
                return self.tree.items[x.index].1;
            }
            result = result + self.tree.items[x.index].1 * (ONE / x.dist);
            wgt = wgt + ONE / x.dist;
        }
        result / wgt
    }
}
