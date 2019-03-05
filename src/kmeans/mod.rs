use crate::linear_space::traits::*;
use num_traits::float::Float;
use std::cmp::Ordering;
use std::ops::{Add, Mul, Sub};
pub fn kmeans_iter<Scalar, T>(points: Vec<Vec<T>>) -> Option<Vec<Vec<T>>>
where
    T: PDInnerProdSpace<Scalar> + std::fmt::Debug,
    for<'a> &'a T: Add<Output = T>,
    for<'a> &'a T: Sub<Output = T>,
    for<'a> &'a T: Mul<Scalar, Output = T>,
    Scalar: Float + std::fmt::Debug,
{
    let k = points.len();

    let seeds: Vec<Option<T>> = points
        .iter()
        .map(|subset| {
            if subset.is_empty() {
                None
            } else {
                Some(
                    &subset
                        .iter()
                        .fold(&subset[0] * Scalar::zero(), |a, b| &a + b)
                        * (Scalar::one() / Scalar::from(subset.len()).unwrap()),
                )
            }
        }).collect();

    //println!("{:?}", seeds);

    if seeds.iter().any(std::option::Option::is_none) {
        None
    } else {
        let mut result: Vec<Vec<T>> = (0..k).map(|_| vec![]).collect();
        let seeds: Vec<_> = seeds.into_iter().map(std::option::Option::unwrap).collect();

        points.into_iter().flatten().for_each(|x| {
            let cid = seeds
                .iter()
                .map(|p| p.distance_to(&x))
                .enumerate()
                .min_by(|a, b| {
                    if a.1 > b.1 {
                        Ordering::Greater
                    } else if a.1 < b.1 {
                        Ordering::Less
                    } else {
                        Ordering::Equal
                    }
                }).unwrap()
                .0;
            result[cid].push(x);
        });

        Some(result)
    }
}
