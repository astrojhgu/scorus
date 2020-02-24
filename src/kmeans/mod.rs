#![allow(clippy::comparison_chain)]
use crate::linear_space::traits::*;
use num_traits::float::Float;
use std::cmp::Ordering;
use std::ops::{Add, Mul, Sub};
pub fn kmeans_iter<Scalar, T>(points: Vec<Vec<T>>) -> Option<Vec<Vec<T>>>
where
    T: PDInnerProdSpace<Scalar> + Clone + std::fmt::Debug,
    for<'a> &'a T: Add<Output = T>,
    for<'a> &'a T: Sub<Output = T>,
    for<'a> &'a T: Mul<Scalar, Output = T>,
    Scalar: Float + std::fmt::Debug,
{
    let seeds: Vec<Option<T>> = calc_centroids(&points);

    //println!("{:?}", seeds);

    if seeds.iter().any(std::option::Option::is_none) {
        None
    } else {
        let seeds: Vec<_> = seeds.into_iter().map(std::option::Option::unwrap).collect();

        let points: Vec<_> = points.into_iter().flatten().collect();
        let result = classify(&points, &seeds);

        Some(result)
    }
}

pub fn classify<Scalar, T>(points: &[T], centroids: &[T]) -> Vec<Vec<T>>
where
    T: PDInnerProdSpace<Scalar> + Clone + std::fmt::Debug,
    for<'a> &'a T: Add<Output = T>,
    for<'a> &'a T: Sub<Output = T>,
    for<'a> &'a T: Mul<Scalar, Output = T>,
    Scalar: Float + std::fmt::Debug,
{
    let k = centroids.len();
    let mut result: Vec<Vec<T>> = (0..k).map(|_| vec![]).collect();
    points.iter().cloned().for_each(|x| {
        let cid = centroids
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
            })
            .unwrap()
            .0;
        result[cid].push(x);
    });

    result
}

pub fn calc_centroids<Scalar, T>(points: &[Vec<T>]) -> Vec<Option<T>>
where
    T: PDInnerProdSpace<Scalar> + Clone + std::fmt::Debug,
    for<'a> &'a T: Add<Output = T>,
    for<'a> &'a T: Sub<Output = T>,
    for<'a> &'a T: Mul<Scalar, Output = T>,
    Scalar: Float + std::fmt::Debug,
{
    points
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
        })
        .collect()
}

pub fn kmeans2<Scalar, T>(points: &[T], centroids: &mut [T], niter: usize) -> Vec<usize>
where
    T: PDInnerProdSpace<Scalar> + Clone + std::fmt::Debug,
    for<'a> &'a T: Add<Output = T>,
    for<'a> &'a T: Sub<Output = T>,
    for<'a> &'a T: Mul<Scalar, Output = T>,
    Scalar: Float + std::fmt::Debug,
{
    let mut p = classify(points, centroids);
    for _i in 0..niter {
        p = kmeans_iter(p).unwrap();
    }

    let c: Vec<_> = calc_centroids(&p)
        .into_iter()
        .map(std::option::Option::unwrap)
        .collect();

    let labels: Vec<_> = points
        .iter()
        .map(|x| {
            centroids
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
                })
                .unwrap()
                .0
        })
        .collect();

    let _ = c.into_iter().zip(centroids.iter_mut()).map(|(c1, c2)| {
        *c2 = c1;
    });
    labels
}
