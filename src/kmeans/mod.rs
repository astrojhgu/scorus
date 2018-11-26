use crate::linear_space::traits::*;
use num_traits::float::Float;
use std::ops::{Add, Mul, Sub};

pub fn kmeans<Scalar, T>(points: Vec<T>, mut seeds: Vec<T>, niter:usize) -> Vec<Vec<T>>
where
    T: PDInnerProdSpace<Scalar>,
    for<'a> &'a T: Add<Output = T>,
    for<'a> &'a T: Sub<Output = T>,
    for<'a> &'a T: Mul<Scalar, Output = T>,
    Scalar: Float,
{
    let k = seeds.len();

    let mut iter_cnt=0;
    let cluster_id=
    loop {
        let cluster_id: Vec<_> = points
            .iter()
            .map(|p| {
                seeds
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (i, s.distance_to(p)))
                    .min_by(|a, b| {
                        if a.1 < b.1 {
                            std::cmp::Ordering::Less
                        } else if a.1 > b.1 {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    }).expect("")
                    .0
            }).collect();
        seeds.iter_mut().for_each(|i| {
            *i = (i as &T) * Scalar::zero();
        });

        let mut cnts = vec![Scalar::zero(); k];

        cluster_id.iter().zip(points.iter()).for_each(|(&cid, p)| {
            seeds[cid] = &seeds[cid] + p;
            cnts[cid] = cnts[cid] + Scalar::one();
        });

        if iter_cnt==niter{
            break cluster_id;
        }

        seeds.iter_mut().zip(cnts.iter()).for_each(|(a, &c)| {
            if c == Scalar::zero() {
                panic!("seed not valid");
            }
            *a = (a as &T) * (Scalar::one() / c);
        });
        iter_cnt+=1;
    };

    let mut result:Vec<Vec<_>>=(0..k).map(|_x|{vec![]}).collect();
    cluster_id.iter().zip(points.into_iter()).for_each(|(&i, p)|{
        result[i].push(p);
    });
    result
}
