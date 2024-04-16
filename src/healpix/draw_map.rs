use num::traits::{Float, FloatConst};
use std::ops::Add;

use crate::{
    coordinates::{SphCoord, Vec3d},
    healpix::{
        pix::{pix2ang_ring, pix2vec_ring},
        utils::{npix2nside, nside2npix},
    },
};

pub fn create_map_sph<T, U, F>(nside: usize, f: F) -> Vec<T>
where
    F: FnMut(SphCoord<U>) -> T,
    U: Float + FloatConst,
{
    (0..nside2npix(nside))
        .map(|ipix| pix2ang_ring(nside, ipix))
        .map(f)
        .collect()
}

pub fn create_map_vec<T, U, F>(nside: usize, f: F) -> Vec<T>
where
    F: FnMut(Vec3d<U>) -> T,
    U: Float + FloatConst,
{
    (0..nside2npix(nside))
        .map(|ipix| pix2vec_ring(nside, ipix))
        .map(f)
        .collect()
}

pub fn add_to_map_sph<T, U, F>(map: &mut [T], mut f: F)
where
    F: FnMut(SphCoord<U>) -> T,
    U: Float + FloatConst,
    T: Add<T, Output = T> + Copy,
{
    let npix = map.len();
    let nside = npix2nside(npix);
    map.iter_mut().enumerate().for_each(|(ipix, x)| {
        let dir = pix2ang_ring(nside, ipix);
        *x = *x + f(dir);
    })
}

pub fn add_to_map_vec<T, U, F>(map: &mut [T], mut f: F)
where
    F: FnMut(Vec3d<U>) -> T,
    U: Float + FloatConst,
    T: Add<T, Output = T> + Copy,
{
    let npix = map.len();
    let nside = npix2nside(npix);
    map.iter_mut().enumerate().for_each(|(ipix, x)| {
        let dir = pix2vec_ring(nside, ipix);
        *x = *x + f(dir);
    })
}
