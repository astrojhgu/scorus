//! calculate the direction of some certain pixel

use super::utils::{
    isqrt, nside2npix, ring2xyf64, xyf2ring64, NB_FACEARRAY, NB_SWAPARRAY, NB_XOFFSET, NB_YOFFSET,
};
use crate::coordinates::{SphCoord, Vec3d};
use num::traits::float::{Float, FloatConst};

fn pix2ang_ring_z_phi<T>(nside: usize, pix: usize) -> (T, T)
where
    T: Float + FloatConst,
{
    let half_pi = T::FRAC_PI_2();
    let pi = T::PI();
    let ncap = nside * (nside - 1) * 2;
    let npix = 12_usize * nside * nside;
    let fact2 = T::from(4.0).unwrap() / T::from(npix).unwrap();
    if pix < ncap {
        let iring = (1 + isqrt(1 + 2 * pix)) / 2;
        let iphi = (pix + 1) - 2 * iring * (iring - 1);

        let z = T::one() - T::from(iring * iring).unwrap() * fact2;
        let phi = (T::from(iphi).unwrap() - T::one() / T::from(2).unwrap()) * half_pi
            / T::from(iring).unwrap();
        (z, phi)
    } else if pix < (npix - ncap) {
        let fact1 = T::from(nside).unwrap() * T::from(2.0).unwrap() * fact2;
        let ip = pix - ncap;
        let iring = ip / (4 * nside) + nside;
        let iphi = ip % (4 * nside) + 1;
        let fodd = if (iring + nside) & 1 != 0 {
            T::one()
        } else {
            T::from(0.5).unwrap()
        };
        let nl2 = 2 * nside;
        let z = (T::from(nl2).unwrap() - T::from(iring).unwrap()) * fact1;
        let phi = (T::from(iphi).unwrap() - fodd) * pi / T::from(nl2).unwrap();
        (z, phi)
    } else {
        let ip = npix - pix;
        let iring = (1 + isqrt(2 * ip - 1)) / 2;
        let iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));

        let z = -T::one() + T::from(iring * iring).unwrap() * fact2;
        let phi =
            (T::from(iphi).unwrap() - T::from(0.5).unwrap()) * half_pi / T::from(iring).unwrap();
        (z, phi)
    }
}

/// pixel to SphCoord point
pub fn pix2ang_ring<T>(nside: usize, ipix: usize) -> SphCoord<T>
where
    T: Float + FloatConst,
{
    let (z, phi) = pix2ang_ring_z_phi::<T>(nside, ipix);
    SphCoord::new(z.acos(), phi)
}

/// pixel to 3D vector
pub fn pix2vec_ring<T>(nside: usize, ipix: usize) -> Vec3d<T>
where
    T: Float + FloatConst,
{
    Vec3d::from_sph_coord(pix2ang_ring(nside, ipix))
}

pub fn pix2ring_ring(nside: usize, ipix: usize) -> usize {
    let ncap = nside * (nside - 1) * 2;
    let npix = nside2npix(nside);
    if ipix < ncap {
        // North Polar cap
        (1 + isqrt(1 + 2 * ipix)) >> 1 // counted from North pole
    } else if ipix < (npix - ncap) {
        // Equatorial region
        (ipix - ncap) / (4 * nside) + nside // counted from North pole
    } else {
        // South Polar cap
        4 * nside - ((1 + isqrt(2 * (npix - ipix) - 1)) >> 1)
    }
}

pub fn ring2z_ring<T>(nside: usize, iring: usize) -> T
where
    T: Float + FloatConst,
{
    let npix = nside2npix(nside) as isize;
    let iring = iring as isize;
    let nside = nside as isize;
    let fact2 = T::from(4).unwrap() / T::from(npix).unwrap();
    let fact1 = T::from(nside).unwrap() * T::from(2).unwrap() * fact2;
    if iring < nside {
        T::one() - T::from(iring.pow(2)).unwrap() * fact2
    } else if iring <= 3 * nside {
        T::from(2 * nside - iring).unwrap() * fact1
    } else {
        let iring = 4 * nside - iring;
        T::from(iring.pow(2)).unwrap() * fact2 - T::one()
    }
}

pub fn neighbors_ring(nside: usize, ipix: usize) -> Vec<usize> {
    //let mut result = [0_isize, 0, 0, 0, 0, 0, 0, 0];
    let (ix, iy, face_num) = ring2xyf64(nside as i64, ipix as i64);
    let nsm1 = nside - 1;
    if ix > 0 && ix < nsm1 as i32 && iy > 0 && iy < nsm1 as i32 {
        (0..8)
            .map(|m| {
                xyf2ring64(
                    nside as i64,
                    ix + NB_XOFFSET[m],
                    iy + NB_YOFFSET[m],
                    face_num,
                ) as usize
            })
            .collect()
    } else {
        (0..8)
            .filter_map(|i| {
                let mut x = ix + NB_XOFFSET[i];
                let mut y = iy + NB_YOFFSET[i];
                let mut nbnum = 4;
                if x < 0 {
                    x += nside as i32;
                    nbnum -= 1;
                } else if x >= nside as i32 {
                    x -= nside as i32;
                    nbnum += 1;
                }
                if y < 0 {
                    y += nside as i32;
                    nbnum -= 3;
                } else if y >= nside as i32 {
                    y -= nside as i32;
                    nbnum += 3;
                }
                let f = NB_FACEARRAY[nbnum][face_num as usize];
                if f >= 0 {
                    let bits = NB_SWAPARRAY[nbnum][face_num as usize >> 2];
                    if bits & 1 != 0 {
                        x = nside as i32 - x - 1;
                    }
                    if bits & 2 != 0 {
                        y = nside as i32 - y - 1;
                    }
                    if bits & 4 != 0 {
                        std::mem::swap(&mut x, &mut y);
                    }
                    Some(xyf2ring64(nside as i64, x, y, f) as usize)
                } else {
                    None
                }
            })
            .collect()
    }
}
