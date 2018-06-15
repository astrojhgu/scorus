//! calculate the direction of some certain pixel

use super::super::coordinates::{SphCoord, Vec3d};
use super::utils::isqrt;
use num_traits::float::Float;
use num_traits::float::FloatConst;

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
