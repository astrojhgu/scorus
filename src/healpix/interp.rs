use num_traits::float::Float;
use num_traits::float::FloatConst;
use num_traits::cast::NumCast;
use super::pix::nside2npix;
use super::super::coordinates::SphCoord;

pub fn ring_above<T>(nside: usize, z: T) -> usize
where
    T: Float,
{
    let twothird = T::from(2.0).unwrap() / T::from(3.0).unwrap();
    let az = z.abs();
    if az < twothird {
        <usize as NumCast>::from(
            T::from(nside).unwrap() * (T::from(2).unwrap() - T::from(1.5).unwrap() * z),
        ).unwrap()
    } else {
        let iring = <usize as NumCast>::from(
            T::from(nside).unwrap() * (T::from(3.0).unwrap() * (T::one() - az)).sqrt(),
        ).unwrap();
        if z > T::zero() {
            iring
        } else {
            4 * nside - iring - 1
        }
    }
}

pub fn get_ring_info2<T>(nside: usize, ring: usize) -> (usize, usize, T, bool)
where
    T: Float + FloatConst,
{
    let pi = T::PI();
    let npix = nside2npix(nside);
    let fact2 = T::from(4.0).unwrap() / T::from(npix).unwrap();
    let fact1 = T::from(nside << 1).unwrap() * fact2;
    let npface = nside * nside;
    let ncap = (npface - nside) << 1;
    let northring = if ring > 2 * nside {
        4 * nside - ring
    } else {
        ring
    };
    let mut theta: T;
    let ringpix: usize;
    let mut startpix: usize;
    let shifted: bool;
    if northring < nside {
        let tmp = T::from(northring).unwrap().powi(2) * fact2;
        let costheta = T::one() - tmp;
        let sintheta = (tmp * (T::from(2).unwrap() - tmp)).sqrt();
        theta = sintheta.atan2(costheta);
        ringpix = 4 * northring;
        shifted = true;
        startpix = 2 * northring * (northring - 1);
    } else {
        theta = ((T::from(2).unwrap() * T::from(nside).unwrap() - T::from(northring).unwrap())
            * fact1)
            .acos();
        ringpix = 4 * nside;
        shifted = (northring - nside) & 1 == 0;
        startpix = ncap + (northring - nside) * ringpix;
    }

    if northring != ring {
        theta = pi - theta;
        startpix = npix - startpix - ringpix;
    }
    (startpix, ringpix, theta, shifted)
}

pub fn get_interpol_ring<T>(nside: usize, ptg: SphCoord<T>) -> (Vec<usize>, Vec<T>)
where
    T: Float + FloatConst,
{
    let npix = nside2npix(nside);
    let two_pi = T::PI() * T::from(2).unwrap();
    let pi = T::PI();
    let z = ptg.pol.cos();
    let ir1 = ring_above(nside, z);
    let ir2 = ir1 + 1;
    let mut pix = vec![0_usize, 0_usize, 0_usize, 0_usize];
    let mut wgt = vec![T::one(), T::one(), T::one(), T::one()];
    let mut theta1 = None;
    let mut theta2 = None;

    if ir1 > 0 {
        let (sp, nr, _theta1, shift) = get_ring_info2::<T>(nside, ir1);
        theta1 = Some(_theta1);
        let dphi = two_pi / T::from(nr).unwrap();
        let tmp = ptg.az / dphi - T::from(if shift { 0.5 } else { 0.0 }).unwrap();
        let mut i1 = if tmp < T::zero() {
            <isize as NumCast>::from(tmp).unwrap() - 1
        } else {
            <isize as NumCast>::from(tmp).unwrap()
        };
        let w1 = (ptg.az
            - (T::from(i1).unwrap() + T::from(if shift { 0.5 } else { 0.0 }).unwrap()) * dphi)
            / dphi;
        let mut i2 = i1 + 1;
        if i1 < 0 {
            i1 = i1 + nr as isize;
        }
        if i2 as usize >= nr {
            i2 = i2 - nr as isize;
        }
        pix[0] = sp + i1 as usize;
        pix[1] = sp + i2 as usize;
        wgt[0] = T::one() - w1;
        wgt[1] = w1;
    }

    if ir2 < (4 * nside) {
        let (sp, nr, _theta2, shift) = get_ring_info2::<T>(nside, ir2);
        theta2 = Some(_theta2);
        let dphi = two_pi / T::from(nr).unwrap();
        let tmp = ptg.az / dphi - T::from(if shift { 0.5 } else { 0.0 }).unwrap();
        let mut i1 = if tmp < T::zero() {
            <isize as NumCast>::from(tmp).unwrap() - 1
        } else {
            <isize as NumCast>::from(tmp).unwrap()
        };
        let w1 = (ptg.az
            - (T::from(i1).unwrap() + T::from(if shift { 0.5 } else { 0.0 }).unwrap()) * dphi)
            / dphi;
        let mut i2 = i1 + 1;

        if i1 < 0 {
            i1 += nr as isize;
        }
        if i2 as usize >= nr {
            i2 -= nr as isize;
        }

        pix[2] = sp + i1 as usize;
        pix[3] = sp + i2 as usize;
        wgt[2] = T::one() - w1;
        wgt[3] = w1;
    }

    if ir1 == 0 {
        let wtheta = ptg.pol / theta2.unwrap();
        wgt[2] = wgt[2] * wtheta;
        wgt[3] = wgt[3] * wtheta;
        let fac = (T::one() - wtheta) * T::from(0.25).unwrap();
        wgt[0] = fac;
        wgt[1] = fac;

        wgt[2] = wgt[2] + fac;
        wgt[3] = wgt[2] + fac;

        pix[0] = (pix[2] + 2_usize) & 3;
        pix[1] = (pix[3] + 2_usize) & 3;
    } else if ir2 == 4 * nside {
        let wtheta = (ptg.pol - theta1.unwrap()) / (pi - theta1.unwrap());
        wgt[0] = wgt[0] * (T::one() - wtheta);
        wgt[1] = wgt[1] * (T::one() - wtheta);
        let fac = wtheta * T::from(0.25).unwrap();
        wgt[0] = wgt[0] + fac;
        wgt[1] = wgt[1] + fac;
        wgt[2] = fac;
        wgt[3] = fac;
        pix[2] = ((pix[0] + 2) & 3) + npix - 4;
        pix[3] = ((pix[1] + 2) & 3) + npix - 4;
    } else {
        let wtheta = (ptg.pol - theta1.unwrap()) / (theta2.unwrap() - theta1.unwrap());
        wgt[0] = wgt[0] * (T::one() - wtheta);
        wgt[1] = wgt[1] * (T::one() - wtheta);
        wgt[2] = wgt[2] * wtheta;
        wgt[3] = wgt[3] * wtheta;
    }
    (pix, wgt)
}