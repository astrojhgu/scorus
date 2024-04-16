#![allow(clippy::needless_range_loop)]
use super::bas_utils::sqr;
use super::linmin::linmin;
use super::opt_errors::OptErr;
use super::tolerance::Tolerance;
use num::traits::{
    cast::NumCast,
    float::Float,
    identities::{one, zero},
};
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

use crate::linear_space::IndexableLinearSpace;

pub fn fmin<F, V, T, O>(
    f: &F,
    p: &V,
    ftol: Tolerance<T>,
    itmax: usize,
    mut observer: Option<&mut O>,
) -> (V, OptErr)
where
    T: Float + NumCast + std::cmp::PartialOrd + Copy + Debug,
    V: Clone + IndexableLinearSpace<T>,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> T,
    O: FnMut(&V, T),
{
    let two = one::<T>() + one::<T>();
    let n = p.dimension();
    let mut xi = Vec::with_capacity(n);
    for i in 0..n {
        let mut xi1 = p.clone();
        for j in 0..n {
            xi1[j] = if i == j { one::<T>() } else { zero::<T>() };
        }
        xi.push(xi1);
    }

    //let tiny = T::epsilon();
    let mut fret = f(p);

    let mut pt = p.clone();
    let mut fp;
    let mut fptt;
    let mut ibig;
    let mut del;
    let mut xit = p.clone();
    let mut ptt = p.clone();
    let mut result = p.clone();
    for iter in 0..=itmax {
        fp = fret;
        ibig = 0;
        del = zero::<T>();
        for i in 0..n {
            for j in 0..n {
                xit[j] = xi[j][i];
            }
            fptt = fret;
            linmin(f, &mut result, &mut xit, &mut fret);
            if (fptt - fret) > del {
                del = fptt - fret;
                ibig = i;
            }
        }

        if ftol.accepted(fp, fret) {
            return (result, OptErr::Normal);
        }

        if iter == itmax {
            return (result, OptErr::MaxIterExceeded);
        }

        for j in 0..n {
            ptt[j] = two * result[j] - pt[j];
            xit[j] = result[j] - pt[j];
            pt[j] = result[j];
        }

        let fptt = f(&ptt);
        if let Some(ref mut obs) = observer {
            obs(&ptt, fptt);
        }
        if fptt < fp {
            let t = two * (fp - two * fret + fptt) * sqr(fp - fret - del) - del * sqr(fp - fptt);
            if t < zero() {
                linmin(f, &mut result, &mut xit, &mut fret);
                for j in 0..n {
                    xi[j][ibig] = xi[j][n - 1];
                    xi[j][n - 1] = xit[j];
                }
            }
        }
    }

    (result, OptErr::Normal)
}
