use num_traits::{Float, NumCast};
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

use super::tolerance::Tolerance;
use crate::linear_space::InnerProdSpace;

pub fn wolf1<T, F, G>(f: &F, fd: &G, alpha: T) -> bool
where
    T: Float + NumCast + std::cmp::PartialOrd + Copy + Debug,
    F: Fn(T) -> T,
    G: Fn(T) -> T,
{
    let c1 = T::from(1e-4).unwrap();
    f(alpha) <= f(T::zero()) + c1 * alpha * fd(alpha)
}

pub fn wolf_strong<T, F, G>(_f: &F, fd: &G, alpha: T, c2: T) -> bool
where
    T: Float + NumCast + std::cmp::PartialOrd + Copy + Debug,
    F: Fn(T) -> T,
    G: Fn(T) -> T,
{
    fd(alpha).abs() <= -c2 * fd(T::zero())
}

pub fn interpolation<T, F, G>(f: &F, fd: &G, mut alpha: T, c2: T) -> T
where
    T: Float + NumCast + std::cmp::PartialOrd + Copy + Debug,
    F: Fn(T) -> T,
    G: Fn(T) -> T,
{
    let mut lo = T::zero();
    let mut hi = T::one();
    let two = T::one() + T::one();
    for _ in 0..20 {
        if wolf1(f, fd, alpha) && wolf_strong(f, fd, alpha, c2) {
            return alpha;
        }
        let half = (lo + hi) / two;
        alpha = -(fd(lo) * hi * hi) / (two * (f(hi) - f(lo) - fd(lo) * hi));
        if alpha < lo || alpha > hi {
            alpha = half;
            if fd(alpha) > T::zero() {
                hi = alpha
            } else {
                lo = alpha
            }
        }
    }
    alpha
}

fn find_step_length<T, V, F, G>(f: &F, fd: &G, x: &V, alpha: T, direction: &V, c2: T) -> T
where
    T: Float + NumCast + std::cmp::PartialOrd + Copy + Debug,
    V: Clone + InnerProdSpace<T>,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> T,
    G: Fn(&V) -> V,
{
    let g = |alpha| f(&(x + &(direction * alpha)));
    let gd = |alpha| direction.dot(&fd(&(x + &(direction * alpha))));
    interpolation(&g, &gd, alpha, c2)
}

pub fn cg_iter<T, V, F, G>(
    fobj: &F,
    grad: &G,
    p0: &mut V,
    d: &mut V,
    g: &mut V,
    fret: &mut T,
    tol: Tolerance<T>,
) -> bool
where
    T: Float + NumCast + std::cmp::PartialOrd + Copy + Debug,
    V: Clone + InnerProdSpace<T>,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> T,
    G: Fn(&V) -> V,
{
    let two = T::one() + T::one();
    let fret_old = *fret;
    //linmin(fobj, p0, d, fret);
    let alpha = find_step_length(fobj, grad, p0, T::one(), d, T::from(0.1).unwrap());
    *p0 = p0 as &V + &(d as &V * alpha);
    *fret = fobj(p0 as &V);
    let g1 = grad(p0);
    let y = &g1 - g as &V;
    //let beta=(&y-&(&d*(&y.dot(&y)*(two/d.dot(&y))))).dot(&g1)*(T::one()/d.dot(&y));
    if d.dot(&y) == T::zero() {
        return true;
    }
    let beta = (&y - &(d as &V * (y.dot(&y) * (two / d.dot(&y))))).dot(&g1) / d.dot(&y);
    *d = &(d as &V * beta) - &g1;
    //*d=g*(-T::one())+(&d*beta);
    tol.accepted(fret_old, *fret)
}
