use num_traits::{Float, NumCast};
use std::ops::{Add, Sub, Mul};
use std::fmt::Debug;

use super::tolerance::Tolerance;
use super::linmin::linmin;
use crate::linear_space::InnerProdSpace;

pub fn cg_iter<T, V, F, G>(fobj:&F, grad: &G, p0: &mut V, d: &mut V,g: &mut V, fret :&mut T, tol: Tolerance<T>)->bool
where 
    T: Float + NumCast + std::cmp::PartialOrd + Copy + Debug,
    V: Clone + InnerProdSpace<T>,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> T,
    G: Fn(&V) -> V,
{
    let two=T::one()+T::one();
    let fret_old=*fret;
    linmin(fobj, p0, d, fret);
    let g1=grad(p0);
    let y=&g1-g as &V;
    //let beta=(&y-&(&d*(&y.dot(&y)*(two/d.dot(&y))))).dot(&g1)*(T::one()/d.dot(&y));
    let beta=(&y-&(d as &V*(y.dot(&y)*(two/d.dot(&y))))).dot(&g1)/d.dot(&y);
    *d=&(d as &V*beta)-&g1;
    //*d=g*(-T::one())+(&d*beta);
    tol.accepted(fret_old, *fret)
}