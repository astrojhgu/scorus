use num_traits::Float;
use std::ops::{Add, Mul, Sub};

use crate::linear_space::traits::LinearSpace;

pub fn leapfrog<T, V, F, G>(q: &mut V, p: &mut V, last_hq_q: &mut V, epsilon: T, h_q: &F, h_p: &G)
where
    T: Float + std::fmt::Debug,
    V: Clone + LinearSpace<T>,
    for<'b> &'b V: Add<Output = V>,
    for<'b> &'b V: Sub<Output = V>,
    for<'b> &'b V: Mul<T, Output = V>,
    F: Fn(&V) -> V,
    G: Fn(&V) -> V,
{
    assert_eq!(q.dimension(), p.dimension());
    assert_eq!(q.dimension(), last_hq_q.dimension());
    let two = T::one() + T::one();
    let half = T::one() / two;

    let p_mid = p as &V - &(last_hq_q as &V * (epsilon * half));

    let qdot = h_p(&p_mid);
    assert_eq!(qdot.dimension(), p.dimension());
    *q = q as &V + &(&qdot * epsilon);

    *last_hq_q = h_q(q);
    assert_eq!(last_hq_q.dimension(), q.dimension());
    *p = &p_mid - &(last_hq_q as &V * (epsilon * half));
}
