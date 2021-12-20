use crate::linear_space::{IndexableLinearSpace, LinearSpace};
use num::traits::Float;

pub fn mean<T, V>(x: &[V]) -> V
where
    T: Float + Clone,
    V: LinearSpace<T> + Clone,
    for<'b> &'b V: std::ops::Add<Output = V>,
    for<'b> &'b V: std::ops::Sub<Output = V>,
    for<'b> &'b V: std::ops::Mul<T, Output = V>,
{
    assert!(!x.is_empty());
    let init = x[0].clone();
    &x.iter().skip(1).fold(init, |a, b| &a + b) * (T::one() / T::from(x.len()).unwrap())
}

pub fn cov<T, V>(x: &[V], cov_result: &mut dyn FnMut(usize, usize, T))
where
    T: Float + Clone,
    V: IndexableLinearSpace<T> + Clone,
    for<'b> &'b V: std::ops::Add<Output = V>,
    for<'b> &'b V: std::ops::Sub<Output = V>,
    for<'b> &'b V: std::ops::Mul<T, Output = V>,
{
    let xm = mean(x);
    let ndim = xm.dimension();
    let mut result = vec![vec![T::zero(); ndim]; ndim];
    for x1 in x.iter().map(|x1| x1 - &xm) {
        for i in 0..ndim {
            for j in 0..ndim {
                result[i][j] = result[i][j] + x1[i] * x1[j]
            }
        }
    }
    for (i, row) in result.iter().enumerate() {
        for (j, &x1) in row.iter().enumerate() {
            let y = x1 / T::from(x.len()).unwrap();
            cov_result(i, j, y);
        }
    }
}
