//use ::std;
use std::marker::Copy;
use num_traits::float::Float;
use num_traits::identities::one;

fn next_p<T>(n: usize, x: T, pn1: T, pn2: T) -> T
where
    T: Float + Copy,
{
    (T::from(2 * n - 1).unwrap() * x * pn1 - T::from(n - 1).unwrap() * pn2) / T::from(n).unwrap()
}

pub fn eval<T>(n: usize, x: T) -> T
where
    T: Float + Copy,
{
    let one = one::<T>();

    if n == 0 {
        return one;
    }
    if n == 1 {
        return x;
    }

    let mut pn1 = x;
    let mut pn2 = one;

    let mut n1 = 2_usize;
    loop {
        let pn = next_p(n1, x, pn1, pn2);
        if n1 == n {
            return pn;
        }
        pn2 = pn1;
        pn1 = pn;
        n1 += 1;
    }
}
