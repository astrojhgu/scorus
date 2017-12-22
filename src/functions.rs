extern crate num_traits;
extern crate special;

use std;
use num_traits::float::Float;
use num_traits::identities::one;

pub fn phi<T>(x: T) -> T
where
    T: special::Error + Float,
{
    let two = one::<T>() + one::<T>();
    let sqrt_2 = two.sqrt();
    (one::<T>() + (x / sqrt_2).erf()) / two
}

pub fn lbeta<T>(x: T, y: T) -> T
where
    T: special::Gamma + Float,
{
    x.ln_gamma().0 + y.ln_gamma().0 - (x + y).ln_gamma().0
}

pub fn log_factorial<T>(x: T) -> T
where
    T: special::Gamma + Float,
{
    (x + one()).ln_gamma().0
}

pub fn log_cn<T>(m: T, n: T) -> T
where
    T: special::Gamma + Float,
{
    log_factorial(m) - log_factorial(n) - log_factorial(m - n)
}

pub fn logdbin<T>(x: T, p: T, n: T) -> T
where
    T: special::Gamma + Float,
{
    log_cn(n, x) + x * p.ln() + (n - x) * (one::<T>() - p).ln()
}
