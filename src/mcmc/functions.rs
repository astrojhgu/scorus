#![cfg(not(target_family = "wasm"))]
use special::{Error, Gamma};

use num::traits::{float::Float, identities::one};

use crate::basic::log_cn;

pub fn phi<T>(x: T) -> T
where
    T: Error + Float,
{
    let two = one::<T>() + one::<T>();
    let sqrt_2 = two.sqrt();
    (one::<T>() + (x / sqrt_2).error()) / two
}

pub fn lbeta<T>(x: T, y: T) -> T
where
    T: Gamma + Float,
{
    x.ln_gamma().0 + y.ln_gamma().0 - (x + y).ln_gamma().0
}

pub fn logdbin<T>(x: T, p: T, n: T) -> T
where
    T: Gamma + Float,
{
    log_cn(n, x) + x * p.ln() + (n - x) * (one::<T>() - p).ln()
}
