#![cfg(not(target_family = "wasm"))]    
use special::Gamma;

use num::traits::{float::Float, identities::one};

pub fn log_factorial<T>(x: T) -> T
where
    T: Gamma + Float,
{
    (x + one()).ln_gamma().0
}

pub fn log_cn<T>(m: T, n: T) -> T
where
    T: Gamma + Float,
{
    log_factorial(m) - log_factorial(n) - log_factorial(m - n)
}
