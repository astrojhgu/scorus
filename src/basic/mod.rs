use special::Gamma;

use num_traits::float::Float;
use num_traits::identities::one;

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
