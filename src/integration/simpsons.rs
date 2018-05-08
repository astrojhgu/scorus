use num_traits::float::Float;
use num_traits::identities::one;
use super::integration_errors::IntegrationError;

fn integrate_aux<F, T>(
    f: &F,
    a: T,
    b: T,
    eps: T,
    s: T,
    fa: T,
    fb: T,
    fc: T,
    bottom: isize,
) -> Result<T, IntegrationError>
where
    F: Fn(T) -> T,
    T: Float + Copy,
{
    let two = T::from(2).unwrap();
    let four = two + two;
    let twelve = T::from(12).unwrap();
    let fifteen = T::from(15).unwrap();
    let c = (a + b) / two;
    let h = b - a;
    let d = (a + c) / two;
    let e = (c + b) / two;
    let fd = f(d);
    let fe = f(e);
    let sleft = (h / twelve) * (fa + four * fd + fc);
    let sright = (h / twelve) * (fc + four * fe + fb);
    let s2 = sleft + sright;
    if bottom <= 0 {
        return Err(IntegrationError::MaxRecReached);
    }
    if bottom <= 0 || (s2 - s).abs() <= fifteen * eps {
        Ok(s2 + (s2 - s) / fifteen)
    } else {
        let a = integrate_aux(f, a, c, eps / two, sleft, fa, fc, fd, bottom - 1)?;
        let b = integrate_aux(f, c, b, eps / two, sright, fc, fb, fe, bottom - 1)?;
        Ok(a + b)
    }
}

pub fn integrate<F, T>(
    f: &F,
    a: T,
    b: T,
    eps: T,
    max_rec_depth: isize,
) -> Result<T, IntegrationError>
where
    F: Fn(T) -> T,
    T: Float + Copy,
{
    let two = one::<T>() + one::<T>();
    let six = T::from(6).unwrap();
    let four = two + two;
    let c = (a + b) / two;
    let h = b - a;
    let fa = f(a);
    let fb = f(b);
    let fc = f(c);
    let s = (h / six) * (fa + four * fc + fb);
    integrate_aux(f, a, b, eps, s, fa, fb, fc, max_rec_depth)
}
