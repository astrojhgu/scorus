use std;
use std::ops::IndexMut;
use num_traits::float::Float;
use num_traits::cast::NumCast;
use num_traits::identities::one;
use num_traits::identities::zero;

use utils::HasLength;

use super::bas_utils::{max, sign, swap, shft3};

pub fn brent<F, T>(ax: T, bx: T, cx: T, f: &F, tol: T) -> (T, T)
where
    T: Float + Copy,
    F: Fn(T) -> T,
{
    let two = one::<T>() + one::<T>();
    let itmax = 100;
    let cgold = T::from(0.3819660).unwrap();
    let zeps = T::epsilon() * T::from(1e-3).unwrap();

    let (mut a, mut b) = if ax < cx { (ax, cx) } else { (cx, ax) };
    let mut d = zero::<T>();
    let mut etemp;
    let mut p;
    let mut q;
    let mut r;
    let mut tol1;
    let mut tol2;
    let mut xm;
    let mut e = zero::<T>();

    let mut u;
    let mut v = bx;
    let mut w = bx;
    let mut x = bx;

    let mut fu;
    let mut fx = f(x);
    let mut fv = fx;
    let mut fw = fx;

    let mut iter = 0;

    while iter < itmax {
        xm = (a + b) / two;
        tol1 = tol * x.abs();
        tol2 = two * (tol1 + zeps);

        if (x - xm).abs() <= (tol2 - (b - a) / two) {
            return (x, fx);
        }

        if e.abs() > tol1 {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = two * (q - r);

            if q > zero() {
                p = -p;
            }

            q = q.abs();
            etemp = e;
            e = d;
            if p.abs() >= (p * etemp / two).abs() || p <= q * (a - x) || p >= q * (b - x) {
                e = if x >= xm { a - x } else { b - x };
                d = cgold * e;
            } else {
                d = p / q;
                u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = sign(tol1, xm - x);
                }
            }
        } else {
            e = if x >= xm { a - x } else { b - x };
            d = cgold * e;
        }
        u = if d.abs() >= tol1 {
            x + d
        } else {
            x + sign(tol1, d)
        };
        fu = f(u);

        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            shft3(&mut v, &mut w, &mut x, u);
            shft3(&mut fv, &mut fw, &mut fx, fu);
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }

        iter += 1;
    }
    (x, fx)
}

pub fn mnbrak<T, F>(ax: &mut T, bx: &mut T, cx: &mut T, fa: &mut T, fb: &mut T, fc: &mut T, f: &F)
where
    T: Float + Copy,
    F: Fn(T) -> T,
{
    let two = one::<T>() + one::<T>();
    let gold = T::from(1.618034).unwrap();
    let glimit = 100;
    let tiny = T::epsilon();
    let mut ulim;
    let mut u;
    let mut r;
    let mut q;
    let mut fu;

    *fa = f(*ax);
    *fb = f(*bx);

    if *fb > *fa {
        swap(ax, bx);
        swap(fa, fb);
    }

    *cx = *bx + gold * (*bx - *ax);
    *fc = f(*cx);

    while *fb > *fc {
        r = (*bx - *ax) * (*fb - *fc);
        q = (*bx - *cx) * (*fb - *fa);
        u = *bx
            - ((*bx - *cx) * q - (*bx - *ax) * r) / (two * sign(max((q - r).abs(), tiny), q - r));

        ulim = *bx + T::from(glimit).unwrap() * (*cx - *bx);

        if (*bx - u) * (u - *cx) > zero::<T>() {
            fu = f(u);

            if fu < *fc {
                *ax = *bx;
                *bx = u;
                *fa = *fb;
                *fb = fu;
                return;
            } else if fu > *fb {
                *cx = u;
                *fc = fu;
                return;
            }

            u = *cx + gold * (*cx - *bx);
            fu = f(u);
        } else if (*cx - u) * (u - ulim) > zero() {
            fu = f(u);
            if fu < *fc {
                let xx = *cx + gold * (*cx - *bx);
                shft3(bx, cx, &mut u, xx);
                shft3(fb, fc, &mut fu, f(u));
            }
        } else if (u - ulim) * (ulim - *cx) >= zero() {
            u = ulim;
            fu = f(u);
        } else {
            u = *cx + gold * (*cx - *bx);
            fu = f(u);
        }
        shft3(ax, bx, cx, u);
        shft3(fa, fb, fc, fu);
    }
}

pub fn linmin<F, V, T>(f: &F, p: &mut V, xi: &mut V, fret: &mut T)
where
    T: Float + NumCast + std::cmp::PartialOrd + Copy,
    V: Clone + IndexMut<usize, Output = T> + HasLength,
    F: Fn(&V) -> T,
{
    let tol = T::epsilon().sqrt();
    let (mut xx, mut fb, mut fa, mut bx, mut ax, mut fx);

    ax = zero::<T>();
    bx = zero::<T>();
    xx = one::<T>();
    fa = zero::<T>();
    fb = zero::<T>();
    fx = zero::<T>();
    let xmin;
    {
        let func_adapter = |x: T| {
            let mut x1 = p.clone();
            for i in 0..x1.length() {
                x1[i] = x1[i] + x * xi[i];
            }
            f(&x1)
        };

        mnbrak(
            &mut ax,
            &mut xx,
            &mut bx,
            &mut fa,
            &mut fx,
            &mut fb,
            &func_adapter,
        );
        let xf = brent(ax, xx, bx, &func_adapter, tol);
        xmin = xf.0;
        *fret = xf.1;
    }
    for j in 0..p.length() {
        xi[j] = xi[j] * xmin;
        p[j] = p[j] + xi[j];
    }
}
