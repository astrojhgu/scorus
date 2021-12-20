#![allow(clippy::comparison_chain)]

use num::traits::float::Float;

pub fn encloses<T>(x: T, x1: T, x2: T) -> bool
where
    T: Float,
{
    if x2 > x1 {
        x >= x1 && x < x2
    } else {
        x >= x2 && x < x1
    }
}

pub fn bisect_search<T>(x: T, xlist: &[T]) -> (usize, usize)
where
    T: Float,
{
    if xlist.len() < 2 {
        panic!("the size of xlist must >=2");
    }
    let mut l = 0;
    let mut r = xlist.len() - 1;
    while r - l > 1 {
        let c = (r + l) >> 1;
        if encloses(x, xlist[c], xlist[r]) {
            l = c;
        } else if encloses(x, xlist[l], xlist[c]) {
            r = c;
        } else {
            unreachable!("should never each here, check if x enclosed by the xlist");
        }
    }
    (l, r)
}

pub fn interp<T>(x: T, xlist: &[T], ylist: &[T]) -> T
where
    T: Copy + Float,
{
    assert!(!xlist.is_empty() && ylist.len() == xlist.len());
    if encloses(x, *xlist.first().unwrap(), *xlist.last().unwrap()) {
        let (l, r) = bisect_search(x, xlist);
        if xlist[r] == xlist[l] {
            ylist[l]
        } else {
            (ylist[r] - ylist[l]) / (xlist[r] - xlist[l]) * (x - xlist[l]) + ylist[l]
        }
    } else {
        for i in 0..(xlist.len() - 1) {
            if (xlist[i] - x) * (xlist[i + 1] - x) <= T::zero() {
                return (ylist[i + 1] - ylist[i]) / (xlist[i + 1] - xlist[i]) * (x - xlist[i])
                    + ylist[i];
            }
        }
        if *xlist.first().unwrap() < *xlist.last().unwrap() {
            if x <= *xlist.first().unwrap() {
                *ylist.first().unwrap()
            } else {
                *ylist.last().unwrap()
            }
        } else if *xlist.first().unwrap() > *xlist.last().unwrap() {
            if x >= *xlist.first().unwrap() {
                *ylist.first().unwrap()
            } else {
                *ylist.last().unwrap()
            }
        } else {
            unreachable!("should never reach here");
        }
    }
}
