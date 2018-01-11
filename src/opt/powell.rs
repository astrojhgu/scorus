use ::std;
use std::ops::IndexMut;
use num_traits::float::Float;
use num_traits::cast::NumCast;

use num_traits::identities::{zero, one};

use super::linmin::linmin;
use super::super::utils::HasLength;
use super::bas_utils::{sqr};

pub fn fmin<F, V, T>(f: &F, p: &V, ftol:T) -> V
where
    T: Float + NumCast + std::cmp::PartialOrd + Copy,
    V: Clone + IndexMut<usize, Output = T> + HasLength + std::marker::Sync + std::marker::Send,
    F: Fn(&V) -> T,
{
    let two=one::<T>()+one::<T>();
    let n=p.length();
    let mut xi=Vec::new();
    xi.reserve(n);
    for i in 0..n{
        let mut xi1=p.clone();
        for j in 0..n{
            xi1[j]=if i==j {one::<T>()} else {zero::<T>()};
        }
        xi.push(xi1);
    }

    let itmax=200;
    let tiny=T::epsilon();
    let mut fret=f(p);

    let mut pt=p.clone();
    let mut fp;
    let mut fptt;
    let mut ibig;
    let mut del;
    let mut xit=p.clone();
    let mut ptt=p.clone();
    let mut result=p.clone();
    for iter in 0..itmax{
        fp=fret;
        ibig=0;
        del=zero::<T>();
        for i in 0..n{
            for j in 0..n{
                xit[j]=xi[j][i];
            }
            fptt=fret;
            linmin(f, &mut result, &mut xit, &mut fret);
            if (fptt-fret) > del {
                del=fptt-fret;
                ibig=i+1;
            }
        }

        if two*(fp-fret) <=ftol*(fp.abs()+fret.abs())+tiny{
            return result;
        }

        if iter==itmax{
            panic!("maximum iter reached");
        }

        for j in 0..n{
            ptt[j]=two*result[j]-pt[j];
            xit[j]=p[j]-pt[j];
            pt[j]=result[j];
        }

        let fptt=f(&ptt);

        if fptt<fp{
            let t=two*(fp-two*fret+fptt)*sqr(fp-fret-del)-del*sqr(fp-fptt);
            if t<zero(){
                linmin(f, &mut result, &mut xit, &mut fret);
                for j in 0..n{
                    xi[j][ibig-1]=xi[j][n-1];
                    xi[j][n-1]=xit[j];
                }
            }
        }
    }

    result
}
