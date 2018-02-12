use num_traits::float::Float;
use num_traits::NumCast;
use std::ops::IndexMut;
use std::clone::Clone;
use std::marker::{Send, Sync};
use std;
use rand::{Rand, Rng};
use rand::distributions::range::SampleRange;

use super::super::utils::{HasLength, ItemSwapable, Resizeable};

use super::{ensemble_sample::sample as esample, ptsample::sample as ptsample};
use super::mcmc_errors::McmcErr;

pub fn esample_closure<T, U, V, W, X, F>(
    flogprob: F,
    mut ensemble_logprob: (W, X),
    mut rng: U,
    a: T,
    nthread: usize,
) -> Box<FnMut(&mut FnMut(&Result<(W,X), McmcErr>))->()>
where
    T: 'static
        + Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + Sync
        + Send
        + std::fmt::Display,
    U: 'static + Rng,
    V: Clone + IndexMut<usize, Output = T> + HasLength + Sync + Send + Sized,
    W: 'static + Clone + IndexMut<usize, Output = V> + HasLength + Sync + Send + Drop + Sized,
    X: 'static
        + Clone
        + IndexMut<usize, Output = T>
        + HasLength
        + Sync
        + Resizeable
        + Send
        + Drop
        + Sized,
    F: 'static + Fn(&V) -> T + Send + Sync,
{
    Box::new(move |handler:&mut FnMut(&Result<(W,X), McmcErr>)| {
        let result = esample(&flogprob, &ensemble_logprob, &mut rng, a, nthread);
        handler(&result);
        match result{
            Ok(x) => ensemble_logprob=x,
            _ => ()
        }
    })
}

pub fn ptsample_closure<T, U, V, W, X, F>(
    flogprob: F,
    mut ensemble_logprob: (W, X),
    mut rng: U,
    beta_list: X,
    a: T,
    nthread: usize,
) -> Box<FnMut(&mut FnMut(&Result<(W,X), McmcErr>), bool)->()>
where
    T: 'static
        + Float
        + NumCast
        + Rand
        + std::cmp::PartialOrd
        + SampleRange
        + Sync
        + Send
        + std::fmt::Display,
    U: 'static + Rng,
    V: Clone + IndexMut<usize, Output = T> + HasLength + Sync + Send + Sized,
    W: 'static
        + Clone
        + IndexMut<usize, Output = V>
        + HasLength
        + Sync
        + Send
        + Drop
        + Sized
        + ItemSwapable,
    X: 'static
        + Clone
        + IndexMut<usize, Output = T>
        + HasLength
        + Sync
        + Resizeable
        + Send
        + Drop
        + Sized
        + ItemSwapable,
    F: 'static + Fn(&V) -> T + Send + Sync,
{
    Box::new(move |handler:&mut FnMut(&Result<(W,X), McmcErr>), sw:bool| {
        let result= ptsample(
            &flogprob,
            &ensemble_logprob,
            &mut rng,
            &beta_list,
            sw,
            a,
            nthread,
        );
        handler(&result);
        match result{
            Ok(x) => ensemble_logprob=x,
            _ => ()
        }
    })
}
