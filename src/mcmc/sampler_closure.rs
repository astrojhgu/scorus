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
) -> Box<FnMut(bool) -> Result<Option<(W, X)>, McmcErr>>
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
    Box::new(move |draw_value: bool| {
        ensemble_logprob = esample(&flogprob, &ensemble_logprob, &mut rng, a, nthread)?;
        if draw_value {
            Ok(Some((
                ensemble_logprob.0.clone(),
                ensemble_logprob.1.clone(),
            )))
        } else {
            Ok(None)
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
) -> Box<FnMut(bool, bool) -> Result<Option<(W, X)>, McmcErr>>
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
    Box::new(move |draw_value: bool, sw: bool| {
        ensemble_logprob = ptsample(
            &flogprob,
            &ensemble_logprob,
            &mut rng,
            &beta_list,
            sw,
            a,
            nthread,
        )?;
        if draw_value {
            Ok(Some((
                ensemble_logprob.0.clone(),
                ensemble_logprob.1.clone(),
            )))
        } else {
            Ok(None)
        }
    })
}
