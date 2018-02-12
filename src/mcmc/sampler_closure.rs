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
use super::{ensemble_sample::sample_st as esample_st, ptsample::sample_st as ptsample_st};
use super::mcmc_errors::McmcErr;





