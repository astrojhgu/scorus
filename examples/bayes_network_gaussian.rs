#![allow(unused_imports)]

extern crate rand;
extern crate scorus;

use rand::thread_rng;
use scorus::mcmc::ensemble_sample::sample;
use scorus::mcmc::graph::graph::Graph;
use scorus::mcmc::graph::graph::ParamObservability::{Observed, UnObserved};
use scorus::mcmc::graph::graph_var::GraphVar;
use scorus::mcmc::graph::nodes::{add_node, const_node, cos_node, normal_node, uniform_node};
use scorus::mcmc::init_ensemble::get_one_init_realization;
use scorus::utils::HasLen;

fn main() {}
