extern crate kxjs;
extern crate rand;

use rand::thread_rng;
use kxjs::mcmc::graph::graph::Graph;
use kxjs::mcmc::graph::graph::ParamObservability::{Observed, UnObserved};
use kxjs::mcmc::graph::graph_var::GraphVar;
use kxjs::mcmc::graph::nodes::{add_node, const_node, cos_node, normal_node, uniform_node};
use kxjs::mcmc::ensemble_sample::sample;
use kxjs::utils::HasLength;
use kxjs::mcmc::init_ensemble::get_one_init_realization;

fn main() {}
