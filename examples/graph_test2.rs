extern crate rand;
extern crate scorus;

//use std;
use rand::distributions::normal::Normal;
use rand::distributions::IndependentSample;
use rand::thread_rng;
use scorus::mcmc::ensemble_sample::sample_st;
use scorus::mcmc::graph::graph::Graph;
use scorus::mcmc::graph::graph::ParamObservability::{Observed, UnObserved};
use scorus::mcmc::graph::nodes::{add_node, const_node, cos_node, normal_node, uniform_node};
use scorus::utils::HasLen;

fn main() {
    let mut g = Graph::new();

    let m_lower = const_node(-1.0).add_to(&mut g, &"m_lower".to_string());
    let m_upper = const_node(1.0).add_to(&mut g, &"m_upper".to_string());

    let s_lower = const_node(0.001).add_to(&mut g, &"s_lower".to_string());
    let s_upper = const_node(10.0).add_to(&mut g, &"s_upper".to_string());

    let m = uniform_node((m_lower, 0), (m_upper, 0))
        .with_all_values(&[UnObserved(1.0)])
        .add_to(&mut g, &"m".to_string());
    let s = uniform_node((s_lower, 0), (s_upper, 0))
        .with_all_values(&[UnObserved(2.0)])
        .add_to(&mut g, &"s".to_string());

    let mut rng = thread_rng();

    let norm = Normal::new(0.0, 2.0);

    for i in 0..1000 {
        let k = format!("n{}", i);
        normal_node((m, 0), (s, 0))
            .with_all_values(&[Observed(norm.ind_sample(&mut rng))])
            .add_to(&mut g, &k);
    }

    //normal_node((m,0),(s,0)).with_all_values(&[UnObserved(1.0)]).add_to(&mut g, &"x1".to_string());
    //normal_node((m,0),(s,0)).with_all_values(&[UnObserved(1.0)]).add_to(&mut g, &"x2".to_string());

    g.seal();
    let mut gv = g.init_gv();

    //g.sample(4, 0, &mut gv, &mut rng, 10);
    //g.sample(5, 0, &mut gv, &mut rng, 10);
    let mut nchanged = 0;
    g.sample_all(&mut gv, &mut rng, 10, &mut nchanged);
    println!(
        "{} {}",
        g.cached_value_of(4, 0, &gv),
        g.cached_value_of(5, 0, &gv)
    );
    println!("{}", nchanged);
}
