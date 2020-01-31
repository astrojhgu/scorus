#![allow(unused_imports)]

extern crate rand;
extern crate scorus;
use rand::thread_rng;
use scorus::mcmc::ensemble_sample::sample;
use scorus::mcmc::graph::graph::Graph;
use scorus::mcmc::graph::graph::ParamObservability::{Observed, UnObserved};
use scorus::mcmc::graph::nodes::{
    add_node, const_node, cos_node, mul_node, normal_node, uniform_node,
};
use scorus::mcmc::HasLen;

fn main() {
    let mut g = Graph::new();
    let m = const_node(0.0).add_to(&mut g, &"m".to_string());
    let s = const_node(1.0).add_to(&mut g, &"s".to_string());
    let x = normal_node((m, 0), (s, 0)).add_to(&mut g, &"g".to_string());
    let _x2 = mul_node((x, 0), (x, 0)).add_to(&mut g, &"g2".to_string());

    g.seal();
    let mut gv = g.init_gv();
    //println!("{}", &gv);

    let mut ensemble = Vec::new();
    ensemble.push(gv.clone());
    gv[0] = 0.1;
    ensemble.push(gv.clone());
    gv[0] = -0.1;
    ensemble.push(gv.clone());
    gv[0] = 0.2;
    ensemble.push(gv.clone());

    let mut lp = Vec::new();

    let mut rng = thread_rng();
    for _i in 0..10000 {
        //let aa = sample_st(&|x| g.logpost_all(x), &(ensemble, lp), &mut rng, 2.0).unwrap();
        //ensemble = aa.0;
        //lp = aa.1;
        sample(
            &|x| g.logpost_all(x),
            &mut ensemble,
            &mut lp,
            &mut rng,
            2.0,
            0.01,
            1,
        );
        //println!("{}", ensemble[0].deterministic_values.borrow()[0]);
        println!("{}", ensemble[0][0]);
    }
}
