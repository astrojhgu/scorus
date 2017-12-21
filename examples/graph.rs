extern crate rsmcmc;

use rsmcmc::graph::graph::Graph;
use rsmcmc::graph::nodes::{add_node, const_node, cos_node, normal_node, uniform_node};

fn main() {
    let mut g = Graph::<String, f64>::new();

    g.add_node("l".to_string(), const_node(-1.0)).done();
    g.add_node("u".to_string(), const_node(1.0)).done();

    g.add_node("x".to_string(), uniform_node())
        .with_parent("l".to_string(), 0)
        .with_parent("u".to_string(), 0)
        .done();

    g.add_node("a1".to_string(), cos_node())
        .with_parent("x".to_string(), 0)
        .done();
    g.add_node("a2".to_string(), cos_node())
        .with_parent("a1".to_string(), 0)
        .done();
    g.add_node("a21".to_string(), add_node())
        .with_parent("x".to_string(), 0)
        .with_parent("a1".to_string(), 0)
        .done();
    g.add_node("a3".to_string(), cos_node())
        .with_parent("a2".to_string(), 0)
        .done();
    g.add_node("a4".to_string(), cos_node())
        .with_parent("a3".to_string(), 0)
        .done();

    g.add_node("y".to_string(), normal_node())
        .with_parent("a3".to_string(), 0)
        .with_parent("a4".to_string(), 0)
        .done();

    g.seal();
    //println!("{:?}", g.enumerate_stochastic_children(2));

    println!("{}", g);
    let mut gv = g.init_gv();
    g.set_value_then_update(2, 0, 0.1, &mut gv);
    g.update_all_deterministic_nodes(&mut gv);
    //g.set_value(5,0, 0.3, &mut gv);
    //g.update_deterministic_children(5, &mut gv);
    //println!("{}",g.calc_logprob(8, & gv));

    println!("{}", &gv);
    println!("{}", g.likelihood(2, &gv));
    println!("{}", g.logprob(2, &gv));
    println!("{}", g.logpost(2, &gv));
    println!("{}", g.logpost_all(&mut gv));
}
