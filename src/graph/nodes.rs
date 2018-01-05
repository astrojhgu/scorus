extern crate special;


use std;
use super::node::Node;
use super::node::NodeContent;
use super::node::BasicNode;
use num_traits::float::Float;
use std::boxed::Box;
use num_traits::identities::one;
use num_traits::identities::zero;
use super::graph::NodeAdder;
use super::graph::NodeHandle;
use super::super::functions::phi;
use num_traits::cast::NumCast;

pub fn const_node<T>(v: T) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 0,
            ndim_output: 1,
        },
        content: NodeContent::DeterministicNode {
            calc: Box::new(move |x| vec![v]),
        },
    };
    NodeAdder::new(n, &[])
}

pub fn add_node<T>(a: (NodeHandle, usize), b: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 2,
            ndim_output: 1,
        },
        content: NodeContent::DeterministicNode {
            calc: Box::new(move |x| vec![x[0] + x[1]]),
        },
    };
    NodeAdder::new(n,&[a, b])
}

pub fn mul_node<T>(a: (NodeHandle, usize), b: (NodeHandle, usize)) -> NodeAdder<T>
    where
        T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 2,
            ndim_output: 1,
        },
        content: NodeContent::DeterministicNode {
            calc: Box::new(move |x| vec![x[0] * x[1]]),
        },
    };
    NodeAdder::new(n,&[a, b])
}


pub fn sub_node<T>(a: (NodeHandle, usize), b: (NodeHandle, usize)) -> NodeAdder<T>
    where
        T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 2,
            ndim_output: 1,
        },
        content: NodeContent::DeterministicNode {
            calc: Box::new(move |x| vec![x[0] / x[1]]),
        },
    };
    NodeAdder::new(n,&[a, b])
}

pub fn div_node<T>(a: (NodeHandle, usize), b: (NodeHandle, usize)) -> NodeAdder<T>
    where
        T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 2,
            ndim_output: 1,
        },
        content: NodeContent::DeterministicNode {
            calc: Box::new(move |x| vec![x[0] / x[1]]),
        },
    };
    NodeAdder::new(n,&[a, b])
}


pub fn cos_node<T>(a: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 1,
            ndim_output: 1,
        },
        content: NodeContent::DeterministicNode {
            calc: Box::new(move |x: &[T]| vec![x[0].cos()]),
        },
    };
    NodeAdder::new(n,&[a])
}

pub fn normal_node<T>(m: (NodeHandle, usize), s: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let two=one::<T>()+one::<T>();
    let four=two+two;
    let pi=one::<T>().atan()*four;
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 2,
            ndim_output: 1,
        },
        content: NodeContent::StochasticNode {
            all_stochastic_children: Vec::new(),
            all_deterministic_children: Vec::new(),
            is_observed: vec![false],
            values: vec![zero()],
            logprob: Box::new(move |x, p| {
                let x = x[0];
                let m = p[0];
                let s = p[1];
                let y=-((x - m) * (x - m) / ((one::<T>() + one::<T>()) * s * s)) - ((two*pi).sqrt()*s).ln();
                //println!("{} {} {} {}", x, m, s, y);
                return y;
            }),
            range: Box::new(move |p|{
                let m=p[0];
                let s=p[1];
                let x1=m-T::from(5).unwrap()*s;
                let x2=m+T::from(5).unwrap()*s;
                vec![(x1,x2)]
            })
        },
    };
    NodeAdder::new(n,&[m, s])
}

pub fn uniform_node<T>(a: (NodeHandle, usize), b: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 2,
            ndim_output: 1,
        },
        content: NodeContent::StochasticNode {
            all_stochastic_children: Vec::new(),
            all_deterministic_children: Vec::new(),
            is_observed: vec![false],
            values: vec![zero()],
            logprob: Box::new(move |x, p| {
                let x1=p[0];
                let x2=p[1];
                let x=x[0];
                if x<=x2 && x>=x1{
                    zero()
                }
                else{
                    <T as Float>::infinity()
                }
            }
                ),
            range: Box::new(move |p| {
                let x1=p[0];
                let x2=p[1];
                vec![(x1,x2)]
            })
        },
    };

    NodeAdder::new(n,&[a, b])
}

pub fn phi_node<T>(x:(NodeHandle, usize))->NodeAdder<T>
where
    T: 'static + Float + special::Error+ Sync + Send + std::fmt::Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: 1,
            ndim_output: 1,
        },
        content: NodeContent::DeterministicNode {
            calc: Box::new(move |x: &[T]| vec![phi(x[0])]),
        },
    };
    NodeAdder::new(n,&[x])
}
