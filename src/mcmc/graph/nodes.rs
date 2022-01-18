use std::fmt::Display;

use special::{Error, Gamma};

use super::super::functions::phi;
use super::graph::NodeAdder;
use super::graph::NodeHandle;
use super::node::BasicNode;
use super::node::Node;
use super::node::NodeContent;
use num::traits::{
    float::{Float, FloatConst},
    identities::{one, zero},
};
use std::boxed::Box;

pub fn const_node<T>(v: T) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + Display,
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
            calc: Box::new(move |_x| vec![v]),
        },
    };
    NodeAdder::new(n, &[])
}

pub fn add_node<T>(a: (NodeHandle, usize), b: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + Display,
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
    NodeAdder::new(n, &[a, b])
}

pub fn mul_node<T>(a: (NodeHandle, usize), b: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + Display,
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
    NodeAdder::new(n, &[a, b])
}

pub fn sub_node<T>(a: (NodeHandle, usize), b: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + Display,
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
    NodeAdder::new(n, &[a, b])
}

pub fn div_node<T>(a: (NodeHandle, usize), b: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + Display,
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
    NodeAdder::new(n, &[a, b])
}

pub fn cos_node<T>(a: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + Display,
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
    NodeAdder::new(n, &[a])
}

pub fn normal_node<T>(m: (NodeHandle, usize), s: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + Display,
{
    let two = one::<T>() + one::<T>();
    let four = two + two;
    let pi = one::<T>().atan() * four;
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
                -((x - m) * (x - m) / ((one::<T>() + one::<T>()) * s * s))
                    - ((two * pi).sqrt() * s).ln()
            }),
            range: Box::new(move |p| {
                let m = p[0];
                let s = p[1];
                let x1 = m - T::from(10).unwrap() * s;
                let x2 = m + T::from(10).unwrap() * s;
                vec![(x1, x2)]
            }),
        },
    };
    NodeAdder::new(n, &[m, s])
}

pub fn uniform_node<T>(a: (NodeHandle, usize), b: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Sync + Send + Display,
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
                let x1 = p[0];
                let x2 = p[1];
                let x = x[0];
                if x <= x2 && x >= x1 {
                    zero()
                } else {
                    <T as Float>::infinity()
                }
            }),
            range: Box::new(move |p| {
                let x1 = p[0];
                let x2 = p[1];
                vec![(x1, x2)]
            }),
        },
    };

    NodeAdder::new(n, &[a, b])
}

pub fn phi_node<T>(x: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Error + Sync + Send + Display,
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
    NodeAdder::new(n, &[x])
}

pub fn scalar_func_node<T>(x: &[(NodeHandle, usize)], func: Box<dyn Fn(&[T]) -> T>) -> NodeAdder<T>
where
    T: 'static + Float + Error + Sync + Send + Display,
{
    let n = Node {
        info: BasicNode {
            parents: Vec::new(),
            children: Vec::new(),
            idx_in_var: Vec::new(),
            value_type: Vec::new(),
            ndim_input: x.len(),
            ndim_output: 1,
        },
        content: NodeContent::DeterministicNode {
            calc: Box::new(move |x: &[T]| vec![func(x)]),
        },
    };
    NodeAdder::new(n, x)
}

pub fn ln_node<T>(x: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Error + Sync + Send + Display,
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
            calc: Box::new(move |x: &[T]| vec![x[0].ln()]),
        },
    };
    NodeAdder::new(n, &[x])
}

pub fn lg_node<T>(x: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + Error + Sync + Send + Display,
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
            calc: Box::new(move |x: &[T]| vec![x[0].log10()]),
        },
    };
    NodeAdder::new(n, &[x])
}

pub fn t_node<T>(mu: (NodeHandle, usize), sigma: (NodeHandle, usize), dof: usize) -> NodeAdder<T>
where
    T: 'static + Float + FloatConst + Error + Gamma + Sync + Send + Display,
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
                let x: T = x[0];
                let m = p[0];
                let s = p[1];

                let tau = T::one() / s.powi(2);
                let k = T::from(dof).unwrap();
                let two = T::one() + T::one();
                T::ln_gamma((k + T::one()) / two).0 - T::ln_gamma(k / two).0
                    + T::ln(tau / k / T::PI()) / two
                    - (k + T::one()) / two * T::ln(T::one() + tau * (x - m).powi(2) / k)
            }),
            range: Box::new(move |p| {
                let m = p[0];
                let s = p[1];
                let x1 = m - T::from(5).unwrap() * s;
                let x2 = m + T::from(5).unwrap() * s;
                vec![(x1, x2)]
            }),
        },
    };
    NodeAdder::new(n, &[mu, sigma])
}

pub fn pareto_node<T>(a: (NodeHandle, usize), c: (NodeHandle, usize)) -> NodeAdder<T>
where
    T: 'static + Float + FloatConst + Error + Gamma + Sync + Send + Display,
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
                let x: T = x[0];
                let a = p[0];
                let c = p[1];

                if x < c {
                    //panic!();
                    -T::infinity()
                } else {
                    let result = a.ln() + a * c.ln() - (a + T::one()) * x.ln();
                    assert!(result.is_finite());
                    result
                }
            }),
            range: Box::new(move |p| {
                let a = p[0];
                let c = p[1];
                let eta = T::from(1e-4).unwrap();
                let x1 = c + T::from(1e-19).unwrap();
                let x2 = c * eta.powf(-T::one() / a);
                assert!(x2 > x1);
                vec![(x1, x2)]
            }),
        },
    };
    NodeAdder::new(n, &[a, c])
}

pub fn trunc_pareto_node<T>(
    a: (NodeHandle, usize),
    c: (NodeHandle, usize),
    xmin: T,
    xmax: T,
) -> NodeAdder<T>
where
    T: 'static + Float + FloatConst + Error + Gamma + Sync + Send + Display,
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
                let x: T = x[0];
                let a = p[0];
                let c = p[1];

                if x < c {
                    //panic!();
                    -T::infinity()
                } else {
                    let result = a.ln() + a * c.ln() - (a + T::one()) * x.ln();
                    assert!(result.is_finite());
                    result
                }
            }),
            range: Box::new(move |p| {
                let a = p[0];
                let c = p[1];
                let eta = T::from(1e-4).unwrap();
                let x1 = c + T::from(1e-19).unwrap();
                let x2 = c * eta.powf(-T::one() / a);
                assert!(x2 > x1);
                vec![(x1.max(xmin), xmax)]
            }),
        },
    };
    NodeAdder::new(n, &[a, c])
}
