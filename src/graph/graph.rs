use std;

use super::node::Node;
use super::node::NodeContent;
use super::node::BasicNode;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use super::node::ValueType;
use std::rc::Rc;
use num_traits::float::Float;
use std::fmt::{Display, Error, Formatter};
use std::iter::FromIterator;
use super::graph_var::GraphVar;
use num_traits::identities::zero;
use std::boxed::Box;

pub struct Graph<K, T>
where
    K: std::hash::Hash + Eq + Clone,
    T: Float + Sync + Send + Display,
{
    key_node_map: HashMap<K, usize>,
    node_key_map: HashMap<usize, K>,
    nodes: Vec<Node<T>>,
    num_of_fixed_vars: usize,
    num_of_deterministic_vars: usize,
    num_of_sampleable_vars: usize,
}

impl<K, T> Display for Graph<K, T>
where
    K: std::hash::Hash + Eq + Clone + Display,
    T: Float + Sync + Send + Display,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "nodes:\n")?;
        for (i, n) in self.nodes.iter().enumerate() {
            write!(f, "Node: {}, {}\n", i, self.node_key_map.get(&i).unwrap());
            write!(f, "{} \n", &n)?;
            write!(f, "========\n")?;
        }

        for (k, id) in &self.key_node_map {
            write!(f, "{}:{} ", k, id)?;
        }

        write!(f, "\nNum of fixed: {}\n", self.num_of_fixed_vars)?;
        write!(
            f,
            "Num of deterministic: {}\n",
            self.num_of_deterministic_vars
        )?;
        write!(f, "Num of sampleable: {}\n", self.num_of_sampleable_vars)?;
        Ok(())
    }
}

pub struct NodeAdder<'a, K, T>
where
    K: std::hash::Hash + Eq + Clone + 'a,
    T: 'a + Float + Sync + Send + Display,
{
    g: &'a mut Graph<K, T>,
    n: Node<T>,
    k: K,
    //parents:Vec<K>
}

impl<'a, K, T> NodeAdder<'a, K, T>
where
    K: std::hash::Hash + Eq + Clone,
    T: 'a + Float + Sync + Send + Display,
{
    pub fn with_parent(mut self, key: K, parent_output_id: usize) -> Self {
        if self.n.info.parents.len() > self.n.info.ndim_input {
            panic!("parents mismatch");
        }
        self.n
            .info
            .parents
            .push((*self.g.key_node_map.get(&key).unwrap(), parent_output_id));
        self
    }

    pub fn with_observed_value(mut self, idx: usize, x: T) -> Self {
        if let NodeContent::StochasticNode {
            ref mut is_observed,
            ref mut values,
            ..
        } = self.n.content
        {
            is_observed[idx] = true;
            values[idx] = x;
        }
        self
    }

    pub fn with_value(mut self, idx: usize, x: T) -> Self {
        if let NodeContent::StochasticNode {
            ref mut is_observed,
            ref mut values,
            ..
        } = self.n.content
        {
            is_observed[idx] = false;
            values[idx] = x;
        } else {
            panic!("It is not a stochastic node");
        }
        self
    }

    pub fn done(mut self) {
        let nid = self.g.nodes.len();

        if self.n.info.parents.len() != self.n.info.ndim_input {
            panic!("parents mismatch");
        }
        match self.n {
            Node {
                info:
                    BasicNode {
                        ref parents,
                        ref mut idx_in_var,
                        ref mut value_type,
                        ..
                    },
                content:
                    NodeContent::StochasticNode {
                        ref is_observed, ..
                    },
            } => {
                idx_in_var.clear();
                value_type.clear();
                for &ob in is_observed {
                    if ob {
                        idx_in_var.push(self.g.num_of_fixed_vars);
                        value_type.push(ValueType::FIXED);
                        self.g.num_of_fixed_vars += 1;
                    } else {
                        idx_in_var.push(self.g.num_of_sampleable_vars);
                        value_type.push(ValueType::SAMPLEABLE);
                        self.g.num_of_sampleable_vars += 1;
                    }
                }
            }
            Node {
                info:
                    BasicNode {
                        ref parents,
                        ref mut idx_in_var,
                        ndim_input,
                        ndim_output,
                        ref mut value_type,
                        ..
                    },
                content: NodeContent::DeterministicNode { .. },
            } => {
                idx_in_var.clear();
                value_type.clear();
                if ndim_input == 0 {
                    for i in 0..ndim_output {
                        idx_in_var.push(self.g.num_of_fixed_vars);
                        value_type.push(ValueType::FIXED);
                        self.g.num_of_fixed_vars += 1;
                    }
                } else {
                    for i in 0..ndim_output {
                        idx_in_var.push(self.g.num_of_deterministic_vars);
                        value_type.push(ValueType::DETERMINISTIC);
                        self.g.num_of_deterministic_vars += 1;
                    }
                }
            }
        }

        for &(p, _) in &self.n.info.parents {
            self.g.nodes[p].info.children.push(nid);
        }

        self.g
            .key_node_map
            .insert(self.k.clone(), self.g.nodes.len());
        self.g
            .node_key_map
            .insert(self.g.nodes.len(), self.k.clone());
        self.g.nodes.push(self.n);
    }
}

impl<K, T> Graph<K, T>
where
    K: std::hash::Hash + Eq + Clone,
    T: Float + Sync + Send + Display,
{
    pub fn new() -> Graph<K, T> {
        Graph {
            key_node_map: HashMap::new(),
            node_key_map: HashMap::new(),
            nodes: Vec::new(),
            num_of_fixed_vars: 0,
            num_of_deterministic_vars: 0,
            num_of_sampleable_vars: 0,
        }
    }

    pub fn add_node(&mut self, key: K, node: Node<T>) -> NodeAdder<K, T> {
        let mut node = node;

        if let Node {
            info:
                BasicNode {
                    ref mut idx_in_var,
                    ref mut value_type,
                    ndim_output,
                    ..
                },
            content:
                NodeContent::StochasticNode {
                    ref mut is_observed,
                    ..
                },
        } = node
        {
            is_observed.resize(ndim_output, false);
            idx_in_var.clear();
            value_type.clear();
        }
        /*
        if let Node{
            info: BasicNode{
                parents, ndim_output,
                idx_in_var,..
            },
            ref content: NodeContent::StochasticNode {
               is_observed,..
            }
        } = node
        {
            is_observed.resize(ndim_output, false);
            idx_in_var.clear();
        }
        */

        NodeAdder {
            g: self,
            n: node,
            k: key,
        }
    }

    pub fn enumerate_children_by_kind(&self, nid: usize) -> (BTreeSet<usize>, BTreeSet<usize>) {
        type Stack = Vec<usize>;
        let mut result0 = BTreeSet::<usize>::new();
        let mut result1 = BTreeSet::<usize>::new();
        let mut stack = Stack::new();
        stack.push(nid);
        while !stack.is_empty() {
            let top = stack.pop().unwrap();

            for i in self.nodes[top].get_children() {
                match self.nodes[*i].content {
                    NodeContent::StochasticNode { .. } => {
                        result0.insert(*i);
                    }
                    NodeContent::DeterministicNode { .. } => {
                        result1.insert(*i);
                        stack.push(*i);
                    }
                }
            }
        }
        (result0, result1)
    }

    pub fn seal(&mut self) {
        for i in 0..self.nodes.len() {
            let (s, d) = self.enumerate_children_by_kind(i);
            match &mut self.nodes[i].content {
                &mut NodeContent::StochasticNode {
                    ref mut all_stochastic_children,
                    ref mut all_deterministic_children,
                    ..
                } => {
                    *all_stochastic_children = Vec::from_iter(s);
                    *all_deterministic_children = Vec::from_iter(d);
                }
                _ => {}
            }
        }
    }

    pub fn init_gv(&self) -> GraphVar<T> {
        let mut gv = GraphVar {
            fixed_values: RefCell::new(Vec::new()),
            deterministic_values: RefCell::new(Vec::new()),
            sampleable_values: Vec::new(),
        };

        gv.fixed_values
            .borrow_mut()
            .resize(self.num_of_fixed_vars, zero());
        gv.deterministic_values
            .borrow_mut()
            .resize(self.num_of_deterministic_vars, zero());
        gv.sampleable_values
            .resize(self.num_of_sampleable_vars, zero());

        for (i, n) in self.nodes.iter().enumerate() {
            match n.content {
                NodeContent::DeterministicNode { .. } => {
                    self.update_deterministic_value_of(i, &mut gv);
                }
                NodeContent::StochasticNode {
                    ref is_observed,
                    ref values,
                    ..
                } => for (j, p) in n.info.idx_in_var.iter().enumerate() {
                    if is_observed[j] {
                        gv.fixed_values.borrow_mut()[*p] = values[j];
                    } else {
                        gv[*p] = values[j];
                    }
                },
            }
        }
        gv
    }

    pub fn cached_value_of(&self, i: usize, j: usize, gv: &GraphVar<T>) -> T {
        match self.nodes[i].info.value_type[j] {
            ValueType::DETERMINISTIC => {
                gv.deterministic_values.borrow_mut()[self.nodes[i].info.idx_in_var[j]]
            }
            ValueType::FIXED => gv.fixed_values.borrow_mut()[self.nodes[i].info.idx_in_var[j]],
            ValueType::SAMPLEABLE => gv.sampleable_values[self.nodes[i].info.idx_in_var[j]],
        }
    }

    pub fn cached_values_of(&self, i: usize, gv: &GraphVar<T>) -> Vec<T> {
        let mut result = Vec::<T>::new();
        result.reserve(self.nodes[i].info.ndim_output);
        for j in 0..self.nodes[i].info.ndim_output {
            result.push(self.cached_value_of(i, j, gv));
        }
        result
    }

    pub fn parent_values_of(&self, i: usize, gv: &GraphVar<T>) -> Vec<T> {
        let mut result = Vec::new();
        let node = &self.nodes[i];
        result.reserve(node.info.ndim_input);
        for j in 0..node.info.ndim_input {
            let (p, k) = node.info.parents[j];
            result.push(self.cached_value_of(p, k, gv));
        }
        result
    }

    pub fn update_deterministic_value_of(&self, i: usize, gv: &GraphVar<T>) {
        let pv = self.parent_values_of(i, gv);
        let node = &self.nodes[i];
        match node.content {
            NodeContent::DeterministicNode { ref calc, .. } => {
                let v = calc(&pv);
                for (m, k) in node.info.idx_in_var.iter().enumerate() {
                    match node.info.value_type[m] {
                        ValueType::FIXED => {
                            gv.fixed_values.borrow_mut()[*k] = v[m];
                            //panic!("Impossible")
                        }
                        ValueType::DETERMINISTIC => {
                            gv.deterministic_values.borrow_mut()[*k] = v[m];
                        }
                        ValueType::SAMPLEABLE => panic!("Impossible"),
                    }
                }
            }
            NodeContent::StochasticNode {
                ref logprob,
                ref values,
                ..
            } => for (m, k) in node.info.idx_in_var.iter().enumerate() {
                match node.info.value_type[m] {
                    ValueType::FIXED => {
                        //panic!("Impossible")
                        gv.fixed_values.borrow_mut()[*k] = values[m];
                    }
                    ValueType::DETERMINISTIC => {
                        panic!("Impossible");
                    }
                    ValueType::SAMPLEABLE => {}
                }
            },
        }
    }

    pub fn logprob(&self, i: usize, gv: &GraphVar<T>) -> T {
        if let NodeContent::StochasticNode { ref logprob, .. } = self.nodes[i].content {
            let x = self.cached_values_of(i, gv);
            let p = self.parent_values_of(i, gv);
            logprob(x.as_slice(), p.as_slice())
        } else {
            panic!("not a stochastic node");
        }
    }

    pub fn likelihood(&self, i: usize, gv: &GraphVar<T>) -> T {
        let mut result = zero();
        if let NodeContent::StochasticNode {
            ref all_stochastic_children,
            ..
        } = self.nodes[i].content
        {
            for n in all_stochastic_children {
                result = result + self.logprob(*n, gv);
            }
        }
        result
    }

    pub fn logpost(&self, i: usize, gv: &GraphVar<T>) -> T {
        let mut result = zero();
        if let NodeContent::StochasticNode {
            ref all_stochastic_children,
            ref logprob,
            ..
        } = self.nodes[i].content
        {
            let x = self.cached_values_of(i, gv);
            let p = self.parent_values_of(i, gv);
            for n in all_stochastic_children {
                result = result + self.logprob(*n, gv);
            }
            result = result + logprob(x.as_slice(), p.as_slice());
        }
        result
    }

    pub fn update_deterministic_children(&self, i: usize, gv: &GraphVar<T>) {
        if let Node {
            content:
                NodeContent::StochasticNode {
                    ref all_deterministic_children,
                    ..
                },
            ..
        } = self.nodes[i]
        {
            for j in all_deterministic_children {
                if let Node {
                    content: NodeContent::DeterministicNode { .. },
                    ..
                } = self.nodes[*j]
                {
                    self.update_deterministic_value_of(*j, gv);
                }
            }
        }
    }

    pub fn update_all_deterministic_nodes(&self, gv: &GraphVar<T>) {
        for i in 0..self.nodes.len() {
            self.update_deterministic_value_of(i, gv);
        }
    }

    pub fn logpost_all(&self, gv: &GraphVar<T>) -> T {
        let mut result = zero();
        self.update_all_deterministic_nodes(gv);
        for n in 0..self.nodes.len() {
            if let NodeContent::StochasticNode { .. } = self.nodes[n].content {
                result = result + self.logprob(n, gv);
            }
        }
        result
    }

    pub fn set_value_then_update(&self, i: usize, j: usize, x: T, gv: &mut GraphVar<T>) {
        if let Node {
            info:
                BasicNode {
                    ref idx_in_var,
                    ref value_type,
                    ..
                },
            content: NodeContent::StochasticNode { .. },
            ..
        } = self.nodes[i]
        {
            match value_type[j] {
                ValueType::DETERMINISTIC => gv.deterministic_values.borrow_mut()[idx_in_var[j]] = x,
                ValueType::FIXED => gv.fixed_values.borrow_mut()[idx_in_var[j]] = x,
                ValueType::SAMPLEABLE => gv.sampleable_values[idx_in_var[j]] = x,
            }
            self.update_deterministic_children(i, gv);
        }
    }

    pub fn set_value_no_update(&self, i: usize, j: usize, x: T, gv: &mut GraphVar<T>) {
        if let Node {
            info:
                BasicNode {
                    ref idx_in_var,
                    ref value_type,
                    ..
                },
            content: NodeContent::StochasticNode { .. },
            ..
        } = self.nodes[i]
        {
            match value_type[j] {
                ValueType::DETERMINISTIC => gv.deterministic_values.borrow_mut()[idx_in_var[j]] = x,
                ValueType::FIXED => gv.fixed_values.borrow_mut()[idx_in_var[j]] = x,
                ValueType::SAMPLEABLE => gv.sampleable_values[idx_in_var[j]] = x,
            }
        }
    }

    //pub fn lambda_adapter<'a>(&'a self)-> Box<Fn(&GraphVar<T>)->T>{
    //    Box::new(|x:&GraphVar<T>|->T{self.logpost_all(x)})
    //}
}

unsafe impl<K, T> std::marker::Sync for Graph<K, T>
where
    K: std::hash::Hash + Eq + Clone,
    T: Float + Sync + Send + std::fmt::Display,
{
}

unsafe impl<K, T> std::marker::Send for Graph<K, T>
where
    K: std::hash::Hash + Eq + Clone,
    T: Float + Sync + Send + std::fmt::Display,
{
}
