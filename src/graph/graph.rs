extern crate std;


use super::node::Node;
use std::collections::{HashMap, HashSet};

use std::rc::Rc;
use num_traits::float::Float;
use std::fmt::{Display, Formatter, Error};
use std::iter::FromIterator;

pub struct Graph<K, T>
where K:std::hash::Hash+Eq+Clone, T:Float+Sync+Send+Display
{
    node_map:HashMap<K, usize>,
    nodes:Vec<Node<T>>,
    num_of_fixed_vars:usize,
    num_of_deterministic_vars:usize,
    num_of_stochastic_vars:usize,
}

impl<K, T> Display for Graph<K,T>
where K:std::hash::Hash+Eq+Clone+Display, T:Float+Sync+Send+Display
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error>{
        write!(f, "nodes:\n")?;
        for n in &self.nodes{
            write!(f, "{} \n", &n)?;
            write!(f, "========\n")?;

        }

        for (k, id) in &self.node_map{
            write!(f, "{}:{} ", k, id)?;
        }

        write!(f, "\nNum of fixed: {}\n", self.num_of_fixed_vars)?;
        write!(f, "Num of deterministic: {}\n", self.num_of_deterministic_vars)?;
        write!(f, "Num of sampleable: {}\n", self.num_of_stochastic_vars)?;
        Ok(())
    }
}


pub struct NodeAdder<'a, K,T>
where K:std::hash::Hash+Eq+Clone+'a, T:'a+Float+Sync+Send+Display
{
    g:&'a mut Graph<K,T>,
    n:Node<T>,
    k:K,

    //parents:Vec<K>
}

impl <'a, K,T> NodeAdder<'a, K,T>
    where K:std::hash::Hash+Eq+Clone, T:'a+Float+Sync+Send+Display
{
    pub fn with_parent(mut self, key:K, parent_output_id:usize)->Self{
        //self.parents.push(key.clone());
        match self.n{
            Node::StochasticNode {ref mut parents,ref mut idx_in_var, ndim_input,..} =>{
                if parents.len()>ndim_input{
                    panic!("parents mismatch");
                }
                parents.push((*self.g.node_map.get(&key).unwrap(), parent_output_id));

            },
            Node::DeterministicNode {ref mut parents,ref mut idx_in_var, ndim_input, ..}=>{
                if parents.len()>ndim_input{
                    panic!("parents mismatch");
                }
                parents.push((*self.g.node_map.get(&key).unwrap(), parent_output_id));
            },
        };
        self
    }

    pub fn with_observed_value(mut self, idx:usize, x:T)->Self{
        if let Node::StochasticNode {ref mut is_observed, ref mut initial_values,..}=self.n{
            is_observed[idx]=true;
            initial_values[idx]=x;
        }
        self
    }

    pub fn done(mut self){
        let nid=self.g.nodes.len();

        match self.n{
            Node::StochasticNode {ref parents,ref mut idx_in_var, ref is_observed, ndim_input,..} =>{
                if parents.len()!=ndim_input{
                    panic!("parents mismatch");
                }
                for ob in is_observed{
                    if *ob{
                        idx_in_var.push(self.g.num_of_fixed_vars);
                        self.g.num_of_fixed_vars+=1;
                    }
                    else{
                        idx_in_var.push(self.g.num_of_stochastic_vars);
                        self.g.num_of_stochastic_vars+=1;
                    }
                }

                for &(p, _ ) in parents{
                    match self.g.nodes[p]{
                        Node::StochasticNode {ref mut children, ..}=> {
                            children.push(nid);
                        }
                        Node::DeterministicNode {ref mut children, ..}=>{
                            children.push(nid);
                        }
                    }
                }
            },
            Node::DeterministicNode {ref parents, ref mut idx_in_var, ndim_input, ndim_output,..}=>{
                if parents.len()!=ndim_input{
                    println!("{} {}", parents.len(), ndim_input);
                    panic!("parents mismatch");
                }
                if ndim_input==0{
                    idx_in_var.clear();
                    for i in 0..ndim_output{
                        idx_in_var.push(self.g.num_of_fixed_vars);
                        self.g.num_of_fixed_vars+=1;
                    }
                }
                else{
                    idx_in_var.clear();
                    for i in 0..ndim_output{
                        idx_in_var.push(self.g.num_of_deterministic_vars);
                        self.g.num_of_deterministic_vars+=1;
                    }
                }

                for &(p, _ ) in parents{
                    match self.g.nodes[p]{
                        Node::StochasticNode {ref mut children, ..}=> {
                            children.push(nid);
                        }
                        Node::DeterministicNode {ref mut children, ..}=>{
                            children.push(nid);
                        }
                    }
                }

            },
        };

        self.g.node_map.insert(self.k.clone(), self.g.nodes.len());
        self.g.nodes.push(self.n);
    }
}



impl<K,T> Graph<K,T>
    where K:std::hash::Hash+Eq+Clone, T:Float+Sync+Send+Display{

    pub fn new() -> Graph<K,T>{
        Graph{node_map:HashMap::new(),
            nodes:Vec::new(),
            num_of_fixed_vars:0,
            num_of_deterministic_vars:0,
            num_of_stochastic_vars:0
        }
    }

    pub fn add_node(&mut self, key:K, node:Node<T>)->NodeAdder<K,T>{
        let mut node=node;

        if let Node::StochasticNode {ref mut is_observed,ref mut idx_in_var, ndim_output,..}=node
        {
            is_observed.resize(ndim_output, false);
            idx_in_var.clear();
        }
        NodeAdder{g:self, n:node, k:key}
    }


    pub fn enumerate_stochastic_children(&self, nid:usize)->HashSet<usize>{
        type Stack=Vec<usize>;
        let mut result=HashSet::<usize>::new();
        let mut stack=Stack::new();
        stack.push(nid);
        while !stack.is_empty(){
            let top=(stack.pop().unwrap());

            for i in self.nodes[top].get_children(){
                match self.nodes[*i]{
                    Node::StochasticNode {..} => {
                        result.insert(*i);
                    }
                    Node::DeterministicNode {..}=>{
                        stack.push(*i);
                    }
                }
            }

        }
        result
    }


    pub fn seal(&mut self){
        for i in 0..self.nodes.len(){
            let s=self.enumerate_stochastic_children(i);
            match &mut self.nodes[i]{
                &mut Node::StochasticNode {ref mut all_stochastic_children,..} =>{
                    *all_stochastic_children=Vec::from_iter(s);
                },
                _ => {}
            }
        }
    }
}
