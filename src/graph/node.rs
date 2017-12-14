extern crate std;
use num_traits::float::Float;
use std::rc::Rc;
use super::graph_var::GraphVar;



pub enum Node<T>
    where T:Float+Sync+Send+std::fmt::Display
{
    StochasticNode {
        parents:Vec<(usize, usize)>,
        children:Vec<usize>,
        all_stochastic_children:Vec<usize>,
        idx_in_var:Vec<usize>,
        ndim_input:usize,
        ndim_output:usize,
        is_observed:Vec<bool>,
        initial_values:Vec<T>,
        logprob:Box<Fn(&[T], &[T])->T>
    },

    DeterministicNode{
        parents:Vec<(usize, usize)>,
        children:Vec<usize>,
        idx_in_var:Vec<usize>,
        ndim_input:usize,
        ndim_output:usize,
        calc: Box<Fn(&[T])->Vec<T>>
    }
}

impl<T> Node<T>
where T:Float+Sync+Send+std::fmt::Display{
    pub fn get_children(&self)->&Vec<usize>{
        match self{
            &Node::StochasticNode {ref children,..}=>children,
            &Node::DeterministicNode {ref children,..}=>children,
        }
    }
}


impl<T> std::fmt::Display for Node<T>
    where T:Float+Sync+Send+std::fmt::Display
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error>
    {
        match self{
            &Node::StochasticNode {ref parents, ref children, ref all_stochastic_children, ref idx_in_var, ref ndim_input, ref ndim_output, ref is_observed, ref initial_values,..} =>{
                write!(f, "Parents:\n")?;
                for &(i,j) in parents{
                    write!(f,"{} - {} ", i,j)?;
                }
                write!(f, "\nChildren:\n")?;
                for i in children{
                    write!(f,"{} ", i)?
                }
                write!(f, "\nall stochastic children\n")?;
                for i in all_stochastic_children{
                    write!(f,"{}", i)?
                }
                write!(f, "\nidx_in_var:\n")?;
                for i in idx_in_var{
                    write!(f, "{} ", i);
                }
                write!(f, "\nninput:{}\n", ndim_input)?;
                write!(f, "\nnoutput:{}\n", ndim_output)?;
                write!(f, "observed:\n")?;
                for i in is_observed{
                    write!(f, "{} ", i)?;
                }
                write!(f, "\ninitial values:\n")?;
                for v in initial_values{
                    write!(f, "{} ", v)?;
                }
                Ok(())
            },
            &Node::DeterministicNode {ref parents, ref children, ref idx_in_var, ref ndim_input, ref ndim_output,..} =>{
                write!(f, "Parents:\n")?;
                for &(i,j) in parents{
                    write!(f,"{} - {} ", i,j)?;
                }
                write!(f, "\nChildren:\n")?;
                for i in children{
                    write!(f,"{} ", i)?
                }
                write!(f, "\nidx_in_var:\n")?;
                for i in idx_in_var{
                    write!(f, "{} ", i);
                }
                write!(f, "\nninput:{}\n", ndim_input)?;
                write!(f, "\nnoutput:{}\n", ndim_output)?;
                Ok(())
            },
        }
    }
}