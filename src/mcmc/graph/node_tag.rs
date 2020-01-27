use std::hash::Hash;
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub enum NodeTag<T>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
{
    Scalar(T),
    Vector(T, usize),
}

impl<T> std::fmt::Display for NodeTag<T>
where
    T: Clone + Eq + Hash + std::fmt::Display + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match *self {
            NodeTag::Scalar(ref x) => write!(f, "{}", x)?,
            NodeTag::Vector(ref x, _) => write!(f, "{}[..]", x)?,
        };
        Ok(())
    }
}
