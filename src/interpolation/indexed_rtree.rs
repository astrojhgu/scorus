use spade::{PointN, rtree::RTree};

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
struct IndexedPoint<P>
    where P: PointN {
    position: P,
    idx: usize,
}

impl<P> PointN for IndexedPoint<P>
    where P: PointN,
{
    type Scalar = P::Scalar;
    fn dimensions() -> usize {
        P::dimensions()
    }

    fn from_value(value: P::Scalar) -> Self {
        IndexedPoint { position: P::from_value(value), idx: 0 }
    }

    fn nth(&self, index: usize) -> &Self::Scalar {
        self.position.nth(index)
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        self.position.nth_mut(index)
    }
}

pub struct IndexedRTree<P>
    where P: PointN
{
    rtree: RTree<IndexedPoint<P>>,
}

impl<P> IndexedRTree<P>
    where P: PointN
{
    pub fn new() -> IndexedRTree<P> {
        IndexedRTree { rtree: RTree::new() }
    }

    pub fn insert(&mut self, p: P) {
        let current_size = self.rtree.size();
        self.rtree.insert(IndexedPoint { position: p, idx: current_size });
    }

    pub fn nearest_n_neighbors(&self, query_point: P, n: usize) -> Vec<usize> {
        let points = self.rtree.nearest_n_neighbors(&IndexedPoint { position: query_point, idx: 0 }, n);
        let mut result = Vec::new();
        for p in points {
            result.push(p.idx);
        }
        result
    }
}
