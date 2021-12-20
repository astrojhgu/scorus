mod pdqselect;
use std::boxed::Box;
use std::cmp::{Ord, Ordering, PartialEq, PartialOrd};
use std::collections::BinaryHeap;
use std::ops::{Add, Sub};

use self::pdqselect::select_by;

use num::traits::Bounded;

pub struct Node<T> {
    index: usize,
    threshold: T,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
}

pub struct IdxDistPair<T>
where
    T: PartialOrd,
{
    pub index: usize,
    pub dist: T,
}

impl<T> PartialEq for IdxDistPair<T>
where
    T: PartialOrd,
{
    fn eq(&self, other: &Self) -> bool {
        self.dist.eq(&(other.dist))
    }
}

impl<T> Eq for IdxDistPair<T> where T: PartialOrd {}

impl<T> PartialOrd for IdxDistPair<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&(other.dist))
    }
}

impl<T> Ord for IdxDistPair<T>
where
    T: PartialOrd,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub trait MetricPoint: Copy {
    type Distance;
    fn distance_to(&self, other: &Self) -> Self::Distance;
}

pub struct VpTree<MP, V = ()>
where
    MP: MetricPoint,
    V: Copy,
{
    root: Option<Box<Node<MP::Distance>>>,
    pub items: Vec<(MP, V)>,
}

impl<MP, V> VpTree<MP, V>
where
    MP: MetricPoint,
    MP::Distance: Bounded
        + PartialOrd
        + Copy
        + Add<MP::Distance, Output = MP::Distance>
        + Sub<MP::Distance, Output = MP::Distance>,
    V: Copy,
{
    pub fn new(items: Vec<(MP, V)>) -> VpTree<MP, V> {
        let nitems = items.len();
        let mut result = VpTree { root: None, items };
        let root = result.build_from_points(0, nitems);
        result.root = root;
        result
    }

    pub fn build_from_points(
        &mut self,
        lower: usize,
        upper: usize,
    ) -> Option<Box<Node<MP::Distance>>> {
        if upper == lower {
            return None;
        }

        let mut node = Node {
            index: lower,
            threshold: MP::Distance::max_value(),
            left: None,
            right: None,
        };
        let p0 = self.items[lower];
        if upper - lower > 1 {
            let median = (upper + lower + 1) / 2;

            //select_by(&mut self.items[lower+1], median)
            select_by(
                &mut self.items[(lower + 1)..upper],
                median - lower - 1,
                |p1, p2| {
                    p0.0.distance_to(&p1.0)
                        .partial_cmp(&p0.0.distance_to(&p2.0))
                        .unwrap()
                },
            );

            node.threshold = p0.0.distance_to(&self.items[median].0);
            node.left = self.build_from_points(lower + 1, median);
            node.right = self.build_from_points(median, upper);
        }
        Some(Box::new(node))
    }

    pub fn search(&self, target: &MP, k: usize) -> BinaryHeap<IdxDistPair<MP::Distance>> {
        let mut heap = BinaryHeap::<IdxDistPair<MP::Distance>>::new();
        let mut tau = MP::Distance::max_value();
        self.inner_search(&self.root, target, k, &mut heap, &mut tau);
        heap
    }

    fn inner_search(
        &self,
        node: &Option<Box<Node<MP::Distance>>>,
        target: &MP,
        k: usize,
        heap: &mut BinaryHeap<IdxDistPair<MP::Distance>>,
        tau: &mut MP::Distance,
    ) {
        match *node {
            None => (),
            Some(ref pn) => {
                let dist = self.items[pn.index].0.distance_to(target);

                if dist < *tau {
                    if heap.len() == k {
                        heap.pop();
                    }
                    heap.push(IdxDistPair {
                        index: pn.index,
                        dist,
                    });
                    if heap.len() == k {
                        *tau = heap.peek().unwrap().dist
                    }
                }

                if pn.left.is_none() && pn.right.is_none() {
                    return;
                }

                if dist < pn.threshold {
                    if dist - *tau <= pn.threshold {
                        self.inner_search(&pn.left, target, k, heap, tau);
                    }

                    if dist + *tau >= pn.threshold {
                        self.inner_search(&pn.right, target, k, heap, tau);
                    }
                } else {
                    if dist + *tau >= pn.threshold {
                        self.inner_search(&pn.right, target, k, heap, tau);
                    }
                    if dist - *tau <= pn.threshold {
                        self.inner_search(&pn.left, target, k, heap, tau);
                    }
                }
            }
        }
    }
}
