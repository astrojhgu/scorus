use num_traits::float::Float;
use spade::{SpadeNum};
use std::marker::Copy;
use super::indexed_rtree::IndexedRTree;
use super::super::coordinates::{sphcoord::SphCoord, vec3d::Vec3d};


pub struct SphNN<T>
    where T: SpadeNum + Copy+Float
{
    tree: IndexedRTree<[T; 3]>,
    points: Vec<Vec3d<T>>
}

impl<T> SphNN<T>
    where T: SpadeNum + Copy + Float
{
    pub fn new()->SphNN<T>{
        SphNN{tree:IndexedRTree::new(), points:Vec::new()}
    }

    pub fn insert(&mut self, sphp: SphCoord<T>) {
        let v = Vec3d::from_sph_coord(sphp);
        self.tree.insert([v.x, v.y, v.z]);
        self.points.push(v);
    }

    pub fn nearest_n_neighbors(&self, query_point: SphCoord<T>, n: usize) -> Vec<(usize, T)> {
        let v = Vec3d::from_sph_coord(query_point);
        let mut result=Vec::new();
        for idx in self.tree.nearest_n_neighbors([v.x, v.y, v.z], n){
            result.push((idx, v.dot(self.points[idx]).acos()));
        }
        result
    }


}