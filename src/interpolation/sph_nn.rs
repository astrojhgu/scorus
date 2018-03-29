use num_traits::float::Float;
use spade::{SpadeNum};
use std::marker::Copy;
use super::indexed_rtree::IndexedRTree;
use super::super::coordinates::{sphcoord::SphCoord, vec3d::Vec3d};
use std::ops::{Add, Mul, Div};

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

pub struct NaturalSphNNInterpolator<T, V>
where T:SpadeNum+Copy+Float,
V:Float+Copy
{
    sphnn:SphNN<T>,
    values:Vec<V>
}

impl<T, V> NaturalSphNNInterpolator<T, V>
where T:SpadeNum+Copy+Float,
V:Float+Copy+Add<T, Output=V>+Mul<T, Output=V>+Div<T, Output=V>{
    pub fn new()->NaturalSphNNInterpolator<T, V>{
        NaturalSphNNInterpolator{sphnn:SphNN::new(), values:Vec::new()}
    }

    pub fn add_point(&mut self, p:SphCoord<T>, v:V)->&mut Self {
        self.sphnn.insert(p);
        self.values.push(v);
        self
    }

    pub fn value_at(&self, p:SphCoord<T>, n:usize)->V{
        let np=self.sphnn.nearest_n_neighbors(p, n);

        let mut result=V::zero();
        let mut wgt=T::zero();
        for idx_dist in np{
            if idx_dist.1==T::zero(){
                return self.values[idx_dist.0];
            }
            let wgt1=T::one()/idx_dist.1;
            wgt=wgt+wgt1;
            result=result+self.values[idx_dist.0]*wgt1;
        }
        result/wgt
    }
}
