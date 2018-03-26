//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;
extern crate spade;
use scorus::map_proj::hammer::{iproj, proj};
use scorus::coordinates::{SphCoord, Vec3d};
use scorus::interpolation::indexed_rtree::IndexedRTree;
use scorus::interpolation::sph_nn::SphNN;
use spade::{PointN, SpadeNum};

fn main() {
    let mut aa=SphNN::new();
    aa.insert(SphCoord::from(Vec3d::new(1.0, 0.0, 0.0)));
    aa.insert(SphCoord::from(Vec3d::new(-1.0, 0.0, 0.0)));
    aa.insert(SphCoord::from(Vec3d::new(0.0, 1.0, 0.0)));
    aa.insert(SphCoord::from(Vec3d::new(0.0, -1.0, 0.0)));
    aa.insert(SphCoord::from(Vec3d::new(0.0, 0.0, 1.0)));
    aa.insert(SphCoord::from(Vec3d::new(0.0, 0.0, -1.0)));
    let nn=aa.nearest_n_neighbors(SphCoord::new(45.0_f64.to_radians(), 45.0_f64.to_radians()), 4);
    println!("{:?}", nn);
}
