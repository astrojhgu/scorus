//extern crate std;
extern crate num_traits;
extern crate quickersort;
extern crate rand;
extern crate scorus;
extern crate spade;

use std::fs::File;
use scorus::map_proj::hammer::{iproj, proj};
use scorus::coordinates::{SphCoord, Vec3d};
use scorus::interpolation::indexed_rtree::IndexedRTree;
use scorus::interpolation::sph_nn::{SphNN, NaturalSphNNInterpolator};
use spade::{PointN, SpadeNum};
use num_traits::float::{Float,FloatConst};
use std::io::Write;

fn main() {
    let mut aa=NaturalSphNNInterpolator::new();


    aa.add_point(SphCoord::from(Vec3d::new(1.0, 0.0, 0.0)), 1.5)
        .add_point(SphCoord::from(Vec3d::new(-1.0, 0.0, 0.0)), 1.5)
        .add_point(SphCoord::from(Vec3d::new(0.0, 1.0, 0.0)), 1.5)
        .add_point(SphCoord::from(Vec3d::new(0.0, -1.0, 0.0)), 1.5)
        .add_point(SphCoord::from(Vec3d::new(0.0, 0.0, 1.0)),1.0)
        .add_point(SphCoord::from(Vec3d::new(0.0, 0.0, -1.0)),1.0);

    let mut beam_file = File::create("beam.txt").unwrap();

    let pol_min = 0.0;
    let pol_max = f64::PI();
    let az_min = -f64::PI();
    let az_max = f64::PI();


    let npol = 100;
    let naz = 100;


    writeln!(beam_file, "{}", npol).unwrap();
    writeln!(beam_file, "{}", naz).unwrap();
    for i in 0..npol {
        let pol = (pol_max - pol_min) / (npol as f64) * (i as f64) + pol_min;
        write!(beam_file, "{} ", pol).unwrap();
    }
    writeln!(beam_file, "").unwrap();
    for j in 0..naz {
        let az = (az_max - az_min) / (naz as f64) * (j as f64) + az_min;
        write!(beam_file, "{} ", az).unwrap();
    }
    writeln!(beam_file, "").unwrap();
    for i in 0..npol {
        let pol = (pol_max - pol_min) / (npol as f64) * (i as f64) + pol_min;
        for j in 0..naz {
            let az = (az_max - az_min) / (naz as f64) * (j as f64) + az_min;

            let sph=SphCoord::new(pol, az);

            let ap=aa.value_at(sph, 3);

            let n = Vec3d::<f64>::from(sph);
            let n = n * ap;
            writeln!(beam_file, "{} {} {} {} {}", i, j, n.x, n.y, n.z).unwrap();
        }
    }

}
