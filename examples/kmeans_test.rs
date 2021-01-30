extern crate rand;
extern crate rand_distr;
extern crate scorus;

use std::fs::File;
use std::io::Write;

use rand::{Rng, distributions::Distribution};
use rand_distr::{
    Normal
};
use rand::thread_rng;

use scorus::kmeans;

use scorus::linear_space::type_wrapper::LsVec;

fn main() {
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut rng = thread_rng();

    let mut points = vec![];
    for _i in 0..1000 {
        let x = rng.sample(normal) + 2.0;
        let y = rng.sample(normal) + 2.0;
        points.push(LsVec(vec![x, y]));
    }

    for _i in 0..1000 {
        let x = rng.sample(normal) - 2.0;
        let y = rng.sample(normal) - 2.0;
        points.push(LsVec(vec![x, y]));
    }

    let mut centroids = vec![LsVec(vec![0., 0.]), LsVec(vec![1., 1.])];
    kmeans::kmeans2(&points, &mut centroids, 100);
    let points = kmeans::classify(&points, &centroids);

    for i in 0..2 {
        let mut f = File::create(format!("cluster_{}.txt", i)).unwrap();
        for p in &points[i] {
            writeln!(f, "{} {}", p[0], p[1]).unwrap();
        }
    }
}
