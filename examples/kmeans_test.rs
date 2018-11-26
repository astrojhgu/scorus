extern crate rand;
extern crate scorus;

use std::fs::File;
use std::io::Write;

use rand::distributions::Normal;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;

use std::ops::{Add, Mul, Sub};

use scorus::kmeans;
use scorus::linear_space;
use scorus::linear_space::type_wrapper::LsVec;

fn main() {
    let normal = Normal::new(0.0, 1.0);

    let mut rng = thread_rng();

    let mut points = vec![vec![], vec![]];
    for i in 0..1000 {
        let x = normal.sample(&mut rng) + 2.0;
        let y = normal.sample(&mut rng) + 2.0;
        let cid = if normal.sample(&mut rng) > 0.0 { 1 } else { 0 };
        points[cid].push(LsVec(vec![x, y]));
    }

    for i in 0..1000 {
        let x = normal.sample(&mut rng) - 2.0;
        let y = normal.sample(&mut rng) - 2.0;
        let cid = if normal.sample(&mut rng) > 0.0 { 1 } else { 0 };
        points[cid].push(LsVec(vec![x, y]));
    }

    for i in 0..10000 {
        points = kmeans::kmeans_iter(points).unwrap();
    }

    for i in 0..2 {
        let mut f = File::create(format!("cluster_{}.txt", i)).unwrap();
        for p in &points[i] {
            writeln!(f, "{} {}", p[0], p[1]);
        }
    }
}