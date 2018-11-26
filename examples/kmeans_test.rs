extern crate scorus;
use std::ops::{Add, Mul, Sub};

use scorus::kmeans;
use scorus::linear_space;

#[derive(Debug)]
struct MyVec(Vec<f64>);

impl<'a> Add for &'a MyVec {
    type Output = MyVec;
    fn add(self, rhs: Self) -> MyVec {
        MyVec(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
        )
    }
}

impl<'a> Sub for &'a MyVec {
    type Output = MyVec;
    fn sub(self, rhs: Self) -> MyVec {
        MyVec(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(&a, &b)| a - b)
                .collect(),
        )
    }
}

impl<'a> Mul<f64> for &'a MyVec {
    type Output = MyVec;

    fn mul(self, rhs: f64) -> MyVec {
        MyVec(self.0.iter().map(|&a| a * rhs).collect())
    }
}

impl linear_space::traits::LinearSpace<f64> for MyVec {
    fn dimension(&self) -> usize {
        self.0.len()
    }
}
impl linear_space::traits::InnerProdSpace<f64> for MyVec {
    fn dot(&self, rhs: &Self) -> f64 {
        self.0.iter().zip(rhs.0.iter()).map(|(&a, &b)| a * b).sum()
    }
}
impl linear_space::traits::PDInnerProdSpace<f64> for MyVec {}

fn main() {
    let points = vec![
        MyVec(vec![1.0, 1.0]),
        MyVec(vec![2.0, 1.0]),
        MyVec(vec![1.0, -1.0]),
        MyVec(vec![2.0, -1.0]),
        MyVec(vec![-1.0, 1.0]),
        MyVec(vec![-2.0, 1.0]),
        MyVec(vec![-1.0, -1.0]),
        MyVec(vec![-2.0, -1.0]),
    ];

    let seeds = vec![MyVec(vec![0.1, 1.0]), MyVec(vec![-0.1, -1.0])];

    let g = kmeans::kmeans(points, seeds, 30);
    for p in g {
        println!("{:?}", p);
    }
}
