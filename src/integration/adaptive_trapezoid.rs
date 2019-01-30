#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]

use num_traits::float::Float;
use std::fmt::Debug;

#[derive(Clone, Copy)]
struct Point<T> {
    x: T,
    f: T,
}

pub fn integrate<T>(func: &dyn Fn(T) -> T, eps: T, init_ticks: &[T]) -> T
    where T: Float + Debug,
{
    if init_ticks.len()<=1{
        return T::zero();
    }
    let two=T::one()+T::one();
    let four=two+two;
    let half=T::one()/two;
    let quarter=half/two;
    let full_width=*init_ticks.last().unwrap()-*init_ticks.first().unwrap();
    let mut areas=Vec::<T>::new();
    let mut points:Vec<_>=init_ticks.iter().map(|&x|Point::<T>{x, f:func(x)}).collect();
    //let mut right=*points.last().unwrap();
    let mut right;
    let eps=eps*four/full_width;
    while points.len()>1 {
        right=points.pop().unwrap();
        let left=*points.last().unwrap();
        let mid=(left.x+right.x)*half;
        let fmid=func(mid);
        if (left.f+right.f-fmid*two).abs()<=eps{
            areas.push((left.f+right.f+fmid*two)*(right.x-left.x)*quarter);
            //points.pop();
            //right=left;
        }
            else{
                //points.pop();
                points.push(Point::<T>{x:mid, f:fmid});
                points.push(right);
            }
    }
    (&mut areas).sort_by(|a, b| {
        //(&mut areas).sort_unstable_by(|a, b| {
        if a.abs() < b.abs() {
            std::cmp::Ordering::Less
        } else if a.abs() > b.abs() {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }});
    areas.into_iter().fold(T::zero(), |a, b| a + b)
}

pub fn integrate_nosort<T>(func: &dyn Fn(T) -> T, eps: T, init_ticks: &[T]) -> T
    where T: Float + Debug,
{
    if init_ticks.len()<=1{
        return T::zero();
    }
    let two=T::one()+T::one();
    let four=two+two;
    let half=T::one()/two;
    let quarter=half/two;
    let full_width=*init_ticks.last().unwrap()-*init_ticks.first().unwrap();
    let mut area=T::zero();
    let mut points:Vec<_>=init_ticks.iter().map(|&x|Point::<T>{x, f:func(x)}).collect();
    //let mut right=*points.last().unwrap();
    let mut right;
    let eps=eps*four/full_width;
    while points.len()>1 {
        right=points.pop().unwrap();
        let left=*points.last().unwrap();
        let mid=(left.x+right.x)*half;
        let fmid=func(mid);
        if (left.f+right.f-fmid*two).abs()<=eps{
            area=area+(left.f+right.f+fmid*two)*(right.x-left.x)*quarter;
            //points.pop();
            //right=left;
        }
            else{
                //points.pop();
                points.push(Point::<T>{x:mid, f:fmid});
                points.push(right);
            }
    }
    area
}
