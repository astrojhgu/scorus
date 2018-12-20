#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]

use std::collections::linked_list::LinkedList;
use std::collections::BinaryHeap;
use std::fmt::Debug;
use num_traits::float::Float;

#[derive(Clone, Debug, PartialEq)]
struct Interval<T>
where T:Float{
    x1:T,
    f1:T,
    x2:T,
    f2:T,
    subsum:T,
}

enum RefineResult<T>
where T:Float
{
    NoMorRefine(Interval<T>),
    Refined(Interval<T>, Interval<T>)
}

impl<T> Interval<T>
where T:Float{
    pub fn try_refine(self, func:&dyn Fn(T)->T, eps:T)->RefineResult<T>
    {
        let one=T::one();
        let two=one+one;
        let xm=(self.x1+self.x2)/two;
        let fm=func(xm);
        let ss1=(self.f1+fm)*(xm-self.x1)/two;
        let ss2=(fm+self.f2)*(self.x2-xm)/two;
        if (ss1+ss2-self.subsum).abs()<=eps{
            RefineResult::NoMorRefine(self)
        }else{
            RefineResult::Refined(Interval{x1:self.x1, f1:self.f1, x2:xm, f2:fm, subsum:ss1},
                                  Interval{x1:xm, f1:fm, x2:self.x2, f2:self.f2, subsum:ss2}
            )
        }
    }
}

impl<T> std::cmp::Eq for Interval<T>
where T:Float
{}

impl<T> std::cmp::PartialOrd for Interval<T>
where T:Float{
    fn partial_cmp(&self, other:&Self)->Option<std::cmp::Ordering>{
        Some(
        if self.subsum.abs()>other.subsum.abs(){
            std::cmp::Ordering::Less
        }else if self.subsum.abs()<other.subsum.abs(){
            std::cmp::Ordering::Greater
        }else{
            std::cmp::Ordering::Equal
        })
    }
}

impl<T> std::cmp::Ord for Interval<T>
where T:Float{
    fn cmp(&self, other:&Self)->std::cmp::Ordering{
        self.partial_cmp(other).unwrap()
    }
}


fn refine_iter<T>(func:&dyn Fn(T)->T, (not_refined, mut sorted_result): (LinkedList<Interval<T>>, BinaryHeap<Interval<T>>), eps:T, original_width:T)->(LinkedList<Interval<T>>, BinaryHeap<Interval<T>>)
where T:Float
{
    let mut updated_not_refined=LinkedList::new();
    for i in not_refined.into_iter(){
        let interval_width=i.x2-i.x1;
        match i.try_refine(func, eps/original_width*interval_width){
            RefineResult::NoMorRefine(i1)=>sorted_result.push(i1),
            RefineResult::Refined(i1,i2)=>{
                updated_not_refined.push_back(i1);
                updated_not_refined.push_back(i2);
            }
        }
    }
    (updated_not_refined, sorted_result)
}

fn refine_until_converged<T>(func:&dyn Fn(T)->T, init_list:LinkedList<Interval<T>>, eps:T)->BinaryHeap<Interval<T>>
where T:Float{
    let original_width=init_list.back().unwrap().x2-init_list.front().unwrap().x1;
    let mut refined_list=init_list;
    let mut sorted_result=BinaryHeap::new();
    loop{
        let refined=refine_iter(func, (refined_list, sorted_result), eps, original_width);
        refined_list=refined.0;
        sorted_result=refined.1;
        if refined_list.is_empty(){
            break sorted_result;
        }
    }
}

fn sum_up<T>(mut bh:BinaryHeap<Interval<T>>)->T
where T:Float+Debug{
    let mut result=T::zero();
    while let Some(i)=bh.pop(){
        result=result+i.subsum;
    }
    result
}

pub fn integrate<T>(func:&dyn Fn(T)->T, eps:T, init_ticks:&[T])->T
where T:Float+Debug
{
    let init_list:LinkedList<_>=init_ticks[1..].iter().zip(init_ticks[0..].iter()).map(
        |(&x2, &x1)|{
            let f1=func(x1);
            let f2=func(x2);
            let subsum=(f2+f1)*(x2-x1)/(T::one()+T::one());
            Interval::<T>{x1, f1, x2, f2, subsum}
        }
    ).collect();
    sum_up(refine_until_converged(func, init_list, eps))
}
