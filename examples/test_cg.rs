extern crate scorus;

use scorus::opt::cg::cg_iter;
use scorus::opt::tolerance::Tolerance;
use scorus::linear_space::InnerProdSpace;
use scorus::linear_space::type_wrapper::LsVec;

fn fobj(x: &LsVec<f64, Vec<f64>>)->f64{
    x.0.iter().map(|&x|x.powi(2)).sum::<f64>()
}

fn grad(x: &LsVec<f64, Vec<f64>>)->LsVec<f64, Vec<f64>>{
    let result=x.iter().map(|&x|2.0*x).collect::<Vec<_>>();
    LsVec(result)
}


fn main(){
    let mut x=LsVec(vec![1., 1., 1.]);
    let mut g=grad(&x);
    let mut d=g.clone();
    let mut fret=fobj(&x);
    for i in 0..10{
        println!("{}",cg_iter(&fobj, &grad, &mut x, &mut d, &mut g, &mut fret, Tolerance::Abs(0.00001)));
        println!("{:?}", x);
    }
}