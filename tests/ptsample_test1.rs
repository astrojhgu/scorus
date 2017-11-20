extern crate num_traits;
extern crate rsmcmc;
use num_traits::float::Float;

fn target_dist(x: &Vec<f64>) -> f64 {
    let x = x[0];
    if x < 0.0 || x > 1.0 {
        -std::f64::INFINITY
    } else {
        if x < 0.5 {
            (0.2).ln()
        } else {
            (0.8).ln()
        }
    }
}



#[test]
fn test() {
    assert_eq!(1, 1);
}
