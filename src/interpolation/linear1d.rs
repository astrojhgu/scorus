use num_traits::float::Float;

pub fn interp<T>(x: T, xlist: &[T], ylist: &[T]) -> T
where
    T: Copy + Float,
{
    assert!(xlist.len() > 1 && ylist.len() == xlist.len());
    for i in 0..(xlist.len() - 1) {
        if (xlist[i] - x) * (xlist[i + 1] - x) <= T::zero() {
            return (ylist[i + 1] - ylist[i]) / (xlist[i + 1] - xlist[i]) * (x - xlist[i])
                + ylist[i];
        }
    }
    if *xlist.first().unwrap() < *xlist.last().unwrap() {
        if x <= *xlist.first().unwrap() {
            return *ylist.first().unwrap();
        } else {
            return *ylist.last().unwrap();
        }
    } else if *xlist.first().unwrap() > *xlist.last().unwrap() {
        if x >= *xlist.first().unwrap() {
            return *ylist.first().unwrap();
        } else {
            return *ylist.last().unwrap();
        }
    } else {
        panic!()
    }
}
