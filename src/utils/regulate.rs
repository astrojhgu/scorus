use num::traits::float::Float;

pub fn regulate<T>(x: T, a: T, b: T) -> T
where
    T: Float,
{
    let width = b - a;
    let n = ((x - a) / width).floor();
    x - n * width
}
