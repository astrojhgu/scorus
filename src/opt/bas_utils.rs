use num_traits::float::Float;

pub fn swap<T>(a: &mut T, b: &mut T)
where
    T: Copy,
{
    let c = *a;
    *a = *b;
    *b = c;
}

pub fn shft3<T>(a: &mut T, b: &mut T, c: &mut T, d: T)
where
    T: Copy,
{
    *a = *b;
    *b = *c;
    *c = d;
}

pub fn shft<T>(a: &mut T, b: &mut T, c: &mut T, d: T)
where
    T: Copy,
{
    *a = *b;
    *b = *c;
    *c = d;
}

pub fn mov3<T>(a: &mut T, b: &mut T, c: &mut T, d: T, e: T, f: T)
where
    T: Copy,
{
    *a = d;
    *b = e;
    *c = f;
}

pub fn sign<T>(a: T, b: T) -> T
where
    T: Float + Copy,
{
    a.abs() * b.signum()
}

pub fn max<T>(a: T, b: T) -> T
where
    T: Float + Copy,
{
    if a > b {
        a
    } else {
        b
    }
}

pub fn sqr<T>(a:T) ->T
    where T:Float+Copy,{
    a*a
}