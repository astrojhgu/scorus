use num_traits::float::Float;

#[derive(Copy, Clone, Debug)]
pub enum Tolerance<T>
where
    T: Float + Copy,
{
    Rel(T),
    Abs(T),
    Both { rel: T, abs: T },
    Either { rel: T, abs: T },
}

impl<T> Tolerance<T>
where
    T: Float + Copy,
{
    pub fn rel(x: T) -> Tolerance<T> {
        Tolerance::Rel(x)
    }

    pub fn abs(x: T) -> Tolerance<T> {
        Tolerance::Abs(x)
    }

    pub fn both(rel: T, abs: T) -> Tolerance<T> {
        Tolerance::Both { rel, abs }
    }

    pub fn either(rel: T, abs: T) -> Tolerance<T> {
        Tolerance::Either { rel, abs }
    }

    pub fn accepted(&self, x1: T, x2: T) -> bool {
        let two: T = T::one() + T::one();
        match *self {
            Tolerance::Rel(x) => (x2 - x1).abs() < (x1 + x2).abs() * two * x,
            Tolerance::Abs(x) => (x2 - x1).abs() < x,
            Tolerance::Both { rel, abs } => {
                ((x2 - x1).abs() < (x1 + x2).abs() * two * rel) && ((x2 - x1).abs() < abs)
            }
            Tolerance::Either { rel, abs } => {
                ((x2 - x1).abs() < (x1 + x2).abs() * two * rel) || ((x2 - x1).abs() < abs)
            }
        }
    }
}
