pub trait HasLength {
    fn length(&self) -> usize;
}

pub trait Resizeable {
    fn resize(&mut self, usize);
}

pub trait ItemSwapable {
    fn swap_items(&mut self, i: usize, j: usize);
}
