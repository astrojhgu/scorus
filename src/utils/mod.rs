pub trait HasLen {
    fn len(&self) -> usize;
}

pub trait HasElement {
    type ElmType;
}

pub trait Resizeable: HasElement {
    fn resize(&mut self, usize, x: Self::ElmType);
}

pub trait ItemSwapable {
    fn swap_items(&mut self, i: usize, j: usize);
}
