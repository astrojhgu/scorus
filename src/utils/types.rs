pub trait HasLen {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait HasElement {
    type ElmType;
}

pub trait Resizeable: HasElement {
    fn resize(&mut self, _: usize, x: Self::ElmType);
}

pub trait ItemSwapable {
    fn swap_items(&mut self, i: usize, j: usize);
}

pub trait InitFromLen: HasLen + HasElement {
    fn init(_: usize) -> Self;
}
