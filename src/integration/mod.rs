pub mod simpsons;
pub use self::simpsons::integrate as simpsons_int;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
