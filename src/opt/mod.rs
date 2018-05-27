pub mod bas_utils;
pub mod linmin;
pub mod opt_errors;
pub mod powell;
pub mod tolerance;
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
