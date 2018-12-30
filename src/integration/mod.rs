pub mod adaptive_trapezoid;
pub mod integration_errors;
pub mod simpsons;
pub use self::adaptive_trapezoid::integrate as adptpd_int;
pub use self::simpsons::integrate as simpsons_int;
