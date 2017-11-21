
#[derive(Debug)]
pub enum McmcErrs {
    NWalkersIsZero,
    NWalkersIsNotEven,
    NWalkersMismatchesNBeta,
    BetaNotInDecrOrd,
}
