#[derive(Debug)]
pub enum McmcErr {
    NWalkersIsZero,
    NWalkersIsNotEven,
    NWalkersMismatchesNBeta,
    BetaNotInDecrOrd,
}
