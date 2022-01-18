//! pixel indexing

use std::fmt::Debug;

use num::traits::{cast::NumCast, int::PrimInt};

/// calculate the sqrt of some integer
pub fn isqrt<T>(x: T) -> T
where
    T: PrimInt + Debug,
{
    <T as NumCast>::from((<f64 as NumCast>::from(x).unwrap() + 0.5).sqrt()).unwrap()
}

const CTAB: [i32; 256] = [
    0, 1, 256, 257, 2, 3, 258, 259, 512, 513, 768, 769, 514, 515, 770, 771, 4, 5, 260, 261, 6, 7,
    262, 263, 516, 517, 772, 773, 518, 519, 774, 775, 1024, 1025, 1280, 1281, 1026, 1027, 1282,
    1283, 1536, 1537, 1792, 1793, 1538, 1539, 1794, 1795, 1028, 1029, 1284, 1285, 1030, 1031, 1286,
    1287, 1540, 1541, 1796, 1797, 1542, 1543, 1798, 1799, 8, 9, 264, 265, 10, 11, 266, 267, 520,
    521, 776, 777, 522, 523, 778, 779, 12, 13, 268, 269, 14, 15, 270, 271, 524, 525, 780, 781, 526,
    527, 782, 783, 1032, 1033, 1288, 1289, 1034, 1035, 1290, 1291, 1544, 1545, 1800, 1801, 1546,
    1547, 1802, 1803, 1036, 1037, 1292, 1293, 1038, 1039, 1294, 1295, 1548, 1549, 1804, 1805, 1550,
    1551, 1806, 1807, 2048, 2049, 2304, 2305, 2050, 2051, 2306, 2307, 2560, 2561, 2816, 2817, 2562,
    2563, 2818, 2819, 2052, 2053, 2308, 2309, 2054, 2055, 2310, 2311, 2564, 2565, 2820, 2821, 2566,
    2567, 2822, 2823, 3072, 3073, 3328, 3329, 3074, 3075, 3330, 3331, 3584, 3585, 3840, 3841, 3586,
    3587, 3842, 3843, 3076, 3077, 3332, 3333, 3078, 3079, 3334, 3335, 3588, 3589, 3844, 3845, 3590,
    3591, 3846, 3847, 2056, 2057, 2312, 2313, 2058, 2059, 2314, 2315, 2568, 2569, 2824, 2825, 2570,
    2571, 2826, 2827, 2060, 2061, 2316, 2317, 2062, 2063, 2318, 2319, 2572, 2573, 2828, 2829, 2574,
    2575, 2830, 2831, 3080, 3081, 3336, 3337, 3082, 3083, 3338, 3339, 3592, 3593, 3848, 3849, 3594,
    3595, 3850, 3851, 3084, 3085, 3340, 3341, 3086, 3087, 3342, 3343, 3596, 3597, 3852, 3853, 3598,
    3599, 3854, 3855,
];

const UTAB: [i32; 256] = [
    0, 1, 4, 5, 16, 17, 20, 21, 64, 65, 68, 69, 80, 81, 84, 85, 256, 257, 260, 261, 272, 273, 276,
    277, 320, 321, 324, 325, 336, 337, 340, 341, 1024, 1025, 1028, 1029, 1040, 1041, 1044, 1045,
    1088, 1089, 1092, 1093, 1104, 1105, 1108, 1109, 1280, 1281, 1284, 1285, 1296, 1297, 1300, 1301,
    1344, 1345, 1348, 1349, 1360, 1361, 1364, 1365, 4096, 4097, 4100, 4101, 4112, 4113, 4116, 4117,
    4160, 4161, 4164, 4165, 4176, 4177, 4180, 4181, 4352, 4353, 4356, 4357, 4368, 4369, 4372, 4373,
    4416, 4417, 4420, 4421, 4432, 4433, 4436, 4437, 5120, 5121, 5124, 5125, 5136, 5137, 5140, 5141,
    5184, 5185, 5188, 5189, 5200, 5201, 5204, 5205, 5376, 5377, 5380, 5381, 5392, 5393, 5396, 5397,
    5440, 5441, 5444, 5445, 5456, 5457, 5460, 5461, 16384, 16385, 16388, 16389, 16400, 16401,
    16404, 16405, 16448, 16449, 16452, 16453, 16464, 16465, 16468, 16469, 16640, 16641, 16644,
    16645, 16656, 16657, 16660, 16661, 16704, 16705, 16708, 16709, 16720, 16721, 16724, 16725,
    17408, 17409, 17412, 17413, 17424, 17425, 17428, 17429, 17472, 17473, 17476, 17477, 17488,
    17489, 17492, 17493, 17664, 17665, 17668, 17669, 17680, 17681, 17684, 17685, 17728, 17729,
    17732, 17733, 17744, 17745, 17748, 17749, 20480, 20481, 20484, 20485, 20496, 20497, 20500,
    20501, 20544, 20545, 20548, 20549, 20560, 20561, 20564, 20565, 20736, 20737, 20740, 20741,
    20752, 20753, 20756, 20757, 20800, 20801, 20804, 20805, 20816, 20817, 20820, 20821, 21504,
    21505, 21508, 21509, 21520, 21521, 21524, 21525, 21568, 21569, 21572, 21573, 21584, 21585,
    21588, 21589, 21760, 21761, 21764, 21765, 21776, 21777, 21780, 21781, 21824, 21825, 21828,
    21829, 21840, 21841, 21844, 21845,
];

const JRLL: [i32; 12] = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
const JPLL: [i32; 12] = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7];

/// get the total number of pixels corresponding to the nside parameter
pub fn nside2npix32(nside: i32) -> i32 {
    12 * nside * nside
}

/// calculate the nside parameter from the total number of pixels
/// if an invalid npix value is given, the program will panic
pub fn npix2nside32(npix: i32) -> i32 {
    let res = isqrt(npix / 12);
    if nside2npix32(res) == npix {
        res
    } else {
        panic!()
    }
}

fn nest2xyf32(nside: i32, mut pix: i32) -> (i32, i32, i32) {
    let npface = nside * nside;
    let face_num = pix / npface;
    pix &= npface - 1;
    let mut raw = (pix & 0x5555) | ((pix & 0x5555_0000) >> 15);
    let ix = CTAB[(raw & 0xff) as usize] | (CTAB[(raw >> 8) as usize] << 4);
    pix >>= 1;
    raw = (pix & 0x5555) | ((pix & 0x5555_0000) >> 15);
    let iy = CTAB[(raw & 0xff) as usize] | (CTAB[(raw >> 8) as usize] << 4);
    (ix, iy, face_num)
}

fn xyf2nest32(nside: i32, ix: i32, iy: i32, face_num: i32) -> i32 {
    face_num * nside * nside
        + (UTAB[(ix & 0xff) as usize]
            | (UTAB[(ix >> 8) as usize] << 16)
            | (UTAB[(iy & 0xff) as usize] << 1)
            | (UTAB[(iy >> 8) as usize] << 17))
}

fn xyf2ring32(nside: i32, ix: i32, iy: i32, face_num: i32) -> i32 {
    let nl4 = 4 * nside;
    let jr = JRLL[face_num as usize] * nside - ix - iy - 1;

    let (nr, n_before, kshift) = if jr < nside {
        let nr = jr;
        (nr, 2 * nr * (nr - 1), 0)
    } else if jr > 3 * nside {
        let nr = nl4 - jr;
        (nr, 12 * nside * nside - 2 * (nr + 1) * nr, 0)
    } else {
        let nr = nside;
        let ncap = 2 * nside * (nside - 1);
        (nr, ncap + (jr - nside) * nl4, (jr - nside) & 1)
    };
    let mut jp = (JPLL[face_num as usize] * nr + ix - iy + 1 + kshift) / 2;
    if jp > nl4 {
        jp -= nl4;
    } else if jp < 1 {
        jp += nl4;
    }
    n_before + jp - 1
}

fn special_div32(mut a: i32, b: i32) -> i32 {
    let t = if a >= (b << 1) { 1 } else { 0 };
    a -= t * (b << 1);
    (t << 1) + if a >= b { 1 } else { 0 }
}

fn ring2xyf32(nside: i32, pix: i32) -> (i32, i32, i32) {
    let ncap = 2 * nside * (nside - 1);
    let npix = 12 * nside * nside;
    let nl2 = 2 * nside;

    let (iring, iphi, kshift, nr, face_num) = if pix < ncap {
        let iring = (1 + isqrt(1 + 2 * pix)) >> 1;
        let iphi = (pix + 1) - 2 * iring * (iring - 1);
        let kshift = 0;
        let nr = iring;
        let face_num = special_div32(iphi - 1, nr);
        (iring, iphi, kshift, nr, face_num)
    } else if pix < (npix - ncap) {
        let ip = pix - ncap;
        let iring = ip / (4 * nside) + nside;
        let iphi = ip % (4 * nside) + 1;
        let kshift = (iring + nside) & 1;
        let nr = nside;
        let ire = iring - nside + 1;
        let irm = nl2 + 2 - ire;
        let ifm = (iphi - ire / 2 + nside - 1) / nside;
        let ifp = (iphi - irm / 2 + nside - 1) / nside;
        let face_num = match ifp {
            _ if ifp == ifm => ifp | 4,
            _ if ifp < ifm => ifp,
            _ => ifm + 8,
        };
        /*
        if ifp == ifm {
            ifp | 4
        } else if ifp < ifm {
            ifp
        } else {
            ifm + 8
        };*/
        (iring, iphi, kshift, nr, face_num)
    } else {
        let ip = npix - pix;
        let iring = (1 + isqrt(2 * ip - 1)) >> 1;
        let iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
        let kshift = 0;
        let nr = iring;
        let iring = 2 * nl2 - iring;
        let face_num = 8 + special_div32(iphi - 1, nr);
        (iring, iphi, kshift, nr, face_num)
    };

    let irt = iring - (JRLL[face_num as usize] * nside) + 1;
    let mut ipt = 2 * iphi - JPLL[face_num as usize] * nr - kshift - 1;
    if ipt >= nl2 {
        ipt -= 8 * nside;
    }

    let ix = (ipt - irt) >> 1;
    let iy = (-(ipt + irt)) >> 1;
    (ix, iy, face_num)
}

/// get the index of a pixel in nest order corresponding to ring order
pub fn ring2nest32(nside: i32, ipring: i32) -> i32 {
    if nside & (nside - 1) != 0 {
        panic!();
    } else {
        let (ix, iy, face_num) = ring2xyf32(nside, ipring);
        xyf2nest32(nside, ix, iy, face_num)
    }
}

/// get the index of a pixel in ring order corresponding to nest order
pub fn nest2ring32(nside: i32, ipnest: i32) -> i32 {
    if nside & (nside - 1) != 0 {
        panic!();
    } else {
        let (ix, iy, face_num) = nest2xyf32(nside, ipnest);
        xyf2ring32(nside, ix, iy, face_num)
    }
}

/// get the total number of pixels corresponding to the nside parameter
pub fn nside2npix64(nside: i64) -> i64 {
    12 * nside * nside
}

/// if an invalid npix value is given, the program will panic
pub fn npix2nside64(npix: i64) -> i64 {
    let res = isqrt(npix / 12);
    if nside2npix64(res) == npix {
        res
    } else {
        panic!()
    }
}

fn special_div64(mut a: i64, b: i64) -> i64 {
    let t: i64 = if a >= (b << 1) { 1 } else { 0 };
    a -= t * (b << 1);
    (t << 1) + if a >= b { 1 } else { 0 }
}

fn compress_bits64(v: i64) -> i64 {
    let mut raw = v & 0x5555_5555_5555_5555_i64;
    raw |= raw >> 15;
    <i64 as std::convert::From<_>>::from(CTAB[(raw & 0xff_i64) as usize])
        | <i64 as std::convert::From<_>>::from(CTAB[((raw >> 8) & 0xff) as usize] << 4)
        | <i64 as std::convert::From<_>>::from(CTAB[((raw >> 32) & 0xff) as usize] << 16)
        | <i64 as std::convert::From<_>>::from(CTAB[((raw >> 40) & 0xff) as usize] << 20)
}

fn spread_bits64(v: i32) -> i64 {
    <i64 as std::convert::From<_>>::from(UTAB[(v & 0xff) as usize])
        | (<i64 as std::convert::From<_>>::from(UTAB[((v >> 8) & 0xff) as usize]) << 16)
        | (<i64 as std::convert::From<_>>::from(UTAB[((v >> 16) & 0xff) as usize]) << 32)
        | (<i64 as std::convert::From<_>>::from(UTAB[((v >> 24) & 0xff) as usize]) << 48)
}

fn nest2xyf64(nside: i64, mut pix: i64) -> (i32, i32, i32) {
    let npface: i64 = nside as i64 * nside as i64;
    let face_num = (pix / npface) as i32;
    pix &= npface - 1;
    let ix = compress_bits64(pix) as i32;
    let iy = compress_bits64(pix >> 1) as i32;
    (ix, iy, face_num)
}

fn xyf2nest64(nside: i64, ix: i32, iy: i32, face_num: i32) -> i64 {
    <i64 as std::convert::From<_>>::from(face_num) * nside * nside
        + spread_bits64(ix)
        + (spread_bits64(iy) << 1)
}

fn xyf2ring64(nside: i64, ix: i32, iy: i32, face_num: i32) -> i64 {
    let nl4 = 4_i64 * nside;
    let jr = <i64 as std::convert::From<_>>::from(JRLL[face_num as usize]) * nside
        - <i64 as std::convert::From<_>>::from(ix)
        - <i64 as std::convert::From<_>>::from(iy)
        - 1_i64;

    let (nr, n_before, kshift) = if jr < nside {
        let nr = jr;
        (nr, 2_i64 * nr * (nr - 1), 0)
    } else if jr > 3_i64 * nside {
        let nr = nl4 - jr;
        (nr, 12 * nside * nside - 2 * (nr + 1) * nr, 0)
    } else {
        let nr = nside;
        let ncap = 2 * nside * (nside - 1);
        (nr, ncap + (jr - nside) * nl4, (jr - nside) & 1)
    };
    let mut jp = (<i64 as std::convert::From<_>>::from(JPLL[face_num as usize]) * nr
        + <i64 as std::convert::From<_>>::from(ix)
        - <i64 as std::convert::From<_>>::from(iy)
        + 1_i64
        + kshift)
        / 2;
    if jp > nl4 {
        jp -= nl4;
    } else if jp < 1 {
        jp += nl4;
    }
    n_before + jp - 1
}

fn ring2xyf64(nside: i64, pix: i64) -> (i32, i32, i32) {
    let ncap = 2 * nside * (nside - 1);
    let npix = 12 * nside * nside;
    let nl2 = 2 * nside;

    let (iring, iphi, kshift, nr, face_num) = if pix < ncap {
        let iring = (1 + isqrt(1 + 2 * pix)) >> 1;
        let iphi = (pix + 1) - 2 * iring * (iring - 1);
        let kshift = 0;
        let nr = iring;
        let face_num = special_div64(iphi - 1, nr);
        (iring, iphi, kshift, nr, face_num)
    } else if pix < (npix - ncap) {
        let ip = pix - ncap;
        let iring = ip / (4 * nside) + nside;
        let iphi = ip % (4 * nside) + 1;
        let kshift = (iring + nside) & 1;
        let nr = nside;
        let ire = iring - nside + 1;
        let irm = nl2 + 2 - ire;
        let ifm = (iphi - ire / 2 + nside - 1) / nside;
        let ifp = (iphi - irm / 2 + nside - 1) / nside;
        let face_num = match ifp {
            _ if ifp == ifm => ifp | 4,
            _ if ifp < ifm => ifp,
            _ => ifm + 8,
        };
        /*
        if ifp == ifm {
            ifp | 4
        } else if ifp < ifm {
            ifp
        } else {
            ifm + 8
        };*/
        (iring, iphi, kshift, nr, face_num)
    } else {
        let ip = npix - pix;
        let iring = (1 + isqrt(2 * ip - 1)) >> 1;
        let iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
        let kshift = 0;
        let nr = iring;
        let iring = 2 * nl2 - iring;
        let face_num = 8 + special_div64(iphi - 1, nr);
        (iring, iphi, kshift, nr, face_num)
    };

    let irt = iring - (<i64 as std::convert::From<_>>::from(JRLL[face_num as usize]) * nside) + 1;
    let mut ipt =
        2 * iphi - <i64 as std::convert::From<_>>::from(JPLL[face_num as usize]) * nr - kshift - 1;
    if ipt >= nl2 {
        ipt -= 8 * nside;
    }

    let ix = (ipt - irt) >> 1;
    let iy = (-(ipt + irt)) >> 1;
    (ix as i32, iy as i32, face_num as i32)
}

/// get the index of nest order pix corresponding to ring order
pub fn ring2nest64(nside: i64, ipring: i64) -> i64 {
    if nside & (nside - 1) != 0 {
        panic!()
    } else {
        let (ix, iy, face_num) = ring2xyf64(nside, ipring);
        xyf2nest64(nside, ix, iy, face_num)
    }
}

/// get the index of ring order pix corresponding to nest order
pub fn nest2ring64(nside: i64, ipnest: i64) -> i64 {
    if nside & (nside - 1) != 0 {
        panic!()
    } else {
        let (ix, iy, face_num) = nest2xyf64(nside, ipnest);
        xyf2ring64(nside, ix, iy, face_num)
    }
}

/// get the index of nest order pix corresponding to ring order
pub fn ring2nest(nside: usize, ipring: usize) -> usize {
    if 12 * nside * nside > i32::max_value() as usize {
        ring2nest64(nside as i64, ipring as i64) as usize
    } else {
        ring2nest32(nside as i32, ipring as i32) as usize
    }
}

/// get the index of ring order pix corresponding to nest order
pub fn nest2ring(nside: usize, ipnest: usize) -> usize {
    if 12 * nside * nside > i32::max_value() as usize {
        nest2ring64(nside as i64, ipnest as i64) as usize
    } else {
        nest2ring32(nside as i32, ipnest as i32) as usize
    }
}

/// get the number of pixels from the nside parameter
pub fn nside2npix(nside: usize) -> usize {
    12 * nside * nside
}

/// get the nside parameter from the total number of pixels, it will panic if an invalid
/// value is given
pub fn npix2nside(npix: usize) -> usize {
    let res = isqrt(npix / 12);
    if nside2npix(res) == npix {
        res
    } else {
        panic!()
    }
}

pub fn nside2nring(nside: usize) -> usize {
    4 * nside - 1
}
