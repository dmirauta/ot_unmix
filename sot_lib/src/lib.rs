#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

mod dists_ffi {
    include!(concat!(env!("OUT_DIR"), "/sot_bindings.rs"));

    impl Signature_t {
        pub fn new(x: &[f64], w: &[f64]) -> Self {
            Self {
                x: x.as_ptr(),
                w: w.as_ptr(),
                n: x.len() as i32,
            }
        }
    }
}

use dists_ffi::{Signature_t, TraceData_t};

struct TraceDataOwned {
    transfer: Vec<f64>,
    path_a: Vec<i32>,
    path_b: Vec<i32>,
}

impl TraceDataOwned {
    fn new(n_trace: usize) -> Self {
        Self {
            transfer: vec![0.0f64; n_trace],
            path_a: vec![0i32; n_trace],
            path_b: vec![0i32; n_trace],
        }
    }
    fn get_td(&mut self) -> TraceData_t {
        TraceData_t {
            transfer: self.transfer.as_mut_ptr(),
            path_a: self.path_a.as_mut_ptr(),
            path_b: self.path_b.as_mut_ptr(),
        }
    }
}

pub fn W22_f64<S: AsRef<[f64]>>(xa: S, wa: S, xb: S, wb: S) -> f64 {
    let xa = xa.as_ref();
    let wa = wa.as_ref();
    let xb = xb.as_ref();
    let wb = wb.as_ref();
    let mut tdo = TraceDataOwned::new(xa.len() + xb.len() - 1);
    unsafe {
        dists_ffi::W22(
            Signature_t::new(xa, wa),
            Signature_t::new(xb, wb),
            tdo.get_td(),
        )
    }
}

pub fn left_bary_f64<S, Sm>(xa: S, wa: S, xb: S, wb: S, mut xba: Sm, bw: f64)
where
    S: AsRef<[f64]>,
    Sm: AsMut<[f64]>,
{
    let xa = xa.as_ref();
    let wa = wa.as_ref();
    let xb = xb.as_ref();
    let wb = wb.as_ref();
    let xba = xba.as_mut();
    let mut tdo = TraceDataOwned::new(xa.len() + xb.len() - 1);
    unsafe {
        dists_ffi::left_bary(
            Signature_t::new(xa, wa),
            Signature_t::new(xb, wb),
            xba.as_mut_ptr(),
            bw,
            tdo.get_td(),
            1e-31,
        )
    }
}

#[test]
fn W22_basic_sanity_check() {
    let xa = vec![0.2, 0.5, 0.8];
    let wa = vec![0.4, 0.2, 0.4];
    let xb = vec![0.3, 0.6];
    let wb = vec![0.3, 0.7];

    let d = W22_f64(xa, wa, xb, wb);
    dbg!(d);
}
