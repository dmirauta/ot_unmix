#![allow(non_snake_case)]
use pyo3::prelude::*;

#[pymodule]
mod sot_rs {
    use super::*;

    // TODO: makes copies? Take bound values instead? numpy arrays will also be cast to list?
    #[pyfunction]
    fn W22_f64(sa: (Vec<f64>, Vec<f64>), sb: (Vec<f64>, Vec<f64>)) -> f64 {
        let (xa, wa) = sa;
        let (xb, wb) = sb;
        sot_lib::W22_f64(xa, wa, xb, wb)
    }
}
