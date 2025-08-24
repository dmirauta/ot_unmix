#![allow(non_snake_case)]
use pyo3::prelude::*;

#[pymodule]
mod sot_rs {
    use super::*;

    // TODO: Declaring lists as Vec<_> makes copies? Take bound values instead?
    /// Expects each signature to be a (positions, weights) tuple, where positions and weights are
    /// lists/arrays of floats.
    #[pyfunction]
    fn W22_f64(signature_a: (Vec<f64>, Vec<f64>), signature_b: (Vec<f64>, Vec<f64>)) -> f64 {
        let (xa, wa) = signature_a;
        let (xb, wb) = signature_b;
        sot_lib::W22_f64(xa, wa, xb, wb)
    }
}
