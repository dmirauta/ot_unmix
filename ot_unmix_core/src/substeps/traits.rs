use ndarray::{Array1, Array2, Array3};

use crate::unmixing::F;

pub trait ComputesBarycenter: Send {
    fn compute(&mut self, variations: Vec<(Array1<F>, Array1<F>, F)>) -> Result<(), String>;
    fn average_positions(&self) -> Array1<F>;
    fn average_weights(&self) -> Array1<F>;
}

pub trait ComputesSeparation: Send {
    fn compute(
        &mut self,
        xc: &Array2<F>,
        wc: &Array2<F>,
        xb: &Array2<F>,
        wb: &Array2<F>,
        wbt: &mut Array3<F>,
        w2: &mut Array1<F>,
    ) -> Result<(), String>;
}
