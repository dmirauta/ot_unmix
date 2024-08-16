//! OpenCL implementation of OT based unmixing

use std::{
    collections::HashMap,
    fs::{self, File},
    io::BufReader,
    ops::Mul,
    path::Path,
    time::{Duration, SystemTime},
};

use log::{error, info};
use ndarray::{array, s, Array1, Array2, Array3, ArrayD, Axis};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use vectoriselp::{
    lp_wrap::{LPOutput, MiniLPSolver},
    ArrayExpr, LinearExpression, LinearProgram,
};

use crate::substeps::traits::{ComputesBarycenter, ComputesSeparation};

use super::data::Signature;

// have opted for relatively flat signature representations since thats how they are passed to OpenCL

// NOTE: Largely unused placeholder option/partial implementation, as we need only really change
// types for OpenCL methods and cast to host precission there.
// #[cfg(fp32)]
// /// floating point precission of crate/host
// pub type F = f32;
//
// #[cfg(not(fp32))]
/// floating point precission of crate/host
pub type F = f64;

pub static EPSILON: F = 1e-9;

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct IterTimings {
    pub avg: Duration,
    pub seg: Duration,
}

// TODO: cost after averaging?
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Iteration {
    /// basis positions in (basis, position) indices
    pub xb: Array2<F>,
    /// basis weights in (basis, position) indices
    pub wb: Array2<F>,
    /// varied basis weights in (mix, basis, position) indices
    // can only have contiguous access for avg or seg,
    // since avg requires sorting anyway, may as well be seg
    pub wbt: Array3<F>,
    pub segmentation_costs: Array1<F>,
    pub timings: IterTimings,
}

impl Iteration {
    #[allow(dead_code)]
    pub fn abundances(&self, mix: usize) -> Array1<F> {
        self.wbt.slice(s![mix, .., ..]).sum_axis(Axis(1))
    }
    pub fn abundances_all(&self) -> Array2<F> {
        self.wbt.sum_axis(Axis(2))
    }
    pub fn get_basis_sig(&self, i: usize) -> Signature {
        (
            self.xb.slice(s![i, ..]).as_slice().unwrap(),
            self.wb.slice(s![i, ..]).as_slice().unwrap(),
        )
            .into()
    }
    pub fn get_comp_sig(&self, i: usize, j: usize, positions: &[F]) -> Signature {
        (positions, self.wbt.slice(s![j, i, ..]).as_slice().unwrap()).into()
    }
    #[allow(dead_code)]
    fn weights_ok(&self) -> bool {
        if let Some(v) = self.wb.iter().find(|v| **v < 0.0) {
            println!("{v} in wb");
            false
        } else if let Some(v) = self.wbt.iter().find(|v| **v < 0.0) {
            println!("{v} in wbt");
            false
        } else {
            true
        }
    }
}

pub fn bson_write(obj: &impl Serialize, path: impl AsRef<Path>) {
    let bytes = bson::to_vec(obj).unwrap();
    fs::write(path, bytes).unwrap();
}

pub fn bson_read<T>(path: impl AsRef<Path>) -> T
where
    T: DeserializeOwned,
{
    bson::from_reader(BufReader::new(File::open(path).unwrap())).unwrap()
}

// TODO: Serialise substep computation details
#[derive(Serialize, Deserialize)]
pub struct OTDecomp {
    pub iterations: Vec<Iteration>,
    /// combined data positions in (mix id, position idx)
    pub xc: Array2<F>,
    /// combined data weights in (mix id, position idx)
    pub wc: Array2<F>,
    /// bases which are not updated
    pub fixed_bases: Vec<bool>,
    pub n_padding: usize,
    pub n_data: usize,
    pub n_bases: usize,
    #[serde(skip)]
    pub avg_compute: Option<Box<dyn ComputesBarycenter>>,
    #[serde(skip)]
    pub sep_compute: Option<Box<dyn ComputesSeparation>>,
}

pub enum InitialBasis {
    Provided {
        pattern: Signature,
        keep_fixed: bool,
    },
    NoPrior,
}

pub type AnomLPOut = (HashMap<String, ArrayD<F>>, F);

impl OTDecomp {
    pub fn new(
        mixed_data: Vec<Signature>,
        init_basis: Vec<InitialBasis>,
        padding: usize,
        avg_compute: Box<dyn ComputesBarycenter>,
        sep_compute: Box<dyn ComputesSeparation>,
    ) -> Self {
        let n_data = mixed_data.len();
        let n_bases = init_basis.len();
        let n_padding = padding;

        let mut xb = Array2::<F>::zeros((n_bases, padding));
        let mut wb = Array2::<F>::zeros((n_bases, padding));
        // TODO: should use simplex uniform abundance vectors?
        let mut wbt = Array3::<F>::random((n_data, n_bases, padding), Uniform::new(0.0, 1.0));
        let mut xc = Array2::<F>::zeros((n_data, padding));
        let mut wc = Array2::<F>::zeros((n_data, padding));
        // TODO: Have used both segmentation/separation nomenclature (previously named fields will
        // be present in existing serialised data)
        let segmentation_costs = Array1::<F>::zeros(n_data);

        // mixtures to decompose
        for (j, sig) in mixed_data.iter().enumerate() {
            for (k, stick) in sig.sticks.iter().take(padding).enumerate() {
                xc[(j, k)] = stick.position;
                wc[(j, k)] = stick.weight;
            }
        }

        // initial components
        for j in 0..n_data {
            for k in 0..padding {
                let mut t = wbt.slice_mut(s![j, .., k]);
                t *= wc[(j, k)] / t.sum();
            }
        }

        //// Begin initialising bases

        let mut fixed_bases = vec![];
        // do not update non-fixed initial data on initialisation
        let mut initially_fixed_bases = vec![];

        for (i, ib) in init_basis.into_iter().enumerate() {
            match ib {
                InitialBasis::Provided {
                    pattern,
                    keep_fixed,
                } => {
                    for (k, stick) in pattern.sticks.iter().take(padding).enumerate() {
                        xb[(i, k)] = stick.position;
                        wb[(i, k)] = stick.weight;
                    }
                    fixed_bases.push(keep_fixed);
                    initially_fixed_bases.push(true);
                }
                InitialBasis::NoPrior => {
                    fixed_bases.push(false);
                    initially_fixed_bases.push(false);
                }
            }
        }

        let mut new = OTDecomp {
            iterations: vec![],
            xc,
            wc,
            fixed_bases: initially_fixed_bases,
            n_padding,
            n_data,
            n_bases,
            avg_compute: Some(avg_compute),
            sep_compute: Some(sep_compute),
        };

        let mut initial_iter = Iteration {
            xb,
            wb,
            wbt,
            segmentation_costs,
            ..Default::default()
        };
        // TODO: Defer basis init, so we do not require compute methods to be registered
        // immediately?
        new.basis_from_comps(&mut initial_iter)
            .expect("failed to initalise basis");
        //// End initialising bases
        new.iterations.push(initial_iter);
        new.fixed_bases = fixed_bases;
        new
    }

    fn basis_from_comps(&mut self, iter: &mut Iteration) -> Result<(), String> {
        let avg_compute = self
            .avg_compute
            .as_mut()
            .ok_or("No barycenter compute method set.".to_string())?;

        let t0 = SystemTime::now();
        for (i, keep_fixed) in self.fixed_bases.iter().enumerate() {
            if !keep_fixed {
                // update basis as average of corresponding variations

                // TODO: more economical to pass through iterator?
                let variations_i = (0..self.n_data)
                    .filter_map(|j| {
                        let wbtij = iter.wbt.slice(s![j, i, ..]).to_owned();
                        let a = wbtij.sum();
                        match a > EPSILON {
                            true => Some((self.xc.slice(s![j, ..]).to_owned(), wbtij, a)),
                            false => None,
                        }
                    })
                    .collect();

                avg_compute.compute(variations_i)?;

                // Collect average
                iter.xb
                    .slice_mut(s![i, ..])
                    .assign(&avg_compute.average_positions());
                iter.wb
                    .slice_mut(s![i, ..])
                    .assign(&avg_compute.average_weights());

                // Ensure normalisation
                let mut wbi = iter.wb.slice_mut(s![i, ..]);
                let s = wbi.sum();
                wbi /= s + EPSILON;
            }
        }
        let t1 = SystemTime::now();
        iter.timings.seg = t1.duration_since(t0).unwrap();

        Ok(())
    }

    fn comps_from_basis(&mut self, iter: &mut Iteration) -> Result<(), String> {
        let sep_compute = self
            .sep_compute
            .as_mut()
            .ok_or("No separation compute method set.".to_string())?;

        let t0 = SystemTime::now();
        sep_compute.compute(
            &self.xc,
            &self.wc,
            &iter.xb,
            &iter.wb,
            &mut iter.wbt,
            &mut iter.segmentation_costs,
        )?;

        // Ensure normalisation
        for j in 0..self.n_data {
            let mut wcj = iter.wbt.slice_mut(s![j, .., ..]);
            let s = wcj.sum();
            wcj /= s + EPSILON;
        }
        let t1 = SystemTime::now();
        iter.timings.avg = t1.duration_since(t0).unwrap();

        Ok(())
    }

    pub fn iteration(&mut self) -> Result<Iteration, String> {
        let mut new_iter = self.iterations.last().unwrap().clone();
        self.comps_from_basis(&mut new_iter)?;
        if !self.separation_only() {
            self.basis_from_comps(&mut new_iter)?;
        } else {
            info!(
                "Basis is constant, multiple iterations not required, only performing separation."
            );
        }
        Ok(new_iter)
    }

    #[inline]
    pub fn separation_only(&self) -> bool {
        !self.fixed_bases.contains(&false)
    }

    pub fn iterate(&mut self, times: usize) {
        for _ in 0..times {
            match self.iteration() {
                Ok(new_iter) => self.iterations.push(new_iter),
                Err(err) => {
                    error!("Error encountered in iteration, stopping. {err:?}");
                    break;
                }
            }
            if self.separation_only() {
                break;
            }
        }
    }

    pub fn anom_lp(
        &self,
        iter_idx: usize,
        j: usize,
        max_avg_displacement: F,
        anom_penalty: F,
    ) -> Result<AnomLPOut, String> {
        // TODO: pre-assign and reuse some containers?
        // TODO: check if variable pruning is required?
        let iter = &self.iterations[iter_idx];

        let mut lp = LinearProgram::<F>::new();
        let mut cost_distances = LinearExpression::<F>::default();
        let mut mix_marginal = ArrayExpr::<F>::new(&[self.n_padding]);

        for i in 0..self.n_bases {
            let basis_key = format!("basis_{i}_plan");

            lp.add_var(basis_key.clone(), &[self.n_padding, self.n_padding]);

            let mut sq_distance_mat = Array2::<F>::zeros((self.n_padding, self.n_padding));

            for p in 0..self.n_padding {
                for q in 0..self.n_padding {
                    sq_distance_mat[(p, q)] = (iter.xb[(i, p)] - self.xc[(j, q)]).powi(2);
                }
            }

            // TODO: masked expression for max displacement limit?
            let plan_expr = lp[&basis_key].into_arr_expr();

            let wbi_neg = iter.wb.slice(s![i, ..]).clone().mul(-1.0).into_dyn();
            let abundance_broadcasted = plan_expr.sum().repeat(&[self.n_padding]);
            let basis_marginal_expr =
                plan_expr.sum_axis(1) + (wbi_neg, &abundance_broadcasted).into();

            // without tolerance
            // lp.add_eq_0_constraints(basis_marginal_expr);

            // allow small tolerance for constraint violation
            lp.add_leq_0_constraints(basis_marginal_expr.scaled(-1.0));
            let cs = EPSILON * Array1::<F>::ones(self.n_padding).into_dyn();
            lp.add_leq_constraints(basis_marginal_expr, cs);

            let prod: ArrayExpr<F> = (sq_distance_mat.into_dyn(), &lp[basis_key]).into();

            // (per basis) average (squared) displacement limit
            let singleton = array![max_avg_displacement * max_avg_displacement].into_dyn();
            lp.add_leq_constraints(prod.sum_axis(0).sum_axis(0), singleton);

            mix_marginal += plan_expr.sum_axis(0);

            cost_distances += prod.sum();
        }

        lp.add_var("anomaly", &[self.n_padding]);
        let cost_full =
            cost_distances.clone() + lp["anomaly"].into_arr_expr().sum().scaled(anom_penalty);
        lp.set_cost(cost_full.clone());

        mix_marginal += lp["anomaly"].into_arr_expr();
        let wc = self.wc.slice(s![j, ..]).to_owned().into_dyn();
        lp.add_eq_constraints(mix_marginal, wc);

        lp.solve::<MiniLPSolver>().map(|LPOutput { sol, .. }| {
            let mut lp_vars = HashMap::new();
            for i in 0..self.n_bases {
                let basis_key = format!("basis_{i}_plan");
                lp_vars.insert(basis_key.clone(), lp[basis_key].shaped_sol(&sol));
            }
            // lp_vars.insert("anomaly".to_string(), lp["anomaly"].shaped_sol(&sol_vec));

            (lp_vars, cost_distances.eval(&sol))
        })
    }

    pub fn unpack_lp_vars(&mut self, iter_idx: usize, j: usize, lp_out: &AnomLPOut) {
        let (lp_vars, seg_cost) = lp_out;
        let iter = &mut self.iterations[iter_idx];
        for i in 0..self.n_bases {
            let basis_key = format!("basis_{i}_plan");
            let plan = &lp_vars[&basis_key];
            iter.wbt
                .slice_mut(s![j, i, ..])
                .assign(&plan.sum_axis(Axis(0)));
        }
        iter.segmentation_costs[j] = *seg_cost;
    }

    #[allow(dead_code)]
    pub fn calc_anom(
        &mut self,
        iter_idx: usize,
        j: usize,
        max_avg_displacement: F,
        anom_penalty: F,
    ) {
        match &self.anom_lp(iter_idx, j, max_avg_displacement, anom_penalty) {
            Ok(lp_vars) => self.unpack_lp_vars(iter_idx, j, lp_vars),
            Err(err) => {
                error!("Calc anomaly failed with: {err:?}");
            }
        }
    }

    pub fn get_anomaly(&self, iter_id: usize, mix_id: usize, tol: F) -> Option<Array1<F>> {
        let mut aw = self.wc.slice(s![mix_id, ..]).to_owned();
        let iter = &self.iterations[iter_id];
        for i in 0..self.n_bases {
            let t = iter.wbt.slice(s![mix_id, i, ..]).to_owned();
            aw -= &t;
        }
        aw.mapv_inplace(|v| if v > 0.0 { v } else { 0.0 });
        match aw.sum() > tol {
            true => Some(aw),
            false => None,
        }
    }

    pub fn substep_methods_are_set(&self) -> bool {
        self.avg_compute.is_some() && self.sep_compute.is_some()
    }
}

/// Serialize [OTDecomp] with additional context
#[derive(Serialize, Deserialize)]
pub struct AnnotatedOTDecomp {
    pub decomp: OTDecomp,
    pub basis_name_map: Vec<String>,
    pub data_name_map: Vec<String>,
}
