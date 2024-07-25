use std::any::type_name;

use crate::{
    substeps::traits::{ComputesBarycenter, ComputesSeparation},
    unmixing::{EPSILON, F},
};
use log::error;
use ndarray::{s, Array1, Array2, Array3};
use ocl::{OclPrm, ProQue};
use serde::{Deserialize, Serialize};
use simple_ocl::{
    prog_que_from_source, selected_device_supports_doubles, DeviceToFrom, PairedBuffers,
    PairedBuffers1, PairedBuffers2,
};

/// The `KF` or Kernel Float parameter attempts to decouple
/// the precision of the kernels run and of the crate/host [F]
#[derive(DeviceToFrom)]
pub struct OCLBarycenterCompute<KF: OclPrm> {
    que: ProQue,
    /// Variants signature positions
    xv: PairedBuffers2<KF>,
    /// Variants signature weights
    #[dev_to_from(from = false)]
    wv: PairedBuffers2<KF>,
    /// Variant (barycenter) weighting
    #[dev_to_from(from = false)]
    w: PairedBuffers1<KF>,
    n_data: usize,
}

static BARY_SOURCE: &str = concat!(
    include_str!("../../../c_opencl_src/shared_defines.h"),
    include_str!("../../../c_opencl_src/dists.c"),
    include_str!("../../../c_opencl_src/lb.c"),
    include_str!("../../../c_opencl_src/barycenter_kernel.cl")
);

fn handle_float_define<KF>(compiler_defines: &mut Vec<String>) {
    let float_type = type_name::<KF>();
    if float_type == "f32" {
        compiler_defines.push("-DUSE_FLOAT".into());
    } else if float_type == "f64"
        && !selected_device_supports_doubles().expect("OCL device should be set here")
    {
        // TODO: When running from GUI, this message will not be visible (in visual log) after panic
        error!("The selected device does not support doubles, but the selected kernel expects double precision buffers.");
    }
}

macro_rules! zero_check {
    ($size: ident) => {
        if $size == 0 {
            println!("{} is zero in substep!!!", stringify!($size));
        }
    };
}

impl<KF> OCLBarycenterCompute<KF>
where
    KF: OclPrm + ndarray_rand::rand_distr::num_traits::Float,
{
    pub fn new(n_data: usize, padding: usize) -> Self {
        zero_check!(n_data);
        zero_check!(padding);

        let mut compiler_defines = vec![
            "-DEXTERNAL_CONCAT".to_string(),
            // format!("-I{OCL_PATH}"),
            format!("-DEPSILON={EPSILON:.3e}"),
            format!("-DN={padding}"),
        ];
        handle_float_define::<KF>(&mut compiler_defines);
        let mut que = prog_que_from_source(
            BARY_SOURCE,
            "barycenter kernel".to_string(),
            compiler_defines,
        );

        let xv_c = Array2::<KF>::zeros((n_data, padding));
        let wv_c = Array2::<KF>::zeros((n_data, padding));
        let w_c = Array1::<KF>::zeros(n_data);

        let xv = PairedBuffers::create_from(xv_c, &mut que);
        let wv = PairedBuffers::create_from(wv_c, &mut que);
        let w = PairedBuffers::create_from(w_c, &mut que);

        Self {
            que,
            xv,
            wv,
            w,
            n_data,
        }
    }

    fn single_reduction_pass(&mut self, n_pairs: usize, skip: usize) -> ocl::Result<()> {
        self.que.set_dims(n_pairs);
        let kernel = self
            .que
            .kernel_builder("multibary_pass")
            .arg(&self.xv.device)
            .arg(&self.wv.device)
            .arg(&self.w.device)
            .arg(skip as i32)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        Ok(())
    }

    pub fn full_reduction(
        &mut self,
        mut variations: Vec<(Array1<KF>, Array1<KF>, KF)>,
    ) -> ocl::Result<()> {
        let n_variations = variations.len();
        assert!(n_variations <= self.n_data);

        variations.sort_by(|(_, _, a1), (_, _, a2)| a2.partial_cmp(a1).expect("Nan weight?"));

        for (j, (positions, weights, total_weight)) in variations.into_iter().enumerate() {
            let mut temp = weights.to_owned();
            temp.mapv_inplace(|v| v / total_weight); // TODO: + EPSILON? Filter out small?
            self.xv.host.slice_mut(s![j, ..]).assign(&positions);
            self.wv.host.slice_mut(s![j, ..]).assign(&temp);
            self.w.host[j] = total_weight;
        }

        self.send_pairedbuffs()?;

        //// Begin average
        let mut m = n_variations;
        let mut skip = 1usize;

        while m > 1 {
            let n_pairs = m / 2;
            self.single_reduction_pass(n_pairs, skip)?;
            m = n_pairs + (m % 2);
            skip *= 2;
        }
        //// End average

        self.retrieve_pairedbuffs()?;
        // TODO: could have a smaller readback buffer for just the barycenter positions

        Ok(())
    }
}

impl ComputesBarycenter for OCLBarycenterCompute<f64> {
    fn compute(&mut self, variations: Vec<(Array1<f64>, Array1<f64>, f64)>) -> Result<(), String> {
        self.full_reduction(variations).map_err(|e| e.to_string())
        // TODO: Would be better to use something like `anyhow` crate errors?
    }

    fn average_positions(&self) -> Array1<f64> {
        self.xv.host.slice(s![0, ..]).to_owned()
    }

    fn average_weights(&self) -> Array1<f64> {
        self.wv.host.slice(s![0, ..]).to_owned()
    }
}

impl ComputesBarycenter for OCLBarycenterCompute<f32> {
    fn compute(&mut self, variations: Vec<(Array1<f64>, Array1<f64>, f64)>) -> Result<(), String> {
        let variations_f32 = variations
            .into_iter()
            .map(|(sp, sw, w)| (sp.mapv(|v| v as f32), sw.mapv(|v| v as f32), w as f32))
            .collect();
        self.full_reduction(variations_f32)
            .map_err(|e| e.to_string())
    }

    fn average_positions(&self) -> Array1<f64> {
        self.xv.host.slice(s![0, ..]).mapv(|v| v as f64)
    }

    fn average_weights(&self) -> Array1<f64> {
        self.wv.host.slice(s![0, ..]).mapv(|v| v as f64)
    }
}

// TODO: restrict to sensible ranges
#[cfg_attr(feature = "gui", derive(egui_inspect::EguiInspect))]
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct OCLSeparationParams {
    n_nm_iter: usize,
    nm_tol: F,
    nm_alpha: F,
    nm_gamma: F,
    nm_rho: F,
    nm_sigma: F,
    sparsity_alpha: F,
}

impl Default for OCLSeparationParams {
    fn default() -> Self {
        Self {
            n_nm_iter: 1000,
            nm_tol: 1e-9,
            nm_alpha: 1.1,
            nm_gamma: 0.5,
            nm_rho: 0.5,
            nm_sigma: 0.5,
            sparsity_alpha: 0.0,
        }
    }
}

/// The `KF` or Kernel Float parameter attempts to decouple
/// the precision of the kernels run and of the crate/host [F]
#[derive(DeviceToFrom)]
pub struct OCLSeparationCompute<KF: OclPrm> {
    que: ProQue,
    /// Sorted concatenated basis signature positions
    #[dev_to_from(from = false)]
    xbf: PairedBuffers1<KF>, // sorted flat
    /// Concatenated basis signature weights (in sorted pos order)
    #[dev_to_from(from = false)]
    wbf: PairedBuffers1<KF>,
    /// Stick basis correspondence
    #[dev_to_from(from = false)]
    karr: PairedBuffers1<i32>,
    /// NM iterations performed
    #[dev_to_from(to = false)]
    i: PairedBuffers1<i32>,
    // non-paired buffers for read/write to multiple sources/destinations
    /// mix positions
    xc_g: ocl::Buffer<KF>,
    /// mix weights
    wc_g: ocl::Buffer<KF>,
    /// transformed bases
    bt_g: ocl::Buffer<KF>,
    /// Wasserstein_2^2 cost
    w2_g: ocl::Buffer<KF>,
    n_bases: usize,
    n_data: usize,
    padding: usize,
}

static SEPARATION_SOURCE: &str = concat!(
    include_str!("../../../c_opencl_src/shared_defines.h"),
    include_str!("../../../c_opencl_src/quicksort.c"),
    include_str!("../../../c_opencl_src/segment_cost.c"),
    include_str!("../../../c_opencl_src/nm_func.c"),
    include_str!("../../../c_opencl_src/nealder_mead.c"),
    include_str!("../../../c_opencl_src/segment_kernel.cl")
);

impl<KF> OCLSeparationCompute<KF>
where
    KF: OclPrm + ndarray_rand::rand_distr::num_traits::Float,
{
    pub fn new(n_bases: usize, n_data: usize, padding: usize, params: OCLSeparationParams) -> Self {
        zero_check!(n_bases);
        zero_check!(n_data);
        zero_check!(padding);

        let mut compiler_defines = vec![
            "-DEXTERNAL_CONCAT".to_string(),
            // format!("-I{OCL_PATH}"),
            format!("-DN={padding}"),
            format!("-DM={n_data}"),
            format!("-DK={n_bases}"),
            format!("-DN_NM_iter={}", params.n_nm_iter),
            format!("-DNM_TOL={:.3e}f", params.nm_tol),
            format!("-DNM_ALPHA={:.3e}f", params.nm_alpha),
            format!("-DNM_GAMMA={:.3e}f", params.nm_gamma),
            format!("-DNM_RHO={:.3e}f", params.nm_rho),
            format!("-DNM_SIGMA={:.3e}f", params.nm_sigma),
            format!("-DSPARSITY_PROMOTION={:.3e}f", params.sparsity_alpha),
            format!("-DEPSILON={EPSILON:.3e}f"),
        ];
        handle_float_define::<KF>(&mut compiler_defines);
        let mut que = prog_que_from_source(
            SEPARATION_SOURCE,
            "separation kernel".to_string(),
            compiler_defines,
        );
        // TODO: Pass params (other than N,M,K) as struct kernel input rather than recompiling kernel...
        // though in practice substep Compute structs are reset with decomps anyway (and probably should not change
        // params for the same decomp)

        let xbf_c = Array1::<KF>::zeros(n_bases * padding);
        let wbf_c = Array1::<KF>::zeros(n_bases * padding);
        let karr_c = Array1::<i32>::zeros(n_bases * padding);
        let i_c = Array1::<i32>::zeros(n_data);

        let xbf = PairedBuffers::create_from(xbf_c, &mut que);
        let wbf = PairedBuffers::create_from(wbf_c, &mut que);
        let karr = PairedBuffers::create_from(karr_c, &mut que);
        let i = PairedBuffers::create_from(i_c, &mut que);

        // NOTE: dims controlls both new buffer and kernel sizes
        que.set_dims(n_data * padding);
        let xc_g = que.create_buffer::<KF>().expect("buffer create error");
        let wc_g = que.create_buffer::<KF>().expect("buffer create error");
        que.set_dims(n_bases * padding * n_data);
        let bt_g = que.create_buffer::<KF>().expect("buffer create error");
        que.set_dims(n_data);
        let w2_g = que.create_buffer::<KF>().expect("buffer create error");

        Self {
            que,
            xbf,
            wbf,
            karr,
            xc_g,
            wc_g,
            bt_g,
            w2_g,
            i,
            n_bases,
            n_data,
            padding,
        }
    }

    pub fn apply_kernel(
        &mut self,
        xc: &Array2<KF>,
        wc: &Array2<KF>,
        xb: &Array2<KF>,
        wb: &Array2<KF>,
        wbt: &mut Array3<KF>,
        w2: &mut Array1<KF>,
    ) -> ocl::Result<()> {
        // unsorted flat
        let xbuf = xb.as_slice().unwrap();
        let wbuf = wb.as_slice().unwrap();

        let mut flat_ids: Vec<_> = (0..self.n_bases * self.padding).collect();
        flat_ids.sort_by(|i, j| xbuf[*i].partial_cmp(&xbuf[*j]).unwrap());

        for (i, fi) in flat_ids.iter().enumerate() {
            self.xbf.host[i] = xbuf[*fi];
            self.wbf.host[i] = wbuf[*fi];
        }

        let mut init_karr = Vec::with_capacity(self.n_bases * self.padding);
        for i in 0..self.n_bases {
            init_karr.append(&mut vec![i; self.padding]);
        }
        for (i, k) in self.karr.host.indexed_iter_mut() {
            *k = init_karr[flat_ids[i]] as i32;
        }

        self.xc_g.write(xc.as_slice().unwrap()).enq()?;
        self.wc_g.write(wc.as_slice().unwrap()).enq()?;
        self.send_pairedbuffs()?;

        self.que.set_dims(self.n_data);
        let kernel = self
            .que
            .kernel_builder("multisegment")
            .arg(&self.xc_g)
            .arg(&self.wc_g)
            .arg(&self.xbf.device)
            .arg(&self.wbf.device)
            .arg(&self.karr.device)
            .arg(&self.bt_g)
            .arg(&self.w2_g)
            .arg(&self.i.device)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        self.bt_g.read(wbt.as_slice_mut().unwrap()).enq()?;
        self.w2_g.read(w2.as_slice_mut().unwrap()).enq()?;
        self.retrieve_pairedbuffs()?;

        Ok(())
    }
}

impl ComputesSeparation for OCLSeparationCompute<f64> {
    fn compute(
        &mut self,
        xc: &Array2<f64>,
        wc: &Array2<f64>,
        xb: &Array2<f64>,
        wb: &Array2<f64>,
        wbt: &mut Array3<f64>,
        w2: &mut Array1<f64>,
    ) -> Result<(), String> {
        self.apply_kernel(xc, wc, xb, wb, wbt, w2)
            .map_err(|e| e.to_string())?;

        Ok(())
    }
}

impl ComputesSeparation for OCLSeparationCompute<f32> {
    fn compute(
        &mut self,
        xc: &Array2<f64>,
        wc: &Array2<f64>,
        xb: &Array2<f64>,
        wb: &Array2<f64>,
        wbt: &mut Array3<f64>,
        w2: &mut Array1<f64>,
    ) -> Result<(), String> {
        let xc_kf = xc.mapv(|v| v as f32);
        let wc_kf = wc.mapv(|v| v as f32);
        let xb_kf = xb.mapv(|v| v as f32);
        let wb_kf = wb.mapv(|v| v as f32);
        let mut wbt_kf = wbt.mapv(|v| v as f32);
        let mut w2_kf = w2.mapv(|v| v as f32);

        self.apply_kernel(&xc_kf, &wc_kf, &xb_kf, &wb_kf, &mut wbt_kf, &mut w2_kf)
            .map_err(|e| e.to_string())?;

        for (i, v) in wbt.indexed_iter_mut() {
            *v = wbt_kf[i] as f64;
        }
        for (i, v) in w2.indexed_iter_mut() {
            *v = w2_kf[i] as f64;
        }

        Ok(())
    }
}
