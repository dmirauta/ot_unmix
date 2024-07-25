//! Data processing actions
#![allow(clippy::needless_range_loop)]

use super::{
    data::{CleaningIntermediate, Signature, Stick, XYData},
    rhai_transform::RhaiTransform,
};
use geo::{ConvexHull, Coord, CoordsIter, LineString, Polygon};
use interp::interp;
use ndarray::Array1;

pub trait XYTransform {
    fn apply(&self, input: &mut CleaningIntermediate);
}

macro_rules! gen_pipeline_item_from_opts {
    ($($option: ident),*) => {
        #[cfg_attr(feature = "gui", derive(egui_inspect::EguiInspect))]
        pub enum PipelineItem {
            $($option {
                #[cfg_attr(feature = "gui", inspect(name = ""))]
                transform: $option,
            },)*
        }
    }
}

gen_pipeline_item_from_opts!(
    XLimit,
    SimpleDownsample,
    ConvexBaseline,
    SlidingEllipseBaseline,
    RhaiTransform,
    XYToSignature,
    XYAsSignature
);

// NOTE: Comparing by variant only, for purposes of dropdown (EguiInspect)
impl PartialEq for PipelineItem {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}

impl Default for PipelineItem {
    fn default() -> Self {
        Self::XLimit {
            transform: Default::default(),
        }
    }
}

impl XYTransform for PipelineItem {
    fn apply(&self, input: &mut CleaningIntermediate) {
        match self {
            PipelineItem::XLimit { transform } => transform.apply(input),
            PipelineItem::SimpleDownsample { transform } => transform.apply(input),
            PipelineItem::ConvexBaseline { transform } => transform.apply(input),
            PipelineItem::SlidingEllipseBaseline { transform } => transform.apply(input),
            PipelineItem::RhaiTransform { transform } => transform.apply(input),
            PipelineItem::XYToSignature { transform } => transform.apply(input),
            PipelineItem::XYAsSignature { transform } => transform.apply(input),
        }
    }
}

#[cfg_attr(feature = "gui", derive(egui_inspect::EguiInspect))]
pub struct XLimit {
    xmin: f64,
    xmax: f64,
}

impl Default for XLimit {
    fn default() -> Self {
        Self {
            xmin: -f64::INFINITY,
            xmax: f64::INFINITY,
        }
    }
}

fn placeholder_message(step: &str) {
    println!("input is unitialised in [{step}], check previous step");
}

impl XYTransform for XLimit {
    fn apply(&self, input: &mut CleaningIntermediate) {
        match input.cleaned_mut() {
            XYData::Loose(xy) => {
                let pop_positions: Vec<_> = (0..xy.len())
                    .filter(|i| self.xmin > xy[*i][0] || xy[*i][0] > self.xmax)
                    .collect();
                for i in pop_positions.into_iter().rev() {
                    xy.remove(i);
                }
            }
            XYData::Placeholder => placeholder_message("XLimit"),
        }
    }
}

#[cfg_attr(feature = "gui", derive(egui_inspect::EguiInspect))]
#[cfg_attr(feature = "gui", inspect(no_border))]
#[derive(Default)]
pub struct ConvexBaseline {}

impl XYTransform for ConvexBaseline {
    fn apply(&self, input: &mut CleaningIntermediate) {
        match input.cleaned_mut() {
            XYData::Loose(xy) => {
                if xy.is_empty() {
                    return;
                }
                let linestring =
                    LineString::new(xy.iter().map(|[x, y]| Coord { x: *x, y: *y }).collect());
                let first = xy[0];
                let last = *xy.last().unwrap();
                let poly = Polygon::new(linestring, vec![]);
                let cv = poly.convex_hull();
                let mut bic: Vec<_> = cv.coords_iter().map(|Coord { x, y }| [x, y]).collect();

                // TODO: make sure we have the fewest necessary number of pushes/pops
                // baseline inner coords
                while bic[0] != first {
                    let cv_first = bic.remove(0);
                    bic.push(cv_first);
                }
                while *bic.last().unwrap() != last {
                    bic.pop();
                }

                let bxv: Vec<_> = bic.iter().map(|[x, _]| *x).collect();
                let byv: Vec<_> = bic.iter().map(|[_, y]| *y).collect();

                let baseline: Vec<_> = xy
                    .iter()
                    .map(|[x, _]| [*x, interp(&bxv, &byv, *x)])
                    .collect();

                input.apply_baseline(baseline);
            }
            XYData::Placeholder => placeholder_message("ConvexBaseline"),
        }
    }
}

#[cfg_attr(feature = "gui", derive(egui_inspect::EguiInspect))]
pub struct SlidingEllipseBaseline {
    horizontal_diameter_fractional: f64,
    vertical_diameter_fractional: f64,
    preview_ellipse: bool,
}

impl Default for SlidingEllipseBaseline {
    fn default() -> Self {
        Self {
            horizontal_diameter_fractional: 0.2,
            vertical_diameter_fractional: 0.05,
            preview_ellipse: false,
        }
    }
}

/// offset at which the stencil becomes "just under" y
fn just_under_offset(y: &[f64], stencil: &[f64], center_at: usize) -> f64 {
    debug_assert!(stencil.len() % 2 == 1);
    debug_assert!(center_at < y.len());
    let r = (stencil.len() - 1) / 2;
    // when not near boundary, we iterate over 2*r+1 points
    let mut m = f64::INFINITY;
    for i in 0..2 * r {
        if let Some(j) = (center_at + r).checked_sub(i) {
            if j < y.len() {
                m = m.min(y[j] - stencil[i]);
            }
        }
    }
    m
}

/// upper part of \[x,y\] = \[0, -height\] + \[cos(t), height*sin(t)\]
fn semi_ellipse(n: usize, height: f64) -> (Vec<f64>, Vec<f64>) {
    let x = Array1::<f64>::linspace(-1.0 + 1e-19, 1.0 - 1e-19, n);
    // TODO: create with array operations
    let y = x
        .iter()
        .map(|xval| height * ((1.0 - xval.powi(2)).sqrt() - 1.0))
        .collect();

    (x.into_iter().collect(), y)
}

// TODO: expose generic sliding shape? defined as [0, 1] -> [0, 1] function?
impl XYTransform for SlidingEllipseBaseline {
    fn apply(&self, input: &mut CleaningIntermediate) {
        let mut baseline_curve = None;
        match input.cleaned_mut() {
            XYData::Loose(xy) => {
                let y: Vec<_> = xy.iter().map(|[_, y]| *y).collect();

                if y.is_empty() {
                    println!("y vector empty, check x limits!");
                    return;
                }

                let ymin = y.iter().copied().reduce(f64::min).unwrap();
                let ymax = y.iter().copied().reduce(f64::max).unwrap();
                let vert_diam = (ymax - ymin) * self.vertical_diameter_fractional;

                let mut stencil_size =
                    (self.horizontal_diameter_fractional * (y.len() as f64)) as usize;
                if stencil_size % 2 == 0 {
                    stencil_size -= 1;
                }
                if stencil_size < 3 {
                    println!("vertical_diameter_fractional is too small");
                    return;
                }
                let (_, stencily) = semi_ellipse(stencil_size, vert_diam);

                if self.preview_ellipse {
                    let x: Vec<_> = xy.iter().map(|[x, _]| *x).collect();
                    let xmin = x.iter().copied().reduce(f64::min).unwrap();
                    let xmax = x.iter().copied().reduce(f64::max).unwrap();
                    let horiz_diam = (xmax - xmin) * self.horizontal_diameter_fractional;
                    let stencilx = x.iter().take(stencil_size).cloned().collect();
                    baseline_curve = Some((
                        format!("ellipse (dx={horiz_diam:.3}, dy={vert_diam:.3})"),
                        XYData::from((stencilx, stencily.clone())),
                    ));
                }

                let mut b: Vec<[f64; 2]> = xy.clone();
                for i in 0..y.len() {
                    b[i][1] = just_under_offset(y.as_slice(), stencily.as_slice(), i);
                }
                input.apply_baseline(b);
            }
            XYData::Placeholder => placeholder_message("SlidingEllipseBaseline"),
        }
        if let Some((k, v)) = baseline_curve {
            input.extra_curves.insert(k, v);
        }
    }
}

#[cfg_attr(feature = "gui", derive(egui_inspect::EguiInspect))]
pub struct SimpleDownsample {
    average_groups_of: u32, // usize slider option not working?
}

impl Default for SimpleDownsample {
    fn default() -> Self {
        Self {
            average_groups_of: 2,
        }
    }
}

fn slice_average(slice: &[f64]) -> f64 {
    slice.iter().copied().sum::<f64>() / (slice.len() as f64)
}

impl XYTransform for SimpleDownsample {
    fn apply(&self, input: &mut CleaningIntermediate) {
        if self.average_groups_of < 2 {
            return;
        }
        match input.cleaned_mut() {
            XYData::Loose(xy) => {
                // TODO: minimise data repacking (from/to [x,y,x,y] and [x,x,y,y])
                let x: Vec<_> = xy.iter().map(|[x, _]| *x).collect();
                let y: Vec<_> = xy.iter().map(|[_, y]| *y).collect();
                let xn = x.chunks(self.average_groups_of as usize).map(slice_average);
                let yn = y.chunks(self.average_groups_of as usize).map(slice_average);
                *xy = xn.into_iter().zip(yn).map(|(x, y)| [x, y]).collect();
            }
            XYData::Placeholder => placeholder_message("SimpleDownsample"),
        }
    }
}

pub enum HSRStopCriteria {
    UntilTargetSize(usize),
    #[allow(dead_code)]
    UntilErrorTol(f64),
}

pub struct HierachicalSignatureReduction {
    sig: Signature,
    merge_costs: Vec<f64>,
    stop_criteria: HSRStopCriteria,
}

impl HierachicalSignatureReduction {
    pub fn new(sig: Signature, stop_criteria: HSRStopCriteria) -> Self {
        let n = sig.sticks.len() - 1;
        let mut new = HierachicalSignatureReduction {
            sig,
            merge_costs: vec![0.0; n],
            stop_criteria,
        };
        for i in 0..n {
            new.update_cost_at(i);
        }
        new
    }

    fn criteria_reached(&self) -> bool {
        match self.stop_criteria {
            HSRStopCriteria::UntilTargetSize(n) => self.sig.sticks.len() == n,
            // TODO: will need to copy and restore previous state to not apply tol breaking iteration
            HSRStopCriteria::UntilErrorTol(_) => todo!(),
        }
    }

    fn update_cost_at(&mut self, i: usize) {
        debug_assert!(i < self.sig.sticks.len());
        let Stick {
            position: p0,
            weight: w0,
        } = self.sig.sticks[i];
        let Stick {
            position: p1,
            weight: w1,
        } = self.sig.sticks[i + 1];
        let w0_sq = w0 * w0;
        let w1_sq = w1 * w1;
        self.merge_costs[i] =
            ((p0 - p1).powi(2) * (w1 * w0_sq + w0 * w1_sq)) / (w0_sq + 2.0 * w0 * w1 + w1_sq);
    }

    pub fn apply(mut self) -> Signature {
        while !self.criteria_reached() {
            let merge_idx = (0..self.merge_costs.len())
                .min_by(|i, j| {
                    self.merge_costs[*i]
                        .partial_cmp(&self.merge_costs[*j])
                        .unwrap()
                })
                .unwrap();

            let Stick {
                position: p0,
                weight: w0,
            } = self.sig.sticks.remove(merge_idx);
            let Stick {
                position: p1,
                weight: w1,
            } = self.sig.sticks.remove(merge_idx);
            let wt = w0 + w1;
            self.sig.sticks.insert(
                merge_idx,
                Stick {
                    position: p0 * w0 / wt + p1 * w1 / wt,
                    weight: wt,
                },
            );

            self.merge_costs.remove(merge_idx);
            if merge_idx > 0 {
                self.update_cost_at(merge_idx - 1)
            }
            if merge_idx < self.merge_costs.len() {
                self.update_cost_at(merge_idx)
            }
        }

        self.sig
    }
}

#[cfg_attr(feature = "gui", derive(egui_inspect::EguiInspect))]
pub struct XYToSignature {
    n_sticks: u32, // usize slider option not working?
}

impl Default for XYToSignature {
    fn default() -> Self {
        Self { n_sticks: 40 }
    }
}

impl XYTransform for XYToSignature {
    fn apply(&self, input: &mut CleaningIntermediate) {
        if self.n_sticks == 0 {
            return;
        }
        match input.cleaned_mut() {
            XYData::Loose(xy) => {
                let hsr = HierachicalSignatureReduction::new(
                    xy.clone().into(),
                    HSRStopCriteria::UntilTargetSize(self.n_sticks as usize),
                );
                input.signature = hsr.apply();
                input.signature.normalise();
            }
            XYData::Placeholder => placeholder_message("XYToSignature"),
        }
    }
}

#[cfg_attr(feature = "gui", derive(egui_inspect::EguiInspect))]
#[cfg_attr(feature = "gui", inspect(no_border))]
#[derive(Default)]
pub struct XYAsSignature {}

impl XYTransform for XYAsSignature {
    fn apply(&self, input: &mut CleaningIntermediate) {
        input.signature = input.cleaned_mut().clone().to_sig().unwrap();
    }
}
