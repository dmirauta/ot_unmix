//! Defines visual elements/subsections of the GUI.

use std::{error::Error, ops::Range, str::FromStr};

use egui_extras::{Column, TableBuilder};
use egui_inspect::egui::{
    self, text::LayoutJob, Align, Color32, ComboBox, FontSelection, RichText, Stroke, Style,
};
use egui_inspect::EguiInspect;
use egui_plot::{Line, Plot, PlotBounds, PlotPoints, PlotUi, Polygon};
use log::warn;

use ot_unmix_core::{
    data::{pairs, Signature, Stick},
    inspect::{setup_plot, SMOOTHING_OPT},
    ndarray::{s, Array1, Array2, Axis},
    substeps::{
        ocl::{OCLBarycenterCompute, OCLSeparationCompute, OCLSeparationParams},
        traits::{ComputesBarycenter, ComputesSeparation},
    },
    unmixing::{InitialBasis, EPSILON},
};

use crate::app::{compute_ongoing, CMAP_FREQS, KNOWN_BASES};

static SEG_COST_INFO: &str =
    "Segmentation cost (without penalties if applicable), indicating amount of overall \"warping\".
    Anom and non-anom versions may not currently be comparable!";
// TODO: Check consistency of anom and non-anom seg costs...

#[derive(Default)]
pub struct IndexSelection {
    pub items: Vec<usize>,
}

impl FromStr for IndexSelection {
    type Err = Box<dyn Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut items = vec![];
        for ss in s.split(',') {
            items.push(ss.trim().parse()?);
        }
        Ok(Self { items })
    }
}

#[derive(Default)]
pub struct IndexSelectionUi {
    input: String,
    pub selection: IndexSelection,
}

impl EguiInspect for IndexSelectionUi {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, label: &str, ui: &mut egui::Ui) {
        ui.label(label);
        if ui.text_edit_singleline(&mut self.input).changed() {
            match self.input.parse() {
                Ok(si) => self.selection = si,
                Err(err) => {
                    let err = err.to_string();
                    if !err.contains("empty string") {
                        warn!("Mixture indices parsing error: {err} (expecting comma separated indices)");
                    }
                }
            };
        }
    }
}

#[derive(EguiInspect)]
pub struct AnomParamsSelection {
    /// maximum average displacement
    pub max_avg_displacement: f64,
    // #[inspect(min = 1.0, max = 100.0)]
    // anom_penalty_frac: f64,
}

impl Default for AnomParamsSelection {
    fn default() -> Self {
        let r = SMOOTHING_OPT.with_borrow(|so| so.xs.last().unwrap() - so.xs.first().unwrap());
        Self {
            max_avg_displacement: 0.05 * r,
            // anom_penalty_frac: 10.0,
        }
    }
}

#[derive(Default, PartialEq)]
pub enum TableSortOption {
    #[default]
    BySegmentationCost,
    ByAnomalousContent,
}

#[derive(Default)]
pub struct SummaryTable {
    pub abundances: Array2<f64>,
    pub seg_costs: Array1<f64>,
    pub anom_amounts: Array1<f64>,
    pub data_name_map: Vec<String>,
    pub basis_name_map: Vec<String>,
    pub table_sort_opt: TableSortOption,
    pub anom_params: AnomParamsSelection,
    pub mixtures_search: String,
    pub anom_decomp_idx: Option<usize>,
    pub anom_decomp_all: bool,
}

impl EguiInspect for SummaryTable {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, _label: &str, ui: &mut egui::Ui) {
        let n_data = self.abundances.shape()[0];
        let n_bases = self.abundances.shape()[1];

        ui.collapsing("Abundances", |ui| {
            ui.horizontal(|ui| {
                self.mixtures_search.inspect_mut("search", ui);
            });

            let mut row_idxs: Vec<_> = match self.mixtures_search.is_empty() {
                true => (0..n_data).collect(),
                false => (0..n_data)
                    .filter(|j| self.data_name_map[*j].contains(&self.mixtures_search))
                    .collect(),
            };

            ui.horizontal(|ui| {
                ui.label("Sort by: ");
                ui.radio_value(
                    &mut self.table_sort_opt,
                    TableSortOption::BySegmentationCost,
                    "separation cost",
                );
                ui.radio_value(
                    &mut self.table_sort_opt,
                    TableSortOption::ByAnomalousContent,
                    "anomalous content",
                );
            });

            match self.table_sort_opt {
                TableSortOption::BySegmentationCost => {
                    row_idxs.sort_by(|i, j| {
                        self.seg_costs[*j].partial_cmp(&self.seg_costs[*i]).unwrap()
                    });
                }
                TableSortOption::ByAnomalousContent => {
                    row_idxs.sort_by(|i, j| {
                        self.anom_amounts[*j]
                            .partial_cmp(&self.anom_amounts[*i])
                            .unwrap()
                    });
                }
            }

            let mut table = TableBuilder::new(ui).striped(true).resizable(true);
            for _ in 0..n_bases + 5 {
                table = table.column(Column::auto());
            }

            table
                .header(20.0, |mut header| {
                    header.col(|ui| {
                        ui.horizontal_top(|ui| {
                            ui.strong("Index");
                        });
                    });
                    header.col(|ui| {
                        ui.strong("Datapoint");
                    });
                    for i in 0..n_bases {
                        header.col(|ui| {
                            ui.horizontal_top(|ui| {
                                ui.strong(format!("Basis {} fraction", i + 1).as_str())
                                    .on_hover_text_at_pointer(self.basis_name_map[i].as_str());
                            });
                        });
                    }
                    header.col(|ui| {
                        ui.strong("Separation cost")
                            .on_hover_text_at_pointer(SEG_COST_INFO);
                    });
                    header.col(|ui| {
                        ui.strong("Anom amount");
                    });
                    header.col(|ui| {
                        ui.horizontal_top(|ui| {
                            ui.strong("Anom LP (for this idx)");
                        });
                    });
                })
                .body(|mut body| {
                    for j in row_idxs {
                        body.row(20.0, |mut row| {
                            row.col(|ui| {
                                ui.label(format!("{j}").as_str());
                            });
                            row.col(|ui| {
                                let name = self.data_name_map[j].clone();
                                if self.mixtures_search.is_empty() {
                                    ui.label(name);
                                } else {
                                    let match_range = substring_range(
                                        name.as_str(),
                                        self.mixtures_search.as_str(),
                                    )
                                    .unwrap_or(0..0);
                                    let prefix = &name[..match_range.start];
                                    let suffix = &name[match_range.end..];
                                    ui.horizontal_top(|ui| {
                                        ui.label(concat_rich_text(vec![
                                            RichText::new(prefix).color(Color32::WHITE),
                                            RichText::new(self.mixtures_search.clone())
                                                .color(Color32::GREEN),
                                            RichText::new(suffix).color(Color32::WHITE),
                                        ]));
                                    });
                                }
                            });
                            for i in 0..n_bases {
                                row.col(|ui| {
                                    ui.label(format!("{:.3}", self.abundances[[j, i]]).as_str());
                                });
                            }
                            row.col(|ui| {
                                ui.label(format!("{:.3}", self.seg_costs[j]).as_str());
                            });
                            row.col(|ui| {
                                let aa = self.anom_amounts[j];
                                ui.label(format!("{aa:.3}").as_str());
                            });
                            row.col(|ui| {
                                if !compute_ongoing() {
                                    ui.menu_button(format!("run for {j}").as_str(), |ui| {
                                        self.anom_params.inspect_mut("", ui);
                                        if ui.button("run").clicked() {
                                            // Would not be visible if opt is None
                                            self.anom_decomp_idx = Some(j);
                                            ui.close_menu();
                                        }
                                    });
                                } else {
                                    "<Compute ongoing>".inspect("", ui);
                                }
                            });
                        });
                    }
                });
        });

        ui.horizontal(|ui| {
            if !compute_ongoing() {
                ui.menu_button("run LP for for all mixtures", |ui| {
                    self.anom_params.inspect_mut("", ui);
                    if ui.button("run").clicked() {
                        self.anom_decomp_all = true;
                    }
                });
            } else {
                "<Compute ongoing>".inspect("", ui);
            }
        });
    }
}

pub fn smoothed_sig(
    sig_positions: &[f64],
    sig_weights: &[f64],
    target_positions: &[f64],
    target_weights: &mut [f64],
    kernel_std: f64,
) {
    let tempsig: Signature = (sig_positions, sig_weights).into();
    tempsig.estimate_density_inplace(target_positions, target_weights, kernel_std);
}

#[derive(Default, EguiInspect)]
pub struct ExportOptions {
    pub basis_signatures: bool,
    pub component_signatures: bool,
    pub basis_kdes: bool,
    pub component_kdes: bool,
    pub abundances: bool,
}

fn concat_rich_text(rtv: Vec<RichText>) -> LayoutJob {
    let style = Style::default();
    let mut layout_job = LayoutJob::default();
    for rt in rtv {
        rt.append_to(
            &mut layout_job,
            &style,
            FontSelection::Default,
            Align::Center,
        );
    }
    layout_job
}

fn substring_range(string: &str, substring: &str) -> Option<Range<usize>> {
    let start = string.find(substring);
    start.map(|si| si..si + substring.len())
}

#[derive(Default, PartialEq)]
pub struct FromKnownOpt {
    pub source: String,
    pub keep_fixed: bool,
}

impl EguiInspect for FromKnownOpt {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, label: &str, ui: &mut egui::Ui) {
        let id = format!("{self:p}");
        let FromKnownOpt { source, keep_fixed } = self;
        KNOWN_BASES.with_borrow(|cleaned| {
            ui.horizontal(|ui| {
                ui.label("selected:");
                ComboBox::new(id, "")
                    .selected_text(source.as_str())
                    .show_ui(ui, |ui| {
                        // will be empty during resets
                        for name in cleaned.keys() {
                            ui.selectable_value(source, name.clone(), name.clone());
                        }
                    });
            });

            // in case options have changed (e.g. during resets)
            if let Some(referenced_sig) = cleaned.get(source) {
                // Note: can give unique names here if we wish to unlink plot objects, but this
                // label will be displayed
                referenced_sig.inspect(label, ui);
                keep_fixed.inspect_mut("keep fixed", ui);
            }
        });
    }
}

/// A precursor for [InitialBasis]
#[derive(Default, PartialEq, EguiInspect)]
pub enum BasisSource {
    // Indirection due to custom Inspect for this option
    FromCleaned {
        data: FromKnownOpt,
    },
    Sketched {
        source: Signature,
        keep_fixed: bool,
    },
    #[default]
    NoPrior,
}

impl BasisSource {
    pub fn desc(&self) -> String {
        match self {
            BasisSource::FromCleaned {
                data: FromKnownOpt { source, keep_fixed },
            } => {
                format!("{source} (fixed={keep_fixed})")
            }
            BasisSource::Sketched { keep_fixed, .. } => {
                format!("sketched (fixed={keep_fixed})")
            }
            BasisSource::NoPrior => "inferred".to_string(),
        }
    }
}

impl From<&BasisSource> for InitialBasis {
    fn from(selection: &BasisSource) -> Self {
        match &selection {
            BasisSource::FromCleaned {
                data: FromKnownOpt { source, keep_fixed },
            } => KNOWN_BASES.with_borrow(|cleaned| Self::Provided {
                pattern: cleaned.get(source).unwrap().clone(),
                keep_fixed: *keep_fixed,
            }),
            BasisSource::Sketched { source, keep_fixed } => {
                let mut pattern = source.clone();
                pattern.normalise();
                Self::Provided {
                    pattern,
                    keep_fixed: *keep_fixed,
                }
            }
            BasisSource::NoPrior => Self::NoPrior,
        }
    }
}

#[derive(EguiInspect)]
#[inspect(on_hover_text = "fp32 option currently has some issues")]
pub struct SubstepParameters {
    #[inspect(no_edit)]
    fp64: bool,
    #[inspect(name = "separation params")]
    sep_params: OCLSeparationParams,
}

impl Default for SubstepParameters {
    fn default() -> Self {
        Self {
            fp64: true,
            sep_params: Default::default(),
        }
    }
}

impl SubstepParameters {
    pub fn barycenter_compute(&self, n_data: usize, padding: usize) -> Box<dyn ComputesBarycenter> {
        match self.fp64 {
            true => Box::new(OCLBarycenterCompute::<f64>::new(n_data, padding)),
            false => Box::new(OCLBarycenterCompute::<f32>::new(n_data, padding)),
        }
    }
    pub fn separation_compute(
        &self,
        n_bases: usize,
        n_data: usize,
        padding: usize,
    ) -> Box<dyn ComputesSeparation> {
        match self.fp64 {
            true => Box::new(OCLSeparationCompute::<f64>::new(
                n_bases,
                n_data,
                padding,
                self.sep_params,
            )),
            false => {
                warn!("Casts from f32 currently seem to be erroneous! Use f64 if possible.");
                Box::new(OCLSeparationCompute::<f32>::new(
                    n_bases,
                    n_data,
                    padding,
                    self.sep_params,
                ))
            }
        }
    }
}

#[derive(Default)]
pub struct InitialParams {
    pub substep_param: SubstepParameters,
    pub initial_basis_selection: Vec<BasisSource>,
}

impl InitialParams {
    pub fn add_all_cleaned(&mut self, keep_fixed: bool) {
        self.initial_basis_selection.clear();
        KNOWN_BASES.with_borrow(|cleaned| {
            for name in cleaned.keys() {
                let data = FromKnownOpt {
                    source: name.clone(),
                    keep_fixed,
                };
                self.initial_basis_selection
                    .push(BasisSource::FromCleaned { data });
            }
        });
    }
}

impl EguiInspect for InitialParams {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, label: &str, ui: &mut egui::Ui) {
        ui.label(label);
        ui.separator();
        ui.columns(2, |columns| {
            self.substep_param
                .inspect_mut("substep params", &mut columns[0]);
            columns[1].horizontal(|ui| {
                if ui.button("Add all loaded").clicked() {
                    self.add_all_cleaned(false);
                }
                if ui.button("Add all loaded as fixed").clicked() {
                    self.add_all_cleaned(true);
                }
            });
            self.initial_basis_selection
                .inspect_mut("initial basis", &mut columns[1]);
        });
        ui.separator();
    }
}

#[derive(Default)]
pub struct PreviewedComponent {
    /// dense position grid for smoothed weights
    pub xs: Array1<f64>,
    /// smoothed component weights
    pub wbts: Array2<f64>,
    pub anom: Option<Array1<f64>>,
}

impl EguiInspect for PreviewedComponent {
    fn inspect(&self, label: &str, ui: &mut egui::Ui) {
        let wbts_shape = self.wbts.shape();
        let n_bases = wbts_shape[0];
        let n_smoothed = wbts_shape[1];

        // TODO: Accept differenct cmap types?
        let cmap = CMAP_FREQS.with_borrow(|cmf| cmf.clone());

        setup_plot(label, ui, true).show(ui, |plot_ui| {
            let mut cumulative_base_bot = Array1::<f64>::zeros(n_smoothed);
            let mut cumulative_base_top = Array1::<f64>::zeros(n_smoothed);
            for i in 0..n_bases {
                // cumulative_base += self.wbts.slice(s![i, ..]);
                for (cbv, wbiv) in cumulative_base_top
                    .iter_mut()
                    .zip(self.wbts.slice(s![i, ..]).iter())
                {
                    *cbv += *wbiv;
                }

                let xy = pairs(
                    self.xs.as_slice().unwrap(),
                    cumulative_base_bot.as_slice().unwrap(),
                );
                let xyt = pairs(
                    self.xs.as_slice().unwrap(),
                    cumulative_base_top.as_slice().unwrap(),
                );

                add_nonconvex_naive(
                    plot_ui,
                    xy,
                    xyt.clone(),
                    cmap.get_color(ColorTarget::Basis(i)),
                    format!("bt {}", i + 1).as_str(),
                );

                let outline = Line::new(PlotPoints::new(xyt.clone()));
                plot_ui.line(outline.color(Color32::BLACK).width(3.0));

                cumulative_base_bot.assign(&cumulative_base_top);
            }

            if let Some(anom) = &self.anom {
                for (cbv, wbiv) in cumulative_base_top.iter_mut().zip(anom.iter()) {
                    *cbv += *wbiv;
                }

                let xy = pairs(
                    self.xs.as_slice().unwrap(),
                    cumulative_base_bot.as_slice().unwrap(),
                );
                let xyt = pairs(
                    self.xs.as_slice().unwrap(),
                    cumulative_base_top.as_slice().unwrap(),
                );

                add_nonconvex_naive(plot_ui, xy, xyt, cmap.get_color(ColorTarget::Anom), "anom");
            }
        });
    }

    // TODO: Implement area painting? (for init or iter biasing)
    fn inspect_mut(&mut self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }
}

enum ColorTarget {
    Basis(usize),
    Anom,
}

trait Cmap {
    fn get_color(&self, ct: ColorTarget) -> Color32;
}

#[derive(EguiInspect, Clone)]
pub struct PeriodicCmapFreqs {
    r: f64,
    g: f64,
    b: f64,
    overrides: Vec<Color32>,
}

impl Default for PeriodicCmapFreqs {
    fn default() -> Self {
        Self {
            r: 1.0,
            g: 16.0,
            b: 3.0,
            overrides: vec![],
        }
    }
}

impl Cmap for PeriodicCmapFreqs {
    fn get_color(&self, ct: ColorTarget) -> Color32 {
        match ct {
            ColorTarget::Basis(i) => {
                if let Some(color) = self.overrides.get(i) {
                    *color
                } else {
                    let j = i as f64;
                    let rs = (self.r * j).sin() + 1.0;
                    let gs = (self.g * j).sin() + 1.0;
                    let bs = (self.b * j).sin() + 1.0;
                    Color32::from_rgb((rs * 80.0) as u8, (gs * 127.0) as u8, (bs * 127.0) as u8)
                }
            }
            ColorTarget::Anom => Color32::RED,
        }
    }
}

#[derive(Default)]
pub struct TransportPlanVis {
    plan: Array2<f64>,
    xb: Array1<f64>,
    xbt: Array1<f64>,
    basis: Signature,
    varied: Signature,
}

impl TransportPlanVis {
    pub fn new(plan: Array2<f64>, xb: Array1<f64>, xbt: Array1<f64>) -> Self {
        let a0_sum = plan.sum_axis(Axis(0));
        let a1_sum = plan.sum_axis(Axis(1));
        let basis = pairs(xb.as_slice().unwrap(), a1_sum.as_slice().unwrap()).into();
        let varied = pairs(xbt.as_slice().unwrap(), a0_sum.as_slice().unwrap()).into();
        Self {
            plan,
            xb,
            xbt,
            basis,
            varied,
        }
    }
}

impl EguiInspect for TransportPlanVis {
    fn inspect(&self, label: &str, ui: &mut egui::Ui) {
        ui.collapsing(label, |ui| {
            let plan_min = self
                .plan
                .iter()
                .filter(|x| **x > 0.0)
                .copied()
                .reduce(f64::min);

            if let Some(plan_min) = plan_min {
                ui.label(format!("fraction: {:.3}", self.plan.sum()).as_str());

                let plan_max = self.plan.iter().copied().reduce(f64::max).unwrap() + EPSILON;

                let (large_side, small_side) = (350.0, 150.0);

                let link_group_id = ui.id().with(format!("{label} marginal_plot_lg"));

                let central = Plot::new(format!("{label} central"))
                    .data_aspect(1.0)
                    .width(large_side)
                    .height(large_side)
                    .link_cursor(link_group_id, true, true)
                    .custom_x_axes(vec![])
                    .custom_y_axes(vec![]);

                let side = Plot::new(format!("{label} side"))
                    .width(small_side)
                    .height(large_side)
                    .link_cursor(link_group_id, false, true)
                    .custom_x_axes(vec![])
                    .y_axis_label("x (source positions, original basis)")
                    .y_axis_position(egui_plot::HPlacement::Right);

                let bottom = Plot::new(format!("{label} bottom"))
                    .width(large_side)
                    .height(small_side)
                    .link_cursor(link_group_id, true, false)
                    .x_axis_label("x' (destination positions, tranformed basis)")
                    .custom_y_axes(vec![]);

                let mut shared_extents = None;

                // let [xmin, xmax] = SMOOTHING_OPT.with(|so| [so.borrow().xmin, so.borrow().xmax]);
                // let r = xmax - xmin;
                // let xdel = r / 2.0;
                // let xmid = xmin + xdel;

                // visible weight range
                fn wr(sig: &Signature) -> [f64; 2] {
                    let sig_scale = (sig.sticks.len() as f64) / sig.total_weight();
                    let wmax = sig
                        .sticks
                        .iter()
                        .map(|Stick { weight, .. }| *weight)
                        .reduce(f64::max)
                        .unwrap()
                        * sig_scale;
                    [-0.05 * wmax, 1.05 * wmax]
                }

                let so = SMOOTHING_OPT.with_borrow(|so| so.clone());

                // Note: cannot set consistent size on ordinary markers (only max)
                fn custom_marker(
                    x: f64,
                    y: f64,
                    v: f64,
                    vmi: f64,
                    vma: f64,
                    stick_delta: f64,
                ) -> Polygon {
                    let r = 3.0 * stick_delta;
                    let f = (v - vmi) / (vma - vmi);
                    let d = (120.0 + 130.0 * f) as u8;
                    let c = Color32::from_rgb(d, d, d);
                    Polygon::new(vec![
                        [x - r, y - r],
                        [x + r, y - r],
                        [x + r, y + r],
                        [x - r, y + r],
                    ])
                    .fill_color(c)
                    .stroke(Stroke::new(0.0, c))
                }

                ui.horizontal(|ui| {
                    central.show(ui, |pui| {
                        for (i, &x) in self.xb.indexed_iter() {
                            for (j, &xt) in self.xbt.indexed_iter() {
                                if self.plan[(i, j)] > 0.0 {
                                    // plotting transposed
                                    pui.polygon(custom_marker(
                                        xt,
                                        x,
                                        self.plan[(i, j)],
                                        plan_min,
                                        plan_max,
                                        so.stick_width,
                                    ));
                                }
                            }
                        }

                        shared_extents = Some(pui.plot_bounds());
                    });

                    side.show(ui, |pui| {
                        let [_, symin] = shared_extents.unwrap().min();
                        let [_, symax] = shared_extents.unwrap().max();

                        let [wmi, wma] = wr(&self.basis);
                        pui.set_plot_bounds(PlotBounds::from_min_max([wmi, symin], [wma, symax]));

                        self.basis.plot_transposed(pui, &so);
                    });
                });

                bottom.show(ui, |pui| {
                    let [sxmin, _] = shared_extents.unwrap().min();
                    let [sxmax, _] = shared_extents.unwrap().max();

                    let [wmi, wma] = wr(&self.varied);
                    pui.set_plot_bounds(PlotBounds::from_min_max([sxmin, wmi], [sxmax, wma]));

                    self.varied.plot(pui, &so);
                });
            } else {
                ui.label("empty");
            }
        });
    }

    fn inspect_mut(&mut self, label: &str, ui: &mut egui::Ui) {
        self.inspect(label, ui);
    }
}

type PF = f64;
type P = [PF; 2];
fn quad_area([x1, y1]: P, [x2, y2]: P, [x3, y3]: P, [x4, y4]: P) -> PF {
    ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (x2 * y1 + x3 * y2 + x4 * y3 + x1 * y4)) / 2.0
}

pub fn add_nonconvex_naive(
    plot_ui: &mut PlotUi,
    xy_bot: Vec<[f64; 2]>,
    xy_top: Vec<[f64; 2]>,
    color: Color32,
    name: &str,
) {
    for (bw, tw) in xy_bot.windows(2).zip(xy_top.windows(2)) {
        // TODO: Check size in terms of pixels instead?
        // Issue is that axes extents are dependent on plot content,
        // but then plot content also become dependent on extents.
        // let t = plot_ui.transform();
        // let a: [[f32; 2]; 4] = [tw[0], bw[0], bw[1], tw[1]].map(|p| {
        //     let tp = t.position_from_point(&p.into());
        //     [tp.x, tp.y]
        // });
        // TODO: Minimum area value is data dependent?
        if quad_area(tw[0], bw[0], bw[1], tw[1]) > 1e-7 {
            let quad = vec![bw[0], tw[0], tw[1], bw[1]];
            plot_ui.polygon(
                Polygon::new(quad)
                    .fill_color(color)
                    .stroke(Stroke::new(1.0, color))
                    .name(name),
            );
        }
    }
}
