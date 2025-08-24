//! Defines eframe::App for initiating and inspecting a decomposition.

#![allow(clippy::needless_range_loop)]

use iter_tools::Itertools;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    fs,
    iter::once,
    ops::DerefMut,
    path::PathBuf,
    sync::{Arc, Mutex},
    thread::{spawn, JoinHandle},
    time::Duration,
};

use egui_inspect::{eframe, logging::LogsView, EguiInspect};
use egui_inspect::{
    egui::{self, CentralPanel, ScrollArea, Slider, TopBottomPanel, Window},
    logging::{setup_mixed_logger, FileLogOption},
};
use log::{error, info, warn};

use ot_unmix_core::{
    data::{pairs, CleaningIntermediate, Signature},
    dataset::{default_pipeline, on_dir_created, Dataset, TaggedSignatures},
    inspect::SMOOTHING_OPT,
    ndarray::{s, Array1, Array2, Axis},
    transforms::PipelineItem,
    unmixing::{bson_read, bson_write, AnnotatedOTDecomp, AnomLPOut, InitialBasis, OTDecomp},
};

use crate::components::{
    smoothed_sig, ExportOptions, IndexSelection, IndexSelectionUi, InitialParams,
    PeriodicCmapFreqs, PreviewedComponent, SubstepParameters, SummaryTable, TransportPlanVis,
};

static CLEANING_PIPELINE_INFO: &str =
    "The default pipeline (with adjusted parameters) should be sufficient for most use cases.
Take care with order of operations.";

static COMPS_IDXS_INFO: &str =
    "for the comps preview, enter indices as comma separated (no spaces), hit reset to apply";

thread_local! {
    pub static CMAP_FREQS: RefCell<PeriodicCmapFreqs> = Default::default();
    pub static KNOWN_BASES: RefCell<TaggedSignatures> = Default::default();
    pub static BACKGROUND_COMPUTE: RefCell<BackgroundComputeState> = Default::default();
}

pub fn compute_ongoing() -> bool {
    BACKGROUND_COMPUTE.with_borrow_mut(|bc| !matches!(bc, BackgroundComputeState::Ready))
}

fn preprocess_just_finished() -> bool {
    BACKGROUND_COMPUTE.with_borrow_mut(|bc| {
        let finished = match bc {
            BackgroundComputeState::CleaningAllData { jh } => jh.is_finished(),
            _ => false,
        };
        if finished {
            info!("Clean all finished");
            *bc = BackgroundComputeState::Ready;
        }
        finished
    })
}

#[derive(PartialEq)]
enum Incremental {
    Busy, // TODO: Should distinguish between active and in a different task?
    QueueNext,
    Finished,
}

fn iterations_compute_state() -> Incremental {
    BACKGROUND_COMPUTE.with_borrow_mut(|bc| {
        let state = match bc {
            BackgroundComputeState::IteratingDecomp {
                jh,
                iterations_remaining,
            } => match jh.is_finished() {
                true => {
                    *iterations_remaining -= 1;
                    match *iterations_remaining > 0 {
                        true => {
                            info!("Iteration complete (more to go).");
                            Incremental::QueueNext
                        }
                        false => {
                            info!("Iterations finished");
                            Incremental::Finished
                        }
                    }
                }
                false => Incremental::Busy, // active
            },
            _ => Incremental::Busy, // in a different task
        };
        if let Incremental::Finished = state {
            *bc = BackgroundComputeState::Ready;
        }
        state
    })
}

// TODO: Compute state overcomplicated?
fn anom_compute_state() -> Incremental {
    BACKGROUND_COMPUTE.with_borrow_mut(|bc| {
        let state = match bc {
            BackgroundComputeState::AnomLP { jho, queued_j } => {
                match jho.as_mut().unwrap().is_finished() {
                    true => {
                        _ = queued_j.pop();
                        match !queued_j.is_empty() {
                            true => {
                                info!("AnomLP finished for single mixture (more to go).");
                                Incremental::QueueNext
                            }
                            false => {
                                info!("AnomLP job complete");
                                Incremental::Finished
                            }
                        }
                    }
                    false => Incremental::Busy, // active
                }
            }
            _ => Incremental::Busy, // in a different task
        };
        state
    })
}

#[derive(Default)]
enum BackgroundComputeState {
    CleaningAllData {
        jh: JoinHandle<()>,
    },
    IteratingDecomp {
        jh: JoinHandle<()>,
        iterations_remaining: usize,
    },
    AnomLP {
        jho: Option<JoinHandle<(usize, AnomLPOut)>>,
        queued_j: Vec<usize>,
    },
    #[default]
    Ready,
}

pub fn update_global_bounds<'a>(sigs: impl Iterator<Item = &'a Signature>) {
    SMOOTHING_OPT.with_borrow_mut(|so| {
        so.set_max_span(sigs);
        so.update_xs();
    })
}

#[derive(Default)]
pub struct DecompGUIApp {
    dataset: Arc<Mutex<Dataset>>,
    cleaning_pipeline: Arc<Mutex<Vec<PipelineItem>>>,
    plot_preview: BTreeMap<String, CleaningIntermediate>,
    n_previewed: usize,
    init_params: InitialParams,
    otdecompui: OTDecompUI,
    /// number of times to iterate
    n_iterations: usize,
    show_cmap_edit: bool,
}

impl DecompGUIApp {
    pub fn new(
        data_dir: Option<PathBuf>,
        decomp_path: Option<PathBuf>,
        log_path: Option<PathBuf>,
    ) -> Self {
        let flo = match log_path {
            Some(log_path) => FileLogOption::FullPath { log_path },
            None => FileLogOption::DefaultTempDir {
                log_name: "ot_unmix_log".into(),
            },
        };
        setup_mixed_logger(flo);

        let mut n = Self {
            cleaning_pipeline: Arc::new(Mutex::new(default_pipeline())),
            n_previewed: 3,
            ..Default::default()
        };
        if let Some(fpath) = decomp_path {
            n.otdecompui.set_decomp(bson_read(fpath));
            info!("Decomp preloaded, skip to relevant section.")
        } else if let Some(dirname) = data_dir {
            n.load_data_from_dir(dirname);
            info!("Dataset preloaded, skip to relevant section.")
        }
        n
    }

    fn update_preview_data(&mut self) {
        self.plot_preview = self
            .dataset
            .lock()
            .unwrap()
            .loaded_data
            .iter()
            .take(self.n_previewed)
            .map(|(n, d)| (n.clone(), d.clone().into()))
            .collect();
    }

    fn load_data_from_dir(&mut self, dirname: PathBuf) {
        if let Ok(mut dataset) = self.dataset.try_lock() {
            dataset.load_dir(dirname);
        } else {
            error!("Dataset used in another thread, so cannot update right now.")
        }
        self.update_preview_data();
    }

    fn loading_and_preprocessing(&mut self, ui: &mut egui::Ui) {
        ui.columns(2, |columns| {
            let ui = &mut columns[0];
            if ui.button("load dataset").clicked() {
                if let Some(dirname) = rfd::FileDialog::new().set_directory(".").pick_folder() {
                    self.load_data_from_dir(dirname);
                }
            }

            if ui
                .button("clean preview subset")
                .on_hover_text("trial clean of currently viewed")
                .clicked()
            {
                // NOTE: Cleaning the preview set is performed in a ui blocking manner, however,
                // if cleaning just 3 or so datapoints is taking long enough to cause a stutter,
                // there is probably something wrong with the chosen operations.

                if self.plot_preview.is_empty() {
                    info!("No preview data is loaded, have you selected a dataset?");
                }

                self.update_preview_data(); // resetting
                let mut no_sigs = false;
                for (_, intermediate) in self.plot_preview.iter_mut() {
                    let pipeline = self.cleaning_pipeline.lock().unwrap();
                    intermediate.apply_tranform_pipeline(&pipeline);
                    no_sigs = no_sigs || intermediate.signature.sticks.is_empty();
                }
                if no_sigs {
                    error!("The last item in the cleaning pipeline should produce a signature.");
                }

                update_global_bounds(self.plot_preview.values().map(|ci| &ci.signature));
            }

            if ui.button("clean all").clicked() {
                let pipeline_arc = self.cleaning_pipeline.clone();
                let dataset_arc = self.dataset.clone();
                info!("Clean all started in background");
                BACKGROUND_COMPUTE.with_borrow_mut(|bc| {
                    *bc = BackgroundComputeState::CleaningAllData {
                        jh: spawn(move || {
                            let pipeline = pipeline_arc.lock().unwrap();
                            let mut dataset = dataset_arc.lock().unwrap();
                            dataset.clean_all(&pipeline);
                        }),
                    };
                });
            }

            if ui
                .button("export cleaned signatures")
                .on_hover_text_at_pointer(
                    "To export cleaned density, use XYAsSignature as final cleaning step.",
                )
                .clicked()
            {
                self.export_cleaned()
            }

            columns[1]
                .vertical(|ui| {
                    self.cleaning_pipeline.inspect_mut("Cleaning pipeline", ui);
                })
                .response
                .on_hover_text_at_pointer(CLEANING_PIPELINE_INFO);
        });
    }

    fn decomp_setup(&mut self, ui: &mut egui::Ui) {
        if self.otdecompui.aotdecomp.is_none() {
            if ui.button("Create decomp object").clicked() {
                if let Ok(dataset) = self.dataset.try_lock() {
                    if self.init_params.initial_basis_selection.is_empty() {
                        error!("Please select some initial basis options first, ordinarily two or more.");
                    } else {
                        let labeled_init_basis = self
                            .init_params
                            .initial_basis_selection
                            .iter()
                            .enumerate()
                            .map(|(i, s)| (format!("basis {}: {}", i + 1, s.desc()), s.into()))
                            .collect();
                        match into_decomp(
                            &dataset,
                            labeled_init_basis,
                            &self.init_params.substep_param,
                        ) {
                            Ok(adecomp) => self.otdecompui.set_decomp(adecomp),
                            Err(e) => error!("{e}"),
                        }
                    }
                } else {
                    warn!("Dataset is used in another thread (still cleaning?).")
                }
            }

            if ui.button("Load decomp").clicked() {
                if let Some(fpath) = rfd::FileDialog::new().set_directory(".").pick_file() {
                    self.otdecompui.set_decomp(bson_read(fpath));
                }
            }
        } else {
            ui.menu_button("Reset decomp", |ui| {
                ui.label("Are you sure? This will discard the current one.");
                if ui.button("Yes").clicked() {
                    self.otdecompui.clear();
                    ui.close_menu();
                }
                if ui.button("No").clicked() {
                    ui.close_menu();
                }
            });
        }

        match &mut self.otdecompui.aotdecomp {
            Some(adecomp) => {
                let mut start_iterations = false;
                if !compute_ongoing() {
                    let mut adecomp = adecomp.lock().unwrap();
                    if adecomp.decomp.substep_methods_are_set() {
                        ui.horizontal(|ui| {
                            if ui.button("Iterate").clicked() {
                                if adecomp.decomp.separation_only() {
                                    self.n_iterations = 1;
                                    info!("Basis is constant, multiple iterations not required, only performing separation.");
                                }
                                start_iterations = true;
                            };
                            self.n_iterations.inspect_mut("", ui);
                            ui.label("times")
                        });
                    } else {
                        let sp = &mut self.init_params.substep_param;
                        let decomp = &mut adecomp.decomp;
                        ui.label("Note: cannot resume iteration until configuring substep compute options (currently not serialised).");
                        sp.inspect_mut("substep_param", ui);
                        if ui.button("apply compute settings").clicked() {
                            decomp.avg_compute =
                                Some(sp.barycenter_compute(decomp.n_data, decomp.n_padding));
                            decomp.sep_compute = Some(sp.separation_compute(
                                decomp.n_bases,
                                decomp.n_data,
                                decomp.n_padding,
                            ));
                        }
                    }
                }

                if start_iterations {
                    let adecomp_arc = adecomp.clone();
                    BACKGROUND_COMPUTE.with_borrow_mut(|bc| {
                        *bc = BackgroundComputeState::IteratingDecomp {
                            jh: spawn(move || adecomp_arc.lock().unwrap().decomp.iterate(1)),
                            iterations_remaining: self.n_iterations,
                        }
                    });
                    ui.ctx().request_repaint_after(Duration::from_millis(2));
                }
                if iterations_compute_state() == Incremental::QueueNext {
                    let adecomp_arc = adecomp.clone();
                    BACKGROUND_COMPUTE.with_borrow_mut(|bc| {
                        if let BackgroundComputeState::IteratingDecomp { jh, .. } = bc {
                            *jh = spawn(move || adecomp_arc.lock().unwrap().decomp.iterate(1));
                        }
                    });
                    ui.ctx().request_repaint_after(Duration::from_millis(2));
                }
            }

            None => self.init_params.inspect_mut("decomp parameters", ui),
        };
    }

    fn export_cleaned(&self) {
        if let Ok(dataset) = self.dataset.try_lock() {
            if dataset.cleaned_data.is_empty() {
                warn!("No cleaned (non-preview) data, make sure to clean all.");
            } else if let Some(dir) = rfd::FileDialog::new().set_directory(".").pick_folder() {
                dataset.export_sigs(dir);
            }
        } else {
            warn!("Cannot export while dataset is in used on another thread.")
        }
    }
}

pub fn into_decomp(
    dataset: &Dataset,
    labeled_init_basis: Vec<(String, InitialBasis)>,
    substep_param: &SubstepParameters,
) -> Result<AnnotatedOTDecomp, String> {
    let n_data = dataset.cleaned_data.len();
    let n_bases = labeled_init_basis.len();

    if let Some(padding) = dataset
        .cleaned_data
        .values()
        .map(|sig| sig.size())
        .reduce(usize::max)
    {
        let data_name_map: Vec<_> = dataset.cleaned_data.keys().cloned().collect();
        let mixed_data = data_name_map
            .iter()
            .map(|k| dataset.cleaned_data[k].clone())
            .collect();

        let mut basis_name_map = vec![];
        let mut init_basis = vec![];
        for (name, prior) in labeled_init_basis {
            basis_name_map.push(name);
            init_basis.push(prior);
        }

        let avg_compute = substep_param.barycenter_compute(n_data, padding);
        let sep_compute = substep_param.separation_compute(n_bases, n_data, padding);
        let decomp = OTDecomp::new(mixed_data, init_basis, padding, avg_compute, sep_compute);

        Ok(AnnotatedOTDecomp {
            decomp,
            basis_name_map,
            data_name_map,
        })
    } else {
        Err("must clean data before creating decomposition".to_string())
    }
}

impl eframe::App for DecompGUIApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // let so_id = egui::Id::from("SmoothingOptions");
        // let so = ctx.memory(|m| m.data.get_temp::<SmoothingOptions>(so_id));
        // if let None = so {
        //     ctx.memory_mut(|m| m.data.insert_temp(so_id, SmoothingOptions::default()));
        // }

        // Global options
        TopBottomPanel::top("top pannel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                SMOOTHING_OPT.with_borrow_mut(|so| {
                    so.inspect_mut("gdes", ui);
                });

                ui.label("    ");
                self.show_cmap_edit.inspect_mut("show cmap edit", ui);
                if self.show_cmap_edit {
                    Window::new("cmap edit").show(ctx, |ui| {
                        CMAP_FREQS.with_borrow_mut(|cf| {
                            ui.vertical(|ui| {
                                cf.inspect_mut("periodic cmap frequencies", ui);
                            });
                        });
                    });
                }
            });

            LogsView.inspect_mut("Feedback:", ui);
        });

        CentralPanel::default().show(ctx, |ui| {
            ScrollArea::vertical().show(ui, |ui| {
                if !compute_ongoing() {
                    ui.collapsing("Data loading and preprocessing", |ui| {
                        self.loading_and_preprocessing(ui);
                    });
                } else {
                    // TODO: Only apply on cleaning finished
                    if preprocess_just_finished() {
                        if let Ok(dataset) = self.dataset.try_lock() {
                            KNOWN_BASES.with_borrow_mut(|basis_copy| {
                                basis_copy.clone_from(&dataset.cleaned_bases)
                            });
                            update_global_bounds(dataset.cleaned_data.values());
                        }
                    }
                }

                self.plot_preview.inspect("Previewed data", ui);

                ui.collapsing("Decomposition", |ui| {
                    self.decomp_setup(ui);
                    self.otdecompui.inspect_mut("decomp visualisation", ui);
                });
            });
        });

        if compute_ongoing() {
            // Ensure that lack of interaction with ui does not prevent us from issuing
            // more background tasks
            ctx.request_repaint_after(Duration::from_millis(2));
        }
    }
}

/// Wraps around base compute object ([ot_unmix_core::unmixing::AnnotatedOTDecomp]) to add gui only data.
#[derive(Default)]
pub struct OTDecompUI {
    pub aotdecomp: Option<Arc<Mutex<AnnotatedOTDecomp>>>,
    // visible
    previewed_iteration: usize,
    previewed_basis: BTreeMap<String, Signature>,
    previewed_components: BTreeMap<String, PreviewedComponent>,
    selected_idxs: IndexSelectionUi,
    table: SummaryTable,
    // LP
    // TODO: Monge transform visualisation?
    lp_vis: BTreeMap<String, TransportPlanVis>,
    lp_vis_mix: String,
    // export
    selected_export_dir: Option<PathBuf>,
    export_options: ExportOptions,
}

impl OTDecompUI {
    pub fn clear(&mut self) {
        self.aotdecomp = None;
        self.previewed_iteration = 0;
        self.previewed_basis = BTreeMap::new();
        self.previewed_components = BTreeMap::new();
    }

    fn update_preview_data(&mut self) {
        if self.aotdecomp.is_none() {
            return;
        }
        // TODO: Tidy repeated pattern?
        let mut aunmix = match self.aotdecomp.as_ref().unwrap().try_lock() {
            Ok(g) => g,
            Err(_) => {
                warn!("Cannot access decomp data while it is used by another thread.");
                return;
            }
        };
        let AnnotatedOTDecomp {
            decomp,
            basis_name_map,
            data_name_map,
        } = aunmix.deref_mut();
        let iter = &mut decomp.iterations[self.previewed_iteration];

        if self.selected_idxs.selection.items.is_empty() {
            self.selected_idxs.selection = IndexSelection {
                items: (0..decomp.n_bases).collect(),
            };
        }

        self.previewed_basis = (0..decomp.n_bases)
            .map(|i| (basis_name_map[i].clone(), iter.get_basis_sig(i)))
            .collect();

        self.table.seg_costs.clone_from(&iter.segmentation_costs);

        self.previewed_components = BTreeMap::new();
        let so = SMOOTHING_OPT.with_borrow(|so| so.clone());
        for j in self.selected_idxs.selection.items.iter().cloned() {
            let anom = decomp
                .get_anomaly(self.previewed_iteration, j, 1e-3)
                .map(|anom| {
                    let mut anom_density = Array1::zeros(so.size);
                    smoothed_sig(
                        decomp.xc.slice(s![j, ..]).as_slice().unwrap(),
                        anom.as_slice().unwrap(),
                        so.xs.as_slice().unwrap(),
                        anom_density.as_slice_mut().unwrap(),
                        so.kernel_std,
                    );
                    anom_density
                });
            let mut comp = PreviewedComponent {
                xs: so.xs.clone(),
                wbts: Array2::zeros((decomp.n_bases, so.size)),
                anom,
            };
            for i in 0..decomp.n_bases {
                smoothed_sig(
                    decomp.xc.slice(s![j, ..]).as_slice().unwrap(),
                    decomp.iterations[self.previewed_iteration]
                        .wbt
                        .slice(s![j, i, ..])
                        .as_slice()
                        .unwrap(),
                    so.xs.as_slice().unwrap(),
                    comp.wbts.slice_mut(s![i, ..]).as_slice_mut().unwrap(),
                    so.kernel_std,
                )
            }
            self.previewed_components
                .insert(data_name_map[j].clone(), comp);
        }

        self.table.abundances = decomp.iterations[self.previewed_iteration].abundances_all();
        self.table.anom_amounts =
            Array1::ones(decomp.n_data) - self.table.abundances.sum_axis(Axis(1));
        self.table.basis_name_map.clone_from(basis_name_map);
        self.table.data_name_map.clone_from(data_name_map);
    }

    pub fn set_decomp(&mut self, aotdecomp: AnnotatedOTDecomp) {
        let sigs: Vec<_> = (0..aotdecomp.decomp.n_data)
            .map(|i| {
                Signature::from(pairs(
                    aotdecomp.decomp.xc.slice(s![i, ..]).as_slice().unwrap(),
                    aotdecomp.decomp.wc.slice(s![i, ..]).as_slice().unwrap(),
                ))
            })
            .collect();
        update_global_bounds(sigs.iter());
        self.aotdecomp = Some(Arc::new(Mutex::new(aotdecomp)));
        self.update_preview_data();
    }

    fn set_lp_vis(&mut self, j: usize, lp_out: AnomLPOut) {
        let mut aunmix = match self.aotdecomp.as_ref().unwrap().try_lock() {
            Ok(g) => g,
            Err(_) => {
                warn!("Cannot access decomp data while it is used by another thread.");
                return;
            }
        };
        let AnnotatedOTDecomp {
            decomp,
            data_name_map,
            ..
        } = aunmix.deref_mut();

        let (lp_vars, _) = lp_out;
        self.lp_vis_mix.clone_from(&data_name_map[j]);
        self.lp_vis = (0..decomp.n_bases)
            .map(|i| {
                let plan = lp_vars[&format!("basis_{i}_plan")]
                    .clone()
                    .into_shape([decomp.n_padding, decomp.n_padding])
                    .unwrap();
                let xb = decomp.iterations[self.previewed_iteration]
                    .xb
                    .slice(s![i, ..])
                    .to_owned();
                let xbt = decomp.xc.slice(s![j, ..]).to_owned();
                (
                    format!("Basis {} Kantorovich plan", i + 1),
                    TransportPlanVis::new(plan, xb, xbt),
                )
            })
            .collect();
    }

    /// Assuming that start signals were only issued if compute was ready
    /// return true if compute was issued
    fn handle_anom_lp_compute_start(&mut self) -> bool {
        let max_avg_displacement = self.table.anom_params.max_avg_displacement; // TODO: May have
                                                                                // changed?
        let pi = self.previewed_iteration; // TODO: Double check that this would not change when
                                           // running incrementally, if yes then need to record in
                                           // compute state instead

        if let Some(j) = self.table.anom_decomp_idx.take() {
            info!("AnomLP job started in background for mixture {j}");
            // TODO: check param scaling appropriateness + add tooltip?
            let old_cost = self.table.seg_costs[j].max(1.0);
            // Just needs to be "large"/kept on a significant scale
            let anom_penalty = 10.0 * old_cost;
            dbg!(&self.table.anom_params.max_avg_displacement, anom_penalty);
            // feedback_info(format!("started anom lp for mixture {j}"));
            let aotdecomp = self.aotdecomp.as_ref().unwrap().clone();
            BACKGROUND_COMPUTE.with_borrow_mut(|bc| {
                *bc = BackgroundComputeState::AnomLP {
                    jho: Some(spawn(move || {
                        let mut aunmix = aotdecomp.lock().expect(
                            "Cannot access decomp data while it is used by another thread.",
                        );
                        let AnnotatedOTDecomp { decomp, .. } = aunmix.deref_mut();
                        let lp_out = decomp
                            .anom_lp(pi, j, max_avg_displacement, anom_penalty)
                            .unwrap();
                        (j, lp_out)
                    })),
                    queued_j: vec![j],
                }
            });

            return true;
        }

        // compute all issued
        if self.table.anom_decomp_all {
            info!("AnomLP job started in background for all mixtures");
            let n_data = self.table.seg_costs.shape()[0];
            for j in 0..n_data {
                // TODO: check param scaling appropriateness + add tooltip?
                let old_cost = self.table.seg_costs[j].max(1.0);
                // Just needs to be "large"/kept on a significant scale
                let anom_penalty = 10.0 * old_cost;
                dbg!(&self.table.anom_params.max_avg_displacement, anom_penalty);

                let aotdecomp = self.aotdecomp.as_ref().unwrap().clone();
                BACKGROUND_COMPUTE.with_borrow_mut(|bc| {
                    *bc = BackgroundComputeState::AnomLP {
                        jho: Some(spawn(move || {
                            let mut aunmix = aotdecomp.lock().expect(
                                "Cannot access decomp data while it is used by another thread.",
                            );
                            let AnnotatedOTDecomp { decomp, .. } = aunmix.deref_mut();
                            let lp_out = decomp
                                .anom_lp(pi, 0, max_avg_displacement, anom_penalty)
                                .unwrap();
                            (0, lp_out)
                        })),
                        queued_j: (0..n_data).rev().collect(),
                    }
                });
                self.table.anom_decomp_all = false;
            }

            return true;
        }

        false
    }

    fn export_current(&self) {
        let dir = self.selected_export_dir.as_ref().unwrap();

        let mut aunmix = self.aotdecomp.as_ref().unwrap().try_lock().unwrap();
        let AnnotatedOTDecomp {
            decomp,
            basis_name_map,
            data_name_map,
        } = aunmix.deref_mut();
        let iter = &decomp.iterations[self.previewed_iteration];

        // TODO: parallel or async save?
        if self.export_options.basis_signatures {
            let _ = on_dir_created(dir.join("basis patterns"), |basis_dir| {
                for i in 0..decomp.n_bases {
                    iter.get_basis_sig(i)
                        .save_to_file(basis_dir, basis_name_map[i].clone())
                }
            });
        }

        if self.export_options.component_signatures {
            let _ = on_dir_created(dir.join("component patterns"), |comp_dir| {
                for j in 0..decomp.n_data {
                    let positions = decomp.xc.slice(s![j, ..]);
                    for i in 0..decomp.n_bases {
                        // TODO: less messy filename?
                        let name =
                            format!("{}_{}", data_name_map[j].clone(), basis_name_map[i].clone());
                        iter.get_comp_sig(i, j, positions.as_slice().unwrap())
                            .save_to_file(comp_dir, name)
                    }
                }
            });
        }

        if self.export_options.basis_kdes {
            let so = SMOOTHING_OPT.with_borrow(|so| so.clone());
            let _ = on_dir_created(dir.join("basis pattern kdes"), |basis_kde_dir| {
                for i in 0..decomp.n_bases {
                    iter.get_basis_sig(i)
                        .estimate_density(so.xs.as_slice().unwrap(), so.kernel_std)
                        .save_to_file(basis_kde_dir, basis_name_map[i].clone())
                }
            });
        }

        if self.export_options.component_kdes {
            let so = SMOOTHING_OPT.with_borrow(|so| so.clone());
            let _ = on_dir_created(dir.join("component pattern kdes"), |comp_kde_dir| {
                for j in 0..decomp.n_data {
                    let positions = decomp.xc.slice(s![j, ..]);
                    for i in 0..decomp.n_bases {
                        // TODO: less messy filename?
                        let name =
                            format!("{}_{}", data_name_map[j].clone(), basis_name_map[i].clone());
                        iter.get_comp_sig(i, j, positions.as_slice().unwrap())
                            .estimate_density(so.xs.as_slice().unwrap(), so.kernel_std)
                            .save_to_file(comp_kde_dir, name)
                    }
                }
            });
        }

        if self.export_options.abundances {
            // TODO: Check overwrite?
            let anom_csv = {
                // TODO: Check names do not contain ","
                let mut table = vec![once(String::new())
                    .chain(basis_name_map.iter().cloned())
                    .join(",")];
                for j in 0..decomp.n_data {
                    table.push(
                        once(data_name_map[j].clone())
                            .chain(
                                (0..decomp.n_bases)
                                    .map(|i| format!("{:.3}", self.table.abundances[[j, i]])),
                            )
                            .join(","),
                    );
                }
                table.join("\n")
            };
            _ = fs::write(dir.join("abundances.csv"), anom_csv);
        }
    }
}

impl EguiInspect for OTDecompUI {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, _label: &str, ui: &mut egui::Ui) {
        if let Some(decomp) = &self.aotdecomp {
            if ui.button("Save decomposition").clicked() {
                if let Some(fpath) = rfd::FileDialog::new().set_directory(".").save_file() {
                    match decomp.try_lock() {
                        Ok(decomp) => bson_write(&*decomp, fpath),
                        Err(e) => warn!("Mutex unlock fail: {e}"),
                    }
                }
            }
        }
        ui.collapsing("Iterations preview", |ui| {
            if self.aotdecomp.is_some() {
                let mut selected_iter_changed = false;
                let mut reset_btn = false;

                let n: Option<usize> = self
                    .aotdecomp
                    .as_ref()
                    .unwrap()
                    .try_lock()
                    .ok()
                    .map(|g| g.decomp.iterations.len());

                ui.horizontal(|ui| {
                    // cannot change iteration data while decomp object used by other thread
                    if let (Some(n), false) = (n, compute_ongoing()) {
                        ui.label("Iteration to preview:");
                        selected_iter_changed = ui
                            .add(Slider::new(&mut self.previewed_iteration, 0..=n - 1))
                            .changed();
                    }
                });

                ui.horizontal(|ui| {
                    ui.horizontal(|ui| {
                        self.selected_idxs
                            .inspect_mut("preview components for mixture indices", ui)
                    })
                    .response
                    .on_hover_text_at_pointer(COMPS_IDXS_INFO);

                    // TODO: could have this update live like signature previews?
                    reset_btn = ui.button("Reset previewed comps").clicked();
                });

                ui.label(format!("Iteration {} preview:", self.previewed_iteration).as_str());

                ui.columns(2, |columns| {
                    self.previewed_basis.inspect("Basis", &mut columns[0]);

                    self.previewed_components
                        .inspect("Components", &mut columns[1]);
                });

                self.table.inspect_mut("", ui);
                self.handle_anom_lp_compute_start();

                let acs = anom_compute_state();
                let pi = self.previewed_iteration;
                if acs == Incremental::QueueNext || acs == Incremental::Finished {
                    BACKGROUND_COMPUTE.with_borrow_mut(|bc| {
                        if let BackgroundComputeState::AnomLP { jho: jh, .. } = bc {
                            let (j, lp_out) = jh.take().unwrap().join().unwrap();
                            // Handle result:
                            let mut aunmix = self.aotdecomp.as_ref().unwrap().lock().expect(
                                "Cannot access decomp data while it is used by another thread.",
                            );
                            let AnnotatedOTDecomp { decomp, .. } = aunmix.deref_mut();
                            decomp.unpack_lp_vars(pi, j, &lp_out);
                            drop(aunmix);
                            self.set_lp_vis(j, lp_out);
                            self.update_preview_data();
                        }
                    });
                }
                if acs == Incremental::QueueNext {
                    let aotdecomp = self.aotdecomp.as_ref().unwrap().clone();
                    BACKGROUND_COMPUTE.with_borrow_mut(|bc| {
                        if let BackgroundComputeState::AnomLP { jho, queued_j } = bc {
                            let j = *queued_j.last().unwrap();
                            let max_avg_displacement = self.table.anom_params.max_avg_displacement;
                            let old_cost = self.table.seg_costs[j].max(1.0);
                            // TODO: Old cost may not be unconstrained cost after first run...
                            let anom_penalty = 10.0 * old_cost;
                            *jho = Some(spawn(move || {
                                let mut aunmix = aotdecomp.lock().expect(
                                    "Cannot access decomp data while it is used by another thread.",
                                );
                                let AnnotatedOTDecomp { decomp, .. } = aunmix.deref_mut();
                                let lp_out = decomp
                                    .anom_lp(pi, j, max_avg_displacement, anom_penalty)
                                    .unwrap();
                                (j, lp_out)
                            }));
                        }
                    });
                    ui.ctx().request_repaint_after(Duration::from_millis(2));
                }
                if acs == Incremental::Finished {
                    BACKGROUND_COMPUTE.with_borrow_mut(|bc| {
                        *bc = BackgroundComputeState::Ready;
                    });
                }

                ui.menu_button("export iteration data", |ui| {
                    if ui.button("select export dir").clicked() {
                        self.selected_export_dir =
                            rfd::FileDialog::new().set_directory(".").pick_folder();
                    }
                    match &self.selected_export_dir {
                        Some(dir) => {
                            format!("Selected {dir:?} for export.").inspect("", ui);

                            self.export_options.inspect_mut("export options", ui);
                            "Note, will use global density estimation settings for kde options."
                                .inspect("", ui);
                            if ui.button("export").clicked() {
                                self.export_current();
                                ui.close_menu();
                            }
                        }
                        None => {
                            ui.label("No export dir selected.");
                        }
                    };
                });

                if !self.lp_vis.is_empty() {
                    self.lp_vis.inspect(
                        format!("Results of last (anom) LP ({})", self.lp_vis_mix).as_str(),
                        ui,
                    );
                }

                if selected_iter_changed || reset_btn {
                    self.update_preview_data();
                }
            } else {
                ui.label("Initialise a decomp object above first.");
            }
        });
    }
}
