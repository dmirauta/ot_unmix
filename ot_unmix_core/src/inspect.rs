//! Defines inspect methods for core structs where possible and inspectable wrappers otherwise
use crate::data::{sort_sticks, CleaningIntermediate, Signature, Stick, XYData};
use egui::{Color32, Stroke};
use egui_inspect::EguiInspect;
use egui_plot::{Legend, Line, Plot, PlotBounds, PlotPoints, PlotUi, Polygon};
use ndarray::Array1;
use std::{cell::RefCell, fs};

static GDES_DESC: &str = "Densities are generated for visualisation purposes, these settings do not have an effect on computation.
Stick width may be increased for easier signature editing (in sketched bases).";

thread_local! {
    pub static SMOOTHING_OPT: RefCell<SmoothingOptions> = Default::default();
}

#[derive(Clone)]
pub struct SmoothingOptions {
    pub xs: Array1<f64>,
    pub xmin: f64,
    pub xmax: f64,
    pub size: usize,
    pub kernel_std: f64,
    pub stick_width: f64,
}

impl Default for SmoothingOptions {
    fn default() -> Self {
        let mut new = Self {
            xs: Default::default(),
            xmin: 0.0,
            xmax: 1.0,
            size: 1000,
            kernel_std: 0.1,
            stick_width: 0.02,
        };
        new.update_xs();
        new
    }
}

impl SmoothingOptions {
    pub fn set_max_span<'a>(&mut self, sigs: impl Iterator<Item = &'a Signature>) {
        self.xmin = f64::INFINITY;
        self.xmax = -f64::INFINITY;
        for sig in sigs.filter(|s| !s.sticks.is_empty()) {
            self.xmin = sig.sticks.first().unwrap().position.min(self.xmin);
            self.xmax = sig.sticks.last().unwrap().position.max(self.xmax);
        }
    }

    pub fn update_xs(&mut self) {
        if self.xmin < self.xmax {
            self.xs = Array1::linspace(
                self.xmin - 2.0 * self.kernel_std,
                self.xmax + 2.0 * self.kernel_std,
                self.size,
            );
        }
    }
}

macro_rules! slider_log_dyn_bounds {
    ($ui:ident, $name: expr, $mut_ref: expr, $min: expr, $max: expr, $on_update: expr) => {
        $ui.horizontal(|ui| {
            ui.label($name);
            if ui
                .add(egui::Slider::new($mut_ref, $min..=$max).logarithmic(true))
                .changed()
            {
                $on_update;
            }
        });
    };
    ($ui:ident, $name: expr, $mut_ref: expr, $min: expr, $max: expr) => {
        slider_log_dyn_bounds!($ui, $name, $mut_ref, $min, $max, {});
    };
}

impl EguiInspect for SmoothingOptions {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, _label: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Density estimation settings:        ");
            slider_log_dyn_bounds!(ui, "grid size", &mut self.size, 50, 2000, self.update_xs());
            let r = self.xs.last().unwrap() - self.xs.first().unwrap();
            slider_log_dyn_bounds!(
                ui,
                "kernel std",
                &mut self.kernel_std,
                r / 1000.0,
                r / 8.0,
                self.update_xs()
            );
            slider_log_dyn_bounds!(
                ui,
                "stick width",
                &mut self.stick_width,
                self.kernel_std / 100.0,
                4.0 * self.kernel_std,
                self.update_xs()
            );
        })
        .response
        .on_hover_text_at_pointer(GDES_DESC);
    }
}

/// generic, shared plot setup
pub fn setup_plot<'a, 'b>(label: &'a str, ui: &'b mut egui::Ui, show_label: bool) -> Plot<'a> {
    if show_label {
        ui.label(label);
    }
    Plot::new(label)
        .view_aspect(2.0)
        .custom_y_axes(vec![])
        .allow_drag(false)
        .legend(Legend::default())
}

fn white_box(poly: Vec<[f64; 2]>) -> Polygon {
    Polygon::new(poly)
        .fill_color(Color32::WHITE)
        .stroke(Stroke::new(1.0, Color32::WHITE)) // can use Stroke::NONE, but this increments
                                                  // line color
}

impl Stick {
    fn poly(&self, scale: f64, stick_width: f64) -> Vec<[f64; 2]> {
        let l = self.position - stick_width;
        let r = self.position + stick_width;
        let h = self.weight * scale;
        vec![[l, 0.0], [r, 0.0], [r, h], [l, h]]
    }

    fn make_plot_poly(&self, scale: f64, stick_width: f64) -> Polygon {
        white_box(self.poly(scale, stick_width))
    }

    fn make_plot_poly_transposed(&self, scale: f64, stick_width: f64) -> Polygon {
        white_box(
            self.poly(scale, stick_width)
                .into_iter()
                .map(|[x, y]| [y, x])
                .collect(),
        )
    }

    fn handle_interaction(
        &mut self,
        plot_ui: &PlotUi,
        request_delete: &mut bool,
        draggable: bool,
        stick_delta: f64,
    ) -> bool {
        let l = self.position - stick_delta;
        let r = self.position + stick_delta;
        if let Some(pointer) = plot_ui.pointer_coordinate() {
            if l < pointer.x && pointer.x < r && 0.0 < pointer.y && pointer.y < self.weight {
                if plot_ui.response().secondary_clicked() {
                    *request_delete = true;
                }
                if draggable {
                    self.position += plot_ui.pointer_coordinate_drag_delta().x as f64;
                    self.weight += plot_ui.pointer_coordinate_drag_delta().y as f64;
                    if self.weight < 0.0 {
                        *request_delete = true;
                    }
                    return true;
                }
            }
        }
        false
    }
}

impl Signature {
    /// for bypassing own inspect method
    pub fn plot(&self, plot_ui: &mut PlotUi, so: &SmoothingOptions) {
        let sig_scale = (self.sticks.len() as f64) / self.total_weight();
        for stick in self.sticks.iter() {
            plot_ui.polygon(
                stick
                    .make_plot_poly(sig_scale, so.stick_width)
                    .name("signature"),
            );
        }
        self.plot_estimated_density(plot_ui, 1.0, so);
    }

    pub fn plot_transposed(&self, plot_ui: &mut PlotUi, so: &SmoothingOptions) {
        let sig_scale = (self.sticks.len() as f64) / self.total_weight();
        for stick in self.sticks.iter() {
            plot_ui.polygon(
                stick
                    .make_plot_poly_transposed(sig_scale, so.stick_width)
                    .name("signature"),
            );
        }
        // TODO: could set up transposed density plot, but ideally need a generic transposing axis
        // self.plot_estimated_density(plot_ui, 1.0);
    }

    pub fn plot_estimated_density(&self, plot_ui: &mut PlotUi, scale: f64, so: &SmoothingOptions) {
        let sig_kde = self.estimate_density(so.xs.as_slice().unwrap(), so.kernel_std);

        let sig_kde_scale = scale / sig_kde.y_mean_height();
        sig_kde.plot("estimated signature density", plot_ui, sig_kde_scale);
    }
}

fn get_so(_ui: &mut egui::Ui) -> SmoothingOptions {
    // let so_id = egui::Id::from("SmoothingOptions");
    // _ui.memory(|m| m.data.get_temp::<SmoothingOptions>(so_id))
    //     .unwrap()
    SMOOTHING_OPT.with_borrow(|so| so.clone())
}

impl EguiInspect for Signature {
    fn inspect(&self, label: &str, ui: &mut egui::Ui) {
        let so = get_so(ui);
        setup_plot(label, ui, true).show(ui, |plot_ui| {
            self.plot(plot_ui, &so);
        });
    }

    fn inspect_mut(&mut self, label: &str, ui: &mut egui::Ui) {
        let so = get_so(ui);
        setup_plot(label, ui, true).show(ui, |plot_ui| {
            plot_ui.set_plot_bounds(PlotBounds::from_min_max([so.xmin, 0.0], [so.xmax, 1.0]));

            let mut to_delete = None;
            let mut drag_occuring = false; // makes sure we only drag one, but not which one

            for (i, stick) in self.sticks.iter_mut().enumerate() {
                plot_ui.polygon(stick.make_plot_poly(1.0, so.stick_width).name("signature"));
                let mut delete_request = false;
                drag_occuring = drag_occuring
                    | stick.handle_interaction(
                        plot_ui,
                        &mut delete_request,
                        !drag_occuring,
                        so.stick_width,
                    );
                if delete_request {
                    to_delete = Some(i);
                }
            }

            if let Some(i) = to_delete {
                self.sticks.remove(i);
            } else if plot_ui.response().secondary_clicked() {
                if let Some(pos) = plot_ui.pointer_coordinate() {
                    if pos.y > 0.0 {
                        self.sticks.push([pos.x, pos.y].into());
                    }
                }
            }

            // ensure sticks are kept sorted
            sort_sticks(&mut self.sticks);

            let sig_mean = self.total_weight() / (self.sticks.len() as f64);
            self.plot_estimated_density(plot_ui, sig_mean, &so);
        });
    }
}

impl XYData {
    fn plot(&self, label: &str, plot_ui: &mut PlotUi, scale: f64) {
        if let XYData::Loose(xy) = self {
            let new_xy = xy.iter().map(|[x, y]| [*x, *y * scale]).collect();
            let l = Line::new(PlotPoints::new(new_xy));
            plot_ui.line(l.name(label));
        }
    }
}

impl EguiInspect for XYData {
    fn inspect(&self, label: &str, ui: &mut egui::Ui) {
        setup_plot(label, ui, true).show(ui, |plot_ui| {
            self.plot(label, plot_ui, 1.0);
        });
    }

    fn inspect_mut(&mut self, _label: &str, _ui: &mut egui::Ui) {
        todo!();
    }
}

impl EguiInspect for CleaningIntermediate {
    fn inspect(&self, label: &str, ui: &mut egui::Ui) {
        ui.label(label);

        let so = get_so(ui);
        ui.columns(2, |columns| {
            setup_plot(label, &mut columns[0], false).show(&mut columns[0], |plot_ui| {
                self.orig.plot(
                    format!("original xy ({} points)", self.orig.size()).as_str(),
                    plot_ui,
                    1.0,
                );
                self.baseline.plot("total baseline", plot_ui, 1.0);

                for (label, curve) in self.extra_curves.iter() {
                    curve.plot(label.as_str(), plot_ui, 1.0);
                }
            });

            setup_plot(format!("{label} cleaned").as_str(), &mut columns[1], false).show(
                &mut columns[1],
                |plot_ui| {
                    let cleaned_scale = 1.0 / self.cleaned.y_mean_height();
                    self.cleaned.plot(
                        format!("cleaned xy ({} points)", self.cleaned.size()).as_str(),
                        plot_ui,
                        cleaned_scale,
                    );

                    self.signature.plot(plot_ui, &so);
                },
            );
        });

        ui.separator();
    }

    fn inspect_mut(&mut self, label: &str, ui: &mut egui::Ui) {
        self.inspect(label, ui);
    }
}

pub fn code_edit(code: &mut String, ui: &mut egui::Ui) {
    let mut layouter = |ui: &egui::Ui, string: &str, wrap_width: f32| {
        let mut layout_job = egui_extras::syntax_highlighting::highlight(
            ui.ctx(),
            &Default::default(),
            string,
            "rust",
        );
        layout_job.wrap.max_width = wrap_width;
        ui.fonts(|f| f.layout_job(layout_job))
    };

    egui::ScrollArea::vertical()
        .max_height(400.0)
        .show(ui, |ui| {
            ui.add(
                egui::TextEdit::multiline(code)
                    .code_editor()
                    .desired_rows(5)
                    .desired_width(f32::INFINITY)
                    .min_size(egui::vec2(300.0, 50.0))
                    .layouter(&mut layouter),
            );
        });

    ui.horizontal(|ui| {
        ui.label("Source: ");
        if ui.button("Save").clicked() {
            if let Some(path_buf) = rfd::FileDialog::new().save_file() {
                fs::write(path_buf, code.as_str()).unwrap();
            }
        }
        if ui.button("Load").clicked() {
            if let Some(path_buf) = rfd::FileDialog::new().pick_file() {
                if let Ok(_code) = fs::read_to_string(path_buf) {
                    *code = _code;
                }
            }
        }
    });
}
