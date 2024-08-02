use egui_inspect::{EguiInspect, egui};
use log::error;
use rhai::{
    packages::{
        ArithmeticPackage, BasicArrayPackage, BasicIteratorPackage, BasicMathPackage,
        LanguageCorePackage, LogicPackage, Package,
    },
    Dynamic, Engine, Scope, AST,
};

use crate::inspect::code_edit;

use super::{data::XYData, transforms::XYTransform};

macro_rules! register_packages {
    ($engine: ident; $($t: ident),+) => {
       $($t::new().register_into_engine(&mut $engine);)*
    };
}

// TODO: Limit how often feedback can be emitted and allow runtime errors again?
pub struct RhaiCommon {
    pub engine: Engine,
    pub ast: Option<AST>,
    pub parsed_code: String,
    pub parse_feedback: String,
}

impl Default for RhaiCommon {
    fn default() -> Self {
        let mut engine = Engine::new_raw();
        register_packages!(engine; LanguageCorePackage, ArithmeticPackage, BasicIteratorPackage, 
            LogicPackage, BasicMathPackage, BasicArrayPackage);

        Self {
            engine,
            ast: Default::default(),
            parsed_code: Default::default(),
            parse_feedback: Default::default(),
        }
    }
}

pub struct RhaiTransform {
    pub code: String,
    pub rc: RhaiCommon,
}

impl Default for RhaiTransform {
    fn default() -> Self {
        Self {
            code: include_str!("default.rh").to_string(),
            rc: Default::default(),
        }
    }
}

impl XYTransform for RhaiTransform {
    fn apply(&self, input: &mut super::data::CleaningIntermediate) {
        if let XYData::Placeholder = input.cleaned {
            input.cleaned = input.orig.clone();
        }
        if let XYData::Loose(xy) = &mut input.cleaned {
            let input: Vec<Dynamic> = xy
                .iter()
                .map(|&[x, y]| {
                    Dynamic::from_array(rhai::Array::from(vec![
                        Dynamic::from_float(x),
                        Dynamic::from_float(y),
                    ]))
                })
                .collect();

            let output_res = match self.rc.ast.as_ref() {
                Some(ast) => self.rc.engine.call_fn::<Vec<Dynamic>>(
                &mut Scope::new(),
                ast,
                "transform",
                (input,),
            ).map_err(|e| e.to_string()),
                None => Err(String::from("Rhai code was not valid, check parse feedback in transform params.")),
            };

            match output_res {
                Ok(output) => {
                    *xy = output
                        .into_iter()
                        .map(|prd| {
                            let pr = prd.into_array().unwrap();
                            [pr[0].as_float().unwrap(), pr[1].as_float().unwrap()]
                        })
                        .collect();
                }
                Err(e) => error!("Rhai run error: {e:?}"),
            }
        }
    }
}

impl EguiInspect for RhaiTransform {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, _label: &str, ui: &mut egui::Ui) {
        ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
            ui.label("Visit the Rhai language reference");
            ui.hyperlink_to(
                "scripting section",
                "https://rhai.rs/book/language/comments.html",
            );
            ui.label("for transform syntax.");
        });
        match self.rc.parse_feedback.is_empty() {
            true => {ui.label("Parse feedback: None (looks good)");},
            false => self.rc.parse_feedback.inspect("Parse feedback", ui),
        }
        code_edit(&mut self.code, ui);
        if self.rc.parsed_code != self.code {
            match self.rc.engine.compile(self.code.as_str()) {
                Ok(ast) => {
                    self.rc.ast = Some(ast);
                    self.rc.parse_feedback = String::new();
                }
                Err(e) => {
                    self.rc.ast = None;
                    self.rc.parse_feedback = format!("Rhai parse error: {e:?}"); 
                },
            }
            self.rc.parsed_code.clone_from(&self.code);
        }
    }
}
