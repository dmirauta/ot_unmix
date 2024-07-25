use std::{collections::HashMap, fs, path::PathBuf};

use rayon::prelude::*;

use super::{
    data::{CleaningIntermediate, Signature, XYData},
    transforms::PipelineItem,
};
use log::{error, info, warn};

pub type TaggedDatapoints = HashMap<String, XYData>;
pub type TaggedSignatures = HashMap<String, Signature>;

pub fn sigs_from_loaded(
    loaded: &TaggedDatapoints,
    cleaning_ops: &[PipelineItem],
) -> TaggedSignatures {
    loaded
        .iter()
        .par_bridge()
        .map(|(name, dat)| {
            let mut intermediate: CleaningIntermediate = dat.clone().into();
            intermediate.apply_tranform_pipeline(cleaning_ops);
            intermediate.signature.normalise();
            (name.clone(), intermediate.signature)
        })
        .collect()
}

#[allow(clippy::result_unit_err)]
pub fn on_dir_created(dir: PathBuf, f: impl Fn(&PathBuf)) -> Result<(), ()> {
    if let Err(e) = fs::create_dir(&dir) {
        error!("{dir:?} may already exist, will not overwrite. ({e})");
        return Err(());
    }
    f(&dir);
    Ok(())
}

#[derive(Default)]
pub struct Dataset {
    pub loaded_data: TaggedDatapoints,
    pub cleaned_data: TaggedSignatures,
    pub loaded_bases: TaggedDatapoints,
    pub cleaned_bases: TaggedSignatures,
}

impl Dataset {
    pub fn load_dir(&mut self, dir: PathBuf) {
        let mixed_dir = dir.join("mixed patterns");
        let basis_dir = dir.join("basis patterns");

        if !mixed_dir.exists() {
            error!("No \"mixed patterns\" folder in {dir:?}.");
            return;
        }

        self.loaded_data = XYData::get_all_in_dir(mixed_dir);
        self.cleaned_data = HashMap::new();

        if !basis_dir.exists() {
            warn!("No \"basis patterns\" folder in {dir:?} (though not strictly required).");
            return;
        }

        self.loaded_bases = XYData::get_all_in_dir(basis_dir);
        self.cleaned_bases = HashMap::new();

        match (self.loaded_data.len(), self.loaded_bases.len()) {
            (0, _) => error!("No mixed data found in {dir:?}."),
            (n, m) => info!("{n} mixed patterns and {m} bases loaded from {dir:?}."),
        }
    }
    pub fn clean_all(&mut self, cleaning_pipeline: &[PipelineItem]) {
        // TODO: Should preprocess in background
        self.cleaned_bases = sigs_from_loaded(&self.loaded_bases, cleaning_pipeline);
        self.cleaned_data = sigs_from_loaded(&self.loaded_data, cleaning_pipeline);
    }
    pub fn export_sigs(&self, dir: PathBuf) {
        let res = on_dir_created(dir.join("basis patterns"), |basis_dir| {
            for (name, sig) in self.cleaned_bases.iter() {
                sig.save_to_file(basis_dir, name);
            }
        });
        if res.is_err() {
            return;
        }
        let _ = on_dir_created(dir.join("mixed patterns"), |mix_dir| {
            for (name, sig) in self.cleaned_data.iter() {
                sig.save_to_file(mix_dir, name);
            }
        });
    }
}

pub fn default_pipeline() -> Vec<PipelineItem> {
    vec![
        PipelineItem::XLimit {
            transform: Default::default(),
        },
        PipelineItem::ConvexBaseline {
            transform: Default::default(),
        },
        PipelineItem::SlidingEllipseBaseline {
            transform: Default::default(),
        },
        PipelineItem::XYToSignature {
            transform: Default::default(),
        },
    ]
}
