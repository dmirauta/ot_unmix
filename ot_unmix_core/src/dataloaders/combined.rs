use super::{plaintext::PlaintextLoader, DataLoader};
use log::warn;
use std::collections::{HashMap, HashSet};

/// Applies multiple [DataLoader] implementors, to handle different file types. Does not expect
/// overlap in handled types.
pub struct CombinedLoader {
    loaders: Vec<Box<dyn DataLoader>>,
}

impl Default for CombinedLoader {
    fn default() -> Self {
        Self {
            loaders: vec![Box::new(PlaintextLoader)],
        }
    }
}

impl DataLoader for CombinedLoader {
    fn handled_extensions(&self) -> HashSet<&'static str> {
        let mut exts = HashSet::new();
        for loader in self.loaders.iter() {
            let new_exts = loader.handled_extensions();
            for ext in exts.intersection(&new_exts) {
                warn!("Two data handlers registered for {ext}.");
            }
            exts.extend(new_exts);
        }
        exts
    }
    fn parse_relevant_files(
        &self,
        all_files: &[std::fs::DirEntry],
    ) -> crate::dataset::TaggedDatapoints {
        let mut data = HashMap::new();
        for loader in self.loaders.iter() {
            // TODO: Handle loaders with overlapping exts?
            data.extend(loader.parse_relevant_files(all_files));
        }
        data
    }
}
