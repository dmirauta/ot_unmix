use std::collections::{HashMap, HashSet};

use log::warn;

use super::{plaintext::PlaintextLoader, traits::DataLoader};

/// Applies multiple [DataLoader] implementors, to handle different file types. Does not expect
/// overlap in handled types.
pub struct CombinedLoader {
    loaders: Vec<Box<dyn DataLoader>>,
}

impl Default for CombinedLoader {
    fn default() -> Self {
        Self {
            loaders: vec![Box::new(PlaintextLoader::default())],
        }
    }
}

impl DataLoader for CombinedLoader {
    fn handled_extensions(&self) -> HashSet<&'static str> {
        let mut exts = HashSet::new();
        for loader in self.loaders.iter() {
            let new_exts = loader.handled_extensions();
            if exts.intersection(&new_exts).count() > 0 {
                warn!("Two data handlers registered for the same extension.");
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
