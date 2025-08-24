pub mod combined;
pub mod plaintext;

use std::{collections::HashSet, fs::DirEntry};

use crate::dataset::TaggedDatapoints;

pub trait DataLoader {
    fn handled_extensions(&self) -> HashSet<&'static str>;
    // NOTE: Same set of files may be passed to multiple loaders, could mark handled/unhandled to
    // prevent double loading?
    fn parse_relevant_files(&self, all_files: &[DirEntry]) -> TaggedDatapoints;
}
