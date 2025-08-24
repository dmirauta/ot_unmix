//! Data structures defines and data loading

use super::DataLoader;
use crate::data::XYData;
use crate::unmixing::F;
use std::collections::{HashMap, HashSet};
use std::fs::{self, DirEntry};
use std::path::Path;

static PLAINTEXT_EXTS: &[&str] = &["xy", "ASC"];

fn collect_numeric_literals(line: &str, max_literals: usize) -> Vec<F> {
    let mut num_begin = None;
    let mut nums = vec![];
    for (i, c) in line
        .chars()
        .chain(['\n']) // ensure line is ended on non-numeric character, so that last number is added
        .enumerate()
    {
        match c {
            '0'..='9' | '.' | '-' | 'e' => {
                if num_begin.is_none() {
                    num_begin = Some(i);
                }
            }
            _ => {
                if let Some(begin) = num_begin {
                    // ignoring ill formed numeric char patterns like "-.-"
                    if let Ok(n) = line[begin..i].parse::<F>() {
                        nums.push(n);
                        num_begin = None;
                    }
                }
            }
        }
        if nums.len() >= max_literals {
            break;
        }
    }
    nums
}

fn parse_xy_lines(file: &str, discard_neg_weights: bool) -> Vec<[F; 2]> {
    let mut out = vec![];
    for (i, line) in file.lines().enumerate() {
        let nums = collect_numeric_literals(line, 2);
        match nums.len() {
            0 | 1 => println!("Too few numbers on line {i}"),
            _ => {
                if !(discard_neg_weights && nums[1] <= 0.0) {
                    out.push([nums[0], nums[1]])
                }
            }
        }
    }
    out
}

#[test]
fn test_parse() {
    let input = r"
 10.000    29477
 10.020    -29415
 1.004e1,    29499
    ";
    let xy = parse_xy_lines(input, true);
    assert_eq!(xy, vec![[10.0, 29477.0], [10.04, 29499.0]]);
}

pub fn parse_xy(filepath: impl AsRef<Path>, discard_neg_weights: bool) -> Vec<[F; 2]> {
    let mut out = vec![];
    if let Ok(file) = fs::read_to_string(filepath.as_ref()) {
        out = parse_xy_lines(file.as_str(), discard_neg_weights);
    } else {
        println!("failed to read {:?}", filepath.as_ref());
    }
    out
}

pub fn parse_files_with_ext<'a>(
    files: impl Iterator<Item = &'a DirEntry>,
    ext: &str,
    discard_neg_weights: bool,
) -> HashMap<String, Vec<[F; 2]>> {
    // TODO: many unwraps here... make more graceful?
    let mut xy_set = HashMap::new();
    for file in files.filter(|f| f.file_name().into_string().unwrap().contains(ext)) {
        let xy = parse_xy(file.path(), discard_neg_weights);
        if !xy.is_empty() {
            xy_set.insert(file.file_name().into_string().unwrap(), xy);
        }
    }
    xy_set
}

#[derive(Default)]
pub struct PlaintextLoader;

impl DataLoader for PlaintextLoader {
    fn handled_extensions(&self) -> HashSet<&'static str> {
        PLAINTEXT_EXTS.iter().cloned().collect()
    }

    fn parse_relevant_files(&self, all_files: &[DirEntry]) -> crate::dataset::TaggedDatapoints {
        let mut data = HashMap::new();
        for &ext in PLAINTEXT_EXTS.iter() {
            data.extend(
                parse_files_with_ext(all_files.iter(), ext, true)
                    .into_iter()
                    .map(|(filename, xy)| {
                        let short_filename = if filename.contains('/') {
                            filename.split('/').next_back().unwrap().to_string()
                        } else {
                            filename
                        };
                        (short_filename, XYData::Loose(xy))
                    }),
            );
        }
        data
    }
}
