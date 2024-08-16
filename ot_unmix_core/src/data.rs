//! Data structures defines and data loading

use std::collections::HashMap;
use std::f64::consts::{FRAC_2_SQRT_PI, SQRT_2};
use std::fs;
use std::path::Path;

use iter_tools::Itertools;

use super::transforms::{PipelineItem, XYTransform};
use super::unmixing::{EPSILON, F};

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

pub fn parse_files_with_ext(
    dir: impl AsRef<Path>,
    ext: &str,
    discard_neg_weights: bool,
) -> HashMap<String, Vec<[F; 2]>> {
    // TODO: many unwraps here... make more graceful?
    let mut xy_set = HashMap::new();
    let dir_entries = fs::read_dir(dir).unwrap().filter_map(|fr| fr.ok());
    let files = dir_entries.filter(|de| de.file_type().unwrap().is_file());
    for file in files.filter(|f| f.file_name().into_string().unwrap().contains(ext)) {
        let xy = parse_xy(file.path(), discard_neg_weights);
        if !xy.is_empty() {
            xy_set.insert(file.file_name().into_string().unwrap(), xy);
        }
    }
    xy_set
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stick {
    pub position: F,
    /// TODO: Enforce positive only?
    pub weight: F,
}

impl Stick {
    fn new(position: F, weight: F) -> Self {
        debug_assert!(weight > 0.0);
        Stick { position, weight }
    }
}

impl From<[F; 2]> for Stick {
    fn from([p, w]: [F; 2]) -> Self {
        Stick::new(p, w)
    }
}

pub fn sort_sticks(sticks: &mut [Stick]) {
    sticks.sort_by(|s1, s2| {
        s1.position
            .partial_cmp(&s2.position)
            .expect("NaNs present in float cmp?")
    });
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Signature {
    pub sticks: Vec<Stick>,
}

impl From<Vec<Stick>> for Signature {
    fn from(_sticks: Vec<Stick>) -> Self {
        let mut sticks = _sticks.clone();
        sort_sticks(&mut sticks);
        Self { sticks }
    }
}

impl From<Vec<[F; 2]>> for Signature {
    fn from(value: Vec<[F; 2]>) -> Self {
        let sticks: Vec<Stick> = value
            .into_iter()
            .filter(|[_, w]| *w > 0.0)
            .map(|arr| arr.into())
            .collect();
        sticks.into()
    }
}

impl From<(&[F], &[F])> for Signature {
    fn from((positions, weights): (&[F], &[F])) -> Self {
        let sticks: Vec<Stick> = positions
            .iter()
            .zip(weights.iter())
            .filter(|(_, w)| **w > 0.0)
            .map(|(p, w)| [*p, *w].into())
            .collect();
        sticks.into()
    }
}

const INV_SQRT_TWOPI: F = FRAC_2_SQRT_PI / (2.0 * SQRT_2);

fn eval_gaussian(x: F, mu: F, sigma: F) -> F {
    INV_SQRT_TWOPI * (-0.5 * ((x - mu) / sigma).powi(2)).exp() / sigma
}

pub fn pairs<T: Clone>(x: &[T], y: &[T]) -> Vec<[T; 2]> {
    x.iter()
        .zip(y.iter())
        .map(|(x, y)| [x.clone(), y.clone()])
        .collect()
}

static FORBIDDEN_CHARS: &[char] = &['<', '>', ':', '"', '/', '\\', '|', '?', '*'];

impl Signature {
    pub fn total_weight(&self) -> F {
        self.sticks.iter().map(|s| s.weight).sum()
    }

    pub fn normalise(&mut self) {
        let tw = self.total_weight();
        for stick in self.sticks.iter_mut() {
            stick.weight /= tw;
        }
    }

    pub fn size(&self) -> usize {
        self.sticks.len()
    }

    /// generate a density via a gaussian kernel of specified standard deviation
    pub fn estimate_density_inplace(&self, x: &[F], y: &mut [F], kernel_sd: F) {
        // TODO: reflect out of bounds gaussian tails to keep all mass inside?
        for (xval, yval) in x.iter().zip(y.iter_mut()) {
            *yval = 0.0;
            for Stick { position, weight } in self.sticks.iter() {
                if *weight > EPSILON {
                    *yval += *weight * eval_gaussian(*xval, *position, kernel_sd);
                }
            }
        }
    }

    /// generate a density via a gaussian kernel of specified standard deviation
    pub fn estimate_density(&self, x: &[F], kernel_sd: F) -> XYData {
        let mut y = vec![0.0; x.len()];
        self.estimate_density_inplace(x, y.as_mut_slice(), kernel_sd);
        XYData::Loose(pairs(x, y.as_slice()))
    }

    pub fn to_xydata(&self) -> XYData {
        XYData::Loose(
            self.sticks
                .iter()
                .cloned()
                .map(|Stick { position, weight }| [position, weight])
                .collect(),
        )
    }

    pub fn save_to_file(&self, dir: &Path, name: impl Into<String>) {
        let xy = self.to_xydata();
        xy.save_to_file(dir, name);
    }
}

// TODO: remove Placeholder variant, or add back in Signature and/or SharedGrid variants?
#[derive(Default, Clone, PartialEq)]
pub enum XYData {
    Loose(Vec<[F; 2]>),
    #[default]
    Placeholder,
}

impl From<(&[F], &[F])> for XYData {
    fn from((xvec, yvec): (&[F], &[F])) -> Self {
        XYData::Loose(pairs(xvec, yvec))
    }
}

impl From<(Vec<F>, Vec<F>)> for XYData {
    fn from((xvec, yvec): (Vec<F>, Vec<F>)) -> Self {
        (xvec.as_slice(), yvec.as_slice()).into()
    }
}

static RECOGNISED_XY_EXT: &[&str] = &["xy", "ASC"];

#[allow(dead_code)]
impl XYData {
    pub fn to_sig(self) -> Option<Signature> {
        match self {
            XYData::Loose(xy) => Some(xy.into()),
            _ => None,
        }
    }

    pub fn to_sig_normed(self) -> Option<Signature> {
        match self {
            XYData::Loose(xy) => {
                let mut sig: Signature = xy.into();
                sig.normalise();
                Some(sig)
            }
            _ => None,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            XYData::Loose(xy) => xy.len(),
            XYData::Placeholder => 0,
        }
    }

    pub fn get_xy(&self) -> Option<Vec<[F; 2]>> {
        match self {
            XYData::Loose(xy) => Some(xy.clone()),
            XYData::Placeholder => None,
        }
    }

    pub fn get_xy_ref(&self) -> Option<&Vec<[F; 2]>> {
        match self {
            XYData::Loose(xy) => Some(xy),
            XYData::Placeholder => None,
        }
    }

    pub fn get_all_in_dir(dir: impl AsRef<Path>) -> HashMap<String, Self> {
        let mut data = HashMap::new();
        for &ext in RECOGNISED_XY_EXT.iter() {
            data.extend(
                parse_files_with_ext(dir.as_ref(), ext, true)
                    .into_iter()
                    .map(|(filename, xy)| {
                        let short_filename = if filename.contains('/') {
                            filename.split('/').last().unwrap().to_string()
                        } else {
                            filename
                        };
                        (short_filename, XYData::Loose(xy))
                    }),
            );
        }
        data
    }

    pub fn save_to_file(&self, dir: &Path, name: impl Into<String>) {
        let sanitised_name = name.into().replace(|c| FORBIDDEN_CHARS.contains(&c), "");
        let path = dir.join(format!("{}.xy", sanitised_name));
        match self {
            XYData::Loose(xy) => {
                let contents = xy.iter().map(|&[x, y]| format!("{x}, {y}")).join("\n");
                let _ = fs::write(path, contents).map_err(|e| eprintln!("{e}"));
            }
            XYData::Placeholder => {}
        }
    }

    /// for comparison with sticks
    pub fn y_mean_height(&self) -> F {
        match self {
            XYData::Loose(xy) => {
                let yv: Vec<_> = xy.iter().map(|[_, y]| *y).filter(|y| *y > 1e-9).collect();
                yv.iter().sum::<F>() / (yv.len() as F)
            }
            XYData::Placeholder => 0.0,
        }
    }
}

#[derive(Default)]
pub struct CleaningIntermediate {
    pub orig: XYData,
    pub baseline: XYData,
    pub cleaned: XYData,
    pub signature: Signature,
    pub extra_curves: HashMap<String, XYData>,
}

impl CleaningIntermediate {
    pub fn apply_baseline(&mut self, baseline: Vec<[F; 2]>) {
        if let XYData::Placeholder = self.baseline {
            self.baseline = XYData::Loose(baseline.iter().map(|[x, _]| [*x, 0.0]).collect());
        }
        match (&mut self.cleaned, &mut self.baseline) {
            (XYData::Loose(xy), XYData::Loose(bxy)) => {
                for ((c, b), bn) in xy.iter_mut().zip(bxy.iter_mut()).zip(baseline.iter()) {
                    b[1] += bn[1];
                    c[1] -= bn[1];
                }
            }
            _ => {
                println!("unimplemented for non Loose variant");
            }
        }
    }

    pub fn apply_tranform_pipeline(&mut self, pipeline: &[PipelineItem]) {
        for transform in pipeline.iter() {
            transform.apply(self);
        }
    }

    pub fn cleaned_mut(&mut self) -> &mut XYData {
        if let XYData::Placeholder = self.cleaned {
            self.cleaned = self.orig.clone();
        }
        &mut self.cleaned
    }
}

impl From<XYData> for CleaningIntermediate {
    fn from(d: XYData) -> Self {
        CleaningIntermediate {
            orig: d,
            ..Default::default()
        }
    }
}
