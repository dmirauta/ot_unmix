extern crate ocl;

pub mod data;
pub mod dataset;
#[cfg(feature = "gui")]
pub mod inspect;
pub mod rhai_transform;
pub mod substeps;
pub mod transforms;
pub mod unmixing;

pub use simple_ocl::{print_ocl_short_info, set_ocl_device};

pub use ndarray;
pub use ndarray_rand;
