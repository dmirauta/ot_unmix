# SOT-rs

Proof of concept python native module generated with pyo3.
In its current form this effectively just uses pyo3 to generate python bindings for sot_lib, which itself provides bindings for `c_opencl_src/*.c`
This could be used to replace components of the older [SOTU](https://github.com/dmirauta/SOTU) project, combining the approachability of the python scripting workflow, and the rigor/maintainability of the c/rust code, while reducing code duplication if elements of both SOTU and ot_unmixing are desired.

## Installation of python module

- Change directory to sot_rs
- Activate the appropriate python environment to install into
- Run `maturin develop` (maturin can be installed with pip)

After this one can simply `import sot_rs` in python.
