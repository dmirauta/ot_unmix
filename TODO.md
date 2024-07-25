# This crate

Global todos, more localised todos embedded in source.

## numerical

- issue with running f32 kernels: producing negative weights
- should not be possible to obtain diracs outside the range of the original (e.g. at pos 5 when all original are between 20 and 30)? yet appears to be happening, though this is mostly just a plotting nuissance

## polish

- unravel gui related code
- ocl/rhai components made independent features
- iterate on subsets of mixtures
- shared hovered plot pos marker?
- fix pan and drag in signature plots
- disable scroll area scroll, when scrolling in subplot?
- more parameter tooltips (and other graphical feedback)?

## possible further features

- pdf/vector export for plots? or at least png.
- read write datasets to/from archives (.zip/.tar)
- data loader trait and loaders for non xy data? (e.g. cornell instance files, matlab files, etc)
- decomp trait and alternative implementations (including basic NMF, etc?), can then compare these
- generalised (alternative ground distance) cpu only decomp implementation
- majority of ocl code can be compiled without ocl with [cc-rs](https://github.com/rust-lang/cc-rs) (and wrapped with bindgen) for cpu (including wasm?)? Though params need to be provided at runtime for this (not compiler defines `-D ...`)
- provide python bindings? [maturin](https://www.maturin.rs/)
- other runtime scripting langs (rhai done) and other scriptable options? E.g. interactive console or custom ground distance
- split off background compute server which interfaces can attach to? Note: remote ocl device driver can already provide this for free, but may incur excessive network use transferring kernel inputs/outputs each substep
