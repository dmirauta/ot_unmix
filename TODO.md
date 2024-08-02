
Global todos, more localised todos embedded in source.

# numerical

- issue with running f32 kernels: producing negative weights
- should not be possible to obtain diracs outside the range of the original (e.g. at pos 5 when all original are between 20 and 30)? yet appears to be happening, though this is mostly just a plotting nuissance

# polish related items

- ocl/rhai components as optional core crate features
- option to iterate on subsets of mixtures (for exedingly large datasets)

## gui
- gui related code could be more tidy
- shared hovered plot pos marker?
- fix pan and drag in signature plots
- disable scroll area scroll, when scrolling in subplot?
- more parameter tooltips (and other graphical feedback)?

# to consider as further features

- non-opencl option on host (could use c bindings)
- pdf/vector export for plots? or at least png.
- read write datasets to/from archives (.zip/.tar)
- data loader trait and loaders for non xy data? (e.g. cornell instance files, matlab files, etc)
- decomp trait to allow use of alternative unmixing methods or implementations thereof
- other runtime scripting langs (rhai done) and other scriptable options? E.g. interactive console or custom ground distance
- independent compute server and interface client? Note: remote ocl device driver can already provide this for free, but may incur excessive network use transferring kernel inputs/outputs each substep
