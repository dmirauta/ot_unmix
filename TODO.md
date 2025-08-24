
Some areas for future work, more localised todos can be found embedded in the source.

# Numerical

- f32 kernels sometimes produce negative weights

# Polish related items

- Make the ocl/rhai components of the core crate optional features
- Option to iterate on random batches of mixtures (for large datasets)

## Gui

- Shared hovered plot pos marker?
- Fix pan and drag in signature plots
- Disable scroll area scroll, when scrolling in subplot?
- More parameter tooltips (and other graphical feedback)?
- Gui related code could be more concise?
- Sometimes plots show diracs outside the expected range?

# To consider as further features

- Non-opencl option on host (could use c bindings)
- Pdf/vector export for plots? or at least png.
- Read write datasets to/from .zip/.tar archives
- Loaders for non xy data? (e.g. Cornell instance files, matlab files, etc)
  - Alternatively, mechanism for using python loaders without going through the file system
- Decomp trait to allow use of alternative unmixing methods or implementations thereof
- Other runtime scripting langs (rhai done) and other scripting options? E.g. an interactive console or custom ground distance
- Independent compute server and interface client? Note: remote ocl device driver can already provide this for free, but may incur excessive network use transferring kernel inputs/outputs each substep
