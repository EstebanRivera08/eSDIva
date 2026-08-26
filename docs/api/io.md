---
icon: lucide/database
---

# I/O

On-disk RF storage and interchange. `RFDataset` is the internal, crash-safe
checkpoint format (one compressed `.npz` per TX event); `save_rf_hdf5` exports a
self-describing, UFF/MATLAB-compatible HDF5 file.

## RFDataset

::: sondi.io.RFDataset
    options:
      members:
        - write_event
        - read_event
        - load_all
        - to_hdf5
        - summary

## save_rf_hdf5

::: sondi.io.save_rf_hdf5
