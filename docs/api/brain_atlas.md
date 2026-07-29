---
icon: lucide/brain
---

# Brain Atlas

BrainGlobe-backed atlas integration. Downloads atlas data on first use and
registers anatomical meshes into the lab coordinate frame. See the
[Brain Atlas user guide](../user-guide/brain-atlas.md).

::: pyfield.utilities.BG_Atlas
    options:
      members:
        - set_bgatlas
        - get_pv_mesh_from_atlas
        - transform
        - reset_mesh
        - summary
        - show_atlases
