

I’ve completed a major refactor of the project:

1. All plotting functions were moved from `utilities` to a new `plotting` module.  
2. Several plotting functions were renamed for consistency, and new ones were added.  
3. The `future` module was renamed to `cache`.  
4. The former `brain_atlas` module is now integrated into `utilities`.  
5. `transformation_functions.py` was moved from `utilities` to `cache`, where `DopplerScan` and other internal tools will live.

Now I need you to:

1. **Merge** `plot2D_mono_pressure` into `plot2D_pressure_slices`.  
   Both functions do similar things but the second already supports time‑dependent inputs for GIF/Matplotlib animations.  
   The final implementation should be fully contained in `plot2D_pressure_slices`, you can use optional internal helper methods for clarity and to make easily readable functions.

2. **Evaluate and outline** the feasibility and expected performance of implementing a `plot3D_transient_pressure` function using PyVista, rendering the transient pressure field frame‑by‑frame in a 3D volume.

3. **Homogenize function descriptions** descriptions' voice or form changes from
   function to function. I want all of them to be written in the same way to increase
   uniformity in the functions.

3. **Fix all minor bugs** introduced by the refactor, especially in the examples.


