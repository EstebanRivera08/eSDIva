

I’ve completed a major refactor of the project:

1. All plotting functions were moved from `utilities` to a new `plotting` module.  
2. Several plotting functions were renamed for consistency, and new ones were added.  
3. The `future` module was renamed to `cache`.  
4. The former `brain_atlas` module is now integrated into `utilities`.  
5. `transformation_functions.py` was moved from `utilities` to `cache`, where `DopplerScan` and other internal tools will live.

I want you to
**Fix all bugs** introduced by the refactor, especially in the examples folder


