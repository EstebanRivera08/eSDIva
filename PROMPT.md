
Create a complete plan for refactoring the **PyField examples**. The goal is to produce the **minimum set of examples** that:

- Exercise **every PyField functionality** (to ensure full test coverage)
- Provide **clear, visual outputs** for users
- Demonstrate **how to use the API** through concise, well‑structured scripts
- Generate asset figures when `config.py` indicates so (this file must be kept)

Use the **existing examples as a base**, but extend them to cover new features (e.g., emission and reception classes).  
Include **parallel examples inspired by FIELD II** to highlight API differences and compare equivalent outputs.

### **General Requirements**
- Follow the same structure as current examples:  
  - Clear parameter declaration  
  - Logical section division  
  - Concise, meaningful comments  
- Each example must begin with a **short description** of what it demonstrates.
- Produce a **README.md** listing all examples with 1–2 sentence descriptions.
- Ensure the collection provides a **complete overview** of PyField capabilities and a **direct comparison** with FIELD II where relevant.

### **Examples to Implement**
These examples form the initial full suite:

1. `visualization_trapezoid_SDI_vs_FWT.py`  
2. `example01_transducer_gallery.py`  
3. `example02_monoelements_monochromatic_CW.py`  
4. `example03_multielements_monochromatic_CW.py`  
5. `example04_lineararray_excitation_DW.py`  
6. `example05_matrixarray_pulsed_steeredPW.py`  
   - Use the **3D transient plotting** function  
7. `example06_concave_PSF.py`  
   - Based on the FIELD II example  
8. `example07_lineararray_TXfocus_RXall.py`  
9. `example08_anotherreceptionexample.py`  
10. `example09_lineararray_imagePSF.py`  
    - FIELD II equivalent  
    - May require a new `pyfield.beamforming` module  
    - Implement a first explicit version inside the example  
11. `example10_intensities_peak_pressure.py`  
    - FIELD II equivalent  
    - Introduce attenuation  
12. `example11_lineararray_attenuations_monochromatic_CW.py`  
    - Show attenuation effect on 2D slice  
    - Shortened version of FIELD II’s long example  

### **“Extras” Section**
These examples highlight advanced or niche capabilities:

12. `example12_txconcave_mousebrain.py`  
13. `example13_txlinear_ratbrainzones.py`  
14. `example14_importstl_petri_dish.py`  
15. `example15_monoelement_petridish.py`  
16. `example16_subdivide_parametric_surface.py`  

### **Additional Task**
evaluate whether anything is missing in the examples.
Propose additional examples that would be valuable for new users or advanced users exploring PyField.

