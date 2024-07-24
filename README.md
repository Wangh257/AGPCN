
# Attention-Guided End-to-End Phase Calculation Network for Accurate High Dynamic Range 3D Reconstruction

##  Overview of AGPCN

<div style="text-align: justify; text-indent: 2em;">
AGPCN is a attention-guided, end-to-end phase calculation network in multi-frequency, multi-step, multi-exposure phase-shifting profilometry (PSP) 3D reconstruction for HDR objects.
</div>

<div style="text-align: center;">
  <img src="images/Network_structure.png" alt="Network_structure">
  <p>Fig 1. Network_structure</p>
</div>


## Dataset


<div style="text-align: justify; text-indent: 2em;">
we compile and release the metallic dataset with HDR problems used in this study. This dataset includes 1700 sets of data from various materials, shapes, standard, and non-standard parts, making it the largest structured light dataset to date.  Click <a href="https://wangh257.github.io/AGPCN/Data_Download.html">here</a>  to download the dataset.The types of defects and data distribution are shown.
</div>

<div style="text-align: center;">
  <img src="images/metal_dataset.png" alt="matal_dataset">
  <p>Fig 2. matal_dataset</p>
</div>

## Experimental results
### metal dataset 


<div style="text-align: center;">
  <p>Table 1. MAE of sine and cosine components, wrapped phase, and absolute phase.</p>
  <img src="images/table1.png" alt="matal_dataset">
</div>

<div style="text-align: center;">
  <img src="images/fig_metal.png" alt="matal_dataset">
  <p>Fig 3. matal dataset result</p>
</div>

### ceramic dataset 


<div style="text-align: center;">
  <p>Table 1. MAE of sine and cosine components, wrapped phase, and absolute phase.</p>
  <img src="images/table2.png" alt="matal_dataset">
</div>

<div style="text-align: center;">
  <img src="images/fig_ceramic.png" alt="matal_dataset">
  <p>Fig 3. ceramic dataset result</p>
</div>

### standard object 


<div style="text-align: center;">
  <p>Table 1. MAE and std of different methods on standard spheres with radii of 15.0086 mm and 12.6975 mm, and on a ceramic plane.</p>
  <img src="images/table3.png" alt="matal_dataset">
</div>

<div style="text-align: center;">
  <img src="images/fig_standard.png" alt="standard object">
  <p>Fig 3. satndard object result</p>
</div>









