This example compares Monte-Carlo estimators of the alpha-particle wall-loss probability in the Landreman-Paul QH stellarator with perturbed coils. 
All methods draw from the same fusion birth pool and trace forward with drag through the same perturbed field, so differences in their estimates and variance come only from how each method chooses and weights its samples. 
The coils and equilibrium are read from the sibling backward_tracing_drag example (backward_tracing_drag/LandremanPaulQH_coils/).

Three estimators are provided as entry points: forward_mc_perturbed.py (baseline forward Monte Carlo, no weighting), uniform_s_is_perturbed.py (importance sampling with a proposal uniform in the Boozer flux label s), and backward_informed_mc_s.py (importance sampling whose proposal is informed by a backward-tracing pilot from the wall, scoring births by Boozer s or by signed distance to the boundary). 
The shared modules perturbed_field_utils.py, birth_pool_utils.py, estimator_utils.py, plot_utils.py, and vtk_utils.py build the perturbed field, prepare the birth pool, and compute the estimator metrics and Paraview exports.

run_three_methods_per_perturbation_s.sh launches the three methods together on one perturbation so their outputs are directly comparable. 
run_forward_mc_gold.sh shards a huge forward run across GPUs and combine_forward_mc.py merges it into a gold-standard reference.
make_comparison_plots.py produces the comparison figures. 
trajectory_viz.py writes high-resolution trajectory polylines for Paraview.

The GPU tracing scripts require a CUDA build of firm3dpp.