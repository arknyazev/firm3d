This example studies fusion-born alpha particles in the Landreman-Paul vacuum quasi-helically-symmetric (QH) stellarator, using a guiding-centre tracer that includes drag. 
The coil and equilibrium files live in LandremanPaulQH_coils/ and are shared with the wall_loss_mc_estimators example.

The numbered subdirectories form a utility pipeline: 1_IC_sample_1e6_points samples initial conditions from the D-T fusion reactivity profile; 2_tracing_gpu traces them forward on the GPU and records the particles lost to the wall; 3_IC_sample_wall samples initial conditions uniformly on the wall surface; 4_robustness repeats the tracing through Gaussian-perturbed coils to probe sensitivity; and 5_get_forward_losses accumulates lost particles over many runs.

backward_tracing_only.py demonstrates the backward use of the drag tracer: starting from states on the wall, it integrates backward in time (energy increasing) until each particle reaches the fusion birth energy, then converts the recovered birth points to Boozer coordinates. 
This is the backward-tracing primitive that the wall_loss_mc_estimators example builds its importance-sampling proposal on.

The GPU tracing scripts require a CUDA build of firm3dpp.