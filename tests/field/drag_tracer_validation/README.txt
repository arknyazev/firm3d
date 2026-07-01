Validation scripts for the drag GPU guiding-centre tracer, run on the NCSX configuration loaded via simsopt.configs.get_data.

zero_drag_verification.py is a regression check: with the slowing-down rate set to zero and energy stopping disabled, the forward and backward drag tracers must reproduce the existing vacuum tracers. 
energy_stop_cond.py checks the energy-evolution law dH/dt = -nu_s H (decreasing forward, increasing backward) and the H >= H_stop energy-stop feature.

These scripts require a CUDA build of firm3dpp and are run manually, not collected by pytest.