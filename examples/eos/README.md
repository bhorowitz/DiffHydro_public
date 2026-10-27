# Isothermal EOS Examples

These scripts compare standard adiabatic hydro against isothermal hydro with an EOS projection force (`EOSProjectionForce`) and save imshow-style timeslice grids.
Each output now has 4 rows:
- adiabatic primary field
- isothermal primary field
- adiabatic temperature
- isothermal temperature

Notes:
- Example time axes are in solver cell units (`dx=1` internally), so physical-looking evolution can require larger `t` than box-normalized intuition.
- By default the helper uses one visible GPU (`CUDA_VISIBLE_DEVICES=0`) unless you override environment variables before launch.

## Run

From the repository root:

```bash
python examples/isothermal/kelvin_helmholtz_compare.py
python examples/isothermal/gaussian_collapse_compare.py
python examples/isothermal/linear_sound_wave_compare.py
```

## Outputs

- `examples/isothermal/kelvin_helmholtz_compare.png` (vorticity comparison)
- `examples/isothermal/gaussian_collapse_compare.png` (`log10(rho)` collapse comparison)
- `examples/isothermal/linear_sound_wave_compare.png` (`rho - <rho>` wave propagation)
