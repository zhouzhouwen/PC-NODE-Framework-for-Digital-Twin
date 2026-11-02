# Data

The dataset is not publicly released due to data confidentiality or usage restrictions. It may be made available upon reasonable request, subject to applicable permissions and agreements.

## Format

The scripts read the dataset from `data/aerosol_data.xlsx` (sheet `Sheet1`). Each row is one aerosol size channel of one UTARTS run, that is, one trajectory of the normalized aerosol concentration C/C<sub>0</sub>. The first 1,365 rows contain the 32 ZrO<sub>2</sub> runs used for training and testing.

| Column | Description |
|---|---|
| `Spray Coverage` | Spray-covered volume divided by the chamber volume (0 to 1) |
| `Spray Flow Rate (L/min)` | Spray flow rate Q (L/min) |
| `Droplet mean size  (μm)` | Droplet mean size of the spray nozzle (μm) |
| `Orfice diameter (mm)` | Orifice diameter of the spray nozzle (mm) |
| `Mist mean size (μm)` | Mist mean size d<sub>m</sub> (μm), 0 without mist |
| `Mist mean concentration (P/cm^3)` | Mist mean concentration C<sub>m</sub> (particles/cm³), 0 without mist |
| `Aerosol density (kg/m3)` | Density of the aerosol material (kg/m³) |
| `Contact Angle` | Water contact angle on the aerosol particle surface (°) |
| `Aerosol size (μm)` | Aerosol size channel d<sub>p</sub> (μm) |
| `0min`, `1min`, `3min`, `5min`, `10min`, `15min`, `20min`, `30min`, `40min`, `50min`, `60min` | Measured C/C<sub>0</sub> at each sampling time, 1 at 0 min, empty if not sampled |
| `Case` | Run number, entered in the first row of each run (the scripts assign run 10 by row position) |

## Example

`example_input.csv` shows the input format for inference with `python scripts/test.py --input data/example_input.csv`. It contains operating conditions only.
