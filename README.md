# NORTEC 700 NDE Viewer

An open-source Python desktop viewer for encoded Eddy Current Array (ECA) data
stored in the Evident NDE Open Format. It is intended for demonstration,
exploration, and format research with data produced by NORTEC 700 instruments.

> [!WARNING]
> This is an exploratory visualization tool, not a calibrated acceptance
> system. Validate results independently before using them in an inspection
> workflow.

## Features

- 2D C-scan views of impedance magnitude and baseline-adjusted X/Y components
- Strip-chart and impedance-plane inspection at the selected channel/position
- Optional interactive 3D surface rendering with VisPy
- Encoder-aware mapping to the discrete scan grid when metadata is available
- Inspection of acquisition settings, processing notes, and HDF5 structure
- Evident/OmniScan XML `.pal` palette support
- Included 32-channel sample scan for immediate verification

## Start on macOS

Double-click `run.command`.

The first launch creates a private `.venv` environment inside this folder and
downloads PySide6, VisPy, h5py, and NumPy. Later launches start immediately.

If macOS blocks the launcher, Control-click `run.command`, choose **Open**, then
confirm.

## Install on Linux or Windows

Python 3.9 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python nde_viewer.py
```

## Use it

1. Click **Open .nde…** and select an NDE file. The included
   `surface-cracks-eca-single-surf25-2026-07-13T04-54-27.nde` is a ready-to-use
   verification sample.
2. Use **C-scan** to switch between impedance change, X, and Y.
3. Choose a palette or use **Load .pal…** for an authorized XML palette.
4. Toggle **Smooth 2D** to switch between interpolation and source grid cells.
5. Click the heatmap or move the controls to inspect a sample.
6. Use the strip chart, impedance plane, and optional 3D surface views.
7. Review instrument metadata under **Setup** and **HDF5 structure**.

Open a file directly from Terminal:

```bash
./run.command "/path/to/inspection.nde"
```

Inspect a file without opening the GUI:

```bash
./.venv/bin/python nde_viewer.py --inspect "/path/to/inspection.nde"
```

## What the viewer reads

- `/Properties`
- `/Public/Setup`
- Declared `Impedance`, `ImpedanceStatus`, and `Encoder` datasets
- Optional `/Private/UNIS/Setup` fields used to identify disabled processes

Complex impedance is scaled using the dataset `dataValue` metadata. Supported
phase/gain transformations are applied, and acquisition cycles are mapped to
the declared spatial grid using encoder positions.

The NDE specification identifies an IIR or FIR filter and its cutoff, but does
not store filter order or coefficients. For enabled temporal IIR filters, this
viewer uses a first-order approximation and reports that limitation in the
**Processing notes** tab.

## Palette files

The viewer accepts the Evident `.pal` XML structure containing
`Palette/MainColors/Color` entries with `R`, `G`, and `B` attributes. Palette
files placed beside `nde_viewer.py` are discovered automatically. The included
`EVDNT_Amplitude.pal` is supplied by the project owner for use with this demo.

## Windows package

The **Windows bundle** GitHub Actions workflow creates a tested, zipped
`NortecNDEViewer.exe` application on a native Windows runner. It can be started
manually from the Actions tab, and version tags such as `v0.1.0` publish the ZIP
as a GitHub Release asset.

PyInstaller builds are platform-specific. A Windows executable must be built on
Windows (or in a Windows virtual machine), so the workflow performs that step
instead of attempting to cross-compile it on macOS.

## Test

The test suite builds a small synthetic NDE/HDF5 file, so no inspection data is
required:

```bash
python -m unittest discover -v
QT_QPA_PLATFORM=offscreen python nde_viewer.py --smoke-test "/path/to/file.nde"
```

## Data privacy

NDE, HDF5, and palette files are ignored by Git by default, except for the two
explicitly authorized demo assets. Review other inspection data for customer,
asset, and location information before sharing it anywhere. See
[`SAMPLE_DATA.md`](SAMPLE_DATA.md) for the sample's scope and limitations.

## Project status

This independent demo is not affiliated with or endorsed by Evident. NORTEC,
Evident, and OmniScan may be trademarks of their respective owners.

## License

Released under the [MIT License](LICENSE).
