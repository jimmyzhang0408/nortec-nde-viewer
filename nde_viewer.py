#!/usr/bin/env python3
"""PySide6 viewer for Evident NDE Open Format eddy-current array files."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QFontMetrics,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from vispy import app as vispy_app
    from vispy import scene
    from vispy.visuals.filters import WireframeFilter

    vispy_app.use_app("pyside6")
    VISPY_AVAILABLE = True
    VISPY_IMPORT_ERROR = ""
except Exception as exc:
    VISPY_AVAILABLE = False
    VISPY_IMPORT_ERROR = str(exc)


APP_NAME = "NDE Open Format ECA Viewer"
APP_ORGANIZATION = "NortecNDEViewer"


def _application_directory() -> Path:
    """Return the source folder or the executable folder when bundled."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


class NDEReadError(RuntimeError):
    """Raised when an NDE file cannot be interpreted by this viewer."""


@dataclass
class NDEModel:
    path: Path
    properties: dict[str, Any]
    setup: dict[str, Any]
    private_setup: dict[str, Any]
    values: dict[str, np.ndarray]
    u_values: np.ndarray
    v_values: np.ndarray
    u_label: str
    v_label: str
    setup_rows: list[tuple[str, str]]
    hdf_objects: list[tuple[str, str, str, str]]
    limits: dict[str, float]
    valid_status_percent: float | None
    notes: list[str] = field(default_factory=list)
    strongest_channel: int = 0
    strongest_position: int = 0

    @property
    def channels(self) -> int:
        return int(self.values["magnitude"].shape[0])

    @property
    def positions(self) -> int:
        return int(self.values["magnitude"].shape[1])

    def summary(self) -> dict[str, Any]:
        file_info = self.properties.get("file", {})
        return {
            "file": self.path.name,
            "formatVersion": file_info.get("formatVersion", "unknown"),
            "methods": self.properties.get("methods", []),
            "channels": self.channels,
            "positions": self.positions,
            "uRange": [
                float(self.u_values[0]),
                float(self.u_values[-1]),
                self.u_label,
            ],
            "vRange": [
                float(self.v_values[0]),
                float(self.v_values[-1]),
                self.v_label,
            ],
            "validStatusPercent": self.valid_status_percent,
            "strongest": {
                "channel": self.strongest_channel + 1,
                "u": float(self.u_values[self.strongest_channel]),
                "position": float(self.v_values[self.strongest_position]),
                "magnitudePercent": float(
                    self.values["magnitude"][
                        self.strongest_channel, self.strongest_position
                    ]
                ),
            },
            "notes": self.notes,
        }


@dataclass
class ColorPalette:
    key: str
    name: str
    colors: np.ndarray
    source: Path | None = None
    missing_color: tuple[int, int, int] = (95, 95, 95)


def _read_json_dataset(dataset: h5py.Dataset) -> dict[str, Any]:
    value = dataset[()]
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise NDEReadError(f"{dataset.name} is not a scalar JSON string.")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise NDEReadError(f"Invalid JSON in {dataset.name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise NDEReadError(f"{dataset.name} must contain a JSON object.")
    return parsed


def _find_group_and_dataset(
    setup: dict[str, Any], data_class: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    for group in setup.get("groups", []):
        for dataset in group.get("datasets", []):
            if dataset.get("dataClass") == data_class:
                return group, dataset
    raise NDEReadError(f"No {data_class} dataset is declared in /Public/Setup.")


def _process_chain(
    group: dict[str, Any], dataset: dict[str, Any]
) -> list[dict[str, Any]]:
    processes = {process["id"]: process for process in group.get("processes", [])}
    ordered: list[dict[str, Any]] = []
    visited: set[int] = set()

    def visit(process_id: int) -> None:
        if process_id in visited or process_id not in processes:
            return
        process = processes[process_id]
        for source in process.get("inputs", []):
            if "groupId" not in source or source.get("groupId") == group.get("id"):
                if "processId" in source:
                    visit(int(source["processId"]))
        visited.add(process_id)
        ordered.append(process)

    for reference in dataset.get("dataTransformations", []):
        if "groupId" not in reference or reference.get("groupId") == group.get("id"):
            if "processId" in reference:
                visit(int(reference["processId"]))
    return ordered


def _private_processes_for_group(
    private_setup: dict[str, Any], group_id: int
) -> dict[int, dict[str, Any]]:
    for group in private_setup.get("groups", []):
        if int(group.get("id", -1)) == group_id:
            return {
                int(process["id"]): process
                for process in group.get("processes", [])
                if "id" in process
            }
    return {}


def _first_order_lowpass(
    values: np.ndarray, cutoff: float, sample_rate: float
) -> np.ndarray:
    if cutoff <= 0 or sample_rate <= 0:
        return values
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff / sample_rate)
    output = np.empty_like(values, dtype=np.float64)
    output[:, 0] = values[:, 0]
    for index in range(1, values.shape[1]):
        output[:, index] = output[:, index - 1] + alpha * (
            values[:, index] - output[:, index - 1]
        )
    return output


def _first_order_highpass(
    values: np.ndarray, cutoff: float, sample_rate: float
) -> np.ndarray:
    if cutoff <= 0 or sample_rate <= 0:
        return values
    alpha = math.exp(-2.0 * math.pi * cutoff / sample_rate)
    output = np.empty_like(values, dtype=np.float64)
    output[:, 0] = 0.0
    for index in range(1, values.shape[1]):
        output[:, index] = alpha * (
            output[:, index - 1] + values[:, index] - values[:, index - 1]
        )
    return output


def _rotate(real: np.ndarray, imag: np.ndarray, degrees: np.ndarray | float):
    radians = np.deg2rad(degrees)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    return real * cosine - imag * sine, real * sine + imag * cosine


def _data_as_channel_cycle(
    raw: np.ndarray, dimensions: list[dict[str, Any]]
) -> np.ndarray:
    axes = [dimension.get("axis") for dimension in dimensions]
    try:
        channel_axis = axes.index("Channel")
        cycle_axis = axes.index("AcquisitionCycle")
    except ValueError as exc:
        raise NDEReadError(
            "The impedance dataset must have Channel and AcquisitionCycle axes."
        ) from exc
    if raw.ndim != 2:
        raise NDEReadError(
            f"Expected a 2-D impedance dataset, but found shape {raw.shape}."
        )
    return np.moveaxis(raw, (channel_axis, cycle_axis), (0, 1))


def _impedance_parts(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if raw.dtype.fields and {"r", "i"}.issubset(raw.dtype.fields):
        return (
            raw["r"].astype(np.float64, copy=False),
            raw["i"].astype(np.float64, copy=False),
        )
    if np.issubdtype(raw.dtype, np.complexfloating):
        return raw.real.astype(np.float64), raw.imag.astype(np.float64)
    raise NDEReadError(
        "The impedance dataset is not a complex type with r/i components."
    )


def _scale_to_declared_unit(
    values: np.ndarray, data_value: dict[str, Any]
) -> np.ndarray:
    required = ("min", "max", "unitMin", "unitMax")
    if not all(key in data_value for key in required):
        return values.astype(np.float64, copy=False)
    raw_min = float(data_value["min"])
    raw_max = float(data_value["max"])
    unit_min = float(data_value["unitMin"])
    unit_max = float(data_value["unitMax"])
    if raw_max == raw_min:
        return values.astype(np.float64, copy=False)
    return (values - raw_min) * (unit_max - unit_min) / (
        raw_max - raw_min
    ) + unit_min


def _find_mapping(
    setup: dict[str, Any], mapping_id: int
) -> dict[str, Any] | None:
    return next(
        (
            mapping
            for mapping in setup.get("dataMappings", [])
            if int(mapping.get("id", -1)) == mapping_id
        ),
        None,
    )


def _find_motion_device(
    setup: dict[str, Any], motion_id: int
) -> dict[str, Any] | None:
    return next(
        (
            device
            for device in setup.get("motionDevices", [])
            if int(device.get("id", -1)) == motion_id
        ),
        None,
    )


def _sensor_positions_mm(
    setup: dict[str, Any], group: dict[str, Any], channels: int
) -> np.ndarray:
    acquisition = next(
        (
            process.get("eddyCurrent")
            for process in group.get("processes", [])
            if "eddyCurrent" in process
        ),
        None,
    )
    if acquisition:
        probe_id = acquisition.get("probeId")
        sensor_group_id = acquisition.get("sensorGroupId")
        for probe in setup.get("probes", []):
            if probe.get("id") != probe_id:
                continue
            for sensor_group in (
                probe.get("eddyCurrentProbe", {}).get("sensorGroups", [])
            ):
                if sensor_group.get("id") != sensor_group_id:
                    continue
                sensor_by_id = {
                    sensor.get("id"): sensor
                    for sensor in sensor_group.get("sensors", [])
                }
                positions: list[float] = []
                for channel in acquisition.get("channels", []):
                    sensor = sensor_by_id.get(channel.get("sensorId"), {})
                    positions.append(float(sensor.get("primaryOffset", 0.0)) * 1000)
                if len(positions) == channels:
                    return np.asarray(positions, dtype=np.float64)
    return np.arange(channels, dtype=np.float64)


def _map_to_spatial_grid(
    handle: h5py.File,
    setup: dict[str, Any],
    group: dict[str, Any],
    chain: list[dict[str, Any]],
    real: np.ndarray,
    imag: np.ndarray,
    notes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, list[dict[str, Any]]]:
    spatial_process = next(
        (
            process
            for process in reversed(chain)
            if "mapToDiscrete" in process or "mapToDescrete" in process
        ),
        None,
    )
    deferred_spatial_filters: list[dict[str, Any]] = []
    if spatial_process is None:
        notes.append(
            "No encoder remapping process was found; the horizontal axis uses acquisition cycles."
        )
        return (
            real,
            imag,
            np.arange(real.shape[1], dtype=np.float64),
            np.ones(real.shape[1], dtype=np.int64),
            "cycle",
            deferred_spatial_filters,
        )

    mapping = _find_mapping(setup, int(spatial_process.get("dataMappingId", -1)))
    if not mapping or "discreteGrid" not in mapping:
        notes.append(
            "The final mapping is not a discreteGrid; the horizontal axis uses acquisition cycles."
        )
        return (
            real,
            imag,
            np.arange(real.shape[1], dtype=np.float64),
            np.ones(real.shape[1], dtype=np.int64),
            "cycle",
            deferred_spatial_filters,
        )

    dimensions = mapping["discreteGrid"].get("dimensions", [])
    spatial_dimension = next(
        (
            dimension
            for dimension in dimensions
            if dimension.get("axis") in {"VCoordinate", "UCoordinate"}
            and "motionDeviceId" in dimension
        ),
        None,
    )
    if spatial_dimension is None:
        notes.append(
            "The discrete grid has no encoder-linked dimension; cycles are shown directly."
        )
        return (
            real,
            imag,
            np.arange(real.shape[1], dtype=np.float64),
            np.ones(real.shape[1], dtype=np.int64),
            "cycle",
            deferred_spatial_filters,
        )

    motion_id = int(spatial_dimension["motionDeviceId"])
    encoder_dataset_metadata = None
    for candidate_mapping in setup.get("dataMappings", []):
        for dataset in candidate_mapping.get("allCycle", {}).get("datasets", []):
            if dataset.get("dataClass") != "Encoder":
                continue
            dimensions_metadata = dataset.get("dimensions", [])
            if any(
                int(dimension.get("motionDeviceId", -1)) == motion_id
                for dimension in dimensions_metadata
            ):
                encoder_dataset_metadata = dataset
                break
        if encoder_dataset_metadata:
            break
    if not encoder_dataset_metadata:
        notes.append("No matching encoder dataset was found; cycles are shown directly.")
        return (
            real,
            imag,
            np.arange(real.shape[1], dtype=np.float64),
            np.ones(real.shape[1], dtype=np.int64),
            "cycle",
            deferred_spatial_filters,
        )

    encoder_path = encoder_dataset_metadata.get("path")
    if not encoder_path or encoder_path not in handle:
        raise NDEReadError(f"Encoder dataset not found at {encoder_path!r}.")
    encoder_values = np.asarray(handle[encoder_path][()], dtype=np.float64).reshape(-1)
    cycles = min(real.shape[1], encoder_values.size)
    real = real[:, :cycles]
    imag = imag[:, :cycles]
    encoder_values = encoder_values[:cycles]

    motion = _find_motion_device(setup, motion_id)
    if not motion or "encoder" not in motion:
        raise NDEReadError(f"Motion device {motion_id} has no encoder definition.")
    encoder = motion["encoder"]
    resolution = float(encoder.get("stepResolution", 0.0))
    if resolution == 0:
        raise NDEReadError("Encoder stepResolution cannot be zero.")
    unit = encoder.get("unit", "StepPerMeter")
    preset = float(encoder.get("preset", 0.0))
    position = (encoder_values - preset) / resolution
    if encoder.get("inverted"):
        position *= -1.0
    if unit == "StepPerMeter":
        position_mm = position * 1000.0
        axis_unit = "mm"
    elif unit == "StepPerRevolution":
        position_mm = position * 360.0
        axis_unit = "°"
    else:
        position_mm = position
        axis_unit = "encoder unit"

    grid_resolution = float(spatial_dimension.get("resolution", 1.0))
    grid_offset = float(spatial_dimension.get("offset", 0.0))
    if axis_unit == "mm":
        grid_resolution *= 1000.0
        grid_offset *= 1000.0
    grid_quantity = int(spatial_dimension.get("quantity", 0))
    if grid_resolution <= 0 or grid_quantity <= 0:
        raise NDEReadError("The encoder-linked grid dimension is invalid.")

    bin_index = np.rint((position_mm - grid_offset) / grid_resolution).astype(int)
    valid = (bin_index >= 0) & (bin_index < grid_quantity)
    if not np.any(valid):
        raise NDEReadError(
            "Encoder positions do not intersect the declared discrete grid."
        )
    counts = np.bincount(bin_index[valid], minlength=grid_quantity).astype(np.int64)

    def average_into_bins(values: np.ndarray) -> np.ndarray:
        result = np.full(
            (values.shape[0], grid_quantity), np.nan, dtype=np.float64
        )
        populated = counts > 0
        for channel in range(values.shape[0]):
            sums = np.bincount(
                bin_index[valid],
                weights=values[channel, valid],
                minlength=grid_quantity,
            )
            result[channel, populated] = sums[populated] / counts[populated]
        return result

    mapped_real = average_into_bins(real)
    mapped_imag = average_into_bins(imag)
    populated_indices = np.flatnonzero(counts > 0)
    first = int(populated_indices[0])
    last = int(populated_indices[-1])
    mapped_real = mapped_real[:, first : last + 1]
    mapped_imag = mapped_imag[:, first : last + 1]
    counts = counts[first : last + 1]
    axis_values = grid_offset + np.arange(first, last + 1) * grid_resolution
    return (
        mapped_real,
        mapped_imag,
        axis_values,
        counts,
        axis_unit,
        deferred_spatial_filters,
    )


def _hdf_object_list(handle: h5py.File) -> list[tuple[str, str, str, str]]:
    objects: list[tuple[str, str, str, str]] = [("/", "Group", "", "")]

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        path = "/" + name
        if isinstance(obj, h5py.Group):
            objects.append((path, "Group", "", ""))
        else:
            objects.append((path, "Dataset", str(obj.shape), str(obj.dtype)))

    handle.visititems(visitor)
    return objects


def _format_number(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}".rstrip("0").rstrip(".")


def load_nde(
    path: str | os.PathLike[str],
    progress: Callable[[str], None] | None = None,
) -> NDEModel:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise NDEReadError(f"File does not exist: {source}")
    notify = progress or (lambda _message: None)
    notes: list[str] = []
    notify("Opening HDF5 container…")

    try:
        handle = h5py.File(source, "r")
    except OSError as exc:
        raise NDEReadError(f"Not a readable HDF5/NDE file: {exc}") from exc

    with handle:
        if "/Properties" not in handle or "/Public/Setup" not in handle:
            raise NDEReadError(
                "The file lacks the required /Properties or /Public/Setup dataset."
            )
        properties = _read_json_dataset(handle["/Properties"])
        setup = _read_json_dataset(handle["/Public/Setup"])
        private_setup: dict[str, Any] = {}
        if "/Private/UNIS/Setup" in handle:
            try:
                private_setup = _read_json_dataset(handle["/Private/UNIS/Setup"])
            except NDEReadError:
                notes.append("Private UNIS setup metadata could not be decoded.")
        hdf_objects = _hdf_object_list(handle)

        group, impedance_metadata = _find_group_and_dataset(setup, "Impedance")
        impedance_path = impedance_metadata.get("path")
        if not impedance_path or impedance_path not in handle:
            raise NDEReadError(
                f"The declared impedance dataset does not exist: {impedance_path!r}"
            )
        notify("Reading complex impedance samples…")
        raw = handle[impedance_path][()]
        raw = _data_as_channel_cycle(
            raw, impedance_metadata.get("dimensions", [])
        )
        if 0 in raw.shape:
            raise NDEReadError("The impedance dataset contains no samples.")
        real_raw, imag_raw = _impedance_parts(raw)
        data_value = impedance_metadata.get("dataValue", {})
        real = _scale_to_declared_unit(real_raw, data_value)
        imag = _scale_to_declared_unit(imag_raw, data_value)
        unit_name = data_value.get("unit", "native unit")

        chain = _process_chain(group, impedance_metadata)
        private_processes = _private_processes_for_group(
            private_setup, int(group.get("id", 0))
        )
        acquisition_rate = float(
            next(
                (
                    unit.get("acquisitionRate")
                    for unit in setup.get("acquisitionUnits", [])
                    if unit.get("acquisitionRate") is not None
                ),
                0.0,
            )
        )

        notify("Applying declared signal processing…")
        processing_descriptions: list[str] = []
        for process in chain:
            process_id = int(process.get("id", -1))
            private_process = private_processes.get(process_id, {})
            if "lowPassFilter" in process:
                configuration = process["lowPassFilter"]
                enabled = private_process.get("lowPassFilter", {}).get(
                    "enabled", True
                )
                cutoff = float(configuration.get("cutoffFrequency", 0.0))
                units = configuration.get("cutoffUnits", "Hz")
                filter_type = configuration.get("filterType", "IIR")
                processing_descriptions.append(
                    f"{filter_type} low-pass {cutoff:g} {units}"
                    + ("" if enabled else " (disabled)")
                )
                if enabled and units == "Hz" and filter_type == "IIR":
                    real = _first_order_lowpass(real, cutoff, acquisition_rate)
                    imag = _first_order_lowpass(imag, cutoff, acquisition_rate)
                    notes.append(
                        "The NDE metadata declares an IIR cutoff but no order or coefficients; "
                        "the viewer uses a first-order approximation."
                    )
                elif enabled:
                    notes.append(
                        f"The {filter_type} low-pass in {units} is shown in setup but not applied."
                    )
            elif "highPassFilter" in process:
                configuration = process["highPassFilter"]
                enabled = private_process.get("highPassFilter", {}).get(
                    "enabled", True
                )
                cutoff = float(configuration.get("cutoffFrequency", 0.0))
                units = configuration.get("cutoffUnits", "Hz")
                filter_type = configuration.get("filterType", "IIR")
                processing_descriptions.append(
                    f"{filter_type} high-pass {cutoff:g} {units}"
                    + ("" if enabled else " (disabled)")
                )
                if enabled and units == "Hz" and filter_type == "IIR":
                    real = _first_order_highpass(real, cutoff, acquisition_rate)
                    imag = _first_order_highpass(imag, cutoff, acquisition_rate)
                    notes.append(
                        "The NDE metadata declares an IIR cutoff but no order or coefficients; "
                        "the viewer uses a first-order approximation."
                    )
                elif enabled:
                    notes.append(
                        f"The {filter_type} high-pass in {units} is shown in setup but not applied."
                    )
            elif "impedanceTransformation" in process:
                configuration = process["impedanceTransformation"]
                channel_settings = {
                    int(item["id"]): item
                    for item in configuration.get("channels", [])
                    if "id" in item
                }
                channel_rotation = np.asarray(
                    [
                        float(channel_settings.get(index, {}).get("rotation", 0.0))
                        for index in range(real.shape[0])
                    ],
                    dtype=np.float64,
                )[:, None]
                channel_gain = np.asarray(
                    [
                        10.0
                        ** (
                            float(channel_settings.get(index, {}).get("gain", 0.0))
                            / 20.0
                        )
                        for index in range(real.shape[0])
                    ],
                    dtype=np.float64,
                )[:, None]
                real, imag = _rotate(real, imag, channel_rotation)
                real *= channel_gain
                imag *= channel_gain
                group_rotation = float(configuration.get("rotation", 0.0))
                real, imag = _rotate(real, imag, group_rotation)
                real_gain = float(configuration.get("realGain", 0.0))
                imaginary_gain = float(configuration.get("imaginaryGain", 0.0))
                real *= 10.0 ** (real_gain / 20.0)
                imag *= 10.0 ** (imaginary_gain / 20.0)
                processing_descriptions.append(
                    f"rotation {group_rotation:g}°, X gain {real_gain:g} dB, "
                    f"Y gain {imaginary_gain:g} dB"
                )

        notify("Mapping acquisition cycles through the encoder…")
        mapped_real, mapped_imag, v_values, counts, v_unit, _ = (
            _map_to_spatial_grid(
                handle, setup, group, chain, real, imag, notes
            )
        )
        u_values = _sensor_positions_mm(
            setup, group, mapped_real.shape[0]
        )
        u_unit = "mm" if not np.array_equal(
            u_values, np.arange(mapped_real.shape[0], dtype=np.float64)
        ) else "channel"

        with np.errstate(all="ignore"):
            baseline_real = np.nanmedian(mapped_real, axis=1, keepdims=True)
            baseline_imag = np.nanmedian(mapped_imag, axis=1, keepdims=True)
        centered_real = mapped_real - baseline_real
        centered_imag = mapped_imag - baseline_imag
        magnitude = np.hypot(centered_real, centered_imag)

        valid_status_percent: float | None = None
        try:
            _, status_metadata = _find_group_and_dataset(
                setup, "ImpedanceStatus"
            )
            status_path = status_metadata.get("path")
            if status_path and status_path in handle:
                status = np.asarray(handle[status_path][()])
                has_data = int(
                    status_metadata.get("dataValue", {}).get("hasData", 1)
                )
                if status.size:
                    valid_status_percent = (
                        100.0
                        * float(np.count_nonzero(status & has_data))
                        / float(status.size)
                    )
        except NDEReadError:
            pass

        if not np.any(np.isfinite(magnitude)):
            raise NDEReadError(
                "The impedance dataset contains no finite samples after processing."
            )
        finite_magnitude = np.nan_to_num(magnitude, nan=-np.inf)
        strongest_flat = int(np.argmax(finite_magnitude))
        strongest_channel, strongest_position = np.unravel_index(
            strongest_flat, magnitude.shape
        )
        limits = {
            "magnitude": max(
                0.001, float(np.nanquantile(magnitude, 0.995))
            ),
            "x": max(
                0.001, float(np.nanquantile(np.abs(centered_real), 0.995))
            ),
            "y": max(
                0.001, float(np.nanquantile(np.abs(centered_imag), 0.995))
            ),
        }

        file_info = properties.get("file", {})
        instrument = (setup.get("acquisitionUnits") or [{}])[0]
        acquisition_process = next(
            (
                process.get("eddyCurrent")
                for process in group.get("processes", [])
                if "eddyCurrent" in process
            ),
            {},
        )
        probe = next(
            (
                candidate
                for candidate in setup.get("probes", [])
                if candidate.get("id") == acquisition_process.get("probeId")
            ),
            {},
        )
        sensor_group = next(
            (
                candidate
                for candidate in probe.get("eddyCurrentProbe", {}).get(
                    "sensorGroups", []
                )
                if candidate.get("id")
                == acquisition_process.get("sensorGroupId")
            ),
            {},
        )
        sensors = sensor_group.get("sensors", [])
        pitch = (
            abs(
                float(sensors[1].get("primaryOffset", 0.0))
                - float(sensors[0].get("primaryOffset", 0.0))
            )
            * 1000.0
            if len(sensors) > 1
            else None
        )
        span = (
            abs(
                float(sensors[-1].get("primaryOffset", 0.0))
                - float(sensors[0].get("primaryOffset", 0.0))
            )
            * 1000.0
            if len(sensors) > 1
            else None
        )

        private_acquisition_unit = (
            (private_setup.get("acquisitionUnits") or [{}])[0]
            .get("eddyCurrentAcquisitionUnit", {})
        )
        hardware_gains = [
            item.get("gain")
            for item in private_acquisition_unit.get("inputs", [])
            if item.get("gain") is not None
        ]
        grid_mapping = next(
            (
                mapping
                for mapping in setup.get("dataMappings", [])
                if "discreteGrid" in mapping
            ),
            {},
        )
        grid_description = "—"
        if grid_mapping:
            dimensions = grid_mapping["discreteGrid"].get("dimensions", [])
            grid_description = " × ".join(
                f'{dimension.get("quantity", "?")} @ '
                f'{_format_number(float(dimension.get("resolution", 0)) * 1000, 4)} mm'
                if dimension.get("axis") in {"UCoordinate", "VCoordinate"}
                else f'{dimension.get("quantity", "?")} {dimension.get("axis", "")}'
                for dimension in dimensions
            )

        setup_rows = [
            (
                "File",
                f'NDE {file_info.get("formatVersion", "unknown")} · '
                f'{file_info.get("notice", "") or "standard"} · '
                f'{", ".join(properties.get("methods", [])) or "method unknown"}',
            ),
            (
                "Application",
                f'{file_info.get("createdByAppName", "unknown")} '
                f'{file_info.get("createdByAppVersion", "")}'.strip(),
            ),
            (
                "Instrument",
                " / ".join(
                    value
                    for value in (
                        str(instrument.get("model", "")),
                        str(instrument.get("platform", "")),
                    )
                    if value
                )
                or "—",
            ),
            (
                "Acquisition",
                f'{_format_number(instrument.get("acquisitionRate"), 0)} cycles/s · '
                f'{raw.shape[1]:,} stored cycles',
            ),
            (
                "Excitation",
                f'{_format_number(acquisition_process.get("frequency"), 0)} Hz · '
                f'{_format_number(acquisition_process.get("driveAmplitude"), 4)} V peak',
            ),
            (
                "Hardware gain",
                " / ".join(_format_number(value, 2) for value in hardware_gains)
                + (" dB" if hardware_gains else "")
                if hardware_gains
                else _format_number(acquisition_process.get("gain"), 2) + " dB",
            ),
            (
                "Probe",
                f'{probe.get("model", "unknown")} · '
                f'{mapped_real.shape[0]} {sensor_group.get("name", "")} channels',
            ),
            (
                "Array",
                f'{_format_number(pitch, 4)} mm pitch · '
                f'{_format_number(span, 4)} mm sensor span',
            ),
            (
                "Processing",
                " · ".join(processing_descriptions) or "No declared processing",
            ),
            (
                "Discrete grid",
                f'{grid_mapping.get("discreteGrid", {}).get("scanPattern", "—")} · '
                f"{grid_description}",
            ),
            (
                "Recorded view",
                f'{mapped_real.shape[0]} × {mapped_real.shape[1]} samples · '
                f'{_format_number(v_values[0], 3)} to '
                f'{_format_number(v_values[-1], 3)} {v_unit}',
            ),
            (
                "Data status",
                f"{_format_number(valid_status_percent, 2)}% hasData"
                if valid_status_percent is not None
                else "No status dataset",
            ),
            ("Stored unit", str(unit_name)),
            ("Impedance path", str(impedance_path)),
        ]

    notify("Ready")
    return NDEModel(
        path=source,
        properties=properties,
        setup=setup,
        private_setup=private_setup,
        values={
            "magnitude": magnitude,
            "x": centered_real,
            "y": centered_imag,
        },
        u_values=u_values,
        v_values=v_values,
        u_label=f"Array position U ({u_unit})",
        v_label=f"Scan position V ({v_unit})",
        setup_rows=setup_rows,
        hdf_objects=hdf_objects,
        limits=limits,
        valid_status_percent=valid_status_percent,
        notes=list(dict.fromkeys(notes)),
        strongest_channel=int(strongest_channel),
        strongest_position=int(strongest_position),
    )


def _jet_palette() -> ColorPalette:
    values = np.linspace(0.0, 1.0, 256)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    colors = np.rint(np.column_stack((red, green, blue)) * 255.0).astype(
        np.uint8
    )
    return ColorPalette("builtin:jet", "Jet (default)", colors)


def _grayscale_palette() -> ColorPalette:
    ramp = np.arange(256, dtype=np.uint8)
    colors = np.column_stack((ramp, ramp, ramp))
    return ColorPalette("builtin:gray", "Grayscale", colors)


def load_palette_file(path: str | os.PathLike[str]) -> ColorPalette:
    source = Path(path).expanduser().resolve()
    try:
        root = ET.parse(source).getroot()
    except (ET.ParseError, OSError) as exc:
        raise NDEReadError(f"Cannot read palette file {source.name}: {exc}") from exc

    def local_name(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    main_colors_element = next(
        (
            element
            for element in root.iter()
            if local_name(element) == "MainColors"
        ),
        None,
    )
    if main_colors_element is None:
        raise NDEReadError(
            f"{source.name} has no <MainColors> palette definition."
        )

    colors: list[tuple[int, int, int]] = []
    for element in main_colors_element:
        if local_name(element) != "Color":
            continue
        try:
            color = tuple(
                max(0, min(255, int(element.attrib[channel])))
                for channel in ("R", "G", "B")
            )
        except (KeyError, ValueError) as exc:
            raise NDEReadError(
                f"{source.name} contains an invalid RGB color entry."
            ) from exc
        colors.append(color)
    if len(colors) < 2:
        raise NDEReadError(
            f"{source.name} must contain at least two <MainColors> entries."
        )

    missing_color = (95, 95, 95)
    special_colors_element = next(
        (
            element
            for element in root.iter()
            if local_name(element) == "SpecialColors"
        ),
        None,
    )
    if special_colors_element is not None:
        special_colors = [
            element
            for element in special_colors_element
            if local_name(element) == "Color"
        ]
        if len(special_colors) >= 3:
            try:
                missing_color = tuple(
                    max(0, min(255, int(special_colors[2].attrib[channel])))
                    for channel in ("R", "G", "B")
                )
            except (KeyError, ValueError):
                pass

    return ColorPalette(
        key=f"file:{source}",
        name=source.stem,
        colors=np.asarray(colors, dtype=np.uint8),
        source=source,
        missing_color=missing_color,
    )


def _apply_palette(values: np.ndarray, palette: ColorPalette) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    scaled = clipped * (len(palette.colors) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.minimum(lower + 1, len(palette.colors) - 1)
    fraction = (scaled - lower)[..., None]
    rgb = (
        palette.colors[lower].astype(np.float64) * (1.0 - fraction)
        + palette.colors[upper].astype(np.float64) * fraction
    )
    return np.rint(rgb).astype(np.uint8)


def _resample_axis(
    values: np.ndarray, target_size: int, axis: int
) -> np.ndarray:
    source_size = values.shape[axis]
    if source_size == target_size:
        return values
    source_coordinates = np.arange(source_size, dtype=np.float64)
    target_coordinates = np.linspace(
        0.0, float(source_size - 1), target_size
    )
    moved = np.moveaxis(values, axis, -1)
    flat = moved.reshape(-1, source_size)
    interpolated = np.empty((flat.shape[0], target_size), dtype=np.float64)
    for row_index, row in enumerate(flat):
        interpolated[row_index] = np.interp(
            target_coordinates, source_coordinates, row
        )
    reshaped = interpolated.reshape(moved.shape[:-1] + (target_size,))
    return np.moveaxis(reshaped, -1, axis)


def _resample_bilinear_2d(
    values: np.ndarray,
    valid: np.ndarray,
    target_height: int,
    target_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    weights = valid.astype(np.float64)
    weighted_values = np.where(valid, values, 0.0)
    for axis, target_size in ((1, target_width), (0, target_height)):
        weighted_values = _resample_axis(
            weighted_values, target_size, axis
        )
        weights = _resample_axis(weights, target_size, axis)
    smoothed = np.zeros_like(weighted_values)
    np.divide(
        weighted_values,
        weights,
        out=smoothed,
        where=weights > 1e-6,
    )
    return smoothed, weights > 1e-3


def _smooth_bilinear_2d(
    values: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = values.shape
    target_width = min(1024, max(width, width * 16))
    target_height = min(2048, max(height, height * 2))
    return _resample_bilinear_2d(
        values, valid, target_height, target_width
    )


class HeatmapWidget(QWidget):
    selection_changed = Signal(int, int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(520, 360)
        self.model: NDEModel | None = None
        self.component = "magnitude"
        self.limit = 1.0
        self.channel = 0
        self.position = 0
        self.color_palette = _jet_palette()
        self.smooth_interpolation = True
        self._image = QImage()
        self._plot_rect = QRectF()

    def sizeHint(self) -> QSize:
        return QSize(760, 450)

    def set_model(self, model: NDEModel) -> None:
        self.model = model
        self.channel = model.strongest_channel
        self.position = model.strongest_position
        self.component = "magnitude"
        self.limit = model.limits["magnitude"]
        self._rebuild_image()

    def set_component(self, component: str, limit: float) -> None:
        self.component = component
        self.limit = max(float(limit), 1e-12)
        self._rebuild_image()

    def set_limit(self, limit: float) -> None:
        self.limit = max(float(limit), 1e-12)
        self._rebuild_image()

    def set_color_palette(self, palette: ColorPalette) -> None:
        self.color_palette = palette
        self._rebuild_image()

    def set_smooth_interpolation(self, enabled: bool) -> None:
        self.smooth_interpolation = bool(enabled)
        self._rebuild_image()

    def set_selection(self, channel: int, position: int) -> None:
        self.channel = channel
        self.position = position
        self.update()

    def _rebuild_image(self) -> None:
        if self.model is None:
            return
        values = self.model.values[self.component].T
        valid = np.isfinite(values)
        if self.component == "magnitude":
            normalized = np.clip(
                np.nan_to_num(values) / self.limit, 0.0, 1.0
            )
        else:
            normalized = (
                np.clip(np.nan_to_num(values) / self.limit, -1.0, 1.0) + 1.0
            ) / 2.0
        if self.smooth_interpolation:
            normalized, valid = _smooth_bilinear_2d(normalized, valid)
        if self.component == "magnitude":
            normalized = np.power(normalized, 0.72)
        rgb = _apply_palette(normalized, self.color_palette)
        rgb[~valid] = self.color_palette.missing_color
        rgb = np.ascontiguousarray(rgb)
        height, width, _ = rgb.shape
        self._image = QImage(
            rgb.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().window())
        if self.model is None or self._image.isNull():
            painter.setPen(self.palette().text().color())
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Open an .nde file to display its ECA C-scan.",
            )
            return

        left, top, right, bottom = 72.0, 24.0, 24.0, 64.0
        self._plot_rect = QRectF(
            left,
            top,
            max(1.0, self.width() - left - right),
            max(1.0, self.height() - top - bottom),
        )
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            self.smooth_interpolation,
        )
        painter.drawImage(self._plot_rect, self._image)
        foreground = self.palette().text().color()
        muted = self.palette().placeholderText().color()
        painter.setPen(QPen(foreground, 1))
        painter.drawRect(self._plot_rect)
        metrics = QFontMetrics(painter.font())

        painter.setPen(muted)
        for fraction in np.linspace(0.0, 1.0, 5):
            x = self._plot_rect.left() + fraction * self._plot_rect.width()
            channel_index = int(
                round(fraction * (self.model.channels - 1))
            )
            label = _format_number(
                float(self.model.u_values[channel_index]), 2
            )
            text_width = metrics.horizontalAdvance(label)
            painter.drawText(
                QPointF(x - text_width / 2, self._plot_rect.bottom() + 22),
                label,
            )
            y = self._plot_rect.top() + fraction * self._plot_rect.height()
            position_index = int(
                round(fraction * (self.model.positions - 1))
            )
            label = _format_number(
                float(self.model.v_values[position_index]), 2
            )
            text_width = metrics.horizontalAdvance(label)
            painter.drawText(
                QPointF(
                    self._plot_rect.left() - text_width - 9,
                    y + metrics.ascent() / 2,
                ),
                label,
            )

        painter.setPen(foreground)
        u_label_width = metrics.horizontalAdvance(self.model.u_label)
        painter.drawText(
            QPointF(
                self._plot_rect.center().x() - u_label_width / 2,
                self.height() - 12,
            ),
            self.model.u_label,
        )
        painter.save()
        painter.translate(17, self._plot_rect.center().y())
        painter.rotate(-90)
        v_label_width = metrics.horizontalAdvance(self.model.v_label)
        painter.drawText(QPointF(-v_label_width / 2, 0), self.model.v_label)
        painter.restore()

        selected_x = self._plot_rect.left() + (
            (self.channel + 0.5) / self.model.channels
        ) * self._plot_rect.width()
        selected_y = self._plot_rect.top() + (
            (self.position + 0.5) / self.model.positions
        ) * self._plot_rect.height()
        selection_color = QColor(255, 255, 255)
        painter.setPen(QPen(selection_color, 1))
        painter.drawLine(
            QPointF(selected_x, self._plot_rect.top()),
            QPointF(selected_x, self._plot_rect.bottom()),
        )
        painter.drawEllipse(QPointF(selected_x, selected_y), 5, 5)

    def _point_to_indices(self, point: QPointF) -> tuple[int, int] | None:
        if self.model is None or not self._plot_rect.contains(point):
            return None
        channel = int(
            (point.x() - self._plot_rect.left())
            / self._plot_rect.width()
            * self.model.channels
        )
        position = int(
            (point.y() - self._plot_rect.top())
            / self._plot_rect.height()
            * self.model.positions
        )
        return (
            max(0, min(self.model.channels - 1, channel)),
            max(0, min(self.model.positions - 1, position)),
        )

    def mouseMoveEvent(self, event) -> None:
        indices = self._point_to_indices(event.position())
        if indices is None or self.model is None:
            QToolTip.hideText()
            return
        channel, position = indices
        value = self.model.values[self.component][channel, position]
        label = {
            "magnitude": "|ΔZ|",
            "x": "X",
            "y": "Y",
        }[self.component]
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"Channel {channel + 1}\n"
            f"U: {_format_number(self.model.u_values[channel], 3)}\n"
            f"V: {_format_number(self.model.v_values[position], 3)}\n"
            f"{label}: {_format_number(value, 4)}%",
            self,
        )

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        indices = self._point_to_indices(event.position())
        if indices is not None:
            self.selection_changed.emit(*indices)


class StripChartWidget(QWidget):
    position_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.model: NDEModel | None = None
        self.component = "magnitude"
        self.channel = 0
        self.position = 0
        self._plot_rect = QRectF()

    def set_state(
        self, model: NDEModel, component: str, channel: int, position: int
    ) -> None:
        self.model = model
        self.component = component
        self.channel = channel
        self.position = position
        self.update()

    def set_selection(self, channel: int, position: int) -> None:
        self.channel = channel
        self.position = position
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().base())
        if self.model is None:
            return
        values = self.model.values[self.component][self.channel]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        if self.component == "magnitude":
            y_min = 0.0
            y_max = max(0.001, float(np.max(finite)) * 1.05)
        else:
            extent = max(0.001, float(np.max(np.abs(finite))) * 1.05)
            y_min, y_max = -extent, extent
        self._plot_rect = QRectF(
            58,
            26,
            max(1, self.width() - 76),
            max(1, self.height() - 66),
        )
        foreground = self.palette().text().color()
        muted = self.palette().placeholderText().color()
        grid = QColor(muted)
        grid.setAlpha(80)
        for fraction in (0.0, 0.5, 1.0):
            y = self._plot_rect.bottom() - fraction * self._plot_rect.height()
            painter.setPen(QPen(grid, 1))
            painter.drawLine(
                QPointF(self._plot_rect.left(), y),
                QPointF(self._plot_rect.right(), y),
            )
            painter.setPen(muted)
            value = y_min + fraction * (y_max - y_min)
            painter.drawText(
                QRectF(0, y - 10, 52, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                _format_number(value, 2),
            )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = self._plot_rect.left() + fraction * self._plot_rect.width()
            index = int(round(fraction * (self.model.positions - 1)))
            painter.setPen(muted)
            painter.drawText(
                QRectF(x - 35, self._plot_rect.bottom() + 8, 70, 20),
                Qt.AlignmentFlag.AlignCenter,
                _format_number(float(self.model.v_values[index]), 2),
            )

        def point(index: int, value: float) -> QPointF:
            x = self._plot_rect.left() + (
                index / max(1, self.model.positions - 1)
            ) * self._plot_rect.width()
            y = self._plot_rect.bottom() - (
                (value - y_min) / (y_max - y_min)
            ) * self._plot_rect.height()
            return QPointF(x, y)

        painter.setClipRect(self._plot_rect.adjusted(-1, -1, 1, 1))
        path = QPainterPath()
        started = False
        step = max(1, self.model.positions // 4000)
        for index in range(0, self.model.positions, step):
            value = values[index]
            if not np.isfinite(value):
                started = False
                continue
            plot_point = point(index, float(value))
            if not started:
                path.moveTo(plot_point)
                started = True
            else:
                path.lineTo(plot_point)
        painter.setPen(QPen(QColor(41, 142, 255), 1.6))
        painter.drawPath(path)
        selected_x = self._plot_rect.left() + (
            self.position / max(1, self.model.positions - 1)
        ) * self._plot_rect.width()
        painter.setPen(QPen(foreground, 1))
        painter.drawLine(
            QPointF(selected_x, self._plot_rect.top()),
            QPointF(selected_x, self._plot_rect.bottom()),
        )
        selected_value = values[self.position]
        if np.isfinite(selected_value):
            selected_point = point(self.position, float(selected_value))
            painter.setBrush(QColor(255, 178, 45))
            painter.drawEllipse(selected_point, 4, 4)
        painter.setClipping(False)
        painter.setPen(foreground)
        label = {"magnitude": "|ΔZ|", "x": "X", "y": "Y"}[self.component]
        painter.drawText(
            QPointF(self._plot_rect.left(), 18),
            f"Channel {self.channel + 1} · {label} (%)",
        )
        metrics = QFontMetrics(painter.font())
        text_width = metrics.horizontalAdvance(self.model.v_label)
        painter.drawText(
            QPointF(
                self._plot_rect.center().x() - text_width / 2,
                self.height() - 8,
            ),
            self.model.v_label,
        )

    def mousePressEvent(self, event) -> None:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self.model is None
            or not self._plot_rect.contains(event.position())
        ):
            return
        fraction = (
            event.position().x() - self._plot_rect.left()
        ) / self._plot_rect.width()
        position = int(round(fraction * (self.model.positions - 1)))
        self.position_selected.emit(
            max(0, min(self.model.positions - 1, position))
        )


class ImpedancePlaneWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.model: NDEModel | None = None
        self.channel = 0
        self.position = 0

    def set_state(self, model: NDEModel, channel: int, position: int) -> None:
        self.model = model
        self.channel = channel
        self.position = position
        self.update()

    def set_selection(self, channel: int, position: int) -> None:
        self.channel = channel
        self.position = position
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().base())
        if self.model is None:
            return
        x_values = self.model.values["x"][self.channel]
        y_values = self.model.values["y"][self.channel]
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(finite):
            return
        extent = max(
            0.001,
            float(np.max(np.abs(x_values[finite]))),
            float(np.max(np.abs(y_values[finite]))),
        ) * 1.05
        plot = QRectF(
            52,
            26,
            max(1, self.width() - 74),
            max(1, self.height() - 64),
        )

        def point(x_value: float, y_value: float) -> QPointF:
            x = plot.left() + ((x_value + extent) / (2 * extent)) * plot.width()
            y = plot.bottom() - ((y_value + extent) / (2 * extent)) * plot.height()
            return QPointF(x, y)

        foreground = self.palette().text().color()
        muted = self.palette().placeholderText().color()
        painter.setPen(QPen(muted, 1))
        origin = point(0, 0)
        painter.drawLine(
            QPointF(origin.x(), plot.top()), QPointF(origin.x(), plot.bottom())
        )
        painter.drawLine(
            QPointF(plot.left(), origin.y()), QPointF(plot.right(), origin.y())
        )
        painter.setClipRect(plot.adjusted(-1, -1, 1, 1))
        path = QPainterPath()
        started = False
        step = max(1, self.model.positions // 4000)
        for index in range(0, self.model.positions, step):
            if not finite[index]:
                started = False
                continue
            plot_point = point(
                float(x_values[index]), float(y_values[index])
            )
            if not started:
                path.moveTo(plot_point)
                started = True
            else:
                path.lineTo(plot_point)
        painter.setPen(QPen(QColor(242, 120, 42), 1.4))
        painter.drawPath(path)
        if finite[self.position]:
            selected = point(
                float(x_values[self.position]), float(y_values[self.position])
            )
            painter.setBrush(QColor(41, 142, 255))
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawEllipse(selected, 4, 4)
        painter.setClipping(False)
        painter.setPen(foreground)
        painter.drawText(QPointF(plot.left(), 18), "Impedance plane (%)")
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 7, plot.width(), 20),
            Qt.AlignmentFlag.AlignCenter,
            "X",
        )
        painter.save()
        painter.translate(14, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QPointF(-5, 0), "Y")
        painter.restore()


class Surface3DWidget(QWidget):
    """Interactive 3D surface for the currently selected C-scan component."""

    MAX_ROWS = 300
    MAX_COLUMNS = 96

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(230)
        self.model: NDEModel | None = None
        self.component = "magnitude"
        self.color_palette = _jet_palette()
        self.smooth_interpolation = True
        self.surface_shape = (0, 0)
        self.surface_range = (math.nan, math.nan)
        self.canvas = None
        self.view = None
        self.camera = None
        self.surface_visual = None
        self.wireframe_filter = None
        self.bounds_visual = None
        self.axis_visual = None
        self._surface_data: np.ndarray | None = None
        self._surface_valid: np.ndarray | None = None
        self._x_coordinates: np.ndarray | None = None
        self._y_coordinates: np.ndarray | None = None
        self._visual_shape = (0, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        controls = QHBoxLayout()
        self.help_label = QLabel(
            "Left-drag rotate · Shift+left-drag pan · wheel zoom"
        )
        controls.addWidget(self.help_label, 1)
        self.wireframe_checkbox = QCheckBox("Wireframe")
        self.wireframe_checkbox.setToolTip(
            "Overlay the sampled surface mesh."
        )
        controls.addWidget(self.wireframe_checkbox)
        self.ortho_checkbox = QCheckBox("Orthographic")
        self.ortho_checkbox.setToolTip(
            "Remove perspective distortion from the 3D view."
        )
        controls.addWidget(self.ortho_checkbox)
        self.reset_button = QPushButton("Reset view")
        controls.addWidget(self.reset_button)
        layout.addLayout(controls)

        self.status_label = QLabel("Open an NDE file to build the 3D surface.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        offscreen = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"
        if VISPY_AVAILABLE and not offscreen:
            self._create_canvas(layout)
        else:
            reason = (
                "VisPy rendering is skipped during the automated off-screen check."
                if VISPY_AVAILABLE
                else f"VisPy is unavailable: {VISPY_IMPORT_ERROR}"
            )
            placeholder = QLabel(reason)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(
                "QLabel { border: 1px solid palette(mid); padding: 24px; }"
            )
            layout.addWidget(placeholder, 1)
            self.wireframe_checkbox.setEnabled(False)
            self.ortho_checkbox.setEnabled(False)
            self.reset_button.setEnabled(False)

    @property
    def renderer_active(self) -> bool:
        return self.canvas is not None

    def _create_canvas(self, layout: QVBoxLayout) -> None:
        self.canvas = scene.SceneCanvas(
            keys=None,
            bgcolor="#f7f7f7",
            show=False,
            vsync=True,
        )
        self.view = self.canvas.central_widget.add_view(
            bgcolor="#f7f7f7"
        )
        self.camera = scene.TurntableCamera(
            up="+z",
            fov=45.0,
            elevation=32.0,
            azimuth=-55.0,
            translate_speed=1.0,
        )
        self.view.camera = self.camera

        self.bounds_visual = scene.visuals.Line(
            pos=np.zeros((24, 3), dtype=np.float32),
            connect="segments",
            color=(0.25, 0.25, 0.25, 0.65),
            width=1.0,
            method="gl",
            parent=self.view.scene,
        )
        self.axis_visual = scene.visuals.XYZAxis(parent=self.view.scene)

        native = self.canvas.native
        native.setMinimumSize(420, 220)
        native.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(native, 1)
        self.canvas_widget = native

        self.wireframe_checkbox.toggled.connect(
            self._update_draw_mode
        )
        self.ortho_checkbox.toggled.connect(
            self._update_projection
        )
        self.reset_button.clicked.connect(self.reset_view)

    def reset_view(self) -> None:
        if self.camera is None:
            return
        self.camera.azimuth = -55.0
        self.camera.elevation = 32.0
        self.camera.roll = 0.0
        self.camera.fov = 0.0 if self.ortho_checkbox.isChecked() else 45.0
        self._fit_camera()

    def _update_draw_mode(self, enabled: bool) -> None:
        if self.wireframe_filter is None:
            return
        self.wireframe_filter.enabled = bool(enabled)
        if self.canvas is not None:
            self.canvas.update()

    def _update_projection(self, orthographic: bool) -> None:
        if self.camera is None:
            return
        self.camera.fov = 0.0 if orthographic else 45.0
        self.camera.view_changed()

    def set_state(
        self,
        model: NDEModel,
        component: str,
        palette: ColorPalette,
        smooth_interpolation: bool,
    ) -> None:
        self.model = model
        self.component = component
        self.color_palette = palette
        self.smooth_interpolation = bool(smooth_interpolation)
        self._rebuild_surface()
        self.reset_view()

    def set_component(self, component: str) -> None:
        self.component = component
        self._rebuild_surface()

    def set_color_palette(self, palette: ColorPalette) -> None:
        self.color_palette = palette
        self._update_visual(fit_camera=False)
        self._refresh_status()

    def set_smooth_interpolation(self, enabled: bool) -> None:
        self.smooth_interpolation = bool(enabled)
        self._rebuild_surface()

    def _update_visual(self, fit_camera: bool = True) -> None:
        if (
            self.view is None
            or self._surface_data is None
            or self._surface_valid is None
            or self._x_coordinates is None
            or self._y_coordinates is None
        ):
            return

        surface = self._surface_data
        valid = self._surface_valid
        minimum, maximum = self.surface_range
        value_span = max(1e-12, maximum - minimum)
        normalized = np.clip((surface - minimum) / value_span, 0.0, 1.0)
        normalized = np.where(valid, normalized, 0.0)
        rgb = _apply_palette(normalized, self.color_palette).astype(
            np.float32
        ) / 255.0
        colors = np.empty(surface.shape + (4,), dtype=np.float32)
        colors[..., :3] = rgb
        colors[..., 3] = valid.astype(np.float32)
        fill_value = float(np.nanmedian(surface[valid]))
        heights = np.where(valid, surface, fill_value).astype(np.float32)
        x_data = np.ascontiguousarray(
            self._x_coordinates, dtype=np.float32
        )
        y_data = np.ascontiguousarray(
            self._y_coordinates, dtype=np.float32
        )
        z_data = np.ascontiguousarray(heights.T)
        color_data = np.ascontiguousarray(colors.transpose(1, 0, 2))

        if self.surface_visual is None or self._visual_shape != z_data.shape:
            if self.surface_visual is not None:
                self.surface_visual.parent = None
            self.surface_visual = scene.visuals.SurfacePlot(
                x=x_data,
                y=y_data,
                z=z_data,
                shading="smooth",
                parent=self.view.scene,
            )
            self.surface_visual.set_data(colors=color_data)
            self.wireframe_filter = WireframeFilter(
                enabled=self.wireframe_checkbox.isChecked(),
                color=(0.12, 0.12, 0.12, 0.65),
                width=1.0,
            )
            self.surface_visual.attach(self.wireframe_filter)
            self._visual_shape = z_data.shape
        else:
            self.surface_visual.set_data(x=x_data, y=y_data, z=z_data)
            self.surface_visual.set_data(colors=color_data)
        self._update_scene_guides()
        if fit_camera:
            self._fit_camera()
        if self.canvas is not None:
            self.canvas.update()

    def _rebuild_surface(self) -> None:
        if self.model is None:
            return

        source = np.asarray(
            self.model.values[self.component].T, dtype=np.float64
        )
        valid = np.isfinite(source)
        if not np.any(valid):
            self.surface_shape = (0, 0)
            self.surface_range = (math.nan, math.nan)
            self.status_label.setText(
                "No finite samples are available for the 3D surface."
            )
            return

        source_rows, source_columns = source.shape
        if self.smooth_interpolation:
            target_rows = min(
                self.MAX_ROWS, max(2, source_rows)
            )
            target_columns = min(
                self.MAX_COLUMNS, max(2, source_columns * 3)
            )
            surface, surface_valid = _resample_bilinear_2d(
                source,
                valid,
                target_rows,
                target_columns,
            )
        else:
            target_rows = min(
                self.MAX_ROWS, max(2, source_rows)
            )
            target_columns = min(
                self.MAX_COLUMNS, max(2, source_columns)
            )
            row_indices = np.rint(
                np.linspace(0, source_rows - 1, target_rows)
            ).astype(np.int32)
            column_indices = np.rint(
                np.linspace(0, source_columns - 1, target_columns)
            ).astype(np.int32)
            surface = source[np.ix_(row_indices, column_indices)]
            surface_valid = valid[np.ix_(row_indices, column_indices)]

        surface = np.where(surface_valid, surface, np.nan)
        finite_surface = surface[np.isfinite(surface)]
        if finite_surface.size == 0:
            self.surface_shape = (0, 0)
            self.surface_range = (math.nan, math.nan)
            self.status_label.setText(
                "No finite samples remain after 3D resampling."
            )
            return

        value_min = float(np.min(finite_surface))
        value_max = float(np.max(finite_surface))
        self.surface_shape = tuple(int(size) for size in surface.shape)
        self.surface_range = (value_min, value_max)
        x_start, x_end = self._coordinate_range(self.model.u_values)
        y_start, y_end = self._coordinate_range(self.model.v_values)
        self._surface_data = surface
        self._surface_valid = surface_valid
        self._x_coordinates = np.linspace(
            x_start, x_end, surface.shape[1], dtype=np.float32
        )
        self._y_coordinates = np.linspace(
            y_start, y_end, surface.shape[0], dtype=np.float32
        )
        self._refresh_status()
        self._update_visual()

    def _refresh_status(self) -> None:
        if (
            self._surface_data is None
            or self._x_coordinates is None
            or self._y_coordinates is None
        ):
            return
        value_min, value_max = self.surface_range
        mode = "bilinear" if self.smooth_interpolation else "source-grid"
        signal_label = {
            "magnitude": "|ΔZ|",
            "x": "X",
            "y": "Y",
        }[self.component]
        self.status_label.setText(
            f"{self._surface_data.shape[1]} × "
            f"{self._surface_data.shape[0]} vertices · "
            f"{mode} · {self.color_palette.name} · "
            f"U {self._x_coordinates[0]:.3g}…"
            f"{self._x_coordinates[-1]:.3g} · "
            f"V {self._y_coordinates[0]:.3g}…"
            f"{self._y_coordinates[-1]:.3g} · "
            f"{signal_label} {value_min:.3g}…{value_max:.3g}%"
        )

    def _fit_camera(self) -> None:
        if (
            self.camera is None
            or self._x_coordinates is None
            or self._y_coordinates is None
            or self._surface_data is None
        ):
            return
        minimum, maximum = self.surface_range
        z_min, z_max = self._padded_range(minimum, maximum)
        self.camera.set_range(
            x=(
                float(np.min(self._x_coordinates)),
                float(np.max(self._x_coordinates)),
            ),
            y=(
                float(np.min(self._y_coordinates)),
                float(np.max(self._y_coordinates)),
            ),
            z=(z_min, z_max),
            margin=0.12,
        )

    def _update_scene_guides(self) -> None:
        if (
            self.bounds_visual is None
            or self.axis_visual is None
            or self._x_coordinates is None
            or self._y_coordinates is None
        ):
            return
        x_min = float(np.min(self._x_coordinates))
        x_max = float(np.max(self._x_coordinates))
        y_min = float(np.min(self._y_coordinates))
        y_max = float(np.max(self._y_coordinates))
        z_min, z_max = self._padded_range(*self.surface_range)
        corners = np.asarray(
            [
                (x_min, y_min, z_min),
                (x_max, y_min, z_min),
                (x_max, y_min, z_min),
                (x_max, y_max, z_min),
                (x_max, y_max, z_min),
                (x_min, y_max, z_min),
                (x_min, y_max, z_min),
                (x_min, y_min, z_min),
                (x_min, y_min, z_min),
                (x_min, y_min, z_max),
                (x_max, y_min, z_min),
                (x_max, y_min, z_max),
                (x_max, y_max, z_min),
                (x_max, y_max, z_max),
                (x_min, y_max, z_min),
                (x_min, y_max, z_max),
                (x_min, y_min, z_max),
                (x_max, y_min, z_max),
                (x_max, y_min, z_max),
                (x_max, y_max, z_max),
                (x_max, y_max, z_max),
                (x_min, y_max, z_max),
                (x_min, y_max, z_max),
                (x_min, y_min, z_max),
            ],
            dtype=np.float32,
        )
        self.bounds_visual.set_data(pos=corners)
        axis_length = max(
            1e-6,
            0.18 * max(x_max - x_min, y_max - y_min, z_max - z_min),
        )
        self.axis_visual.transform = scene.transforms.STTransform(
            scale=(axis_length, axis_length, axis_length),
            translate=(x_min, y_min, z_min),
        )

    def closeEvent(self, event) -> None:
        if self.canvas is not None:
            self.canvas.close()
        super().closeEvent(event)

    @staticmethod
    def _coordinate_range(values: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        start = float(finite[0])
        end = float(finite[-1])
        if math.isclose(start, end):
            end = start + 1.0
        return start, end

    @staticmethod
    def _padded_range(
        minimum: float, maximum: float
    ) -> tuple[float, float]:
        if math.isclose(minimum, maximum):
            padding = max(0.001, abs(minimum) * 0.05)
        else:
            padding = (maximum - minimum) * 0.04
        if minimum >= 0.0:
            return 0.0, maximum + padding
        return minimum - padding, maximum + padding


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1280, 850)
        self.settings = QSettings(APP_ORGANIZATION, APP_NAME)
        self.model: NDEModel | None = None
        self.color_palettes: dict[str, ColorPalette] = {}

        self._build_actions()
        self._build_ui()
        self._build_menu()
        self.setStatusBar(QStatusBar())
        self._initialize_palettes()
        self.statusBar().showMessage("Open an .nde file to begin.")

    def _build_actions(self) -> None:
        self.open_action = QAction("Open NDE File…", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.open_file_dialog)
        self.reload_action = QAction("Reload", self)
        self.reload_action.setShortcut("Ctrl+R")
        self.reload_action.setEnabled(False)
        self.reload_action.triggered.connect(self.reload_file)
        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)
        self.about_action = QAction("About", self)
        self.about_action.triggered.connect(self.show_about)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.reload_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.about_action)

    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)

        file_controls = QHBoxLayout()
        self.open_button = QPushButton("Open .nde…")
        self.open_button.clicked.connect(self.open_file_dialog)
        file_controls.addWidget(self.open_button)
        self.path_label = QLabel("No file loaded")
        self.path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        file_controls.addWidget(self.path_label, 1)
        root_layout.addLayout(file_controls)

        display_controls = QHBoxLayout()
        display_controls.addWidget(QLabel("C-scan:"))
        self.component_combo = QComboBox()
        self.component_combo.addItem("Impedance change |ΔZ|", "magnitude")
        self.component_combo.addItem("Baseline-adjusted X", "x")
        self.component_combo.addItem("Baseline-adjusted Y", "y")
        self.component_combo.setEnabled(False)
        self.component_combo.currentIndexChanged.connect(
            self.component_changed
        )
        display_controls.addWidget(self.component_combo)
        display_controls.addWidget(QLabel("Palette:"))
        self.palette_combo = QComboBox()
        self.palette_combo.setMinimumContentsLength(12)
        self.palette_combo.currentIndexChanged.connect(
            self.palette_changed
        )
        display_controls.addWidget(self.palette_combo)
        self.load_palette_button = QPushButton("Load .pal…")
        self.load_palette_button.clicked.connect(self.load_palette_dialog)
        display_controls.addWidget(self.load_palette_button)
        self.smoothing_checkbox = QCheckBox("Smooth 2D")
        self.smoothing_checkbox.setChecked(True)
        self.smoothing_checkbox.setToolTip(
            "Apply bilinear interpolation in both scan dimensions."
        )
        self.smoothing_checkbox.toggled.connect(self.smoothing_changed)
        display_controls.addWidget(self.smoothing_checkbox)
        display_controls.addStretch(1)
        display_controls.addWidget(QLabel("Display max:"))
        self.limit_spin = QDoubleSpinBox()
        self.limit_spin.setDecimals(3)
        self.limit_spin.setRange(0.001, 1_000_000.0)
        self.limit_spin.setSuffix(" %")
        self.limit_spin.setEnabled(False)
        self.limit_spin.valueChanged.connect(self.limit_changed)
        display_controls.addWidget(self.limit_spin)
        root_layout.addLayout(display_controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.heatmap = HeatmapWidget()
        self.heatmap.selection_changed.connect(self.select_sample)
        left_layout.addWidget(self.heatmap, 3)

        selection_controls = QHBoxLayout()
        selection_controls.addWidget(QLabel("Channel:"))
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(1, 1)
        self.channel_spin.setEnabled(False)
        self.channel_spin.valueChanged.connect(self.channel_changed)
        selection_controls.addWidget(self.channel_spin)
        selection_controls.addWidget(QLabel("Scan position:"))
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setEnabled(False)
        self.position_slider.valueChanged.connect(self.position_changed)
        selection_controls.addWidget(self.position_slider, 1)
        self.selection_label = QLabel("—")
        self.selection_label.setMinimumWidth(300)
        self.selection_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        selection_controls.addWidget(self.selection_label)
        left_layout.addLayout(selection_controls)

        self.detail_tabs = QTabWidget()
        self.strip_chart = StripChartWidget()
        self.strip_chart.position_selected.connect(
            lambda position: self.select_sample(
                self.channel_spin.value() - 1, position
            )
        )
        self.impedance_plane = ImpedancePlaneWidget()
        self.surface_3d = Surface3DWidget()
        self.detail_tabs.addTab(self.strip_chart, "Strip chart")
        self.detail_tabs.addTab(self.impedance_plane, "Impedance plane")
        self.detail_tabs.addTab(self.surface_3d, "3D surface")
        left_layout.addWidget(self.detail_tabs, 2)
        splitter.addWidget(left_panel)

        metadata_tabs = QTabWidget()
        self.setup_table = QTableWidget(0, 2)
        self.setup_table.setHorizontalHeaderLabels(["Setting", "Value"])
        self.setup_table.verticalHeader().setVisible(False)
        self.setup_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.setup_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.setup_table.setWordWrap(True)
        self.setup_table.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.setup_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.setup_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        metadata_tabs.addTab(self.setup_table, "Setup")

        notes_widget = QWidget()
        notes_layout = QVBoxLayout(notes_widget)
        self.notes_label = QLabel("No notes.")
        self.notes_label.setWordWrap(True)
        self.notes_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        notes_layout.addWidget(self.notes_label)
        notes_layout.addStretch(1)
        metadata_tabs.addTab(notes_widget, "Processing notes")

        self.hdf_tree = QTreeWidget()
        self.hdf_tree.setHeaderLabels(["Object", "Type", "Shape", "Data type"])
        self.hdf_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for column in (1, 2, 3):
            self.hdf_tree.header().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        metadata_tabs.addTab(self.hdf_tree, "HDF5 structure")
        splitter.addWidget(metadata_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([800, 480])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _register_palette(
        self, palette: ColorPalette, select: bool = False
    ) -> None:
        existing_index = self.palette_combo.findData(palette.key)
        self.color_palettes[palette.key] = palette
        if existing_index < 0:
            self.palette_combo.addItem(palette.name, palette.key)
            existing_index = self.palette_combo.count() - 1
        else:
            self.palette_combo.setItemText(existing_index, palette.name)
        if select:
            self.palette_combo.setCurrentIndex(existing_index)

    def _initialize_palettes(self) -> None:
        self.palette_combo.blockSignals(True)
        self.palette_combo.clear()
        self.color_palettes.clear()
        jet = _jet_palette()
        self._register_palette(jet)
        self._register_palette(_grayscale_palette())
        application_directory = _application_directory()
        for candidate in sorted(application_directory.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() == ".pal":
                try:
                    self._register_palette(load_palette_file(candidate))
                except NDEReadError:
                    continue
        saved_palette = self.settings.value("lastPaletteFile", "")
        if saved_palette:
            saved_path = Path(str(saved_palette)).expanduser()
            if saved_path.is_file():
                try:
                    self._register_palette(load_palette_file(saved_path))
                except NDEReadError:
                    pass
        jet_index = self.palette_combo.findData(jet.key)
        self.palette_combo.setCurrentIndex(max(0, jet_index))
        self.palette_combo.blockSignals(False)
        self.heatmap.set_color_palette(jet)
        self.surface_3d.set_color_palette(jet)

    def load_palette_dialog(self) -> None:
        initial_directory = self.settings.value(
            "lastPaletteDirectory", str(_application_directory())
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Palette File",
            str(initial_directory),
            "Evident palette files (*.pal);;XML files (*.xml);;All files (*)",
        )
        if not path:
            return
        try:
            palette = load_palette_file(path)
        except NDEReadError as exc:
            QMessageBox.critical(self, "Unable to load palette", str(exc))
            return
        self._register_palette(palette, select=True)
        self.settings.setValue("lastPaletteFile", str(palette.source))
        self.settings.setValue(
            "lastPaletteDirectory", str(Path(path).resolve().parent)
        )
        self.statusBar().showMessage(
            f"Loaded palette {palette.name} ({len(palette.colors)} colors).",
            5000,
        )

    def palette_changed(self) -> None:
        palette = self.current_color_palette()
        self.heatmap.set_color_palette(palette)
        self.surface_3d.set_color_palette(palette)

    def smoothing_changed(self, enabled: bool) -> None:
        self.heatmap.set_smooth_interpolation(enabled)
        self.surface_3d.set_smooth_interpolation(enabled)

    def open_file_dialog(self) -> None:
        initial = self.settings.value("lastDirectory", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open NDE File",
            str(initial),
            "NDE files (*.nde);;HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if path:
            self.open_path(path)

    def open_path(self, path: str | os.PathLike[str]) -> bool:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.statusBar().showMessage("Opening NDE file…")
            QApplication.processEvents()
            model = load_nde(
                path,
                lambda message: (
                    self.statusBar().showMessage(message),
                    QApplication.processEvents(),
                ),
            )
            self.set_model(model)
            self.settings.setValue("lastDirectory", str(model.path.parent))
            return True
        except Exception as exc:
            self.statusBar().showMessage("Could not open file.")
            QMessageBox.critical(self, "Unable to open NDE file", str(exc))
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def reload_file(self) -> None:
        if self.model is not None:
            self.open_path(self.model.path)

    def set_model(self, model: NDEModel) -> None:
        self.model = model
        self.path_label.setText(str(model.path))
        self.path_label.setToolTip(str(model.path))
        self.reload_action.setEnabled(True)
        self.component_combo.setEnabled(True)
        self.limit_spin.setEnabled(True)
        self.channel_spin.setEnabled(True)
        self.position_slider.setEnabled(True)
        self.channel_spin.blockSignals(True)
        self.channel_spin.setRange(1, model.channels)
        self.channel_spin.setValue(model.strongest_channel + 1)
        self.channel_spin.blockSignals(False)
        self.position_slider.blockSignals(True)
        self.position_slider.setRange(0, model.positions - 1)
        self.position_slider.setValue(model.strongest_position)
        self.position_slider.blockSignals(False)
        self.component_combo.blockSignals(True)
        self.component_combo.setCurrentIndex(0)
        self.component_combo.blockSignals(False)
        self.limit_spin.blockSignals(True)
        self.limit_spin.setValue(model.limits["magnitude"])
        self.limit_spin.blockSignals(False)
        self.heatmap.set_model(model)
        self.strip_chart.set_state(
            model,
            "magnitude",
            model.strongest_channel,
            model.strongest_position,
        )
        self.impedance_plane.set_state(
            model, model.strongest_channel, model.strongest_position
        )
        self.surface_3d.set_state(
            model,
            "magnitude",
            self.current_color_palette(),
            self.smoothing_checkbox.isChecked(),
        )
        self._populate_setup(model)
        self._populate_hdf_tree(model)
        self.notes_label.setText(
            "\n\n".join(f"• {note}" for note in model.notes)
            if model.notes
            else "No processing limitations were reported."
        )
        self.update_selection_label()
        file_info = model.properties.get("file", {})
        status = (
            f" · {model.valid_status_percent:.1f}% hasData"
            if model.valid_status_percent is not None
            else ""
        )
        self.statusBar().showMessage(
            f'NDE {file_info.get("formatVersion", "unknown")} · '
            f"{model.channels} channels × {model.positions} positions{status}"
        )
        self.setWindowTitle(f"{model.path.name} — {APP_NAME}")

    def _populate_setup(self, model: NDEModel) -> None:
        self.setup_table.setRowCount(len(model.setup_rows))
        for row, (label, value) in enumerate(model.setup_rows):
            self.setup_table.setItem(row, 0, QTableWidgetItem(label))
            value_item = QTableWidgetItem(value)
            value_item.setToolTip(value)
            self.setup_table.setItem(row, 1, value_item)
        self.setup_table.resizeRowsToContents()
        QTimer.singleShot(0, self.setup_table.resizeRowsToContents)

    def _populate_hdf_tree(self, model: NDEModel) -> None:
        self.hdf_tree.clear()
        items: dict[str, QTreeWidgetItem] = {}
        for path, kind, shape, dtype in model.hdf_objects:
            if path == "/":
                root = QTreeWidgetItem(["/", kind, shape, dtype])
                self.hdf_tree.addTopLevelItem(root)
                items[path] = root
                continue
            parent_path, name = path.rsplit("/", 1)
            parent_path = parent_path or "/"
            item = QTreeWidgetItem([name, kind, shape, dtype])
            parent = items.get(parent_path)
            if parent is None:
                self.hdf_tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            items[path] = item
        if self.hdf_tree.topLevelItemCount():
            self.hdf_tree.topLevelItem(0).setExpanded(True)

    def current_component(self) -> str:
        return str(self.component_combo.currentData() or "magnitude")

    def current_color_palette(self) -> ColorPalette:
        key = str(self.palette_combo.currentData() or "builtin:jet")
        palette = self.color_palettes.get(key)
        if palette is None:
            palette = self.color_palettes.get("builtin:jet")
        return palette if palette is not None else _jet_palette()

    def component_changed(self) -> None:
        if self.model is None:
            return
        component = self.current_component()
        self.limit_spin.blockSignals(True)
        self.limit_spin.setValue(self.model.limits[component])
        self.limit_spin.blockSignals(False)
        self.heatmap.set_component(component, self.limit_spin.value())
        self.strip_chart.set_state(
            self.model,
            component,
            self.channel_spin.value() - 1,
            self.position_slider.value(),
        )
        self.surface_3d.set_component(component)
        self.update_selection_label()

    def limit_changed(self, value: float) -> None:
        self.heatmap.set_limit(value)

    def channel_changed(self, value: int) -> None:
        self.select_sample(value - 1, self.position_slider.value())

    def position_changed(self, value: int) -> None:
        self.select_sample(self.channel_spin.value() - 1, value)

    def select_sample(self, channel: int, position: int) -> None:
        if self.model is None:
            return
        channel = max(0, min(self.model.channels - 1, channel))
        position = max(0, min(self.model.positions - 1, position))
        self.channel_spin.blockSignals(True)
        self.channel_spin.setValue(channel + 1)
        self.channel_spin.blockSignals(False)
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)
        self.heatmap.set_selection(channel, position)
        self.strip_chart.set_selection(channel, position)
        self.impedance_plane.set_selection(channel, position)
        self.update_selection_label()

    def update_selection_label(self) -> None:
        if self.model is None:
            return
        channel = self.channel_spin.value() - 1
        position = self.position_slider.value()
        component = self.current_component()
        value = self.model.values[component][channel, position]
        signal_label = {
            "magnitude": "|ΔZ|",
            "x": "X",
            "y": "Y",
        }[component]
        self.selection_label.setText(
            f"Ch {channel + 1} · "
            f"U {_format_number(self.model.u_values[channel], 3)} · "
            f"V {_format_number(self.model.v_values[position], 3)} · "
            f"{signal_label} {_format_number(value, 4)}%"
        )

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            "A desktop reader for HDF5-based NDE Open Format eddy-current "
            "array data.\n\n"
            "The viewer reads /Properties and /Public/Setup, scales the complex "
            "r/i impedance dataset, applies supported processing metadata, and "
            "reconstructs encoded scans onto their declared spatial grid.\n\n"
            "Exploratory visualization only; not a substitute for a qualified "
            "inspection or acceptance workflow.",
        )


def run_gui(path: str | None = None, smoke_test: bool = False) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORGANIZATION)
    window = MainWindow()
    if path:
        if smoke_test:
            model = load_nde(path)
            window.set_model(model)
        else:
            window.open_path(path)
    window.show()
    if smoke_test:
        app.processEvents()
        if window.model is None:
            raise RuntimeError("Smoke test did not load a model.")
        result = {
            "windowTitle": window.windowTitle(),
            "windowSize": [window.width(), window.height()],
            "palettes": [
                window.palette_combo.itemText(index)
                for index in range(window.palette_combo.count())
            ],
            "selectedPalette": window.current_color_palette().name,
            "setupRows": window.setup_table.rowCount(),
            "hdfTopLevelItems": window.hdf_tree.topLevelItemCount(),
            "surface3D": {
                "backend": "VisPy",
                "moduleAvailable": VISPY_AVAILABLE,
                "rendererActive": window.surface_3d.renderer_active,
                "shape": list(window.surface_3d.surface_shape),
                "range": list(window.surface_3d.surface_range),
            },
            **window.model.summary(),
        }
        print(json.dumps(result, indent=2))
        window.close()
        app.processEvents()
        return 0
    return app.exec()


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file",
        nargs="?",
        help="Optional .nde file to open at startup.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Read the file and print a JSON summary without opening the GUI.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Open the GUI offscreen, load the file, and exit after verification.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv or sys.argv[1:])
    if arguments.inspect:
        if not arguments.file:
            raise SystemExit("--inspect requires a file path.")
        model = load_nde(arguments.file)
        print(json.dumps(model.summary(), indent=2))
        return 0
    if arguments.smoke_test and not arguments.file:
        raise SystemExit("--smoke-test requires a file path.")
    return run_gui(arguments.file, arguments.smoke_test)


if __name__ == "__main__":
    raise SystemExit(main())
