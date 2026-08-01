from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from nde_viewer import (
    NDEReadError,
    _data_as_channel_cycle,
    _resample_bilinear_2d,
    _scale_to_declared_unit,
    load_nde,
    load_palette_file,
)


class ReaderHelperTests(unittest.TestCase):
    def test_data_is_reordered_to_channel_cycle(self) -> None:
        source = np.arange(6).reshape(3, 2)
        dimensions = [
            {"axis": "AcquisitionCycle"},
            {"axis": "Channel"},
        ]

        result = _data_as_channel_cycle(source, dimensions)

        np.testing.assert_array_equal(result, source.T)

    def test_declared_unit_scaling(self) -> None:
        source = np.asarray([0.0, 50.0, 100.0])
        metadata = {"min": 0, "max": 100, "unitMin": -1, "unitMax": 1}

        result = _scale_to_declared_unit(source, metadata)

        np.testing.assert_allclose(result, [-1.0, 0.0, 1.0])

    def test_bilinear_resampling_preserves_missing_mask(self) -> None:
        source = np.asarray([[1.0, np.nan], [3.0, 5.0]])
        valid = np.isfinite(source)

        values, result_valid = _resample_bilinear_2d(
            source, valid, target_height=3, target_width=3
        )

        self.assertEqual(values.shape, (3, 3))
        self.assertTrue(result_valid[0, 0])
        self.assertFalse(result_valid[0, 2])
        self.assertAlmostEqual(values[2, 2], 5.0)


class PaletteTests(unittest.TestCase):
    def test_evident_xml_palette_is_loaded(self) -> None:
        document = """<?xml version="1.0"?>
        <Palette>
          <MainColors>
            <Color R="0" G="10" B="20" />
            <Color R="255" G="245" B="235" />
          </MainColors>
        </Palette>
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pal"
            path.write_text(document, encoding="utf-8")

            palette = load_palette_file(path)

        self.assertEqual(palette.name, "example")
        np.testing.assert_array_equal(
            palette.colors,
            np.asarray([[0, 10, 20], [255, 245, 235]], dtype=np.uint8),
        )


class EndToEndReaderTests(unittest.TestCase):
    @staticmethod
    def _write_minimal_nde(path: Path, samples: np.ndarray) -> None:
        setup = {
            "groups": [
                {
                    "id": 1,
                    "datasets": [
                        {
                            "dataClass": "Impedance",
                            "path": "/Data/Impedance",
                            "dimensions": [
                                {"axis": "Channel"},
                                {"axis": "AcquisitionCycle"},
                            ],
                            "dataValue": {
                                "min": -100,
                                "max": 100,
                                "unitMin": -100,
                                "unitMax": 100,
                                "unit": "Percent",
                            },
                        }
                    ],
                    "processes": [],
                }
            ],
            "acquisitionUnits": [{"acquisitionRate": 1000}],
        }
        properties = {
            "file": {"formatVersion": "4.3.0"},
            "methods": ["ET"],
        }
        with h5py.File(path, "w") as handle:
            handle.create_dataset("Properties", data=json.dumps(properties))
            public = handle.create_group("Public")
            public.create_dataset("Setup", data=json.dumps(setup))
            data = handle.create_group("Data")
            data.create_dataset("Impedance", data=samples)

    def test_minimal_nde_loads_without_external_fixture(self) -> None:
        dtype = np.dtype([("r", "<f4"), ("i", "<f4")])
        samples = np.zeros((2, 4), dtype=dtype)
        samples[0]["r"] = [0, 1, 4, 1]
        samples[1]["i"] = [0, 2, 0, -2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimal.nde"
            self._write_minimal_nde(path, samples)

            model = load_nde(path)

        self.assertEqual(model.channels, 2)
        self.assertEqual(model.positions, 4)
        self.assertEqual(model.v_label, "Scan position V (cycle)")
        self.assertIn("magnitude", model.values)

    def test_empty_impedance_data_has_clear_error(self) -> None:
        dtype = np.dtype([("r", "<f4"), ("i", "<f4")])
        samples = np.zeros((2, 0), dtype=dtype)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.nde"
            self._write_minimal_nde(path, samples)

            with self.assertRaisesRegex(NDEReadError, "contains no samples"):
                load_nde(path)


if __name__ == "__main__":
    unittest.main()
