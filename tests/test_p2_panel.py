"""Pruebas sintéticas de disponibilidad temporal; no requieren MLflow ni Parquet."""

import unittest

import pandas as pd

from mibici.p2_panel import build_panel


class P2PanelTests(unittest.TestCase):
    def setUp(self):
        self.weeks = pd.date_range("2025-01-06", periods=36, freq="7D")
        self.stations = pd.DataFrame({"station_id": [1, 2, 3], "capacity": [10, 0, 12]})
        self.counts = pd.DataFrame({"station_id": 1, "week_start": self.weeks,
                                    "departures": [5, 30] * 18, "arrivals": 5})

    def build(self, counts=None, stations=None):
        return build_panel(self.counts if counts is None else counts,
                           self.stations if stations is None else stations,
                           self.weeks, validation_weeks=4, test_weeks=4)

    def test_features_precede_target_and_partition_boundaries(self):
        panel, features, metadata = self.build()
        self.assertTrue((panel.target_week == panel.week_start + pd.Timedelta(weeks=1)).all())
        self.assertEqual(panel.iloc[0].departures_mean4, 17.5)
        self.assertEqual(panel.iloc[0].target, 0)
        self.assertEqual(panel.iloc[1].target, 1)
        self.assertEqual(metadata["cohort_stations"], 1)
        self.assertEqual(metadata["catalog_nonpositive_or_missing_capacity"], 1)
        self.assertEqual(set(panel.station_id), {1})
        self.assertNotIn("target", features)
        for earlier, later in (("train", "validation"), ("validation", "test")):
            early = panel[panel.split == earlier]
            late = panel[panel.split == later]
            self.assertEqual(early.target_week.max() + pd.Timedelta(weeks=1), late.target_week.min())

    def test_future_label_does_not_modify_its_features(self):
        before, features, _ = self.build()
        changed = self.counts.copy()
        target_week = self.weeks[-3]
        changed.loc[changed.week_start == target_week, "departures"] = 10000
        after, _, _ = self.build(changed)
        pd.testing.assert_frame_equal(before.loc[before.target_week == target_week, features],
                                      after.loc[after.target_week == target_week, features])

    def test_no_synthetic_history_before_first_observation(self):
        panel, _, _ = self.build(self.counts.iloc[6:].copy())
        self.assertEqual(panel.week_start.min(), self.weeks[9])

    def test_station_first_seen_in_test_is_excluded(self):
        new = self.counts.iloc[-4:].copy()
        new["station_id"] = 3
        panel, _, metadata = self.build(pd.concat([self.counts, new]))
        self.assertEqual(set(panel.station_id), {1})
        self.assertIn(3, metadata["stations_without_sufficient_training_history"])

    def test_unknown_station_is_reported(self):
        unknown = self.counts.copy()
        unknown["station_id"] = 405
        _, _, metadata = self.build(pd.concat([self.counts, unknown]))
        self.assertEqual(metadata["unknown_station_ids"], [405])

    def test_weeks_must_be_contiguous(self):
        with self.assertRaises(ValueError):
            build_panel(self.counts, self.stations, self.weeks.delete(5), 4, 4)


if __name__ == "__main__":
    unittest.main()
