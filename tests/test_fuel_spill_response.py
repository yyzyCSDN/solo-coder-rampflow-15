import unittest
from datetime import datetime, timezone

from rampflow.fuel_spill_response import (
    FuelSpillCoordinator,
    ResponseAsset,
    SpillReport,
    SpillSeverity,
    SpillPhase,
    CleaningReading,
    CORDON_RADIUS_M,
)
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

UTC = timezone.utc
T = lambda h, m=0: datetime(2026, 5, 1, h, m, tzinfo=UTC)


def coordinator():
    assets = [
        ResponseAsset("ARFF-1", "fire-truck", frozenset({"fire-standby", "fuel-spill-response"})),
        ResponseAsset("CLEAN-1", "spill-cart", frozenset({"fuel-spill-response"})),
        ResponseAsset("HAZ-1", "hazmat-team", frozenset({"hazmat", "fuel-spill-response"})),
        ResponseAsset("CLEAN-2", "spill-cart", frozenset({"fuel-spill-response"})),
    ]
    return FuelSpillCoordinator(assets)


class FuelSpillCordonTests(unittest.TestCase):
    def test_cordon_sized_from_severity(self):
        c = coordinator()
        d = c.report_incident("S1", SpillReport("B12", 10.0, 20.0, 80.0, SpillSeverity.MODERATE, T(8)))
        self.assertTrue(d.accepted)
        self.assertEqual(c.incidents["S1"].cordon_radius_m, CORDON_RADIUS_M[SpillSeverity.MODERATE])
        d = c.establish_cordon("S1", T(8, 2))
        self.assertTrue(d.accepted)
        self.assertEqual(c.incidents["S1"].phase, SpillPhase.CORDONED)
        d2 = c.establish_cordon("S1", T(8, 3))
        self.assertFalse(d2.accepted)
        self.assertEqual(d2.code, "cordon-already-established")

    def test_cordon_can_only_grow(self):
        c = coordinator()
        c.report_incident("S1", SpillReport("B12", 0, 0, 200.0, SpillSeverity.MAJOR, T(8)))
        c.establish_cordon("S1", T(8, 1))
        self.assertFalse(c.expand_cordon("S1", 10.0, T(8, 5)).accepted)
        ok = c.expand_cordon("S1", 75.0, T(8, 5), "sheen beyond estimate")
        self.assertTrue(ok.accepted)
        self.assertEqual(c.incidents["S1"].cordon_radius_m, 75.0)


class FuelSpillResourceTests(unittest.TestCase):
    def _opened(self, c, spill_id="S1", severity=SpillSeverity.MODERATE):
        c.report_incident(spill_id, SpillReport("B12", 0, 0, 80.0, severity, T(8)))
        c.establish_cordon(spill_id, T(8, 1))

    def test_incomplete_or_unqualified_roster_rejected(self):
        c = coordinator()
        self._opened(c)
        missing = c.muster_resources("S1", {"fuel-spill-response": "CLEAN-1"}, T(8, 5))
        self.assertFalse(missing.accepted)
        self.assertEqual(missing.code, "roster-incomplete")
        self.assertIn("fire-standby", missing.details["missing_capabilities"])
        bad = c.muster_resources(
            "S1",
            {"fuel-spill-response": "CLEAN-1", "fire-standby": "CLEAN-1"},
            T(8, 5),
        )
        self.assertFalse(bad.accepted)
        self.assertEqual(bad.code, "asset-double-rostered")
        unqualified = c.muster_resources(
            "S1",
            {"fuel-spill-response": "CLEAN-1", "fire-standby": "HAZ-1"},
            T(8, 5),
        )
        self.assertFalse(unqualified.accepted)
        self.assertEqual(unqualified.code, "unqualified-asset")

    def test_qualified_roster_moves_to_cleaning(self):
        c = coordinator()
        self._opened(c)
        ok = c.muster_resources(
            "S1",
            {"fuel-spill-response": "CLEAN-1", "fire-standby": "ARFF-1"},
            T(8, 5),
        )
        self.assertTrue(ok.accepted)
        self.assertEqual(c.incidents["S1"].phase, SpillPhase.CLEANING)
        self.assertEqual(c.asset_holder("ARFF-1"), "S1")

    def test_major_spill_requires_hazmat(self):
        c = coordinator()
        self._opened(c, severity=SpillSeverity.MAJOR)
        no_haz = c.muster_resources(
            "S1",
            {"fuel-spill-response": "CLEAN-1", "fire-standby": "ARFF-1"},
            T(8, 5),
        )
        self.assertFalse(no_haz.accepted)
        self.assertEqual(no_haz.details["missing_capabilities"], ["hazmat"])
        roster = {
            "fuel-spill-response": "CLEAN-1",
            "fire-standby": "ARFF-1",
            "hazmat": "HAZ-1",
        }
        self.assertTrue(c.muster_resources("S1", roster, T(8, 5)).accepted)

    def test_second_incident_cannot_reuse_occupied_asset(self):
        c = coordinator()
        self._opened(c, "S1", SpillSeverity.MODERATE)
        c.muster_resources(
            "S1",
            {"fuel-spill-response": "CLEAN-1", "fire-standby": "ARFF-1"},
            T(8, 5),
        )
        c.report_incident("S2", SpillReport("B18", 5, 5, 60.0, SpillSeverity.MODERATE, T(9)))
        c.establish_cordon("S2", T(9, 1))
        clash = c.muster_resources(
            "S2",
            {"fuel-spill-response": "CLEAN-2", "fire-standby": "ARFF-1"},
            T(9, 5),
        )
        self.assertFalse(clash.accepted)
        self.assertEqual(clash.code, "asset-occupied")
        self.assertEqual(clash.details["holder_spill_id"], "S1")
        # ARFF-1 still belongs to S1; S2 can commit the alternate qualified asset
        self.assertEqual(c.asset_holder("ARFF-1"), "S1")

    def test_assets_release_after_reopen(self):
        c = coordinator()
        self._opened(c, "S1", SpillSeverity.MODERATE)
        c.muster_resources(
            "S1",
            {"fuel-spill-response": "CLEAN-1", "fire-standby": "ARFF-1"},
            T(8, 5),
        )
        c.record_reading("S1", CleaningReading(T(8, 30), "P1", 4.0))
        c.record_reading("S1", CleaningReading(T(8, 40), "P2", 3.0))
        self.assertTrue(c.reopen("S1", T(8, 45)).accepted)
        self.assertIsNone(c.asset_holder("ARFF-1"))


class CleaningVerificationTests(unittest.TestCase):
    def _cleaning(self, c, spill_id="S1"):
        c.report_incident(spill_id, SpillReport("B12", 0, 0, 80.0, SpillSeverity.MODERATE, T(8)))
        c.establish_cordon(spill_id, T(8, 1))
        c.muster_resources(
            spill_id,
            {"fuel-spill-response": "CLEAN-1", "fire-standby": "ARFF-1"},
            T(8, 5),
        )

    def test_consecutive_compliant_readings_required(self):
        c = coordinator()
        self._cleaning(c)
        c.record_reading("S1", CleaningReading(T(8, 30), "P1", 4.0))
        decision = c.reopening_decision("S1")
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "readings-not-consecutive")
        self.assertEqual(decision.details["compliant_streak"], 1)
        # an over-limit reading resets the streak
        c.record_reading("S1", CleaningReading(T(8, 40), "P1", 18.0))
        c.record_reading("S1", CleaningReading(T(8, 50), "P1", 5.0))
        decision = c.reopening_decision("S1")
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.details["compliant_streak"], 1)
        c.record_reading("S1", CleaningReading(T(9, 0), "P2", 5.0))
        decision = c.reopening_decision("S1")
        self.assertTrue(decision.accepted)

    def test_readings_must_be_chronicled_in_order(self):
        c = coordinator()
        self._cleaning(c)
        c.record_reading("S1", CleaningReading(T(8, 40), "P1", 4.0))
        stale = c.record_reading("S1", CleaningReading(T(8, 35), "P1", 3.0))
        self.assertFalse(stale.accepted)
        self.assertEqual(stale.code, "reading-out-of-order")

    def test_samples_too_close_block_reopen(self):
        tight = FuelSpillCoordinator(
            [
                ResponseAsset("ARFF-1", "fire-truck", frozenset({"fire-standby", "fuel-spill-response"})),
                ResponseAsset("CLEAN-1", "spill-cart", frozenset({"fuel-spill-response"})),
            ],
            min_reading_gap_min=10,
        )
        self._cleaning(tight)
        tight.record_reading("S1", CleaningReading(T(8, 30), "P1", 4.0))
        tight.record_reading("S1", CleaningReading(T(8, 35), "P2", 4.0))
        decision = tight.reopening_decision("S1")
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "sampling-too-close")

    def test_reopen_requires_cordon_then_muster_then_verification(self):
        c = coordinator()
        c.report_incident("S1", SpillReport("B12", 0, 0, 10.0, SpillSeverity.MINOR, T(8)))
        self.assertFalse(c.record_reading("S1", CleaningReading(T(8, 10), "P1", 1.0)).accepted)
        c.establish_cordon("S1", T(8, 1))
        self.assertFalse(c.record_reading("S1", CleaningReading(T(8, 10), "P1", 1.0)).accepted)
        c.muster_resources("S1", {"fuel-spill-response": "CLEAN-1"}, T(8, 5))
        c.record_reading("S1", CleaningReading(T(8, 20), "P1", 2.0))
        c.record_reading("S1", CleaningReading(T(8, 30), "P2", 2.0))
        ok = c.reopen("S1", T(8, 35))
        self.assertTrue(ok.accepted)
        self.assertEqual(c.incidents["S1"].phase, SpillPhase.REOPENED)
        # the full cleanup/detection record is retained
        self.assertEqual(len(c.incidents["S1"].readings), 2)
        self.assertEqual(len(c.incidents["S1"].journal), 6)

    def test_records_survive_reopen_but_area_cannot_be_reworked(self):
        c = coordinator()
        self._cleaning(c)
        c.record_reading("S1", CleaningReading(T(8, 30), "P1", 4.0))
        c.record_reading("S1", CleaningReading(T(8, 40), "P2", 3.0))
        c.reopen("S1", T(8, 45))
        self.assertFalse(c.record_reading("S1", CleaningReading(T(9), "P1", 1.0)).accepted)
        self.assertFalse(c.expand_cordon("S1", 99.0, T(9)).accepted)


class DuplicateReportTests(unittest.TestCase):
    def test_duplicate_report_allocates_no_resources(self):
        c = coordinator()
        first = c.report_incident("S1", SpillReport("B12", 0, 0, 80.0, SpillSeverity.MODERATE, T(8)))
        self.assertTrue(first.accepted)
        dup = c.report_incident("S2", SpillReport("B12", 0, 0, 90.0, SpillSeverity.MODERATE, T(8, 4)))
        self.assertFalse(dup.accepted)
        self.assertEqual(dup.code, "duplicate-report")
        self.assertEqual(dup.details["active_spill_id"], "S1")
        self.assertNotIn("S2", c.incidents)
        self.assertIsNone(c.asset_holder("ARFF-1"))

    def test_report_after_reopen_is_a_new_incident(self):
        c = coordinator()
        c.report_incident("S1", SpillReport("B12", 0, 0, 5.0, SpillSeverity.MINOR, T(8)))
        c.establish_cordon("S1", T(8, 1))
        c.muster_resources("S1", {"fuel-spill-response": "CLEAN-1"}, T(8, 5))
        c.record_reading("S1", CleaningReading(T(8, 20), "P1", 2.0))
        c.record_reading("S1", CleaningReading(T(8, 30), "P2", 1.0))
        c.reopen("S1", T(8, 35))
        again = c.report_incident("S3", SpillReport("B12", 0, 0, 5.0, SpillSeverity.MINOR, T(8, 40)))
        self.assertTrue(again.accepted)


class AffectedFlightTests(unittest.TestCase):
    def _resolved(self, c, spill_id):
        c.report_incident(spill_id, SpillReport("B12", 0, 0, 5.0, SpillSeverity.MINOR, T(8)))
        c.establish_cordon(spill_id, T(8, 1))
        self.assertTrue(c.restrict_flight(spill_id, "CA123").accepted)
        self.assertFalse(c.flight_clearance("CA123").accepted)
        c.muster_resources(spill_id, {"fuel-spill-response": "CLEAN-1"}, T(8, 5))
        c.record_reading(spill_id, CleaningReading(T(8, 20), "P1", 2.0))
        c.record_reading(spill_id, CleaningReading(T(8, 30), "P2", 1.0))

    def test_flights_resume_only_after_reopen(self):
        c = coordinator()
        self._resolved(c, "S1")
        too_soon = c.resume_flights("S1")
        self.assertFalse(too_soon.accepted)
        self.assertEqual(too_soon.code, "cordon-active")
        reopened = c.reopen("S1", T(8, 35))
        self.assertTrue(reopened.accepted)
        self.assertEqual(reopened.details["released_flights"], ("CA123",))
        self.assertTrue(c.flight_clearance("CA123").accepted)
        self.assertTrue(c.resume_flights("S1").accepted)


class SpillWorkflowTests(unittest.TestCase):
    def test_cordon_event_published_only_after_domain_change(self):
        c = coordinator()
        c.report_incident("S1", SpillReport("B12", 0, 0, 80.0, SpillSeverity.MODERATE, T(8)))
        wf = RampWorkflowControl()
        receipt = wf.commit_fuel_spill_cordon("op-1", "S1", T(8, 1), lambda: c.establish_cordon("S1", T(8, 1)))
        self.assertEqual(receipt.event.topic, "fuel.spill.cordon.committed")
        # replay returns the same receipt, the domain action is not re-run
        replay = wf.commit_fuel_spill_cordon("op-1", "S1", T(8, 1), lambda: c.establish_cordon("S1", T(8, 1)))
        self.assertIs(replay, receipt)
        self.assertEqual(c.incidents["S1"].version, 2)

    def test_rejected_reopen_publishes_nothing(self):
        c = coordinator()
        c.report_incident("S1", SpillReport("B12", 0, 0, 5.0, SpillSeverity.MINOR, T(8)))
        c.establish_cordon("S1", T(8, 1))
        c.muster_resources("S1", {"fuel-spill-response": "CLEAN-1"}, T(8, 5))
        c.record_reading("S1", CleaningReading(T(8, 20), "P1", 2.0))
        wf = RampWorkflowControl()
        with self.assertRaises(WorkflowRejected) as ctx:
            wf.reopen_fuel_spill_area(
                "op-x",
                "S1",
                T(8, 25),
                lambda: c.reopen("S1", T(8, 25)),
                requirements=(c.reopening_decision("S1"),),
            )
        self.assertEqual(ctx.exception.decisions[0].code, "readings-not-consecutive")
        self.assertEqual(len(wf.stream._events), 0)


if __name__ == "__main__":
    unittest.main()
