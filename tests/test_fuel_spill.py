import unittest
from datetime import datetime, timezone, timedelta
from dataclasses import replace

from rampflow.apron_safety import ApronSafetyMonitor
from rampflow.fuel_spill import (
    ARFF, CLEANUP, SAFETY, ENVIRONMENTAL,
    FOAM, FUEL_SPILL, SPILL_SUPERVISOR, ENV_MONITOR,
    SpillResponder, SpillReport, CleanupReading, SpillState,
    FuelSpillCoordinator, classify_tier, SpillTier,
)
from rampflow.models import Decision
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


def responders(counts):
    out = []
    kinds = (
        (ARFF, FOAM),
        (CLEANUP, FUEL_SPILL),
        (SAFETY, SPILL_SUPERVISOR),
        (ENVIRONMENTAL, ENV_MONITOR),
    )
    for (kind, qual), count in zip(kinds, counts):
        for i in range(1, count + 1):
            out.append(SpillResponder(f"{kind}-{i}", kind, frozenset({qual})))
    return out


def full_pool():
    # 4 ARFF, 4 cleanup, 1 safety, 1 environmental covers critical tier.
    return responders((4, 4, 1, 1))


class FuelSpillTests(unittest.TestCase):
    def setUp(self):
        self.monitor = ApronSafetyMonitor()

    def test_report_raises_graded_safety_zone_and_musters_qualified_only(self):
        pool = full_pool() + [SpillResponder("arff-x", ARFF, frozenset())]
        coord = FuelSpillCoordinator(pool, self.monitor)

        decision = coord.report(SpillReport("R1", "K1", D(10), 0.0, 0.0, radius_m=8.0, volume_l=50.0, affected_flights=frozenset({"F1"})))

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.code, "containment-established")
        self.assertEqual(decision.details["radius_m"], 15.0)  # minor policy floor
        zone = self.monitor.zones["fuel-spill:K1"]
        self.assertEqual(zone.center_x, 0.0)
        self.assertIn(zone.zone_id, self.monitor.zones)
        assigned = set(decision.details["assigned"])
        self.assertEqual(len(assigned), 3)  # 1 ARFF + 1 cleanup + 1 safety officer
        self.assertNotIn("arff-x", assigned)  # missing foam qualification
        self.assertEqual(len(decision.details["arff_standby"]), 1)
        self.assertEqual(coord.incidents["K1"].state, SpillState.CONTAINED)
        self.assertEqual(coord.held_flights(), frozenset({"F1"}))

    def test_resource_shortage_blocks_zone_and_retry_after_muster(self):
        coord = FuelSpillCoordinator(responders((1, 1, 0, 0)), self.monitor)

        denied = coord.report(SpillReport("R1", "K1", D(10), 0.0, 0.0, radius_m=5.0, affected_flights=frozenset({"F1"})))
        self.assertFalse(denied.accepted)
        self.assertEqual(denied.code, "qualified-resources-short")
        self.assertEqual(denied.details["missing"], {"safety": 1})
        self.assertEqual(coord.incidents["K1"].state, SpillState.REPORTED)
        self.assertNotIn("fuel-spill:K1", self.monitor.zones)  # no zone before qualified muster
        self.assertEqual(coord.held_flights(), frozenset({"F1"}))  # flight held regardless

        coord.responders["safety-1"] = SpillResponder("safety-1", SAFETY, frozenset({SPILL_SUPERVISOR}))
        ok = coord.establish_containment("K1", D(10, 5))
        self.assertTrue(ok.accepted)
        self.assertIn("fuel-spill:K1", self.monitor.zones)

    def test_duplicate_report_links_without_second_resource_set(self):
        coord = FuelSpillCoordinator(full_pool(), self.monitor)
        first = coord.report(SpillReport("R1", "K1", D(10), 0.0, 0.0, radius_m=5.0, affected_flights=frozenset({"F1"})))
        occupied_after_first = dict(coord.occupied)

        second = coord.report(SpillReport("R2", "K1", D(10, 2), 0.0, 0.0, radius_m=5.0, affected_flights=frozenset({"F2"})))
        replay = coord.report(SpillReport("R2", "K1", D(10, 2), 0.0, 0.0, radius_m=5.0, affected_flights=frozenset({"F2"})))

        self.assertEqual(second.code, "duplicate-report-linked")
        self.assertEqual(replay.code, "duplicate-report")
        self.assertEqual(coord.occupied, occupied_after_first)  # no extra vehicle/person
        inc = coord.incidents["K1"]
        self.assertEqual(inc.perimeter.assigned, frozenset(first.details["assigned"]))
        self.assertEqual(inc.reports, ("R1", "R2"))
        self.assertEqual(coord.held_flights(), frozenset({"F1", "F2"}))

    def test_release_requires_continuous_compliant_readings(self):
        coord = FuelSpillCoordinator(full_pool(), self.monitor)
        coord.report(SpillReport("R1", "K1", D(10), 0.0, 0.0, radius_m=8.0, affected_flights=frozenset({"F1"})))
        self.assertTrue(coord.begin_cleanup("K1").accepted)

        # No readings -> cannot release.
        self.assertFalse(coord.can_release("K1", D(10, 30)).accepted)

        r1 = coord.record_reading("K1", CleanupReading("M1", D(10, 20), 0.0, 0.0, 10.0, 50.0))
        self.assertEqual(r1.details["streak"], 1)
        self.assertEqual(coord.can_release("K1", D(10, 25)).code, "readings-not-clear")  # minor needs 2

        # A failing reading resets the consecutive streak.
        coord.record_reading("K1", CleanupReading("M2", D(10, 30), 0.0, 0.0, 80.0, 50.0))
        coord.record_reading("K1", CleanupReading("M3", D(10, 40), 0.0, 0.0, 12.0, 50.0))
        self.assertEqual(coord.incidents["K1"].streak, 1)
        coord.record_reading("K1", CleanupReading("M4", D(10, 50), 0.0, 0.0, 9.0, 50.0))
        self.assertEqual(coord.incidents["K1"].streak, 2)

        # Stale compliant evidence cannot open the zone.
        stale = coord.release("K1", D(12, 21))
        self.assertEqual(stale.code, "readings-not-clear")
        self.assertFalse(coord.flight_recoverable("F1").accepted)

        released = coord.release("K1", D(11, 10))
        self.assertTrue(released.accepted)
        self.assertEqual(released.details["recovered_flights"], ("F1",))
        self.assertNotIn("fuel-spill:K1", self.monitor.zones)  # perimeter lifted
        self.assertEqual(coord.occupied, {})  # responders freed
        self.assertEqual(coord.held_flights(), frozenset())
        self.assertEqual(coord.recovered_flights(), frozenset({"F1"}))
        self.assertTrue(coord.flight_recoverable("F1").accepted)
        self.assertEqual(coord.release("K1", D(11, 10)).code, "already-released")

    def test_cleanup_blocked_when_arff_standby_degrades(self):
        coord = FuelSpillCoordinator(full_pool(), self.monitor)
        coord.report(SpillReport("R1", "K1", D(10), 0.0, 0.0, radius_m=5.0))
        coord.responders[ARFF + "-1"] = replace(coord.responders[ARFF + "-1"], active=False)

        denied = coord.begin_cleanup("K1")
        self.assertFalse(denied.accepted)
        self.assertEqual(denied.code, "standby-degraded")

        coord.responders[ARFF + "-1"] = replace(coord.responders[ARFF + "-1"], active=True)
        self.assertEqual(coord.begin_cleanup("K1").code, "cleanup-begun")

    def test_larger_duplicate_adds_only_delta_resources(self):
        coord = FuelSpillCoordinator(full_pool(), self.monitor)
        coord.report(SpillReport("R1", "K1", D(10), 0.0, 0.0, radius_m=5.0, affected_flights=frozenset({"F1"})))
        initial = frozenset(coord.incidents["K1"].perimeter.assigned)

        escalated = coord.report(SpillReport("R2", "K1", D(10, 10), 1.0, 1.0, radius_m=20.0, affected_flights=frozenset({"F2"})))
        self.assertEqual(escalated.code, "duplicate-report-escalated")
        self.assertEqual(escalated.details["radius_m"], 30.0)  # moderate perimeter
        current = coord.incidents["K1"].perimeter.assigned
        self.assertTrue(initial <= current)
        self.assertEqual(len(current - initial), 2)  # only +1 ARFF, +1 cleanup
        self.assertEqual(self.monitor.zones["fuel-spill:K1"].radius_m, 30.0)
        self.assertEqual(coord.held_flights(), frozenset({"F1", "F2"}))

    def test_escalation_without_capacity_keeps_original_perimeter(self):
        coord = FuelSpillCoordinator(responders((1, 1, 1, 0)), self.monitor)
        coord.report(SpillReport("R1", "K1", D(10), 0.0, 0.0, radius_m=5.0))
        before = coord.incidents["K1"].perimeter.assigned

        denied = coord.report(SpillReport("R2", "K1", D(10, 10), 0.0, 0.0, radius_m=20.0))
        self.assertFalse(denied.accepted)
        self.assertEqual(denied.code, "escalation-resources-short")
        self.assertEqual(coord.incidents["K1"].tier, SpillTier.MINOR)  # tier rolled back
        self.assertEqual(coord.incidents["K1"].perimeter.assigned, before)
        self.assertEqual(self.monitor.zones["fuel-spill:K1"].radius_m, 15.0)

        # Re-reporting the enlarged extent once augmentation exists re-escalates.
        coord.responders[ARFF + "-2"] = SpillResponder(ARFF + "-2", ARFF, frozenset({FOAM}))
        coord.responders[CLEANUP + "-2"] = SpillResponder(CLEANUP + "-2", CLEANUP, frozenset({FUEL_SPILL}))
        re_escalated = coord.report(SpillReport("R3", "K1", D(10, 20), 0.0, 0.0, radius_m=20.0))
        self.assertEqual(re_escalated.code, "duplicate-report-escalated")
        self.assertEqual(coord.incidents["K1"].tier, SpillTier.MODERATE)

    def test_one_responder_set_cannot_cover_two_open_incidents(self):
        coord = FuelSpillCoordinator(responders((1, 1, 1, 0)), self.monitor)
        self.assertTrue(coord.report(SpillReport("R1", "K1", D(10), 0.0, 0.0, radius_m=5.0)).accepted)

        second = coord.report(SpillReport("R2", "K2", D(10, 1), 100.0, 100.0, radius_m=5.0))
        self.assertFalse(second.accepted)
        self.assertEqual(second.code, "qualified-resources-short")

        # Release K1 (minor tier: two compliant readings) and its pool frees up.
        coord.record_reading("K1", CleanupReading("M1", D(10, 20), 0.0, 0.0, 5.0, 50.0))
        coord.record_reading("K1", CleanupReading("M2", D(10, 30), 0.0, 0.0, 4.0, 50.0))
        self.assertTrue(coord.release("K1", D(10, 40)).accepted)
        self.assertTrue(coord.establish_containment("K2", D(10, 41)).accepted)

    def test_readings_must_be_ordered_and_unique(self):
        coord = FuelSpillCoordinator(full_pool(), self.monitor)
        coord.report(SpillReport("R1", "K1", D(10), 0.0, 0.0, radius_m=5.0))
        coord.record_reading("K1", CleanupReading("M1", D(10, 20), 0.0, 0.0, 5.0, 50.0))
        with self.assertRaises(ValueError):
            coord.record_reading("K1", CleanupReading("M1", D(10, 20), 0.0, 0.0, 5.0, 50.0))
        with self.assertRaises(ValueError):
            coord.record_reading("K1", CleanupReading("M2", D(10, 19), 0.0, 0.0, 5.0, 50.0))

    def test_tier_classification(self):
        self.assertEqual(classify_tier(5.0, 10.0), SpillTier.MINOR)
        self.assertEqual(classify_tier(10.0, 2000.0), SpillTier.MODERATE)
        self.assertEqual(classify_tier(25.0, 5000.0), SpillTier.MAJOR)
        self.assertEqual(classify_tier(60.0, 0.0), SpillTier.CRITICAL)

    def test_workflow_publishes_containment_and_release_once(self):
        control = RampWorkflowControl()
        coord = FuelSpillCoordinator(full_pool(), self.monitor)
        action = lambda: coord.report(SpillReport("R1", "K1", D(10), 0.0, 0.0, radius_m=5.0, affected_flights=frozenset({"F1"})))

        first = control.commit_fuel_spill_containment("OP-1", "K1", D(10), action)
        second = control.commit_fuel_spill_containment("OP-1", "K1", D(10), action)
        self.assertIs(first, second)
        self.assertEqual(first.event.topic, "fuel.spill.containment.committed")
        self.assertEqual(control.stream.head, 1)

        coord.record_reading("K1", CleanupReading("M1", D(10, 20), 0.0, 0.0, 5.0, 50.0))
        coord.record_reading("K1", CleanupReading("M2", D(10, 30), 0.0, 0.0, 5.0, 50.0))
        release = control.release_fuel_spill_zone(
            "OP-2", "K1", D(10, 40), lambda: coord.release("K1", D(10, 40)),
        )
        self.assertEqual(release.event.topic, "fuel.spill.zone.released")
        self.assertEqual(control.stream.head, 2)

        # A rejected gate prevents both the domain change and the outward event.
        with self.assertRaises(WorkflowRejected):
            control.release_fuel_spill_zone(
                "OP-3", "K1", D(10, 41), lambda: self.fail("must not run"),
                requirements=(Decision(False, "readings-not-clear", "streak broken"),),
            )
        self.assertEqual(control.stream.head, 2)


if __name__ == "__main__":
    unittest.main()
