from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable

from .apron_safety import ApronSafetyMonitor, SafetyZone
from .models import Decision
from .timeutil import Window, utc

# Responder roles (vehicles and personnel share one qualified-resource pool).
ARFF = "arff"
CLEANUP = "cleanup"
SAFETY = "safety"
ENVIRONMENTAL = "environmental"

FOAM = "foam"
FUEL_SPILL = "fuel-spill"
SPILL_SUPERVISOR = "fuel-spill-supervisor"
ENV_MONITOR = "environmental-monitor"

ROLE_QUALIFICATION = {
    ARFF: FOAM,
    CLEANUP: FUEL_SPILL,
    SAFETY: SPILL_SUPERVISOR,
    ENVIRONMENTAL: ENV_MONITOR,
}

PERIMETER_BUFFER_M = 5.0
ZONE_ACTIVITY = "fuel-spill"


class SpillState(str, Enum):
    REPORTED = "reported"
    CONTAINED = "contained"
    CLEANING = "cleaning"
    MONITORING = "monitoring"
    RELEASED = "released"
    CANCELLED = "cancelled"


class SpillTier(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass(frozen=True)
class TierPolicy:
    tier: SpillTier
    max_radius_m: float
    max_volume_l: float
    perimeter_radius_m: float
    arff: int
    cleanup: int
    safety: int
    environmental: int
    required_readings: int


_TIERS: tuple[TierPolicy, ...] = (
    TierPolicy(SpillTier.MINOR, 10.0, 200.0, 15.0, 1, 1, 1, 0, 2),
    TierPolicy(SpillTier.MODERATE, 25.0, 2000.0, 30.0, 2, 2, 1, 0, 2),
    TierPolicy(SpillTier.MAJOR, 50.0, 10000.0, 50.0, 3, 3, 1, 1, 3),
    TierPolicy(SpillTier.CRITICAL, float("inf"), float("inf"), 100.0, 4, 4, 1, 1, 3),
)

_TIER_INDEX = {p.tier: p for p in _TIERS}
_TIER_ORDER = {tier: i for i, tier in enumerate(SpillTier)}


def classify_tier(radius_m: float, volume_l: float) -> SpillTier:
    if radius_m < 0 or volume_l < 0:
        raise ValueError("spill extent cannot be negative")
    for policy in _TIERS:
        if radius_m <= policy.max_radius_m and volume_l <= policy.max_volume_l:
            return policy.tier
    return SpillTier.CRITICAL


def policy_for(tier: SpillTier) -> TierPolicy:
    return _TIER_INDEX[tier]


def perimeter_radius(policy: TierPolicy, spill_radius_m: float) -> float:
    return max(policy.perimeter_radius_m, spill_radius_m + PERIMETER_BUFFER_M)


@dataclass(frozen=True)
class SpillReport:
    report_id: str
    incident_key: str
    at: datetime
    x: float
    y: float
    radius_m: float = 0.0
    volume_l: float = 0.0
    affected_flights: frozenset[str] = frozenset()
    source: str = ""

    def __post_init__(self):
        object.__setattr__(self, "at", utc(self.at))
        if not self.report_id or not self.incident_key:
            raise ValueError("report and incident identity required")
        if self.radius_m < 0 or self.volume_l < 0:
            raise ValueError("spill extent cannot be negative")


@dataclass(frozen=True)
class SpillResponder:
    """A vehicle (ARFF tender, cleanup unit) or person (officer) on duty."""

    responder_id: str
    kind: str
    qualifications: frozenset[str]
    location: str = ""
    active: bool = True

    def __post_init__(self):
        if not self.responder_id:
            raise ValueError("responder id required")
        if self.kind not in ROLE_QUALIFICATION:
            raise ValueError(f"unknown responder role {self.kind}")


@dataclass(frozen=True)
class CleanupReading:
    reading_id: str
    at: datetime
    x: float
    y: float
    value: float
    limit: float
    method: str = ""

    def __post_init__(self):
        object.__setattr__(self, "at", utc(self.at))
        if not self.reading_id:
            raise ValueError("reading id required")
        if self.value < 0 or self.limit <= 0:
            raise ValueError("invalid cleanup reading")

    @property
    def compliant(self) -> bool:
        return self.value <= self.limit


@dataclass(frozen=True)
class ContainmentPerimeter:
    incident_id: str
    tier: SpillTier
    zone: SafetyZone
    assigned: frozenset[str]
    established_at: datetime


@dataclass(frozen=True)
class FuelSpillIncident:
    incident_id: str
    state: SpillState
    tier: SpillTier
    x: float
    y: float
    radius_m: float
    volume_l: float
    reports: tuple[str, ...]
    affected_flights: frozenset[str]
    perimeter: ContainmentPerimeter | None
    readings: tuple[CleanupReading, ...] = ()
    streak: int = 0
    released_at: datetime | None = None
    version: int = 1

    def __post_init__(self):
        if self.released_at is not None:
            object.__setattr__(self, "released_at", utc(self.released_at))


class FuelSpillCoordinator:
    """Owns the full apron fuel-spill lifecycle for one airport.

    Containment perimeters are pushed into an optional ``ApronSafetyMonitor`` so
    vehicle intrusions are alerted against the same safety zone that holds the
    flights.  A responder (vehicle or person) can only be mustered onto one open
    incident at a time, and duplicate reports of the same incident never occupy
    a second resource set.
    """

    def __init__(
        self,
        responders: Iterable[SpillResponder] = (),
        monitor: ApronSafetyMonitor | None = None,
        perimeter_window_hours: int = 12,
    ):
        self.responders = {r.responder_id: r for r in responders}
        if len(self.responders) != len(list(responders)):
            raise ValueError("duplicate responder")
        self.monitor = monitor
        self.perimeter_window_hours = perimeter_window_hours
        self.incidents: dict[str, FuelSpillIncident] = {}
        self.occupied: dict[str, str] = {}
        self._report_incident: dict[str, str] = {}

    # ------------------------------------------------------------------ intake

    def report(self, report: SpillReport) -> Decision:
        """Register a spill report; duplicate reports link, never re-muster."""
        prior_incident = self._report_incident.get(report.report_id)
        if prior_incident is not None:
            if prior_incident != report.incident_key:
                raise ValueError("report id reused for a different incident")
            inc = self.incidents[prior_incident]
            return Decision(
                True,
                "duplicate-report",
                "report already recorded for this incident",
                {"incident_id": inc.incident_id, "state": inc.state.value},
            )

        inc = self.incidents.get(report.incident_key)
        if inc is None:
            inc = FuelSpillIncident(
                incident_id=report.incident_key,
                state=SpillState.REPORTED,
                tier=classify_tier(report.radius_m, report.volume_l),
                x=report.x,
                y=report.y,
                radius_m=report.radius_m,
                volume_l=report.volume_l,
                reports=(report.report_id,),
                affected_flights=frozenset(report.affected_flights),
                perimeter=None,
            )
            self.incidents[inc.incident_id] = inc
            self._report_incident[report.report_id] = inc.incident_id
            return self.establish_containment(inc.incident_id, report.at)

        # Another caller already reported the same physical incident: link the
        # extra evidence (flights, extent) without touching any resource set.
        self._report_incident[report.report_id] = inc.incident_id
        grew = report.radius_m > inc.radius_m
        inc = replace(
            inc,
            reports=inc.reports + (report.report_id,),
            affected_flights=inc.affected_flights | frozenset(report.affected_flights),
            radius_m=max(inc.radius_m, report.radius_m),
            volume_l=max(inc.volume_l, report.volume_l),
            x=report.x if grew else inc.x,
            y=report.y if grew else inc.y,
            version=inc.version + 1,
        )
        new_tier = classify_tier(inc.radius_m, inc.volume_l)
        details = {
            "incident_id": inc.incident_id,
            "existing_reports": len(inc.reports),
            "tier": inc.tier.value,
        }
        if _TIER_ORDER[new_tier] > _TIER_ORDER[inc.tier]:
            prior = inc
            self.incidents[inc.incident_id] = replace(inc, tier=new_tier, version=inc.version + 1)
            decision = self.establish_containment(inc.incident_id, report.at)
            if not decision.accepted:
                # Keep the original tier and perimeter until augmentation exists.
                self.incidents[inc.incident_id] = prior
                return Decision(
                    False,
                    "escalation-resources-short",
                    "linked report enlarges the spill but augmentation is short",
                    {**details, "missing": decision.details.get("missing", {})},
                )
            return Decision(
                True,
                "duplicate-report-escalated",
                "linked report enlarged the containment perimeter",
                {**details, "tier": new_tier.value, "radius_m": decision.details["radius_m"]},
            )
        self.incidents[inc.incident_id] = inc
        return Decision(
            True,
            "duplicate-report-linked",
            "report linked to existing incident; resource set unchanged",
            details,
        )

    # ------------------------------------------------------------- containment

    def _role_counts(self, policy: TierPolicy) -> tuple[tuple[str, int], ...]:
        return (
            (ARFF, policy.arff),
            (CLEANUP, policy.cleanup),
            (SAFETY, policy.safety),
            (ENVIRONMENTAL, policy.environmental),
        )

    def _qualified(self, responder: SpillResponder, role: str) -> bool:
        qualification = ROLE_QUALIFICATION[role]
        return (
            responder.active
            and responder.kind == role
            and qualification in responder.qualifications
        )

    def establish_containment(self, incident_id: str, at: datetime) -> Decision:
        """Muster the qualified resource set and raise the safety zone.

        Reconciles an already-contained incident (e.g. an escalated tier): the
        responders already on scene stay, and only the shortfall is added.
        """
        at = utc(at)
        inc = self.incidents.get(incident_id)
        if inc is None:
            raise KeyError(incident_id)
        if inc.state in {SpillState.RELEASED, SpillState.CANCELLED}:
            return Decision(False, "invalid-state", "incident is closed", {"state": inc.state.value})

        policy = policy_for(inc.tier)
        assigned: set[str] = set()
        if inc.perimeter is not None:
            assigned = {
                rid
                for rid in inc.perimeter.assigned
                if rid in self.responders and self.occupied.get(rid) == incident_id
            }

        missing: dict[str, int] = {}
        additions: list[str] = []
        for role, required in self._role_counts(policy):
            have = sum(1 for rid in assigned if self.responders[rid].kind == role)
            short = required - have
            if short <= 0:
                continue
            candidates = sorted(
                rid
                for rid, responder in self.responders.items()
                if rid not in assigned
                and self.occupied.get(rid, incident_id) == incident_id
                and self._qualified(responder, role)
            )
            if len(candidates) < short:
                missing[role] = short - len(candidates)
                continue
            additions.extend(candidates[:short])

        if missing:
            return Decision(
                False,
                "qualified-resources-short",
                "not enough qualified responders to establish containment",
                {"incident_id": incident_id, "tier": inc.tier.value, "missing": missing},
            )

        assigned.update(additions)
        radius = perimeter_radius(policy, inc.radius_m)
        window = Window(at, at + timedelta(hours=self.perimeter_window_hours))
        zone = SafetyZone(
            zone_id=f"fuel-spill:{incident_id}",
            center_x=inc.x,
            center_y=inc.y,
            radius_m=radius,
            activity=ZONE_ACTIVITY,
            window=window,
            severity=_TIER_ORDER[inc.tier] + 1,
        )
        perimeter = ContainmentPerimeter(
            incident_id=incident_id,
            tier=inc.tier,
            zone=zone,
            assigned=frozenset(assigned),
            established_at=at,
        )
        for rid in assigned:
            self.occupied[rid] = incident_id
        if self.monitor is not None:
            self.monitor.set_zone(zone)

        preserved = inc.state in {SpillState.CLEANING, SpillState.MONITORING}
        inc = replace(
            inc,
            state=inc.state if preserved else SpillState.CONTAINED,
            perimeter=perimeter,
            version=inc.version + 1,
        )
        self.incidents[incident_id] = inc
        arff = sorted(rid for rid in assigned if self.responders[rid].kind == ARFF)
        return Decision(
            True,
            "containment-established",
            "safety zone raised and qualified responders mustered",
            {
                "incident_id": incident_id,
                "tier": inc.tier.value,
                "radius_m": radius,
                "assigned": tuple(sorted(assigned)),
                "added": tuple(sorted(additions)),
                "arff_standby": tuple(arff),
            },
        )

    def arff_standby(self, incident_id: str) -> tuple[str, ...]:
        inc = self.incidents[incident_id]
        if inc.perimeter is None:
            return ()
        return tuple(
            sorted(
                rid
                for rid in inc.perimeter.assigned
                if rid in self.responders
                and self.responders[rid].kind == ARFF
                and self.responders[rid].active
                and self.occupied.get(rid) == incident_id
            )
        )

    def begin_cleanup(self, incident_id: str) -> Decision:
        inc = self.incidents[incident_id]
        if inc.state != SpillState.CONTAINED or inc.perimeter is None:
            return Decision(False, "invalid-state", "cleanup requires an active containment", {"state": inc.state.value})
        required = policy_for(inc.tier).arff
        standby = self.arff_standby(incident_id)
        if len(standby) < required:
            return Decision(
                False,
                "standby-degraded",
                "ARFF foam standby must cover the spill during cleanup",
                {"required": required, "standing_by": len(standby)},
            )
        self.incidents[incident_id] = replace(inc, state=SpillState.CLEANING, version=inc.version + 1)
        return Decision(True, "cleanup-begun", "cleanup underway under ARFF standby", {"arff_standby": standby})

    def begin_monitoring(self, incident_id: str) -> Decision:
        inc = self.incidents[incident_id]
        if inc.state != SpillState.CLEANING:
            return Decision(False, "invalid-state", "monitoring follows cleanup", {"state": inc.state.value})
        self.incidents[incident_id] = replace(inc, state=SpillState.MONITORING, version=inc.version + 1)
        return Decision(True, "monitoring-begun", "post-cleanup detection monitoring underway")

    # ------------------------------------------------------------ detection log

    def record_reading(self, incident_id: str, reading: CleanupReading) -> Decision:
        """Append a cleanup-detection reading; any failure resets the streak."""
        inc = self.incidents.get(incident_id)
        if inc is None:
            raise KeyError(incident_id)
        if inc.perimeter is None or inc.state not in {
            SpillState.CONTAINED,
            SpillState.CLEANING,
            SpillState.MONITORING,
        }:
            return Decision(False, "invalid-state", "readings require active containment", {"state": inc.state.value})
        if any(r.reading_id == reading.reading_id for r in inc.readings):
            raise ValueError("duplicate cleanup reading")
        if inc.readings and reading.at < inc.readings[-1].at:
            raise ValueError("out-of-order cleanup reading")
        streak = inc.streak + 1 if reading.compliant else 0
        inc = replace(
            inc,
            readings=inc.readings + (reading,),
            streak=streak,
            version=inc.version + 1,
        )
        self.incidents[incident_id] = inc
        return Decision(
            True,
            "reading-recorded",
            "reading within limit" if reading.compliant else "reading exceeds limit; streak reset",
            {
                "compliant": reading.compliant,
                "streak": streak,
                "required_consecutive": policy_for(inc.tier).required_readings,
            },
        )

    def can_release(self, incident_id: str, at: datetime, max_reading_age_min: int = 30) -> Decision:
        at = utc(at)
        inc = self.incidents.get(incident_id)
        if inc is None:
            raise KeyError(incident_id)
        if inc.state == SpillState.RELEASED:
            return Decision(True, "already-released", "zone already released", {"incident_id": incident_id})
        if inc.state == SpillState.CANCELLED:
            return Decision(False, "invalid-state", "incident was cancelled")
        if inc.perimeter is None:
            return Decision(False, "not-contained", "no containment perimeter established")
        required = policy_for(inc.tier).required_readings
        if not inc.readings:
            return Decision(False, "readings-not-clear", "no cleanup readings recorded", {"streak": 0, "required": required})
        last = inc.readings[-1]
        age_min = (at - last.at).total_seconds() / 60.0
        details = {
            "streak": inc.streak,
            "required": required,
            "last_compliant": last.compliant,
            "last_reading_age_min": age_min,
        }
        if inc.streak < required:
            return Decision(False, "readings-not-clear", "consecutive compliant readings below requirement", details)
        if age_min < 0:
            return Decision(False, "readings-not-clear", "release precedes last reading", details)
        if age_min > max_reading_age_min:
            return Decision(False, "readings-not-clear", "last compliant reading is stale", details)
        return Decision(True, "clear-to-release", "detection streak supports reopening", details)

    def release(self, incident_id: str, at: datetime, max_reading_age_min: int = 30) -> Decision:
        """Lift the perimeter, free every responder and clear the flight holds."""
        at = utc(at)
        inc = self.incidents[incident_id]
        if inc.state == SpillState.RELEASED:
            return Decision(True, "already-released", "zone already released", {"incident_id": incident_id})
        gate = self.can_release(incident_id, at, max_reading_age_min)
        if not gate.accepted:
            return gate
        if self.monitor is not None:
            self.monitor.close_zone(inc.perimeter.zone.zone_id)
        freed = tuple(sorted(rid for rid, owner in self.occupied.items() if owner == incident_id))
        for rid in freed:
            del self.occupied[rid]
        self.incidents[incident_id] = replace(
            inc,
            state=SpillState.RELEASED,
            released_at=at,
            version=inc.version + 1,
        )
        return Decision(
            True,
            "released",
            "containment lifted; affected flights may resume",
            {"incident_id": incident_id, "freed": freed, "recovered_flights": tuple(sorted(inc.affected_flights))},
        )

    def cancel(self, incident_id: str) -> Decision:
        inc = self.incidents.get(incident_id)
        if inc is None:
            raise KeyError(incident_id)
        if inc.state == SpillState.CANCELLED:
            return Decision(True, "already-cancelled", "incident already cancelled")
        if inc.perimeter is not None or inc.state == SpillState.RELEASED:
            return Decision(False, "invalid-state", "contained incidents must be released, not cancelled")
        self.incidents[incident_id] = replace(inc, state=SpillState.CANCELLED, version=inc.version + 1)
        return Decision(True, "cancelled", "unconfirmed report cancelled")

    # ------------------------------------------------------------------ flights

    def held_flights(self) -> frozenset[str]:
        held: set[str] = set()
        for inc in self.incidents.values():
            if inc.state in {SpillState.RELEASED, SpillState.CANCELLED}:
                continue
            held.update(inc.affected_flights)
        return frozenset(held)

    def recovered_flights(self) -> frozenset[str]:
        recovered: set[str] = set()
        for inc in self.incidents.values():
            if inc.state == SpillState.RELEASED:
                recovered.update(inc.affected_flights)
        return frozenset(f for f in recovered if f not in self.held_flights())

    def flight_recoverable(self, flight_id: str) -> Decision:
        for inc in self.incidents.values():
            if flight_id in inc.affected_flights and inc.state not in {SpillState.RELEASED, SpillState.CANCELLED}:
                return Decision(
                    False,
                    "flight-held",
                    "flight remains inside an active spill perimeter",
                    {"incident_id": inc.incident_id, "state": inc.state.value},
                )
        return Decision(True, "flight-clear", "no active spill perimeter affects this flight")
