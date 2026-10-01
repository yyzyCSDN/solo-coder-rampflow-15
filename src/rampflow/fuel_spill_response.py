"""Apron fuel-spill response.

One coordinator owns the whole incident lifecycle that is otherwise split
across desks:

* the safety zone (cordon) is sized from the reported spill extent and can
  only grow while the incident is active;
* cleanup, ARFF standby and hazmat cover must be supplied by *qualified*
  assets, and an asset already committed to another active spill is refused;
* vapor (LEL) readings are kept in recording order as the continuous
  decontamination record, and the area reopens only after a configurable
  streak of consecutive compliant readings;
* a duplicate report for the same physical spill is rejected with the
  active incident id, so a second alert never occupies a second set of
  vehicles and crew;
* flights blocked by the cordon are released for resumption only when the
  area reopens.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Mapping

from .models import Decision
from .timeutil import utc


class SpillSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class SpillPhase(str, Enum):
    REPORTED = "reported"
    CORDONED = "cordoned"
    CLEANING = "cleaning"
    VERIFYING = "verifying"
    REOPENED = "reopened"
    CANCELLED = "cancelled"


ACTIVE_PHASES = frozenset(
    {
        SpillPhase.REPORTED,
        SpillPhase.CORDONED,
        SpillPhase.CLEANING,
        SpillPhase.VERIFYING,
    }
)

# Initial exclusion radius in metres, keyed by reported severity.
CORDON_RADIUS_M: Mapping[SpillSeverity, float] = {
    SpillSeverity.MINOR: 15.0,
    SpillSeverity.MODERATE: 30.0,
    SpillSeverity.MAJOR: 50.0,
}

# Capabilities that must be covered by the roster before cleanup starts.
REQUIRED_CAPABILITIES: Mapping[SpillSeverity, frozenset[str]] = {
    SpillSeverity.MINOR: frozenset({"fuel-spill-response"}),
    SpillSeverity.MODERATE: frozenset({"fuel-spill-response", "fire-standby"}),
    SpillSeverity.MAJOR: frozenset(
        {"fuel-spill-response", "fire-standby", "hazmat"}
    ),
}

# Reopening threshold: percent of lower explosive limit.
DEFAULT_READING_LIMIT_PCT_LEL = 10.0
DEFAULT_CONSECUTIVE_READINGS = 2
DEFAULT_READING_GAP_MIN = 5
DEFAULT_DEDUP_BUCKET_MIN = 15


@dataclass(frozen=True)
class SpillReport:
    location: str
    x: float
    y: float
    estimated_liters: float
    severity: SpillSeverity
    reported_at: datetime

    def __post_init__(self):
        if not self.location:
            raise ValueError("spill location required")
        if self.estimated_liters <= 0:
            raise ValueError("estimated spill volume must be positive")
        if not isinstance(self.severity, SpillSeverity):
            raise ValueError("invalid spill severity")
        object.__setattr__(self, "reported_at", utc(self.reported_at))


@dataclass(frozen=True)
class ResponseAsset:
    """A vehicle or crew team that can be committed to a spill response."""

    asset_id: str
    kind: str
    capabilities: frozenset[str]

    def __post_init__(self):
        if not self.asset_id or not self.kind:
            raise ValueError("asset id and kind required")


@dataclass(frozen=True)
class CleaningReading:
    at: datetime
    point: str
    pct_lel: float
    analyst: str = ""

    def __post_init__(self):
        object.__setattr__(self, "at", utc(self.at))
        if not self.point:
            raise ValueError("sampling point required")
        if self.pct_lel < 0:
            raise ValueError("LEL reading cannot be negative")


@dataclass(frozen=True)
class SpillJournalEntry:
    at: datetime
    code: str
    detail: str

    def __post_init__(self):
        object.__setattr__(self, "at", utc(self.at))


@dataclass(frozen=True)
class FuelSpillIncident:
    spill_id: str
    report_key: str
    location: str
    x: float
    y: float
    estimated_liters: float
    severity: SpillSeverity
    reported_at: datetime
    phase: SpillPhase
    cordon_radius_m: float
    roster: Mapping[str, str] = field(default_factory=dict)
    readings: tuple[CleaningReading, ...] = ()
    affected_flights: frozenset[str] = frozenset()
    journal: tuple[SpillJournalEntry, ...] = ()
    version: int = 1


class FuelSpillCoordinator:
    """Owns cordon, qualified assets, verification and flight resumption."""

    def __init__(
        self,
        assets=(),
        reading_limit_pct_lel: float = DEFAULT_READING_LIMIT_PCT_LEL,
        required_consecutive_readings: int = DEFAULT_CONSECUTIVE_READINGS,
        min_reading_gap_min: int = DEFAULT_READING_GAP_MIN,
        dedup_bucket_min: int = DEFAULT_DEDUP_BUCKET_MIN,
    ):
        if reading_limit_pct_lel < 0:
            raise ValueError("reading limit cannot be negative")
        if required_consecutive_readings < 1:
            raise ValueError("at least one compliant reading required")
        if min_reading_gap_min < 0 or dedup_bucket_min < 1:
            raise ValueError("invalid timing configuration")
        self.reading_limit = reading_limit_pct_lel
        self.required_consecutive_readings = required_consecutive_readings
        self.min_reading_gap = timedelta(minutes=min_reading_gap_min)
        self.dedup_bucket_min = dedup_bucket_min
        self.assets: dict[str, ResponseAsset] = {}
        self.incidents: dict[str, FuelSpillIncident] = {}
        self._report_keys: dict[str, str] = {}
        self._asset_holder: dict[str, str] = {}
        self._flight_holder: dict[str, str] = {}
        for asset in assets:
            self.register_asset(asset)

    # ------------------------------------------------------------------ setup

    def register_asset(self, asset: ResponseAsset) -> None:
        if asset.asset_id in self.assets:
            raise ValueError(f"duplicate response asset {asset.asset_id}")
        self.assets[asset.asset_id] = asset

    def default_report_key(self, report: SpillReport) -> str:
        """Correlation key for reports describing the same physical spill."""
        at = utc(report.reported_at)
        bucket = int(at.timestamp() // (self.dedup_bucket_min * 60))
        return f"{report.location}|{report.severity.value}|{bucket}"

    # ------------------------------------------------------------------ report

    def report_incident(
        self,
        spill_id: str,
        report: SpillReport,
        correlation_key: str | None = None,
    ) -> Decision:
        if spill_id in self.incidents:
            return Decision(False, "duplicate-incident-id", "spill id already used", {"spill_id": spill_id})
        key = correlation_key or self.default_report_key(report)
        active = self._report_keys.get(key)
        if active is not None and self.is_active(active):
            # Same event reported again: point at the existing incident and
            # deliberately allocate nothing.
            return Decision(
                False,
                "duplicate-report",
                "an active incident already covers this report",
                {"active_spill_id": active, "report_key": key},
            )
        inc = FuelSpillIncident(
            spill_id=spill_id,
            report_key=key,
            location=report.location,
            x=report.x,
            y=report.y,
            estimated_liters=report.estimated_liters,
            severity=report.severity,
            reported_at=report.reported_at,
            phase=SpillPhase.REPORTED,
            cordon_radius_m=CORDON_RADIUS_M[report.severity],
            journal=(SpillJournalEntry(report.reported_at, "reported", key),),
        )
        self.incidents[spill_id] = inc
        self._report_keys[key] = spill_id
        return Decision(
            True,
            "ok",
            "spill incident opened",
            {"spill_id": spill_id, "cordon_radius_m": inc.cordon_radius_m},
        )

    # ------------------------------------------------------------------ cordon

    def establish_cordon(self, spill_id: str, at: datetime) -> Decision:
        inc = self._require(spill_id)
        at = utc(at)
        if inc.phase != SpillPhase.REPORTED:
            return self._reject(inc, "cordon-already-established", "cordon established once")
        inc = self._next(
            inc,
            SpillPhase.CORDONED,
            SpillJournalEntry(at, "cordon-established", f"r={inc.cordon_radius_m:g}m"),
        )
        self.incidents[spill_id] = inc
        return Decision(True, "ok", "safety zone cordoned", self._incident_details(inc))

    def expand_cordon(self, spill_id: str, radius_m: float, at: datetime, reason: str = "") -> Decision:
        inc = self._require(spill_id)
        at = utc(at)
        if inc.phase not in ACTIVE_PHASES:
            return self._reject(inc, "incident-closed", "cannot resize a closed cordon")
        if radius_m <= inc.cordon_radius_m:
            return self._reject(
                inc,
                "cordon-must-grow",
                "cordon can only be expanded while active",
                {"current_radius_m": inc.cordon_radius_m, "requested_radius_m": radius_m},
            )
        inc = replace(
            inc,
            cordon_radius_m=radius_m,
            version=inc.version + 1,
            journal=inc.journal
            + (SpillJournalEntry(at, "cordon-expanded", f"{reason or 'spill extent reassessed'} r={radius_m:g}m"),),
        )
        self.incidents[spill_id] = inc
        return Decision(True, "ok", "safety zone expanded", self._incident_details(inc))

    # ------------------------------------------------------------------ assets

    def muster_resources(self, spill_id: str, roster: Mapping[str, str], at: datetime) -> Decision:
        """Commit qualified assets.

        ``roster`` maps required capability to asset id.  An asset may cover
        more than one capability but may never appear twice in one roster,
        and an asset already held by another active spill is refused.
        """
        inc = self._require(spill_id)
        at = utc(at)
        if inc.phase not in {SpillPhase.CORDONED, SpillPhase.CLEANING}:
            return self._reject(inc, "roster-wrong-phase", "resources muster against a cordoned incident")
        required = REQUIRED_CAPABILITIES[inc.severity]
        merged = dict(inc.roster)
        asset_ids = list(roster.values())
        if len(set(asset_ids)) != len(asset_ids):
            return self._reject(inc, "asset-double-rostered", "one asset cannot cover two roster slots")
        missing = required - frozenset(merged) - frozenset(roster)
        if missing:
            return self._reject(
                inc,
                "roster-incomplete",
                "qualified cover missing",
                {"missing_capabilities": sorted(missing)},
            )
        for capability, asset_id in roster.items():
            if capability in merged and merged[capability] != asset_id:
                return self._reject(
                    inc,
                    "capability-reassigned",
                    "roster slot already covered by another asset",
                    {"capability": capability, "assigned_asset": merged[capability]},
                )
            asset = self.assets.get(asset_id)
            if asset is None:
                return self._reject(inc, "unknown-asset", "asset not registered", {"asset_id": asset_id})
            if capability not in asset.capabilities:
                return self._reject(
                    inc,
                    "unqualified-asset",
                    "asset lacks the required qualification",
                    {"asset_id": asset_id, "capability": capability},
                )
            holder = self._asset_holder.get(asset_id)
            if holder is not None and holder != spill_id and self.is_active(holder):
                return self._reject(
                    inc,
                    "asset-occupied",
                    "asset already committed to another active spill",
                    {"asset_id": asset_id, "holder_spill_id": holder},
                )
        for capability, asset_id in roster.items():
            merged.setdefault(capability, asset_id)
            self._asset_holder[asset_id] = spill_id
        phase = SpillPhase.CLEANING if inc.phase == SpillPhase.CORDONED else inc.phase
        inc = self._next(
            replace(inc, roster=merged),
            phase,
            SpillJournalEntry(at, "resources-mustered", ",".join(sorted(merged))),
        )
        self.incidents[spill_id] = inc
        return Decision(True, "ok", "qualified response resources committed", self._incident_details(inc))

    def asset_holder(self, asset_id: str) -> str | None:
        holder = self._asset_holder.get(asset_id)
        return holder if holder is not None and self.is_active(holder) else None

    # ------------------------------------------------------------- verification

    def record_reading(self, spill_id: str, reading: CleaningReading) -> Decision:
        inc = self._require(spill_id)
        if inc.phase not in {SpillPhase.CLEANING, SpillPhase.VERIFYING}:
            return self._reject(inc, "reading-wrong-phase", "readings follow resource muster")
        if inc.readings and reading.at <= inc.readings[-1].at:
            return self._reject(
                inc,
                "reading-out-of-order",
                "detection readings must be recorded chronologically",
                {"last_reading_at": inc.readings[-1].at.isoformat(), "reading_at": reading.at.isoformat()},
            )
        compliant = reading.pct_lel <= self.reading_limit
        phase = SpillPhase.VERIFYING
        inc = replace(
            inc,
            phase=phase,
            readings=inc.readings + (reading,),
            version=inc.version + 1,
            journal=inc.journal
            + (
                SpillJournalEntry(
                    reading.at,
                    "reading-compliant" if compliant else "reading-exceeded",
                    f"{reading.point}={reading.pct_lel:g}%LEL",
                ),
            ),
        )
        self.incidents[spill_id] = inc
        streak = self._compliant_streak(inc)
        return Decision(
            True,
            "ok",
            "detection reading recorded",
            {**self._incident_details(inc), "compliant": compliant, "compliant_streak": len(streak)},
        )

    def reopening_decision(self, spill_id: str) -> Decision:
        inc = self._require(spill_id)
        if inc.phase == SpillPhase.REOPENED:
            return Decision(True, "already-open", "area already reopened", self._incident_details(inc))
        if inc.phase == SpillPhase.CANCELLED:
            return self._reject(inc, "incident-cancelled", "cancelled incident cannot reopen")
        if inc.phase in {SpillPhase.REPORTED, SpillPhase.CORDONED}:
            return self._reject(inc, "cleanup-not-started", "no detection readings on record")
        streak = self._compliant_streak(inc)
        need = self.required_consecutive_readings
        if len(streak) < need:
            last = inc.readings[-1] if inc.readings else None
            return self._reject(
                inc,
                "readings-not-consecutive",
                "reopening requires consecutive compliant readings",
                {
                    "compliant_streak": len(streak),
                    "required_streak": need,
                    "limit_pct_lel": self.reading_limit,
                    "last_pct_lel": None if last is None else last.pct_lel,
                },
            )
        window = streak[-need:]
        for prev, nxt in zip(window, window[1:]):
            if nxt.at - prev.at < self.min_reading_gap:
                return self._reject(
                    inc,
                    "sampling-too-close",
                    "consecutive compliant readings are too close together",
                    {
                        "previous_at": prev.at.isoformat(),
                        "next_at": nxt.at.isoformat(),
                        "min_gap_min": self.min_reading_gap.total_seconds() / 60,
                    },
                )
        return Decision(
            True,
            "ok",
            "detection criteria satisfied; area may reopen",
            self._incident_details(inc),
        )

    def reopen(self, spill_id: str, at: datetime) -> Decision:
        decision = self.reopening_decision(spill_id)
        if not decision.accepted:
            return decision
        inc = self.incidents[spill_id]
        inc = self._next(inc, SpillPhase.REOPENED, SpillJournalEntry(at, "reopened", "cordon lifted"))
        self._release(inc)
        self.incidents[spill_id] = inc
        flights = tuple(sorted(inc.affected_flights))
        return Decision(
            True,
            "ok",
            "area reopened and affected flights released",
            {**self._incident_details(inc), "released_flights": flights},
        )

    def cancel(self, spill_id: str, at: datetime, reason: str = "") -> Decision:
        inc = self._require(spill_id)
        if inc.phase not in ACTIVE_PHASES:
            return self._reject(inc, "incident-closed", "incident already closed")
        inc = self._next(
            inc,
            SpillPhase.CANCELLED,
            SpillJournalEntry(at, "cancelled", reason or "response stood down"),
        )
        self._release(inc)
        self.incidents[spill_id] = inc
        return Decision(True, "ok", "incident cancelled; resources released", self._incident_details(inc))

    # ----------------------------------------------------------------- flights

    def restrict_flight(self, spill_id: str, flight_id: str) -> Decision:
        inc = self._require(spill_id)
        if inc.phase not in ACTIVE_PHASES:
            return self._reject(inc, "incident-closed", "cannot restrict flights to a closed incident")
        holder = self._flight_holder.get(flight_id)
        if holder is not None and holder != spill_id and self.is_active(holder):
            return self._reject(
                inc,
                "flight-restricted-elsewhere",
                "flight already held by another active spill",
                {"flight_id": flight_id, "holder_spill_id": holder},
            )
        if flight_id in inc.affected_flights:
            return Decision(True, "ok", "flight already restricted", self._incident_details(inc))
        self._flight_holder[flight_id] = spill_id
        inc = replace(
            inc,
            affected_flights=inc.affected_flights | {flight_id},
            version=inc.version + 1,
        )
        self.incidents[spill_id] = inc
        return Decision(True, "ok", "flight restricted within cordon", self._incident_details(inc))

    def flight_clearance(self, flight_id: str) -> Decision:
        holder = self._flight_holder.get(flight_id)
        if holder is not None and self.is_active(holder):
            return Decision(
                False,
                "flight-held-by-cordon",
                "flight remains inside an active safety zone",
                {"flight_id": flight_id, "spill_id": holder},
            )
        return Decision(True, "ok", "flight clear to resume", {"flight_id": flight_id})

    def resume_flights(self, spill_id: str) -> Decision:
        inc = self._require(spill_id)
        if inc.phase != SpillPhase.REOPENED:
            return self._reject(inc, "cordon-active", "flights resume only after the area reopens")
        flights = tuple(sorted(inc.affected_flights))
        return Decision(
            True,
            "ok",
            "affected flights cleared for resumption",
            {**self._incident_details(inc), "flights": flights},
        )

    # ----------------------------------------------------------------- helpers

    def is_active(self, spill_id: str) -> bool:
        inc = self.incidents.get(spill_id)
        return inc is not None and inc.phase in ACTIVE_PHASES

    def _compliant_streak(self, inc: FuelSpillIncident) -> tuple[CleaningReading, ...]:
        streak: list[CleaningReading] = []
        for reading in inc.readings:
            if reading.pct_lel <= self.reading_limit:
                streak.append(reading)
            else:
                streak = []
        return tuple(streak)

    def _release(self, inc: FuelSpillIncident) -> None:
        committed = set(inc.roster.values())
        for asset_id in committed:
            if self._asset_holder.get(asset_id) == inc.spill_id:
                self._asset_holder.pop(asset_id, None)
        for flight_id in inc.affected_flights:
            if self._flight_holder.get(flight_id) == inc.spill_id:
                self._flight_holder.pop(flight_id, None)
        self._report_keys.pop(inc.report_key, None)

    def _next(self, inc: FuelSpillIncident, phase: SpillPhase, entry: SpillJournalEntry) -> FuelSpillIncident:
        return replace(
            inc,
            phase=phase,
            version=inc.version + 1,
            journal=inc.journal + (entry,),
        )

    def _require(self, spill_id: str) -> FuelSpillIncident:
        try:
            return self.incidents[spill_id]
        except KeyError:
            raise KeyError(f"unknown spill incident {spill_id}") from None

    def _reject(self, inc: FuelSpillIncident, code: str, reason: str, details: Mapping[str, object] | None = None) -> Decision:
        payload = {"spill_id": inc.spill_id, "phase": inc.phase.value, "version": inc.version}
        if details:
            payload.update(details)
        return Decision(False, code, reason, payload)

    def _incident_details(self, inc: FuelSpillIncident) -> dict[str, object]:
        return {
            "spill_id": inc.spill_id,
            "phase": inc.phase.value,
            "version": inc.version,
            "cordon_radius_m": inc.cordon_radius_m,
            "roster": dict(inc.roster),
            "affected_flights": sorted(inc.affected_flights),
            "reading_count": len(inc.readings),
        }
