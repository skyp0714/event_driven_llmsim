"""Session-level arrival and closed-backlog admission configuration."""

import argparse
from dataclasses import dataclass
import math


SESSION_ARRIVAL_MODES = ("trace", "poisson", "backlog")
MEASUREMENT_COHORT_SELECTIONS = (
    "completion_order",
    "admission_order",
)


@dataclass(frozen=True)
class SessionAdmissionConfig:
    """Control when agentic sessions enter the online simulator.

    ``trace`` preserves first-call arrival timestamps already present in the
    workload. ``poisson`` replaces only those first-call arrivals with a
    deterministic exponential inter-arrival process. When
    ``max_active_sessions`` is positive, offered Poisson sessions wait in a
    FIFO admission backlog for a slot. ``backlog`` ignores first-call trace
    arrivals and maintains a closed population of at most
    ``max_active_sessions`` sessions. A session keeps its slot through every
    recorded human/tool gap and releases it only when its final LLM request
    completes.
    """

    mode: str = "trace"
    max_active_sessions: int = 0
    backlog_epochs: int = 1
    session_arrival_rate_sps: float = 0.0
    session_arrival_seed: int = 42
    warmup_completions: int = 0
    measure_completions: int = 0
    stop_after_measurement: bool = False
    measurement_cohort_selection: str = "completion_order"

    def __post_init__(self):
        mode = str(self.mode).strip().lower()
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self, "max_active_sessions", int(self.max_active_sessions))
        object.__setattr__(self, "backlog_epochs", int(self.backlog_epochs))
        object.__setattr__(
            self,
            "session_arrival_rate_sps",
            float(self.session_arrival_rate_sps),
        )
        object.__setattr__(
            self, "session_arrival_seed", int(self.session_arrival_seed))
        object.__setattr__(
            self, "warmup_completions", int(self.warmup_completions))
        object.__setattr__(
            self, "measure_completions", int(self.measure_completions))
        object.__setattr__(
            self, "stop_after_measurement", bool(self.stop_after_measurement))
        measurement_cohort_selection = str(
            self.measurement_cohort_selection).strip().lower()
        object.__setattr__(
            self,
            "measurement_cohort_selection",
            measurement_cohort_selection,
        )
        if mode not in SESSION_ARRIVAL_MODES:
            raise ValueError(
                f"Unknown session arrival mode {self.mode!r}; expected one "
                f"of {', '.join(SESSION_ARRIVAL_MODES)}")
        if self.max_active_sessions < 0:
            raise ValueError("max_active_sessions must be non-negative")
        if self.backlog_epochs <= 0:
            raise ValueError("backlog_epochs must be positive")
        if self.warmup_completions < 0:
            raise ValueError("warmup_completions must be non-negative")
        if self.measure_completions < 0:
            raise ValueError("measure_completions must be non-negative")
        if self.stop_after_measurement and self.measure_completions <= 0:
            raise ValueError(
                "stop_after_measurement requires measure_completions > 0")
        if measurement_cohort_selection not in MEASUREMENT_COHORT_SELECTIONS:
            raise ValueError(
                "Unknown measurement cohort selection "
                f"{measurement_cohort_selection!r}; expected one of "
                f"{', '.join(MEASUREMENT_COHORT_SELECTIONS)}")
        if (measurement_cohort_selection == "admission_order"
                and mode != "backlog"):
            raise ValueError(
                "admission_order measurement cohorts require backlog mode")
        if mode == "backlog" and self.max_active_sessions <= 0:
            raise ValueError(
                "backlog mode requires max_active_sessions > 0")
        if (mode == "poisson"
                and (not math.isfinite(self.session_arrival_rate_sps)
                     or self.session_arrival_rate_sps <= 0)):
            raise ValueError(
                "poisson mode requires session_arrival_rate_sps > 0")


def add_session_admission_arguments(parser):
    """Add online session-load CLI arguments to an argparse parser."""
    parser.add_argument(
        "--session-arrival-mode",
        choices=SESSION_ARRIVAL_MODES,
        default="trace",
        help=(
            "session admission model: trace preserves workload first-call "
            "timestamps, poisson generates exponential inter-arrivals, and "
            "backlog keeps a closed population of at most "
            "--max-active-sessions sessions"
        ),
    )
    parser.add_argument(
        "--session-arrival-rate-sps",
        type=float,
        default=0.0,
        help=(
            "Poisson first-session arrival rate in sessions/second; required "
            "and positive in poisson mode"
        ),
    )
    parser.add_argument(
        "--session-arrival-seed",
        type=int,
        default=42,
        help="random seed for deterministic Poisson session arrivals",
    )
    parser.add_argument(
        "--max-active-sessions",
        type=int,
        default=0,
        help=(
            "closed-backlog session population K; required and positive in "
            "backlog mode; optional positive FIFO admission limit in "
            "poisson mode"
        ),
    )
    parser.add_argument(
        "--session-backlog-epochs",
        type=int,
        default=1,
        help=(
            "number of deterministic passes over the session template set "
            "in backlog mode (default: 1)"
        ),
    )
    parser.add_argument(
        "--session-warmup-completions",
        type=int,
        default=0,
        help=(
            "warmup size: completion_order excludes this many completed "
            "sessions, while backlog admission_order excludes this many "
            "fixed epoch-major admission-prefix sessions (default: 0)"
        ),
    )
    parser.add_argument(
        "--session-measure-completions",
        type=int,
        default=0,
        help=(
            "number of completed sessions included after warmup; 0 includes "
            "all remaining completions"
        ),
    )
    parser.add_argument(
        "--session-stop-after-measurement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "stop the online ASTRA simulation at the first batch boundary "
            "that reaches warmup + measured session completions; requests "
            "outside the fixed completion window are not drained"
        ),
    )
    parser.add_argument(
        "--session-measurement-cohort-selection",
        choices=MEASUREMENT_COHORT_SELECTIONS,
        default="completion_order",
        help=(
            "select measured sessions by completion order (legacy default) "
            "or by deterministic closed-backlog admission order; "
            "admission_order interprets warmup as a fixed admission-prefix "
            "exclusion and requires backlog mode"
        ),
    )
    return parser


def session_admission_from_args(args):
    """Build validated session admission configuration from CLI arguments."""
    return SessionAdmissionConfig(
        mode=args.session_arrival_mode,
        max_active_sessions=args.max_active_sessions,
        backlog_epochs=args.session_backlog_epochs,
        session_arrival_rate_sps=args.session_arrival_rate_sps,
        session_arrival_seed=args.session_arrival_seed,
        warmup_completions=args.session_warmup_completions,
        measure_completions=args.session_measure_completions,
        stop_after_measurement=getattr(
            args, "session_stop_after_measurement", False),
        measurement_cohort_selection=getattr(
            args,
            "session_measurement_cohort_selection",
            "completion_order",
        ),
    )
