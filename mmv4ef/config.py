"""Configuration shared by the causal weather-forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time


VALIDATION_DATES = (
    date(2024, 9, 20),
    date(2024, 9, 23),
    date(2024, 9, 24),
    date(2024, 9, 26),
    date(2024, 10, 1),
    date(2024, 10, 2),
    date(2024, 10, 3),
    date(2024, 10, 4),
    date(2024, 10, 7),
    date(2024, 10, 8),
    date(2024, 10, 9),
    date(2024, 10, 10),
)

TEST_DATES = (
    date(2024, 10, 15),
    date(2024, 10, 16),
    date(2024, 10, 17),
    date(2024, 10, 18),
    date(2024, 10, 21),
    date(2024, 10, 22),
    date(2024, 10, 24),
    date(2024, 10, 28),
    date(2024, 10, 29),
    date(2024, 10, 30),
)

WET_NEA_CODES = frozenset({"LR", "RA", "LS", "SH", "PS", "HS", "TL", "HT", "HG"})

LOCAL_HISTORY_COLUMNS = (
    "temperature_c",
    "rh_pct",
    "wind_u_ms",
    "wind_v_ms",
    "solar_wm2",
    "rain_observed",
)

CONTINUOUS_TARGET_COLUMNS = LOCAL_HISTORY_COLUMNS[:5]
ECMWF_COLUMNS = (
    "temperature_c",
    "rh_pct",
    "wind_u_ms",
    "wind_v_ms",
    "solar_wm2",
    "cloud_fraction",
)


@dataclass(frozen=True)
class ForecastConfig:
    history_minutes: int = 60
    horizon_minutes: int = 60
    step_minutes: int = 5
    first_decision: time = time(8, 30)
    last_decision: time = time(18, 0)
    availability_delay_hours: float = 6.0
    timezone: str = "Asia/Singapore"
    validation_dates: tuple[date, ...] = VALIDATION_DATES
    test_dates: tuple[date, ...] = TEST_DATES

    def __post_init__(self) -> None:
        if self.history_minutes <= 0 or self.horizon_minutes <= 0 or self.step_minutes <= 0:
            raise ValueError("History, horizon, and step durations must be positive")
        if self.history_minutes % self.step_minutes or self.horizon_minutes % self.step_minutes:
            raise ValueError("History and horizon must be divisible by step_minutes")
        if self.first_decision > self.last_decision:
            raise ValueError("first_decision must not be later than last_decision")
        validation = set(self.validation_dates)
        test = set(self.test_dates)
        if len(validation) != len(self.validation_dates) or len(test) != len(self.test_dates):
            raise ValueError("Split date lists contain duplicates")
        if validation & test:
            raise ValueError("Validation and test dates overlap")

    @property
    def history_steps(self) -> int:
        return self.history_minutes // self.step_minutes

    @property
    def horizon_steps(self) -> int:
        return self.horizon_minutes // self.step_minutes


def split_for_date(value: date, cfg: ForecastConfig) -> str | None:
    if value in cfg.validation_dates:
        return "validation"
    if value in cfg.test_dates:
        return "test"
    if value < min(cfg.validation_dates):
        return "train"
    return None
