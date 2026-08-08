"""Pinned Open Power System Data ingestion.

The raw 130 MB CSV is never retained by default.  A network source is streamed
to a temporary file while its SHA256 is computed, selected columns are written
to a compact NPZ, and the temporary file is removed.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np


OPSD_DATA_VERSION = "2020-10-06"
OPSD_DATA_DOI = "10.25832/time_series/2020-10-06"
OPSD_DATA_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)
OPSD_EXPECTED_CONTENT_LENGTH = 130_339_665

OPSD_MARKET_COLUMNS = {
    "AT": {
        "load_actual": "AT_load_actual_entsoe_transparency",
        "load_forecast": "AT_load_forecast_entsoe_transparency",
        "price": "AT_price_day_ahead",
        "solar": "AT_solar_generation_actual",
        "wind": "AT_wind_onshore_generation_actual",
    },
    "DK_1": {
        "load_actual": "DK_1_load_actual_entsoe_transparency",
        "load_forecast": "DK_1_load_forecast_entsoe_transparency",
        "price": "DK_1_price_day_ahead",
        "solar": "DK_1_solar_generation_actual",
        "wind": "DK_1_wind_generation_actual",
    },
    "DK_2": {
        "load_actual": "DK_2_load_actual_entsoe_transparency",
        "load_forecast": "DK_2_load_forecast_entsoe_transparency",
        "price": "DK_2_price_day_ahead",
        "solar": "DK_2_solar_generation_actual",
        "wind": "DK_2_wind_generation_actual",
    },
    "GB_GBN": {
        "load_actual": "GB_GBN_load_actual_entsoe_transparency",
        "load_forecast": "GB_GBN_load_forecast_entsoe_transparency",
        "price": "GB_GBN_price_day_ahead",
        "solar": "GB_GBN_solar_generation_actual",
        "wind": "GB_GBN_wind_generation_actual",
    },
    "DE_LU": {
        "load_actual": "DE_LU_load_actual_entsoe_transparency",
        "load_forecast": "DE_LU_load_forecast_entsoe_transparency",
        "price": "DE_LU_price_day_ahead",
        "solar": "DE_LU_solar_generation_actual",
        "wind": "DE_LU_wind_generation_actual",
    },
    "IE_sem": {
        "load_actual": "IE_sem_load_actual_entsoe_transparency",
        "load_forecast": "IE_sem_load_forecast_entsoe_transparency",
        "price": "IE_sem_price_day_ahead",
        "solar": None,
        "wind": "IE_sem_wind_onshore_generation_actual",
    },
    **{
        market: {
            "load_actual": f"{market}_load_actual_entsoe_transparency",
            "load_forecast": f"{market}_load_forecast_entsoe_transparency",
            "price": f"{market}_price_day_ahead",
            "solar": f"{market}_solar_generation_actual",
            "wind": f"{market}_wind_onshore_generation_actual",
        }
        for market in (
            "IT_CNOR", "IT_CSUD", "IT_NORD", "IT_SARD", "IT_SICI",
            "IT_SUD",
        )
    },
    **{
        market: {
            "load_actual": f"{market}_load_actual_entsoe_transparency",
            "load_forecast": f"{market}_load_forecast_entsoe_transparency",
            "price": f"{market}_price_day_ahead",
            "solar": None,
            "wind": f"{market}_wind_onshore_generation_actual",
        }
        for market in (
            "NO_1", "NO_2", "NO_3", "NO_4", "NO_5",
            "SE_1", "SE_2", "SE_3", "SE_4",
        )
    },
}

DEFAULT_MARKETS = ("AT", "DK_1", "DK_2", "GB_GBN")
EXTENDED_MARKETS = tuple(OPSD_MARKET_COLUMNS)
DEFAULT_YEARS = (2017, 2018, 2019)


def _sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), int(size)


@contextmanager
def _materialized_source(source):
    source = str(source)
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"}:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest, size = _sha256_file(path)
        yield path, {
            "kind": "local_file",
            "source": str(path),
            "sha256": digest,
            "content_length": size,
            "etag": None,
            "last_modified": None,
        }
        return

    temporary = tempfile.NamedTemporaryFile(
        prefix="opsd_time_series_", suffix=".csv", delete=False)
    path = Path(temporary.name)
    digest = hashlib.sha256()
    size = 0
    headers = None
    try:
        with temporary, urlopen(source) as response:
            headers = response.headers
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                temporary.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if source == OPSD_DATA_URL and size != OPSD_EXPECTED_CONTENT_LENGTH:
            raise ValueError(
                "pinned OPSD source length changed: "
                f"observed {size}, expected {OPSD_EXPECTED_CONTENT_LENGTH}"
            )
        yield path, {
            "kind": "https",
            "source": source,
            "sha256": digest.hexdigest(),
            "content_length": int(size),
            "etag": headers.get("ETag") if headers is not None else None,
            "last_modified": (
                headers.get("Last-Modified") if headers is not None else None
            ),
        }
    finally:
        path.unlink(missing_ok=True)


def _validate_markets(markets: Iterable[str]):
    values = tuple(str(value) for value in markets)
    unknown = sorted(set(values) - set(OPSD_MARKET_COLUMNS))
    if unknown:
        raise ValueError(f"unsupported OPSD markets: {unknown}")
    if not values:
        raise ValueError("at least one OPSD market is required")
    return values


def preprocess_opsd(
    output,
    *,
    source=OPSD_DATA_URL,
    markets=DEFAULT_MARKETS,
    years=DEFAULT_YEARS,
    interpolation_limit=6,
):
    """Create a compact, provenance-bearing OPSD market archive."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("OPSD preprocessing requires pandas") from exc

    markets = _validate_markets(markets)
    years = tuple(sorted({int(value) for value in years}))
    if not years:
        raise ValueError("at least one OPSD year is required")
    interpolation_limit = max(0, int(interpolation_limit))
    wanted = {"utc_timestamp"}
    for market in markets:
        wanted.update(
            value for value in OPSD_MARKET_COLUMNS[market].values()
            if value is not None
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    market_metadata = {}
    with _materialized_source(source) as (source_path, source_meta):
        frame = pd.read_csv(
            source_path,
            usecols=lambda name: name in wanted,
            low_memory=False,
        )
        frame["utc_timestamp"] = pd.to_datetime(
            frame["utc_timestamp"], utc=True, errors="raise")
        frame = frame[frame["utc_timestamp"].dt.year.isin(years)].copy()
        frame.sort_values("utc_timestamp", inplace=True)
        for market in markets:
            columns = OPSD_MARKET_COLUMNS[market]
            available = {
                key: value for key, value in columns.items()
                if value is not None and value in frame.columns
            }
            required = ("load_actual", "load_forecast", "price")
            missing_required = [
                name for name in required if name not in available
            ]
            if missing_required:
                raise ValueError(
                    f"OPSD source lacks required {market} fields: "
                    f"{missing_required}"
                )
            local = frame[["utc_timestamp", *available.values()]].rename(
                columns={value: key for key, value in available.items()}
            ).copy()
            optional = tuple(
                name for name in ("solar", "wind") if name in available)
            interpolated = ("price", *optional)
            before = {
                name: int(local[name].isna().sum()) for name in interpolated
            }
            if interpolation_limit > 0:
                local[list(interpolated)] = local[list(interpolated)].interpolate(
                    method="linear",
                    limit=interpolation_limit,
                    limit_direction="both",
                    limit_area="inside",
                )
            after = {
                name: int(local[name].isna().sum()) for name in interpolated
            }
            local.dropna(subset=list(required), inplace=True)
            if local.empty:
                raise ValueError(f"no complete OPSD rows remain for {market}")
            hours = (
                local["utc_timestamp"].astype("int64").to_numpy(dtype=np.int64)
                // 3_600_000_000_000
            )
            prefix = f"{market}__"
            arrays[prefix + "timestamp_hour"] = hours
            for name in (*required, *optional):
                values = local[name].to_numpy(dtype=np.float64)
                if name in optional:
                    values = np.where(np.isfinite(values), values, 0.0)
                if not np.all(np.isfinite(values)):
                    raise FloatingPointError(
                        f"non-finite OPSD values remain for {market}/{name}"
                    )
                arrays[prefix + name] = values
            gaps = int(np.sum(np.diff(hours) != 1))
            market_metadata[market] = {
                "row_count": int(len(local)),
                "first_timestamp_hour": int(hours[0]),
                "last_timestamp_hour": int(hours[-1]),
                "non_hourly_gap_count": gaps,
                "missing_before_interpolation": before,
                "missing_after_interpolation": after,
                "interpolated_counts": {
                    name: int(before[name] - after[name])
                    for name in interpolated
                },
                "columns": dict(available),
                "unused_optional_fields": ["solar", "wind"],
                "absent_optional_fields": [
                    name for name in ("solar", "wind")
                    if name not in available
                ],
            }

        metadata = {
            "schema_version": 2,
            "dataset": "Open Power System Data time_series",
            "version": OPSD_DATA_VERSION,
            "doi": OPSD_DATA_DOI,
            "official_url": OPSD_DATA_URL,
            "source": source_meta,
            "markets": list(markets),
            "years": list(years),
            "interpolation_limit_hours": interpolation_limit,
            "outcome_dependent_filtering": False,
            "market_metadata": market_metadata,
        }
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True), dtype="U")

        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(output)

    output_sha, output_size = _sha256_file(output)
    metadata["output"] = {
        "path": str(output),
        "sha256": output_sha,
        "content_length": output_size,
    }
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest)
    return metadata


@dataclass(frozen=True)
class OPSDMarketSeries:
    market: str
    timestamp_hour: np.ndarray
    load_actual: np.ndarray | None
    load_forecast: np.ndarray
    price: np.ndarray
    solar: np.ndarray | None
    wind: np.ndarray | None
    metadata: dict

    def __post_init__(self):
        names = ["timestamp_hour", "load_forecast", "price"]
        if self.load_actual is not None:
            names.append("load_actual")
        if self.solar is not None:
            names.append("solar")
        if self.wind is not None:
            names.append("wind")
        lengths = []
        for name in names:
            value = np.asarray(getattr(self, name))
            object.__setattr__(self, name, value)
            lengths.append(len(value))
        if len(set(lengths)) != 1 or lengths[0] == 0:
            raise ValueError("OPSD market arrays must be nonempty and aligned")
        if np.any(np.diff(self.timestamp_hour.astype(np.int64)) <= 0):
            raise ValueError("OPSD timestamps must be strictly increasing")
        numeric_columns = [self.load_forecast, self.price]
        if self.load_actual is not None:
            numeric_columns.append(self.load_actual)
        if self.solar is not None:
            numeric_columns.append(self.solar)
        if self.wind is not None:
            numeric_columns.append(self.wind)
        numeric = np.column_stack(numeric_columns)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("OPSD market arrays must be finite")

    def mask_year(self, year):
        start = np.datetime64(f"{int(year):04d}-01-01T00", "h").astype(np.int64)
        stop = np.datetime64(f"{int(year) + 1:04d}-01-01T00", "h").astype(np.int64)
        return (self.timestamp_hour >= start) & (self.timestamp_hour < stop)

    def valid_window_starts(self, horizon, start, stop):
        """Return starts whose full hourly window lies in `[start, stop)`."""

        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("window horizon must be positive")
        start_hour = np.datetime64(str(start), "h").astype(np.int64)
        stop_hour = np.datetime64(str(stop), "h").astype(np.int64)
        hours = self.timestamp_hour.astype(np.int64)
        candidates = np.flatnonzero(
            (hours >= start_hour) & (hours + horizon <= stop_hour)
        )
        candidates = candidates[candidates + horizon <= len(hours)]
        if len(candidates) == 0:
            return np.zeros(0, dtype=np.int64)
        breaks = (np.diff(hours) != 1).astype(np.int64)
        prefix = np.concatenate([[0], np.cumsum(breaks)])
        ends = candidates + horizon - 1
        contiguous = (prefix[ends] - prefix[candidates]) == 0
        return candidates[contiguous].astype(np.int64)


def load_opsd_market(path, market, *, include_outcomes=True):
    """Load one market from a compact archive without pickle support."""

    market = str(market)
    if market not in OPSD_MARKET_COLUMNS:
        raise ValueError(f"unsupported OPSD market {market!r}")
    prefix = f"{market}__"
    with np.load(Path(path), allow_pickle=False) as archive:
        names = ["timestamp_hour", "load_forecast", "price"]
        if include_outcomes:
            names.append("load_actual")
        required = [prefix + name for name in names]
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"OPSD archive is missing keys: {missing}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {
            name: np.asarray(archive[prefix + name]).copy() for name in names
        }
        arrays["solar"] = (
            np.asarray(archive[prefix + "solar"]).copy()
            if prefix + "solar" in archive else None
        )
        arrays["wind"] = (
            np.asarray(archive[prefix + "wind"]).copy()
            if prefix + "wind" in archive else None
        )
        if not include_outcomes:
            arrays["load_actual"] = None
    if market not in metadata.get("markets", []):
        raise ValueError("OPSD archive metadata does not contain requested market")
    return OPSDMarketSeries(market=market, metadata=metadata, **arrays)
