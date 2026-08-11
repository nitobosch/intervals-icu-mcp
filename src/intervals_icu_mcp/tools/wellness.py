"""Wellness and health tracking tools for Intervals.icu MCP server."""

from datetime import datetime, timedelta
from typing import Annotated, Any

from fastmcp import Context

from ..auth import ICUConfig
from ..client import ICUAPIError, ICUClient
from ..response_builder import ResponseBuilder

# Scale labels for subjective metrics — surfaced in response metadata so LLM
# clients that only see the JSON payload (not the tool docstring) can still
# interpret the values correctly. Direction matters: sleep_quality is inverted
# (1=Great, 5=Poor) while every other 1-5 metric is "higher = more"; hydration
# uses a 1-4 range in the same higher-is-worse direction.
WELLNESS_SCALES: dict[str, str] = {
    "fatigue": "1-5 (1=very low, 5=very high)",
    "soreness": "1-5 (1=very low, 5=very high)",
    "stress": "1-5 (1=very low, 5=very high)",
    "mood": "1-5 (1=very poor, 5=very good)",
    "motivation": "1-5 (1=very low, 5=very high)",
    "injury": "1-5 (1=none, 5=severe)",
    "hydration": "1-4 (1=well hydrated, 4=very dehydrated)",
    "sleep_quality": "1-5 (1=Great, 5=Poor — inverted scale)",
    "sleep_score": "0-100 (device-specific, higher is better)",
    "readiness": "0-100 (higher is better)",
}

# Garmin Body Battery is stored in Intervals.icu as athlete-defined wellness
# fields, so the API only returns it when those field codes are requested.
CUSTOM_WELLNESS_FIELDS: list[str] = [
    "id",
    "BodyBatteryMin",
    "BodyBatteryMax",
]


def _get_custom_wellness_value(
    record: Any,
    custom_record: Any | None,
    field: str,
) -> Any:
    """Return an athlete-defined wellness value from an enriched or base record."""
    for source in (custom_record, record):
        if source is None:
            continue

        # Wellness uses Pydantic extra="allow", therefore athlete-defined API
        # fields are preserved in model_extra instead of being discarded.
        model_extra: dict[str, Any] = getattr(source, "model_extra", None) or {}
        if field in model_extra:
            return model_extra[field]

        # Keep this fallback in case a future model version declares the field
        # explicitly instead of leaving it in model_extra.
        value = getattr(source, field, None)
        if value is not None:
            return value

    return None


async def _get_custom_wellness_by_date(
    client: ICUClient,
    athlete_id: str | None,
    oldest: str,
    newest: str,
) -> dict[str, Any]:
    """Fetch optional athlete-defined wellness fields, indexed by date."""
    try:
        records = await client.get_wellness(
            athlete_id=athlete_id,
            oldest=oldest,
            newest=newest,
            fields=CUSTOM_WELLNESS_FIELDS,
        )
    except ICUAPIError:
        # Custom wellness fields are optional. Failure to retrieve them must not
        # make the standard wellness tools unusable.
        return {}

    return {record.id: record for record in records}


def _format_wellness_record(
    record: Any,
    date_id: str,
    custom_record: Any | None = None,
) -> dict[str, Any]:
    """Format a wellness record object into a structured dictionary."""
    day_data: dict[str, Any] = {"date": date_id}

    # Sleep
    sleep: dict[str, Any] = {}
    if getattr(record, "sleep_secs", None):
        sleep["duration_seconds"] = record.sleep_secs
    if getattr(record, "sleep_quality", None):
        sleep["quality"] = record.sleep_quality
    if getattr(record, "sleep_score", None):
        sleep["score"] = round(record.sleep_score, 0)
    if getattr(record, "avg_sleeping_hr", None):
        sleep["avg_sleeping_hr"] = round(record.avg_sleeping_hr, 0)
    if sleep:
        day_data["sleep"] = sleep

    # Heart metrics
    heart: dict[str, Any] = {}
    if getattr(record, "hrv", None):
        heart["hrv_rmssd"] = round(record.hrv, 1)
    if getattr(record, "hrv_sdnn", None):
        heart["hrv_sdnn"] = round(record.hrv_sdnn, 1)
    if getattr(record, "resting_hr", None):
        heart["resting_hr"] = record.resting_hr
    if getattr(record, "baevsky_si", None):
        heart["baevsky_si"] = round(record.baevsky_si, 1)
    if heart:
        day_data["heart"] = heart

    # Garmin Body Battery (Intervals.icu athlete-defined wellness fields)
    body_battery: dict[str, Any] = {}
    body_battery_min = _get_custom_wellness_value(
        record,
        custom_record,
        "BodyBatteryMin",
    )
    body_battery_max = _get_custom_wellness_value(
        record,
        custom_record,
        "BodyBatteryMax",
    )
    if body_battery_min is not None:
        body_battery["min"] = body_battery_min
    if body_battery_max is not None:
        body_battery["max"] = body_battery_max
    if body_battery:
        day_data["body_battery"] = body_battery

    # Subjective feelings
    subjective: dict[str, Any] = {}
    for field in ["fatigue", "soreness", "stress", "mood", "motivation", "injury", "hydration"]:
        if getattr(record, field, None):
            subjective[field] = getattr(record, field)
    if getattr(record, "readiness", None):
        subjective["readiness"] = round(record.readiness, 0)
    if subjective:
        day_data["subjective"] = subjective

    # Body metrics
    body: dict[str, Any] = {}
    if getattr(record, "weight", None):
        body["weight_kg"] = record.weight
    if getattr(record, "body_fat", None):
        body["body_fat_percent"] = round(record.body_fat, 1)
    if getattr(record, "abdomen", None):
        body["abdomen_cm"] = round(record.abdomen, 1)
    if getattr(record, "vo2max", None):
        body["vo2max"] = round(record.vo2max, 1)
    if body:
        day_data["body"] = body

    # Vital signs
    vitals: dict[str, Any] = {}
    if getattr(record, "systolic", None):
        vitals["systolic_mmhg"] = record.systolic
    if getattr(record, "diastolic", None):
        vitals["diastolic_mmhg"] = record.diastolic
    if getattr(record, "spo2", None):
        vitals["spo2_percent"] = round(record.spo2, 1)
    if getattr(record, "respiration", None):
        vitals["respiration_rate"] = round(record.respiration, 1)
    if vitals:
        day_data["vitals"] = vitals

    # Activity
    activity: dict[str, Any] = {}
    if getattr(record, "steps", None):
        activity["steps"] = record.steps
    if activity:
        day_data["activity"] = activity

    # Nutrition
    nutrition: dict[str, Any] = {}
    if getattr(record, "kcal_consumed", None):
        nutrition["calories_consumed"] = record.kcal_consumed
    if getattr(record, "carbohydrates", None):
        nutrition["carbohydrates_g"] = round(record.carbohydrates, 1)
    if getattr(record, "protein", None):
        nutrition["protein_g"] = round(record.protein, 1)
    if getattr(record, "fat_total", None):
        nutrition["fat_total_g"] = round(record.fat_total, 1)
    if getattr(record, "hydration_volume", None):
        nutrition["hydration_liters"] = round(record.hydration_volume, 1)
    if nutrition:
        day_data["nutrition"] = nutrition

    # Training load
    training: dict[str, Any] = {}
    for field in ["ctl", "atl", "tsb", "ramp_rate"]:
        if getattr(record, field, None):
            training[field] = round(getattr(record, field), 1)
    if training:
        day_data["training"] = training

    # Per-sport context (eFTP, W', Pmax per sport type)
    sport_info_raw: list[Any] = getattr(record, "sport_info", None) or []
    sport_info_list: list[dict[str, Any]] = []
    for sport in sport_info_raw:
        entry: dict[str, Any] = {}
        if getattr(sport, "type", None):
            entry["type"] = sport.type
        if getattr(sport, "eftp", None):
            entry["eftp"] = round(sport.eftp, 1)
        if getattr(sport, "w_prime", None):
            entry["w_prime"] = round(sport.w_prime, 1)
        if getattr(sport, "p_max", None):
            entry["p_max"] = round(sport.p_max, 1)
        if entry:
            sport_info_list.append(entry)
    if sport_info_list:
        day_data["sport_info"] = sport_info_list

    # State flags (record locked, fallback values in use)
    state_flags: dict[str, Any] = {}
    for field in ["locked", "temp_weight", "temp_resting_hr"]:
        value = getattr(record, field, None)
        if value is not None:
            state_flags[field] = value
    if state_flags:
        day_data["state_flags"] = state_flags

    # Other metrics
    other: dict[str, Any] = {}
    if getattr(record, "blood_glucose", None):
        other["blood_glucose_mmol_per_l"] = round(record.blood_glucose, 1)
    if getattr(record, "lactate", None):
        other["lactate_mmol_per_l"] = round(record.lactate, 1)
    if getattr(record, "menstrual_phase", None):
        other["menstrual_phase"] = record.menstrual_phase
    if getattr(record, "menstrual_phase_predicted", None):
        other["menstrual_phase_predicted"] = record.menstrual_phase_predicted
    if other:
        day_data["other"] = other

    # Comments
    if getattr(record, "comments", None):
        day_data["comments"] = record.comments

    # Athlete-defined wellness fields are not part of the static Wellness
    # schema. Pydantic keeps them in model_extra (Wellness uses extra="allow");
    # surface them verbatim instead of silently dropping them while formatting.
    # Body Battery is exposed above in a dedicated structure, so exclude it here
    # to avoid duplicating the same values in the response.
    custom_fields: dict[str, Any] = {}
    for source in (record, custom_record):
        if source is None:
            continue
        model_extra: dict[str, Any] = getattr(source, "model_extra", None) or {}
        for name, value in model_extra.items():
            if name in {"BodyBatteryMin", "BodyBatteryMax"}:
                continue
            if value is not None:
                custom_fields[name] = value
    if custom_fields:
        day_data["custom_fields"] = custom_fields

    return day_data


def _scales_for_records(records: list[dict[str, Any]]) -> dict[str, str]:
    """Return scale labels only for subjective metrics actually present in output."""
    present: set[str] = set()
    for record in records:
        sleep = record.get("sleep", {})
        if "quality" in sleep:
            present.add("sleep_quality")
        if "score" in sleep:
            present.add("sleep_score")
        subjective = record.get("subjective", {})
        for key in subjective:
            if key in WELLNESS_SCALES:
                present.add(key)
    return {k: WELLNESS_SCALES[k] for k in present}



def _recovery_metric_value(
    record: Any | None,
    custom_record: Any | None,
    metric: str,
) -> Any:
    """Return one raw recovery metric without converting missing values to zero."""
    if metric == "body_battery_min":
        return _get_custom_wellness_value(
            record,
            custom_record,
            "BodyBatteryMin",
        )

    if metric == "body_battery_max":
        return _get_custom_wellness_value(
            record,
            custom_record,
            "BodyBatteryMax",
        )

    if record is None:
        return None

    field_by_metric = {
        "sleep_duration": "sleep_secs",
        "sleep_score": "sleep_score",
        "sleep_quality": "sleep_quality",
        "hrv": "hrv",
        "resting_hr": "resting_hr",
    }

    field = field_by_metric.get(metric)
    if field is None:
        return None

    return getattr(record, field, None)


def _numeric_values_for_dates(
    metric: str,
    date_ids: list[str],
    records_by_date: dict[str, Any],
    custom_by_date: dict[str, Any],
) -> list[float]:
    """Return available numeric values for explicit calendar dates."""
    values: list[float] = []

    for date_id in date_ids:
        value = _recovery_metric_value(
            records_by_date.get(date_id),
            custom_by_date.get(date_id),
            metric,
        )

        if value is not None and isinstance(value, (int, float)):
            values.append(float(value))

    return values


def _average(
    values: list[float],
    digits: int = 1,
) -> float | None:
    """Average available values only; missing values are never treated as zero."""
    if not values:
        return None

    return round(sum(values) / len(values), digits)


def _baseline_comparison(
    metric: str,
    recent_dates: list[str],
    previous_dates: list[str],
    records_by_date: dict[str, Any],
    custom_by_date: dict[str, Any],
    digits: int = 1,
) -> dict[str, float | None]:
    """Compare equal recent and previous calendar windows for one metric."""
    recent_values = _numeric_values_for_dates(
        metric,
        recent_dates,
        records_by_date,
        custom_by_date,
    )

    previous_values = _numeric_values_for_dates(
        metric,
        previous_dates,
        records_by_date,
        custom_by_date,
    )

    recent_average = _average(recent_values, digits)
    previous_average = _average(previous_values, digits)

    change: float | None = None

    if recent_average is not None and previous_average is not None:
        change = round(
            recent_average - previous_average,
            digits,
        )

    return {
        "recent_average": recent_average,
        "previous_average": previous_average,
        "change": change,
    }


def _metric_coverage(
    metric: str,
    recent_dates: list[str],
    previous_dates: list[str],
    records_by_date: dict[str, Any],
    custom_by_date: dict[str, Any],
) -> dict[str, Any]:
    """Describe how much real data backs a recovery metric."""

    def available(date_id: str) -> bool:
        value = _recovery_metric_value(
            records_by_date.get(date_id),
            custom_by_date.get(date_id),
            metric,
        )
        return value is not None

    recent_present = [
        date_id
        for date_id in recent_dates
        if available(date_id)
    ]

    previous_present = [
        date_id
        for date_id in previous_dates
        if available(date_id)
    ]

    all_dates = previous_dates + recent_dates

    return {
        "recent_available": len(recent_present),
        "recent_expected": len(recent_dates),
        "previous_available": len(previous_present),
        "previous_expected": len(previous_dates),
        "total_available": len(recent_present) + len(previous_present),
        "total_expected": len(all_dates),
        "missing_dates": [
            date_id
            for date_id in all_dates
            if not available(date_id)
        ],
    }


async def get_recovery_analysis(
    comparison_window_days: Annotated[
        int,
        (
            "Calendar days per comparison block. Default 7 means "
            "recent 7 days versus previous 7 days."
        ),
    ] = 7,
    reference_date: Annotated[
        str | None,
        "Reference date in YYYY-MM-DD format. Defaults to today.",
    ] = None,
    athlete_id: Annotated[
        str | None,
        "Athlete ID (for coaches managing multiple athletes)",
    ] = None,
    ctx: Context | None = None,
) -> str:
    """Build a deterministic recovery baseline from Intervals.icu wellness data.

    Two equal calendar windows are compared. Missing values are excluded from
    averages and are never converted to zero.

    This tool deliberately does not generate a recovery/readiness score,
    physiological diagnosis, or training recommendation.
    """
    assert ctx is not None
    config: ICUConfig = await ctx.get_state("config")

    if comparison_window_days < 2 or comparison_window_days > 30:
        return ResponseBuilder.build_error_response(
            "comparison_window_days must be between 2 and 30.",
            error_type="validation_error",
        )

    if reference_date is None:
        reference_dt = datetime.now()
    else:
        try:
            reference_dt = datetime.strptime(
                reference_date,
                "%Y-%m-%d",
            )
        except ValueError:
            return ResponseBuilder.build_error_response(
                "Invalid reference_date format. Please use YYYY-MM-DD format.",
                error_type="validation_error",
            )

    reference_date_id = reference_dt.strftime("%Y-%m-%d")

    recent_start_dt = reference_dt - timedelta(
        days=comparison_window_days - 1
    )

    previous_end_dt = recent_start_dt - timedelta(days=1)

    previous_start_dt = previous_end_dt - timedelta(
        days=comparison_window_days - 1
    )

    recent_dates = [
        (
            recent_start_dt + timedelta(days=i)
        ).strftime("%Y-%m-%d")
        for i in range(comparison_window_days)
    ]

    previous_dates = [
        (
            previous_start_dt + timedelta(days=i)
        ).strftime("%Y-%m-%d")
        for i in range(comparison_window_days)
    ]

    all_dates = previous_dates + recent_dates

    oldest = previous_dates[0]
    newest = recent_dates[-1]

    try:
        async with ICUClient(config) as client:
            wellness_records = await client.get_wellness(
                athlete_id=athlete_id,
                oldest=oldest,
                newest=newest,
            )

            custom_by_date = await _get_custom_wellness_by_date(
                client=client,
                athlete_id=athlete_id,
                oldest=oldest,
                newest=newest,
            )

        records_by_date = {
            record.id: record
            for record in wellness_records
        }

        today_record = records_by_date.get(reference_date_id)
        today_custom = custom_by_date.get(reference_date_id)

        sleep_seconds = _recovery_metric_value(
            today_record,
            today_custom,
            "sleep_duration",
        )

        sleep_score = _recovery_metric_value(
            today_record,
            today_custom,
            "sleep_score",
        )

        sleep_quality = _recovery_metric_value(
            today_record,
            today_custom,
            "sleep_quality",
        )

        hrv = _recovery_metric_value(
            today_record,
            today_custom,
            "hrv",
        )

        resting_hr = _recovery_metric_value(
            today_record,
            today_custom,
            "resting_hr",
        )

        body_battery_min = _recovery_metric_value(
            today_record,
            today_custom,
            "body_battery_min",
        )

        body_battery_max = _recovery_metric_value(
            today_record,
            today_custom,
            "body_battery_max",
        )

        today = {
            "date": reference_date_id,
            "sleep_duration": {
                "seconds": sleep_seconds,
                "hours": (
                    round(float(sleep_seconds) / 3600, 2)
                    if sleep_seconds is not None
                    else None
                ),
                "source_field": "sleepSecs",
            },
            "sleep_score": sleep_score,
            "sleep_quality": sleep_quality,
            "hrv": {
                "value_ms": hrv,
            },
            "resting_hr": {
                "value_bpm": resting_hr,
            },
            "body_battery": {
                "min": body_battery_min,
                "max": body_battery_max,
            },
        }

        hrv_baseline = _baseline_comparison(
            "hrv",
            recent_dates,
            previous_dates,
            records_by_date,
            custom_by_date,
        )

        resting_hr_baseline = _baseline_comparison(
            "resting_hr",
            recent_dates,
            previous_dates,
            records_by_date,
            custom_by_date,
        )

        sleep_score_baseline = _baseline_comparison(
            "sleep_score",
            recent_dates,
            previous_dates,
            records_by_date,
            custom_by_date,
        )

        recent_sleep_values = _numeric_values_for_dates(
            "sleep_duration",
            recent_dates,
            records_by_date,
            custom_by_date,
        )

        previous_sleep_values = _numeric_values_for_dates(
            "sleep_duration",
            previous_dates,
            records_by_date,
            custom_by_date,
        )

        recent_sleep_seconds = _average(
            recent_sleep_values,
            0,
        )

        previous_sleep_seconds = _average(
            previous_sleep_values,
            0,
        )

        sleep_change_seconds: float | None = None

        if (
            recent_sleep_seconds is not None
            and previous_sleep_seconds is not None
        ):
            sleep_change_seconds = round(
                recent_sleep_seconds - previous_sleep_seconds,
                0,
            )

        sleep_duration_baseline = {
            "recent_average_seconds": recent_sleep_seconds,
            "recent_average_hours": (
                round(recent_sleep_seconds / 3600, 2)
                if recent_sleep_seconds is not None
                else None
            ),
            "previous_average_seconds": previous_sleep_seconds,
            "previous_average_hours": (
                round(previous_sleep_seconds / 3600, 2)
                if previous_sleep_seconds is not None
                else None
            ),
            "change_seconds": sleep_change_seconds,
            "change_hours": (
                round(sleep_change_seconds / 3600, 2)
                if sleep_change_seconds is not None
                else None
            ),
        }

        baseline = {
            "hrv": hrv_baseline,
            "resting_hr": resting_hr_baseline,
            "sleep_duration": sleep_duration_baseline,
            "sleep_score": sleep_score_baseline,
        }

        coverage = {
            metric: _metric_coverage(
                metric,
                recent_dates,
                previous_dates,
                records_by_date,
                custom_by_date,
            )
            for metric in (
                "sleep_duration",
                "sleep_score",
                "sleep_quality",
                "hrv",
                "resting_hr",
                "body_battery_min",
                "body_battery_max",
            )
        }

        today_fields = {
            "sleep_duration": sleep_seconds,
            "sleep_score": sleep_score,
            "sleep_quality": sleep_quality,
            "hrv": hrv,
            "resting_hr": resting_hr,
            "body_battery_min": body_battery_min,
            "body_battery_max": body_battery_max,
        }

        data_quality = {
            "requested_days": len(all_dates),
            "records_available": len(records_by_date),
            "nights_available": (
                coverage["sleep_duration"]["total_available"]
            ),
            "coverage": coverage,
            "missing_fields": [
                name
                for name, value in today_fields.items()
                if value is None
            ],
        }

        data = {
            "source": "intervals_icu",
            "period": {
                "reference_date": reference_date_id,
                "comparison_window_days": comparison_window_days,
                "previous_start": previous_dates[0],
                "previous_end": previous_dates[-1],
                "recent_start": recent_dates[0],
                "recent_end": recent_dates[-1],
            },
            "today": today,
            "baseline": baseline,
            "data_quality": data_quality,
        }

        return ResponseBuilder.build_response(
            data=data,
            metadata={
                "method": (
                    "Two equal calendar windows. Averages exclude "
                    "null/missing values. "
                    "change = recent_average - previous_average."
                ),
                "interpretation": (
                    "No recovery score, physiological diagnosis, or "
                    "training recommendation is generated by this tool."
                ),
                "sleep_duration_note": (
                    "sleep_duration is derived from Intervals.icu sleepSecs. "
                    "Its semantics may differ from Garmin effective sleep "
                    "duration, so it must not be assumed to be identical to "
                    "Garmin sleep time."
                ),
            },
            query_type="recovery_analysis",
        )

    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(
            e.message,
            error_type="api_error",
        )

    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected error: {str(e)}",
            error_type="internal_error",
        )



async def get_wellness_data(
    days_back: Annotated[int, "Number of days to look back"] = 7,
    athlete_id: Annotated[str | None, "Athlete ID (for coaches managing multiple athletes)"] = None,
    ctx: Context | None = None,
) -> str:
    """Fetch wellness records over a RANGE of recent days (default last 7).

    Use for trends, weekly summaries, "how has my sleep been this week?",
    recovery curves. For PMC/fitness chart CTL/ATL/TSB series use
    icu_get_fitness_chart. For a single specific date use
    icu_get_wellness_for_date instead — this tool always returns a multi-day list.
    """
    assert ctx is not None
    config: ICUConfig = await ctx.get_state("config")

    try:
        # Calculate date range
        oldest_date = datetime.now() - timedelta(days=days_back)
        oldest = oldest_date.strftime("%Y-%m-%d")
        newest = datetime.now().strftime("%Y-%m-%d")

        async with ICUClient(config) as client:
            wellness_records = await client.get_wellness(
                athlete_id=athlete_id,
                oldest=oldest,
                newest=newest,
            )

            if not wellness_records:
                return ResponseBuilder.build_response(
                    data={"wellness_data": [], "count": 0},
                    metadata={"message": f"No wellness data found for the last {days_back} days"},
                )

            custom_wellness_by_date = await _get_custom_wellness_by_date(
                client=client,
                athlete_id=athlete_id,
                oldest=oldest,
                newest=newest,
            )

            # Sort by date (most recent first)
            wellness_records.sort(key=lambda x: x.id, reverse=True)

            wellness_data: list[dict[str, Any]] = []
            for record in wellness_records:
                wellness_data.append(
                    _format_wellness_record(
                        record,
                        record.id,
                        custom_wellness_by_date.get(record.id),
                    )
                )

            # Calculate trends if we have multiple days
            trends: dict[str, Any] = {}
            if len(wellness_records) > 1:
                # HRV trend
                hrv_values = [r.hrv for r in wellness_records if r.hrv is not None]
                if len(hrv_values) >= 2:
                    trends["hrv"] = {
                        "current": round(hrv_values[0], 1),
                        "change": round(hrv_values[0] - hrv_values[-1], 1),
                    }

                # Resting HR trend
                rhr_values = [r.resting_hr for r in wellness_records if r.resting_hr is not None]
                if len(rhr_values) >= 2:
                    trends["resting_hr"] = {
                        "current": rhr_values[0],
                        "change": rhr_values[0] - rhr_values[-1],
                    }

                # Sleep quality trend
                sleep_values = [
                    r.sleep_quality for r in wellness_records if r.sleep_quality is not None
                ]
                if len(sleep_values) >= 2:
                    trends["avg_sleep_quality"] = round(sum(sleep_values) / len(sleep_values), 1)

                # Weight trend
                weight_values = [r.weight for r in wellness_records if r.weight is not None]
                if len(weight_values) >= 2:
                    trends["weight"] = {
                        "current": weight_values[0],
                        "change": round(weight_values[0] - weight_values[-1], 1),
                    }

            result_data: dict[str, Any] = {
                "wellness_data": wellness_data,
                "count": len(wellness_data),
            }
            if trends:
                result_data["trends"] = trends

            metadata: dict[str, Any] = {}
            scales = _scales_for_records(wellness_data)
            if scales:
                metadata["scales"] = scales

            return ResponseBuilder.build_response(
                data=result_data,
                metadata=metadata or None,
                query_type="wellness_data",
            )

    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(e.message, error_type="api_error")
    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected error: {str(e)}", error_type="internal_error"
        )


async def get_wellness_for_date(
    date: Annotated[str, "Date in YYYY-MM-DD format"],
    athlete_id: Annotated[str | None, "Athlete ID (for coaches managing multiple athletes)"] = None,
    ctx: Context | None = None,
) -> str:
    """Fetch the wellness record for ONE specific date.

    Use when the user names a date — "show my HRV for Monday",
    "wellness on 2026-03-15", "how did I sleep last Thursday?". For
    ranges, weeks, or trends use icu_get_wellness_data.
    """
    assert ctx is not None
    config: ICUConfig = await ctx.get_state("config")

    # Validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return ResponseBuilder.build_error_response(
            "Invalid date format. Please use YYYY-MM-DD format.",
            error_type="validation_error",
        )

    try:
        async with ICUClient(config) as client:
            wellness = await client.get_wellness_for_date(date=date, athlete_id=athlete_id)
            custom_wellness_by_date = await _get_custom_wellness_by_date(
                client=client,
                athlete_id=athlete_id,
                oldest=date,
                newest=date,
            )

            wellness_data = _format_wellness_record(
                wellness,
                date,
                custom_wellness_by_date.get(date),
            )

            metadata: dict[str, Any] = {}
            scales = _scales_for_records([wellness_data])
            if scales:
                metadata["scales"] = scales

            return ResponseBuilder.build_response(
                data=wellness_data,
                metadata=metadata or None,
                query_type="wellness_for_date",
            )

    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(e.message, error_type="api_error")
    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected error: {str(e)}", error_type="internal_error"
        )


async def update_wellness(
    date: Annotated[str, "Date in YYYY-MM-DD format"],
    weight: Annotated[float | None, "Weight in kg"] = None,
    resting_hr: Annotated[int | None, "Resting heart rate in bpm"] = None,
    hrv: Annotated[float | None, "HRV (rMSSD) value"] = None,
    sleep_secs: Annotated[int | None, "Sleep duration in seconds"] = None,
    sleep_quality: Annotated[int | None, "Sleep quality (1-5 scale)"] = None,
    fatigue: Annotated[int | None, "Fatigue level (1-5 scale)"] = None,
    soreness: Annotated[int | None, "Soreness level (1-5 scale)"] = None,
    stress: Annotated[int | None, "Stress level (1-5 scale)"] = None,
    mood: Annotated[int | None, "Mood level (1-5 scale)"] = None,
    motivation: Annotated[int | None, "Motivation level (1-5 scale)"] = None,
    injury: Annotated[int | None, "Injury severity (1-5 scale: 1=none, 5=severe)"] = None,
    hydration: Annotated[
        int | None, "Subjective hydration rating (1-4: 1=well hydrated, 4=very dehydrated)"
    ] = None,
    readiness: Annotated[float | None, "Readiness score (0-100)"] = None,
    body_fat: Annotated[float | None, "Body fat percentage"] = None,
    abdomen: Annotated[float | None, "Abdominal circumference in cm"] = None,
    vo2max: Annotated[float | None, "VO2max (ml/kg/min) — lab result or device estimate"] = None,
    systolic: Annotated[int | None, "Systolic blood pressure in mmHg"] = None,
    diastolic: Annotated[int | None, "Diastolic blood pressure in mmHg"] = None,
    spo2: Annotated[float | None, "Blood oxygen saturation percentage (SpO2)"] = None,
    respiration: Annotated[float | None, "Respiration rate in breaths per minute"] = None,
    blood_glucose: Annotated[float | None, "Blood glucose in mmol/L"] = None,
    lactate: Annotated[float | None, "Blood lactate in mmol/L — lab result"] = None,
    menstrual_phase: Annotated[
        str | None, "Menstrual phase (e.g. FOLLICULAR, OVULATING, LUTEAL, MENSTRUAL)"
    ] = None,
    locked: Annotated[
        bool | None, "Lock record to prevent device sync from overwriting manual entries"
    ] = None,
    calories_consumed: Annotated[int | None, "Calories consumed (kcal)"] = None,
    carbohydrates: Annotated[float | None, "Carbohydrates consumed (grams)"] = None,
    protein: Annotated[float | None, "Protein consumed (grams)"] = None,
    fat_total: Annotated[float | None, "Total fat consumed (grams)"] = None,
    hydration_liters: Annotated[float | None, "Hydration volume (liters)"] = None,
    comments: Annotated[str | None, "Comments or notes"] = None,
    athlete_id: Annotated[str | None, "Athlete ID (for coaches managing multiple athletes)"] = None,
    ctx: Context | None = None,
) -> str:
    """Upsert wellness data for ONE specific date — creates the record if missing, otherwise updates the fields you pass.

    Only provided fields are sent. Subjective scales (fatigue, soreness,
    stress, mood, motivation, injury) are 1-5. Pass `locked=True` to stop
    device sync from overwriting manual entries. Writes to the authenticated
    athlete unless athlete_id names a managed athlete.
    """
    assert ctx is not None
    config: ICUConfig = await ctx.get_state("config")

    # Validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return ResponseBuilder.build_error_response(
            "Invalid date format. Please use YYYY-MM-DD format.",
            error_type="validation_error",
        )

    try:
        # Build wellness data (only include provided fields)
        wellness_data: dict[str, Any] = {"id": date}

        if weight is not None:
            wellness_data["weight"] = weight
        if resting_hr is not None:
            wellness_data["restingHR"] = resting_hr
        if hrv is not None:
            wellness_data["hrv"] = hrv
        if sleep_secs is not None:
            wellness_data["sleepSecs"] = sleep_secs
        if sleep_quality is not None:
            wellness_data["sleepQuality"] = sleep_quality
        if fatigue is not None:
            wellness_data["fatigue"] = fatigue
        if soreness is not None:
            wellness_data["soreness"] = soreness
        if stress is not None:
            wellness_data["stress"] = stress
        if mood is not None:
            wellness_data["mood"] = mood
        if motivation is not None:
            wellness_data["motivation"] = motivation
        if injury is not None:
            wellness_data["injury"] = injury
        if hydration is not None:
            wellness_data["hydration"] = hydration
        if readiness is not None:
            wellness_data["readiness"] = readiness
        if body_fat is not None:
            wellness_data["bodyFat"] = body_fat
        if abdomen is not None:
            wellness_data["abdomen"] = abdomen
        if vo2max is not None:
            wellness_data["vo2max"] = vo2max
        if systolic is not None:
            wellness_data["systolic"] = systolic
        if diastolic is not None:
            wellness_data["diastolic"] = diastolic
        if spo2 is not None:
            wellness_data["spO2"] = spo2
        if respiration is not None:
            wellness_data["respiration"] = respiration
        if blood_glucose is not None:
            wellness_data["bloodGlucose"] = blood_glucose
        if lactate is not None:
            wellness_data["lactate"] = lactate
        if menstrual_phase is not None:
            wellness_data["menstrualPhase"] = menstrual_phase
        if locked is not None:
            wellness_data["locked"] = locked
        if calories_consumed is not None:
            wellness_data["kcalConsumed"] = calories_consumed
        if carbohydrates is not None:
            wellness_data["carbohydrates"] = carbohydrates
        if protein is not None:
            wellness_data["protein"] = protein
        if fat_total is not None:
            wellness_data["fatTotal"] = fat_total
        if hydration_liters is not None:
            wellness_data["hydrationVolume"] = hydration_liters
        if comments is not None:
            wellness_data["comments"] = comments

        if len(wellness_data) == 1:  # Only has 'id'
            return ResponseBuilder.build_error_response(
                "No wellness data provided. Please specify at least one metric to update.",
                error_type="validation_error",
            )

        async with ICUClient(config) as client:
            wellness = await client.update_wellness(wellness_data, athlete_id=athlete_id)

            result_data = _format_wellness_record(wellness, date)

            metadata: dict[str, Any] = {"message": f"Successfully updated wellness for {date}"}
            scales = _scales_for_records([result_data])
            if scales:
                metadata["scales"] = scales

            return ResponseBuilder.build_response(
                data=result_data,
                query_type="update_wellness",
                metadata=metadata,
            )

    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(e.message, error_type="api_error")
    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected error: {str(e)}", error_type="internal_error"
        )