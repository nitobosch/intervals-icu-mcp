"""Tests for wellness tools."""

import json
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

from httpx import Response

from intervals_icu_mcp.tools.wellness import (
    get_recovery_analysis,
    get_wellness_data,
    get_wellness_for_date,
    update_wellness,
)


class TestWellnessTools:
    """Tests for wellness tools."""

    async def test_update_wellness_success(self, mock_config, respx_mock):
        """Test successful wellness record update."""
        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(return_value=mock_config)

        respx_mock.put("/athlete/i123456/wellness").mock(
            return_value=Response(
                200,
                json={"id": "2026-03-17", "weight": 71.5, "restingHR": 45},
            )
        )

        result = await update_wellness(
            date="2026-03-17",
            weight=71.5,
            resting_hr=45,
            ctx=mock_ctx,
        )

        response = json.loads(result)
        assert "data" in response
        assert response["data"]["date"] == "2026-03-17"
        assert response["data"]["body"]["weight_kg"] == 71.5
        assert response["data"]["heart"]["resting_hr"] == 45
        assert response["metadata"]["message"] == "Successfully updated wellness for 2026-03-17"

    async def test_update_wellness_validation_error(self, mock_config, respx_mock):
        """Test updating wellness with invalid date."""
        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(return_value=mock_config)

        result = await update_wellness(
            date="invalid-date",
            weight=71.5,
            ctx=mock_ctx,
        )

        response = json.loads(result)
        assert "error" in response
        assert "Invalid date format" in response["error"]["message"]

    async def test_update_wellness_no_data(self, mock_config, respx_mock):
        """Test updating wellness with no data fields."""
        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(return_value=mock_config)

        result = await update_wellness(
            date="2026-03-17",
            ctx=mock_ctx,  # No other args provided
        )

        response = json.loads(result)
        assert "error" in response
        assert "No wellness data provided" in response["error"]["message"]

    async def test_get_wellness_for_date_surfaces_new_fields(self, mock_config, respx_mock):
        """Previously-dropped API fields and `spO2` alias must reach the output."""
        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(return_value=mock_config)

        respx_mock.get("/athlete/i123456/wellness/2026-04-20").mock(
            return_value=Response(
                200,
                json={
                    "id": "2026-04-20",
                    "weight": 71.0,
                    "spO2": 97.5,
                    "vo2max": 54.2,
                    "abdomen": 82.4,
                    "carbohydrates": 320.5,
                    "protein": 140.2,
                    "fatTotal": 70.1,
                    "menstrualPhase": "FOLLICULAR",
                    "menstrualPhasePredicted": "OVULATING",
                    "locked": True,
                    "tempWeight": False,
                    "tempRestingHR": True,
                    "sportInfo": [
                        {"type": "Ride", "eftp": 252.3, "wPrime": 18000.0, "pMax": 1100.5},
                    ],
                },
            )
        )

        result = await get_wellness_for_date(date="2026-04-20", ctx=mock_ctx)
        response = json.loads(result)
        data = response["data"]

        assert data["body"]["vo2max"] == 54.2
        assert data["body"]["abdomen_cm"] == 82.4
        assert data["vitals"]["spo2_percent"] == 97.5
        assert data["nutrition"]["carbohydrates_g"] == 320.5
        assert data["nutrition"]["protein_g"] == 140.2
        assert data["nutrition"]["fat_total_g"] == 70.1
        assert data["other"]["menstrual_phase"] == "FOLLICULAR"
        assert data["other"]["menstrual_phase_predicted"] == "OVULATING"
        assert data["state_flags"] == {
            "locked": True,
            "temp_weight": False,
            "temp_resting_hr": True,
        }
        assert data["sport_info"] == [
            {"type": "Ride", "eftp": 252.3, "w_prime": 18000.0, "p_max": 1100.5}
        ]

    async def test_get_wellness_for_date_emits_scales(self, mock_config, respx_mock):
        """Scale labels must appear in metadata for subjective metrics in output."""
        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(return_value=mock_config)

        respx_mock.get("/athlete/i123456/wellness/2026-04-20").mock(
            return_value=Response(
                200,
                json={
                    "id": "2026-04-20",
                    "fatigue": 4,
                    "mood": 3,
                    "sleepQuality": 2,
                    "sleepScore": 82,
                    "readiness": 75,
                },
            )
        )

        result = await get_wellness_for_date(date="2026-04-20", ctx=mock_ctx)
        response = json.loads(result)

        scales = response["metadata"]["scales"]
        assert "fatigue" in scales and "1-5" in scales["fatigue"]
        assert "mood" in scales
        assert "sleep_quality" in scales and "inverted" in scales["sleep_quality"]
        assert "sleep_score" in scales and "0-100" in scales["sleep_score"]
        assert "readiness" in scales and "0-100" in scales["readiness"]
        # Scales for fields not present should be absent
        assert "soreness" not in scales

    async def test_get_wellness_for_date_surfaces_hydration_rating(self, mock_config, respx_mock):
        """The subjective 1-4 hydration rating was silently dropped by the
        formatter (it is enumerated on the model, so it never reached the
        custom_fields fallback either). It must surface with its scale."""
        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(return_value=mock_config)

        respx_mock.get("/athlete/i123456/wellness/2026-04-21").mock(
            return_value=Response(
                200,
                json={
                    "id": "2026-04-21",
                    "hydration": 2,
                    "hydrationVolume": 2.5,
                },
            )
        )

        result = await get_wellness_for_date(date="2026-04-21", ctx=mock_ctx)
        response = json.loads(result)

        data = response["data"]
        assert data["subjective"]["hydration"] == 2
        # the liters-drunk volume stays a separate nutrition field
        assert data["nutrition"]["hydration_liters"] == 2.5
        scales = response["metadata"]["scales"]
        assert "hydration" in scales and "1-4" in scales["hydration"]

    async def test_update_wellness_writes_hydration_rating(self, mock_config, respx_mock):
        """The subjective rating is writable and distinct from hydrationVolume."""
        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(return_value=mock_config)

        route = respx_mock.put("/athlete/i123456/wellness").mock(
            return_value=Response(
                200,
                json={"id": "2026-04-22", "hydration": 3, "hydrationVolume": 1.5},
            )
        )

        result = await update_wellness(
            date="2026-04-22", hydration=3, hydration_liters=1.5, ctx=mock_ctx
        )

        sent = json.loads(route.calls[0].request.content)
        assert sent["hydration"] == 3
        assert sent["hydrationVolume"] == 1.5
        response = json.loads(result)
        assert response["data"]["subjective"]["hydration"] == 3
        assert response["data"]["nutrition"]["hydration_liters"] == 1.5

    async def test_get_wellness_data_includes_scales_and_extra_fields(
        self, mock_config, respx_mock
    ):
        """List endpoint should also surface new fields and scale labels."""
        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(return_value=mock_config)

        respx_mock.get("/athlete/i123456/wellness").mock(
            return_value=Response(
                200,
                json=[
                    {
                        "id": "2026-04-20",
                        "fatigue": 3,
                        "carbohydrates": 250.0,
                        "vo2max": 53.0,
                    },
                    {
                        "id": "2026-04-19",
                        "fatigue": 2,
                        "protein": 130.0,
                    },
                ],
            )
        )

        result = await get_wellness_data(days_back=2, ctx=mock_ctx)
        response = json.loads(result)

        assert response["data"]["count"] == 2
        days = response["data"]["wellness_data"]
        # Most recent first
        assert days[0]["date"] == "2026-04-20"
        assert days[0]["nutrition"]["carbohydrates_g"] == 250.0
        assert days[0]["body"]["vo2max"] == 53.0
        assert days[1]["nutrition"]["protein_g"] == 130.0
        assert "fatigue" in response["metadata"]["scales"]

    async def test_update_wellness_new_fields(self, mock_config, respx_mock):
        """New fields (injury, body metrics, vitals, lab, menstrual, locked) round-trip correctly."""
        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(return_value=mock_config)

        respx_mock.put("/athlete/i123456/wellness").mock(
            return_value=Response(
                200,
                json={
                    "id": "2026-04-29",
                    "injury": 2,
                    "bodyFat": 18.5,
                    "abdomen": 80.0,
                    "vo2max": 55.0,
                    "systolic": 118,
                    "diastolic": 76,
                    "spO2": 98.0,
                    "respiration": 14.5,
                    "bloodGlucose": 5.2,
                    "lactate": 1.8,
                    "menstrualPhase": "FOLLICULAR",
                    "locked": True,
                },
            )
        )

        result = await update_wellness(
            date="2026-04-29",
            injury=2,
            body_fat=18.5,
            abdomen=80.0,
            vo2max=55.0,
            systolic=118,
            diastolic=76,
            spo2=98.0,
            respiration=14.5,
            blood_glucose=5.2,
            lactate=1.8,
            menstrual_phase="FOLLICULAR",
            locked=True,
            ctx=mock_ctx,
        )

        response = json.loads(result)
        data = response["data"]
        assert data["subjective"]["injury"] == 2
        assert data["body"]["body_fat_percent"] == 18.5
        assert data["body"]["abdomen_cm"] == 80.0
        assert data["body"]["vo2max"] == 55.0
        assert data["vitals"]["systolic_mmhg"] == 118
        assert data["vitals"]["diastolic_mmhg"] == 76
        assert data["vitals"]["spo2_percent"] == 98.0
        assert data["vitals"]["respiration_rate"] == 14.5
        assert data["other"]["blood_glucose_mmol_per_l"] == 5.2
        assert data["other"]["lactate_mmol_per_l"] == 1.8
        assert data["other"]["menstrual_phase"] == "FOLLICULAR"
        assert data["state_flags"]["locked"] is True
        assert "injury" in response["metadata"]["scales"]

    async def test_wellness_model_preserves_unknown_fields(self):
        """`extra=allow` keeps future API additions accessible on the model."""
        from intervals_icu_mcp.models import Wellness

        record = Wellness.model_validate({"id": "2026-04-20", "weight": 70.0, "futureMetric": 42})
        # Unknown field is preserved (not silently dropped)
        assert getattr(record, "futureMetric", None) == 42

    async def test_get_wellness_for_date_surfaces_custom_fields(self, mock_config, respx_mock):
        """Athlete-defined wellness fields survive MCP response formatting."""
        custom_fields = {
            "REMSleep": 7200,
            "DeepSleep": 5400,
            "LightSleep": 14400,
            "ActiveEnergy": 850,
            "RestingEnergy": 1750,
            "LeanBodyMass": 61.5,
        }
        respx_mock.get("/athlete/i123456/wellness/2026-04-20").mock(
            return_value=Response(
                200,
                json={"id": "2026-04-20", **custom_fields},
            )
        )

        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(return_value=mock_config)
        result = await get_wellness_for_date(date="2026-04-20", ctx=mock_ctx)
        response = json.loads(result)

        assert response["data"]["custom_fields"] == custom_fields

    async def test_get_wellness_data_surfaces_custom_fields_including_zero(
        self, mock_config, respx_mock
    ):
        """Range responses retain valid falsy custom-field values."""
        respx_mock.get("/athlete/i123456/wellness").mock(
            return_value=Response(
                200,
                json=[
                    {
                        "id": "2026-04-20",
                        "REMSleep": 0,
                        "ActiveEnergy": 0,
                        "LeanBodyMass": 61.5,
                    }
                ],
            )
        )

        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(return_value=mock_config)
        result = await get_wellness_data(days_back=1, ctx=mock_ctx)
        response = json.loads(result)

        assert response["data"]["wellness_data"][0]["custom_fields"] == {
            "REMSleep": 0,
            "ActiveEnergy": 0,
            "LeanBodyMass": 61.5,
        }


    async def test_get_recovery_analysis_compares_equal_calendar_windows(
        self,
        mock_config,
        respx_mock,
    ):
        """Recovery analysis uses equal windows, ignores nulls, and reports coverage."""
        records = []
        start = date(2026, 7, 29)

        for i in range(14):
            current = start + timedelta(days=i)
            recent = i >= 7

            record = {
                "id": current.isoformat(),
                "sleepSecs": 25200 if recent else 28800,
                "sleepScore": 75 if recent else 80,
                "sleepQuality": 2,
                "hrv": 70 + (i - 7) if recent else 60 + i,
                "restingHR": 47 - (i - 7) if recent else 50 - i,
            }

            if current.isoformat() == "2026-08-09":
                record["sleepScore"] = None

            if current.isoformat() == "2026-08-11":
                record["BodyBatteryMin"] = 29
                record["BodyBatteryMax"] = 89

            records.append(record)

        respx_mock.get("/athlete/i123456/wellness").mock(
            return_value=Response(
                200,
                json=records,
            )
        )

        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(
            return_value=mock_config
        )

        result = await get_recovery_analysis(
            comparison_window_days=7,
            reference_date="2026-08-11",
            ctx=mock_ctx,
        )

        response = json.loads(result)
        data = response["data"]

        assert data["source"] == "intervals_icu"

        assert data["period"] == {
            "reference_date": "2026-08-11",
            "comparison_window_days": 7,
            "previous_start": "2026-07-29",
            "previous_end": "2026-08-04",
            "recent_start": "2026-08-05",
            "recent_end": "2026-08-11",
        }

        assert data["today"]["sleep_duration"] == {
            "seconds": 25200,
            "hours": 7.0,
            "source_field": "sleepSecs",
        }

        assert data["today"]["sleep_score"] == 75
        assert data["today"]["hrv"]["value_ms"] == 76
        assert data["today"]["resting_hr"]["value_bpm"] == 41

        assert data["today"]["body_battery"] == {
            "min": 29,
            "max": 89,
        }

        assert data["baseline"]["hrv"] == {
            "recent_average": 73.0,
            "previous_average": 63.0,
            "change": 10.0,
        }

        assert data["baseline"]["resting_hr"] == {
            "recent_average": 44.0,
            "previous_average": 47.0,
            "change": -3.0,
        }

        assert data["baseline"]["sleep_duration"] == {
            "recent_average_seconds": 25200.0,
            "recent_average_hours": 7.0,
            "previous_average_seconds": 28800.0,
            "previous_average_hours": 8.0,
            "change_seconds": -3600.0,
            "change_hours": -1.0,
        }

        assert data["baseline"]["sleep_score"] == {
            "recent_average": 75.0,
            "previous_average": 80.0,
            "change": -5.0,
        }

        sleep_score_coverage = (
            data["data_quality"]["coverage"]["sleep_score"]
        )

        assert sleep_score_coverage["recent_available"] == 6
        assert sleep_score_coverage["previous_available"] == 7
        assert sleep_score_coverage["total_available"] == 13

        assert sleep_score_coverage["missing_dates"] == [
            "2026-08-09"
        ]

        assert data["data_quality"]["requested_days"] == 14
        assert data["data_quality"]["nights_available"] == 14
        assert data["data_quality"]["missing_fields"] == []

        assert "recovery_score" not in data
        assert "recommended_training" not in data

    async def test_get_recovery_analysis_validates_window(
        self,
        mock_config,
    ):
        mock_ctx = MagicMock()
        mock_ctx.get_state = AsyncMock(
            return_value=mock_config
        )

        result = await get_recovery_analysis(
            comparison_window_days=1,
            reference_date="2026-08-11",
            ctx=mock_ctx,
        )

        response = json.loads(result)

        assert response["error"]["type"] == "validation_error"


    def test_wellness_handles_null_sport_info(self):
        """API returns sportInfo: null for days without computed sport metrics —
        this should not crash parsing."""
        from intervals_icu_mcp.models import Wellness

        raw = {
            "id": "2026-07-09",
            "sportInfo": None,
        }
        wellness = Wellness.model_validate(raw)
        assert wellness.sport_info == []
