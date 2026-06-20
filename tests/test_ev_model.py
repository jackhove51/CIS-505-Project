import pytest

from ev_model import VehicleParams, level_kwh, reserve_level, start_level, drive_levels, drive_weight, charge_step_weight


def test_w_time_multiplies_value_of_time_by_w_cost():
    params = VehicleParams(w_cost=1.5, value_of_time_per_hour=20.0)
    assert params.w_time == pytest.approx(30.0)


def test_level_kwh_basic_division():
    params = VehicleParams(battery_capacity_kwh=75.0, soc_levels=20)
    assert level_kwh(params) == pytest.approx(3.75)


def test_level_kwh_times_soc_levels_equals_capacity():
    params = VehicleParams(battery_capacity_kwh=75.0, soc_levels=20)
    assert params.soc_levels * level_kwh(params) == pytest.approx(params.battery_capacity_kwh)


def test_reserve_level_no_rounding():
    params = VehicleParams(reserve_soc_frac=0.20, soc_levels=20)
    assert reserve_level(params) == 4


def test_reserve_level_ceil():
    params = VehicleParams(reserve_soc_frac=0.12, soc_levels=20)
    assert reserve_level(params) == 3


def test_reserve_level_zero():
    params = VehicleParams(reserve_soc_frac=0.0, soc_levels=20)
    assert reserve_level(params) == 0


def test_reserve_level_never_exceeds_soc_levels():
    for i in range(6):
        params = VehicleParams(reserve_soc_frac=i * 0.2, soc_levels=20)  # 0.2, 0.4, ... , 1.0
        assert reserve_level(params) <= params.soc_levels


def test_start_level_rounding_down():
    params = VehicleParams(start_soc_frac=0.67, soc_levels=20)
    assert start_level(params) == 13


def test_start_level_rounding_up():
    params = VehicleParams(start_soc_frac=0.33, soc_levels=20)
    assert start_level(params) == 7


def test_start_level_rounding_midpoint():
    params = VehicleParams(start_soc_frac=0.475, soc_levels=20)
    assert start_level(params) == 10


def test_start_level_zero():
    params = VehicleParams(start_soc_frac=0.0, soc_levels=20)
    assert start_level(params) == 0


def test_drive_level_rounding():
    params = VehicleParams(battery_capacity_kwh=80.0, soc_levels=20, consumption_kwh_per_km=0.2)
    assert drive_levels(50.0, params) == 3


def test_drive_level_no_rounding():
    params = VehicleParams(battery_capacity_kwh=80.0, soc_levels=20, consumption_kwh_per_km=0.2)
    assert drive_levels(40.0, params) == 2


def test_drive_weight_whole_number():
    params = VehicleParams(avg_speed_kmh=50.0, value_of_time_per_hour=20.0, w_cost=1.0)
    assert drive_weight(100.0, params) == pytest.approx(40.0)


def test_drive_weight_float():
    params = VehicleParams(avg_speed_kmh=50.0, value_of_time_per_hour=20.0, w_cost=1.0)
    assert drive_weight(51.0, params) == pytest.approx(20.4)


def test_charge_step_weight_paid_charger():
    params = VehicleParams(battery_capacity_kwh=80.0, soc_levels=20, value_of_time_per_hour=20.0, w_cost=1.0)
    assert charge_step_weight(5.0, 1.0, params) == pytest.approx(100.0)


def test_charge_step_weight_free_charger():
    params = VehicleParams(battery_capacity_kwh=80.0, soc_levels=20, value_of_time_per_hour=20.0, w_cost=1.0)
    assert charge_step_weight(0.0, 1.0, params) == pytest.approx(80.0)