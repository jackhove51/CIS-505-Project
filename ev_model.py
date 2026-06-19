from dataclasses import dataclass
from math import ceil


@dataclass
class VehicleParams:
    battery_capacity_kwh: float = 75.0
    consumption_kwh_per_km: float = 0.20
    start_soc_frac: float = 1.0
    reserve_soc_frac: float = 0.10
    session_fee: float = 0.0
    avg_speed_kmh: float = 100.0
    soc_levels: int = 20
    w_cost: float = 1.0
    value_of_time_per_hour: float = 18.0  # USD/hr, roughly half the US median wage

    @property
    def w_time(self) -> float:
        return self.value_of_time_per_hour * self.w_cost


def level_kwh(p: VehicleParams) -> float:
    return p.battery_capacity_kwh / p.soc_levels


def reserve_level(p: VehicleParams) -> int:
    return ceil(p.reserve_soc_frac * p.soc_levels)


def start_level(p: VehicleParams) -> int:
    return round(p.start_soc_frac * p.soc_levels)


def drive_levels(dist_km: float, p: VehicleParams) -> int:
    return ceil(dist_km * p.consumption_kwh_per_km / level_kwh(p))


def drive_weight(dist_km: float, p: VehicleParams) -> float:
    return p.w_time * (dist_km / p.avg_speed_kmh)


def charge_step_weight(price: float, rate_kw: float, p: VehicleParams) -> float:
    energy = level_kwh(p)
    return p.w_cost * energy * price + p.w_time * (energy / rate_kw)
