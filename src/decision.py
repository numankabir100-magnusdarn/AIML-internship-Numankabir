from dataclasses import dataclass
from typing import Dict, Optional

from .risk import PredictedInteraction, RiskReport, RiskTrend, compute_risk_trend


@dataclass(frozen=True)
class Decision:
    track_id: int
    action: str
    current_risk: float
    time_to_threshold: float | None
    trend: str
    reason: str


class DecisionEngine:
    def __init__(
        self,
        high_risk_threshold: float = 0.25,
        monitor_risk: float = 0.15,
        caution_risk: float = 0.4,
        alert_risk: float = 0.7,
        alert_within_seconds: float = 1.0,
        caution_within_seconds: float = 2.0,
        monitor_within_seconds: float = 3.0,
    ) -> None:
        self.high_risk_threshold = high_risk_threshold
        self.monitor_risk = monitor_risk
        self.caution_risk = caution_risk
        self.alert_risk = alert_risk
        self.alert_within_seconds = alert_within_seconds
        self.caution_within_seconds = caution_within_seconds
        self.monitor_within_seconds = monitor_within_seconds

    def update(self, report: RiskReport) -> Dict[int, Decision]:
        worst_per_track = self._worst_interaction_per_track(report)
        decisions: Dict[int, Decision] = {}
        for track_id, interaction in worst_per_track.items():
            risk_trend = compute_risk_trend(
                interaction.risk_timeline,
                self.high_risk_threshold,
                per_horizon_risk=interaction.per_horizon_risk,
            )
            action = self._decide(
                current_risk=risk_trend.current_risk,
                time_to_threshold=risk_trend.time_to_threshold,
                trend=risk_trend.trend,
            )
            decisions[track_id] = Decision(
                track_id=track_id,
                action=action,
                current_risk=risk_trend.current_risk,
                time_to_threshold=risk_trend.time_to_threshold,
                trend=risk_trend.trend,
                reason=self._reason(interaction, risk_trend, action),
            )
        return decisions

    def _decide(
        self,
        current_risk: float,
        time_to_threshold: Optional[float],
        trend: str,
    ) -> str:
        if current_risk >= self.alert_risk:
            return "alert"
        if time_to_threshold is not None and time_to_threshold <= self.alert_within_seconds:
            return "alert"
        if current_risk >= self.caution_risk:
            return "caution"
        if time_to_threshold is not None and time_to_threshold <= self.caution_within_seconds:
            return "caution"
        if current_risk >= self.monitor_risk:
            return "monitor"
        if time_to_threshold is not None and time_to_threshold <= self.monitor_within_seconds:
            return "monitor"
        return "no_action"

    @staticmethod
    def _reason(
        interaction: PredictedInteraction,
        risk_trend: RiskTrend,
        action: str,
    ) -> str:
        thresholds = {
            "alert": "risk {current:.2f} or crossing high-risk by {ttt:.1f}s",
            "caution": "risk {current:.2f} or crossing high-risk by {ttt:.1f}s",
            "monitor": "risk {current:.2f} or crossing high-risk by {ttt:.1f}s",
            "no_action": "no significant projected contact",
        }
        ttt = risk_trend.time_to_threshold if risk_trend.time_to_threshold is not None else 0.0
        return thresholds[action].format(current=risk_trend.current_risk, ttt=ttt)

    def _worst_interaction_per_track(
        self,
        report: RiskReport,
    ) -> Dict[int, PredictedInteraction]:
        worst: Dict[int, PredictedInteraction] = {}
        for interaction in report.interactions:
            for track_id in (interaction.track_a, interaction.track_b):
                current = worst.get(track_id)
                if current is None or interaction.max_risk > current.max_risk:
                    worst[track_id] = interaction
        return worst