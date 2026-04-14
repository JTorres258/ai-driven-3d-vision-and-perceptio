"""
Shadow Mode Telemetry Module.

Simulates an MLOps pipeline for fleet-wide monitoring where an experimental 
'Shadow Model' runs silently alongside a 'Production Model'. 

Telemetry logs (images/states) are heavily bandwidth-constrained on drones. 
Therefore, data is strictly uploaded as JSON 'Flight Events' only when 
significant disparity, confidence degradation, or manual interventions arise.
"""

import json
import random
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional


class ProductionModel:
    """Simulates the currently deployed, active perception backbone."""
    def predict(self) -> Tuple[float, float]:
        """
        Returns:
            distance (float): Predicted distance to obstacle in meters.
            confidence (float): Uncertainty score from softmax/logits (0.0 to 1.0).
        """
        # Base realistic dummy values for flight
        distance = round(random.uniform(5.0, 50.0), 2)
        # Weighted higher to simulate a stable model that occasionally degrades
        confidence = round(random.uniform(0.65, 0.99), 2)
        return distance, confidence


class ShadowModel:
    """
    Simulates the experimental perception model running in silent shadow mode.
    Does not control the drone, only predicts concurrently for telemetry comparisons.
    """
    def predict(self, base_distance: float) -> Tuple[float, float]:
        """
        Returns:
            distance (float): Predicted distance. Injected with artificial drift.
            confidence (float): Uncertainty score.
        """
        # Simulate the shadow model occasionally diverging from production outputs.
        # e.g., A 25% chance of severe hallucination or systematic shift.
        drift = random.choice([0.9, 1.0, 1.0, 1.0, 1.35]) 
        distance = round(base_distance * drift, 2)
        confidence = round(random.uniform(0.70, 0.99), 2)
        return distance, confidence


def check_telemetry_trigger(
    prod_pred: float, 
    prod_conf: float, 
    shadow_pred: float, 
    manual_override: bool
) -> Tuple[bool, Optional[str]]:
    """
    Evaluates business/MLOps rules to determine if edge data should be 
    packaged and uploaded to the centralized datalake.
    
    Args:
        prod_pred (float): Production predicted distance.
        prod_conf (float): Production confidence score.
        shadow_pred (float): Shadow predicted distance.
        manual_override (bool): True if the human pilot assumed control.
        
    Returns:
        trigger (bool): True if emergency/interesting upload is required.
        reason (str/None): Description of the exact trigger condition.
    """
    # Rule 1: Human Intervention (High Priority Ground Truth Source)
    if manual_override:
         return True, "SYSTEM_OVERRIDE_INITIATED"
         
    # Rule 2: Production Model indicates low certainty
    if prod_conf < 0.75:
         return True, "PROD_CONFIDENCE_DEGRADATION"
         
    # Rule 3: Architectural Disagreement (> 20% disparity between predictions)
    if prod_pred > 0:
        disparity = abs(prod_pred - shadow_pred) / prod_pred
        if disparity > 0.20:
             return True, f"SHADOW_DISPARITY_EXCEEDED_{disparity*100:.1f}%"
             
    # Default: Discard edge buffer to save telemetry bandwidth
    return False, None


def log_flight_event(
    timestamp: str,
    prod_dist: float,
    prod_conf: float,
    shadow_dist: float,
    reason: str,
    sensor_state: Dict[str, Any]
) -> None:
    """
    Builds a bandwidth-efficient JSON payload mapping the flight context 
    exactly at the moment of the trigger.
    """
    event_payload = {
        "timestamp": timestamp,
        "trigger_reason": reason,
        "predictions": {
            "production": {
                "distance_m": prod_dist,
                "confidence": prod_conf
            },
            "shadow": {
                "distance_m": shadow_dist
            }
        },
        "sensor_state": sensor_state
    }
    
    # Simulate an MQTT/HTTP POST to the control tower or AWS IoT endpoint
    print(f"\n[EDGE TELEMETRY] Packaging Flight Event Buffer (Size: ~{len(json.dumps(event_payload))} bytes)")
    print(json.dumps(event_payload, indent=2))
    print("-" * 65)


def simulate_fleet_operation(iterations: int = 15) -> None:
    """Runs a simulated drone flight demonstrating shadow mode edge checks."""
    print(f"Starting Drone Flight Simulation ({iterations} frames)...\n" + "="*65)
    
    prod_model = ProductionModel()
    shadow_model = ShadowModel()
    
    for frame in range(1, iterations + 1):
        # 10% chance of the remote pilot actively forcing a trajectory correction
        manual_override = random.random() < 0.10
        
        # Inference pipeline (Prod + Shadow)
        prod_dist, prod_conf = prod_model.predict()
        shadow_dist, _ = shadow_model.predict(base_distance=prod_dist)
        
        # MLOps Telemetry validation check
        trigger, reason = check_telemetry_trigger(
            prod_dist, prod_conf, shadow_dist, manual_override
        )
        
        if trigger:
            timestamp = datetime.utcnow().isoformat() + "Z"
            # Simulated IMU/Battery status at the exact moment of trigger
            sensor_state = {
                "battery_pct": round(random.uniform(40.0, 95.0), 1),
                "gps_locked": True,
                "velocity_ms": round(random.uniform(5.0, 15.0), 1)
            }
            print(f"Frame {frame:03d} | ALERT DETECTED: Upload protocol engaged.")
            log_flight_event(timestamp, prod_dist, prod_conf, shadow_dist, reason, sensor_state)
        else:
            print(f"Frame {frame:03d} | Nominal. Discarding cyclic buffer to save bandwidth.")
            
        time.sleep(0.4)


if __name__ == "__main__":
    simulate_fleet_operation()
