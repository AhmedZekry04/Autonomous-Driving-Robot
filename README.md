# Self-Driving Navigation System

This package implements autonomous robot navigation using lane detection and traffic sign recognition. The system supports two main modes of operation: lane-following with sequence-based navigation or time-based routing with odometry tracking.

## Overview

The self-driving system allows the Hiwonder robot to autonomously navigate a predefined parcours using:
- Yellow lane detection for line-following navigation
- YOLO object detection for traffic sign recognition (stop signs, direction indicators, parking signs)
- PID control for smooth steering
- Sequential step execution or timed route execution

## System Architecture

### Key Components

1. Lane Detection: Detects yellow lines and calculates lane position
   - lane_detect.py: For timed mode (self_driving_timed_light.py)
   - lane_detect_seq.py: For sequence mode (self_driving_seq.py)
2. Main Navigation Scripts:
   - self_driving_seq.py (self_driving3.launch): Step-by-step sequence-based navigation
   - self_driving_timed_light.py (self_driving4.launch): Time/distance-based navigation with YOLO signs
3. Base Launch (self_driving_base.launch): Loads YOLO detection and camera configuration
4. Parking (parking_test.py): Lateral shift maneuver execution (called from timed mode)
5. ROS Services: Allow external control and monitoring

### Data Flow

```
Camera Feed
    |
Lane Detection (HSV-based)
    |
Navigation Script (self_driving_seq.py or self_driving_timed_light.py)
    |
PID Controller
    |
Motor Commands (Twist messages to /hiwonder_controller/cmd_vel)
```

## Running the System

### Prerequisites

1. Robot hardware initialized and connected
2. Camera stream running and publishing to /<camera_name>/rgb/image_raw
3. YOLO detection node initialized (if using detection features)
4. ROS environment sourced

### Important: Set Camera Parameter

Before launching either mode, set the depth camera parameter:

```bash
rosparam set /depth_camera_name robot_1/depth_cam
```

This parameter is used by both self_driving_seq.py and self_driving_timed_light.py to locate the camera feed. If not set, the scripts will fail to subscribe to camera images.

### Option 1: Sequence-Based Navigation (self_driving3.launch)

Use this mode for predefined routes with waypoint-based turns.

```bash
roslaunch hiwonder_example self_driving3.launch
```

**Features:**
- Step-by-step navigation through a fixed route
- Detects turn points when lane position exceeds thresholds
- Supports inside turns (where robot must back up and realign)
- Uses lane_detect_seq.py for improved lane detection
- Configurable route sequence in self_driving_seq.py

**Key Parameters (edit in self_driving_seq.py):**
- ROUTE: Tuple list of (follow_side, turn_action) pairs
- START_STEP: Skip first N steps for testing (default: 3)
- TURN_THRESHOLD: Lane position threshold to detect turns
- REVERSE_PARAMETERS: Backup timing for lane correction

**Expected Flow:**
1. Robot follows left lane with defined rules
2. Detects turn when lane_x > threshold (default 150 on left)
3. Applies steering while turn is detected
4. Advances to next step when lane returns to normal
5. Repeats for all steps in ROUTE

### Option 2: Time-Based Navigation with Odometry (self_driving4.launch)

Use this mode for distance-based segment execution with YOLO traffic sign detection.

```bash
roslaunch hiwonder_example self_driving4.launch
```

**Features:**
- Initial phase: Lane following with turn counting
- After N turns: Switches to timed route execution
- Uses odometry distance measurements for straight segments
- Responds to traffic signs (speed limits, stop signs, parking indicators)
- Supports speed regulation based on signs
- Launches parking sequence automatically at end

**Key Parameters (edit in self_driving_timed_light.py):**
- SWITCH_AFTER_TURNS: Number of lanes to follow before switching modes (default: 3)
- ROUTE_SEQUENCE: List of (name, speed_x, steering_z, duration/distance, mode)
- STOP_SIGN_MIN_AREA: Minimum area to recognize stop sign

**Expected Flow:**
1. Phase 1 (Lane Following): Robot counts completed lane turns
2. Phase 2 (Timed Route): After 3 turns, execute predefined route sequence
3. Phase 3 (Final Route): Optional second sequence execution
4. Phase 4 (Parking): Auto-launch parking_test.py for parking behavior

## Navigation Modes Explained

### Lane Following (Both Modes)

The robot uses PID control to maintain position relative to detected lane:
- Left lane: target position = 100 pixels from left edge
- Right lane: target position = 620 pixels (from left edge)
- PID gains tuned for smooth steering in Ackermann steering model

### Turn Detection (self_driving3.launch - self_driving_seq.py)

**Outside Turns** (follow_side not equal to turn_direction):
- Detected when lane_x exceeds threshold
- Robot applies fixed steering angle
- Completes when lane_x returns below threshold for N seconds

**Inside Turns** (follow_side equals turn_direction):
1. Robot backs up for alignment
2. Enters settle loop: forward/backward micro-adjustments
3. Applies forced steering for minimum duration
4. Returns to lane following

### Timed Route Execution (self_driving4.launch - self_driving_timed_light.py)

Each segment specifies:
- mode="dist": Drive until odometry distance traveled >= value (meters)
- mode="time": Publish command for value seconds
- Can also specify detection classes to stop on (red/stop signs)

### Parking Maneuvers (parking_test.py)

Automatically launched at end of Phase 3 in timed mode. Implements lateral shift parking with:
- Parking sign detection via YOLO (class_name='parking')
- Multi-step shift maneuvers: turn + straight + counter-turn + reverse
- Left and right shift support
- Target position centering (PARK_TARGET_CX = 78, tolerance = 40 pixels)
- Area matching for parking spot size (PARK_TARGET_AREA = 10000, tolerance = 2000)

## File Structure

```
self_driving/
├── README.md                    # This file
├── lane_detect.py               # Lane detection for timed_light mode
├── lane_detect_seq.py           # Enhanced lane detection for seq mode
├── parking_test.py              # Parking maneuver execution (called by timed mode)
├── self_driving_seq.py          # Main navigation script (sequence mode)
├── self_driving_timed_light.py  # Main navigation script (timed mode)
├── self_driving3.launch         # Launch file for sequence mode
├── self_driving4.launch         # Launch file for timed mode
└── self_driving_base.launch     # Base configuration (YOLO + camera)
```

## Configuration

### ROI (Region of Interest) Definition

Lane detection uses multiple horizontal stripes for robust detection:

```python
LEFT_ROIS = (
    (450, 480, 0, 320, 0.7),     # Near camera: high weight
    (390, 420, 0, 320, 0.2),     # Middle distance
    (330, 360, 0, 320, 0.1)      # Far distance: low weight
)
```

Format: (y_min, y_max, x_min, x_max, weight)

### YOLO Configuration

Edit in self_driving_base.launch:
- engine: Model file to use (default: traffic_signs_640s_7_0.engine)
- conf_thresh: Confidence threshold (default: 0.8)
- classes: List of detectable objects

## Control Services

Once launched, you can control the robot via ROS services:

```bash
# Start robot
rosservice call /self_driving/enter

# Run/stop navigation
rosservice call /self_driving/set_running "data: true"

# Stop and cleanup
rosservice call /self_driving/exit
```

## Tuning Guide

### Lane Detection Issues (Both Modes)

1. Line not detected:
   - Adjust HSV range in lane_detect.py (timed mode) or lane_detect_seq.py (seq mode)
   - Check ROI coverage matches lane position

2. False detections:
   - Increase saturation threshold in HSV
   - Reduce ROI weights for noisy regions

### Turn Detection Issues (self_driving3.launch - self_driving_seq.py)

1. Turns not detected:
   - Lower TURN_THRESHOLD value
   - Increase TURN_DETECT_GRACE period

2. Extra turns detected:
   - Increase TURN_END_TIMEOUT to require longer straight
   - Increase TURN_COOLDOWN between turns

### Inside Turn Tuning (self_driving3.launch - self_driving_seq.py)

1. Robot overshoots settle zone:
   - Reduce SETTLE_FORWARD_SPEED
   - Increase SETTLE_PID_RANGE

2. Robot gets stuck in settle loop:
   - Increase SETTLE_FORWARD_DURATION
   - Adjust SETTLE_TOLERANCE window

3. Robot not completing inside turn:
   - Increase INSIDE_TURN_MIN_STEER_DURATION
   - Increase INSIDE_TURN_BACKUP_DURATION for better rotation

### Speed and Steering Tuning (self_driving4.launch - self_driving_timed_light.py)

Edit _SPD, _R_ANG, _SL_R, _SL_L in self_driving_timed_light.py:
```python
_SPD = 0.15               # Normal forward speed
_SPD_S = 0.10            # Slower speed
_R_ANG = -0.6 steering   # Full right steering angle
_SL_R = 0.05             # Slight right steering
_SL_L = -0.05            # Slight left steering
```

## Debugging

### View Lane Detection Output

Lane detection creates image visualization:
```bash
# Displays real-time lane detection with drawn lines
rostopic echo /self_driving/image_result
```

### View YOLO Detections

Enable image output:
```bash
rostopic echo /yolov5/object_image
```

### Log Messages

Navigation outputs detailed logs for each phase:
```
[FOLLOW] Step 0: lane_x=105 threshold=150
[TURN] Detected! Step 1 (right/right)
[INSIDE_FWD] Forward 0.2/0.0s
[SETTLE_FWD] lane_x=620 setpoint=620 dist=0 good=8/8
[SETTLE_STR] Straightening 0.5/1.5s
```

## Performance Notes

- Frame rate: 33 Hz typical (30ms per frame)
- Lane detection: Robust in good lighting, affected by shadows
- Turn accuracy: +/- 10 degrees steering typical
- Distance accuracy: Within 5% for odometry-based routing

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Robot not moving | Services not called | Call /self_driving/enter then set_running true |
| Erratic steering | Lane detection noise | Improve lighting, clean lens |
| Turns not triggering | Threshold too high | Lower TURN_THRESHOLD value |
| Robot spins in place | Ackermann model mismatch | Check wheelbase constant (0.213m) |
| YOLO not detecting signs | Model not loaded | Check /yolov5/init_finish parameter |

## Next Steps

1. Choose mode: Use self_driving3.launch for sequence-based, self_driving4.launch for timed/odometry-based
2. Calibrate lane detection: Run with appropriate lane_detect settings and verify yellow line visibility
3. Tune for seq mode: Adjust turn detection thresholds (self_driving_seq.py) based on actual parcours layout
4. Tune for timed mode: Measure actual distances and times for route segments, adjust speeds and steering
5. Test parking: Verify parking sign detection and shift maneuvers (timed mode only)
6. Integration: Combine with other modules (manipulation, etc.)

## References

- Lane Detection: HSV-based yellow detection with multi-region weighting
- YOLO Traffic Signs: TensorRT-optimized YOLOv5 inference
- Steering Model: Ackermann steering with wheelbase = 0.213m
- PID Tuning: Proportional control with fixed gains (Kp=0.01)
