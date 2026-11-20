# OpenArm V10 startup changes (2026-11-02)

## Scope and zero reference

The standard mechanical zero remains seven zero-radian arm targets per arm,
with normalized gripper target zero (closed). The archived reference is
`/home/yundu1/零位/零位信息.json`. No encoder-zero command is sent and that file
is not used as a calibration offset. Its measured values are historical
feedback, not offsets. A hanging arm is not proof of correct encoder zero.

The hardware driver no longer calls `return_to_zero()` on activation. It holds
the measured startup pose, including the current gripper opening. Holding the
startup pose and deliberately returning to standard zero are separate actions.
The previous startup gripper command also passed a joint-space constant directly
as motor radians; that entire automatic-homing path has been removed.

## Startup behavior

1. Before opening CAN, acquire an abstract Unix socket lock named
   `openarm.hardware.can.<interface>`. A second instance using this modified
   driver and the same interface/network namespace fails before sending CAN.
   Separate arms on `can0` and `can1` can coexist. The lock is held even while
   inactive and is released when the hardware object/process is destroyed.
2. Configuration requests state only. A new receive-only CAN observer avoids
   treating the underlying library's default zero-filled cache as feedback.
   Each motor, including the gripper, must supply at least three decoded state
   replies during a minimum 100 ms observation, all no older than 100 ms.
   Timeout is one second. Missing/invalid replies reject startup.
3. Motor speed above 0.1 rad/s or drift above 0.02 motor radians rejects startup.
   Joint positions must lie within the loaded URDF bounds, allowing 0.01 rad
   arm / 0.001 m finger boundary tolerance for feedback only. No pose is snapped
   to zero or clamped to a limit. These checks do not change encoder calibration.
4. Activation repeats the checks. It seeds ROS position commands from the
   measured pose, zeros velocity/feedforward commands, preloads the measured
   hold command, then enables motors. It verifies fresh feedback again; change
   above 0.02 arm radians / 0.001 finger meters aborts activation and attempts
   disable. On success it seeds the final measured hold pose.
5. Inactive `write()` sends no commands. Non-finite active commands are rejected.
   Deactivation, lifecycle error/shutdown, and normal destruction attempt motor
   disable if activation was attempted. They do not send a homing trajectory.

## Important limits

- The lock is cooperative: OLD binaries, `set_zero.sh`, CAN utilities, and other
  programs that do not take the lock can still transmit. Stop all such programs
  before any supervised hardware test. This is not a bus-level interlock.
- CAN enable/disable and hold transmissions are not firmware-level safety
  acknowledgements. Loss of communications/power or SIGKILL may prevent disable.
  Software exit is not an emergency stop. Removing holding torque can let arms
  fall; use appropriate mechanical support, not hands in the motion path.
- These are startup checks, not full runtime feedback/fault monitoring, collision
  avoidance, or proof of encoder calibration. Existing runtime gains and mapping
  are retained. Firmware response to preload/enable has NOT been hardware-tested.
- The two hardware components activate sequentially, not atomically. Failure of
  one arm may occur after the other has enabled; keep physical emergency stop
  available and inspect both arms before resuming.
- Do not resume the 15-second automatic recording takeover immediately after
  installing this change. First perform a supervised stationary startup check,
  compare actual joint feedback with physical pose and the saved zero reference,
  and verify one driver and one command source. Do not write new encoder zeros
  to conceal a discrepancy.

## Offline verification / build

No real hardware launch is part of these commands. A fresh build directory is
used because the old CMake cache references `/home/openarm/ros2_ws`.

```bash
source /opt/ros/humble/setup.bash
cmake -S /home/yundu1/ros2_ws/src/openarm_ros2/openarm_hardware \
  -B /home/yundu1/ros2_ws/build_startup_safety/openarm_hardware \
  -DOpenArmCAN_DIR=/home/yundu1/ros2_ws/install/openarm_can/lib/cmake/OpenArmCAN \
  -DCMAKE_INSTALL_PREFIX=/home/yundu1/ros2_ws/install/openarm_hardware \
  -DBUILD_TESTING=ON
cmake --build /home/yundu1/ros2_ws/build_startup_safety/openarm_hardware -j2
ctest --test-dir /home/yundu1/ros2_ws/build_startup_safety/openarm_hardware --output-on-failure
```

Unit tests exercise synthetic feedback and fake activation callbacks; ownership
tests use test-named Unix sockets and child processes only, never CAN interfaces.
Offline passing tests do not establish real-machine safety.

### Receive-filter correction

The first installed revision incorrectly included `CAN_ERR_FLAG` in the
`CAN_RAW_FILTER` mask. On this host that subscribed to the error receive path
and missed normal motor replies, causing configuration to fail before enable.
The mask is now only `CAN_SFF_MASK | CAN_EFF_FLAG | CAN_RTR_FLAG`; error-frame
reception is not enabled. A regression test checks this exact mask.

Read-state-only diagnostics on both physical CAN interfaces confirmed all 16
motors replied (three replies each). A simultaneous filter comparison received
zero replies through the original mask and one reply per interface through the
corrected mask. These diagnostics sent only existing 0x7ff / 0xCC read-state
queries, not enable, position, parameter-write, or encoder-zero commands.
They verify communications/filtering, NOT enabled-motion behavior or zero pose.

After the correction, `startup_feedback_probe` called the actual driver's
`on_init` and `on_configure` for both arms using the installed robot description
and successfully read all 16 joint positions. It never calls `on_activate` or
`write` and is not registered with ctest. The 17 offline unit tests also passed.
The observed hanging pose differed from the archived zero (for example right
joint 3 was about -0.1513 rad), so this was not a zero-pose acceptance test.

Install only with every hardware process stopped, after preserving the previous
installed package. `cmake --install` deploys files; it does not launch hardware.

```bash
cmake --install /home/yundu1/ros2_ws/build_startup_safety/openarm_hardware
```
