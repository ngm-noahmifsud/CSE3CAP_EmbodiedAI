"""
Capstone Project
Robot : Hello Robot Stretch 2
Model    : mujoco_menagerie/hello_robot_stretch/stretch.xml
Task  : Pick cubes one-by-one from a table and place them on a moving conveyor.

Layout (top-down view, +x right, +y toward robot from table):
  Robot starts to the left of the table, works at y = ROBOT_WORK_Y for all ops.
  Table centre   : x=0,   y=-0.9,  surface z=0.70
  Conveyor centre: x=2.0, y=-1.20, surface z=0.31
  Robot work y   : -0.40  (clear of both table and conveyor faces)

Pick sequence (extend FIRST so arm sweeps over cube, THEN lower):
  1. NAV_TO_PICK  — align robot x with cube x
  2. FACE_PICK    — face +x (arm extends in -y)
  3. OPEN_GRIP    — spread fingers before extending
  4. EXTEND_PICK  — extend arm at HOME height (gripper above cube level)
  5. LOWER_PICK   — lower lift to cube height; kinematically snap cube to gripper
  6. CLOSE_GRIP   — close fingers (visual) + activate kinematic carry
  7. RAISE        — raise lift above table (cube follows)
  8. RETRACT_PICK — retract arm

Run:  python main.py
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

# ── Scene constants (must match scene.xml) ────────────────────────────────────
NUM_CUBES   = 4
CUBE_HALF   = 0.040      # cube half-edge (m) — 8 cm cubes

TABLE_X     =  0.0
TABLE_Y     = -0.9
TABLE_TOP_Z =  0.70      # table surface z (geom centre 0.68 + half 0.02)

CONV_X      =  2.0       # conveyor moved closer (was 2.8)
CONV_Y      = -1.20      # belt centre — accounts for grip_offset_y=-0.148 so cube
                          # lands at belt middle (slider at y=-1.052, cube at y=-1.20)
CONV_TOP_Z  =  0.31      # top of flat belt surface z

# Robot parks at this y for every arm operation; just moves in x between tasks.
# y=-0.40 keeps the robot body ~17 cm clear of the conveyor face (at y≈-0.74)
# and ~20 cm clear of the table face (at y≈-0.80), eliminating visual clipping.
ROBOT_WORK_Y = -0.40

# Gripper structural offset: at arm_joint=0 the slider is 0.192 m behind robot
# centre in -y.  Robot at y=-0.40:
#   fingertips to TABLE_Y=-0.9: ext = 0.9 - 0.40 - 0.192 - 0.148(grip_off_y) = 0.160 m
#   To reach slider_y=-1.052: extension = 1.052 - 0.40 - 0.192 = 0.460 m → cube_y=-1.200=CONV_Y
GRIPPER_Y_HOME   = -0.592   # gripper y when arm=0, robot at y=-0.40
# grip_offset_y = -0.148 (fingertips are 0.148 m further in -y than the slider).
# To bring fingertips to TABLE_Y=-0.9 with zero y-teleport at snap time:
#   arm_ext = TABLE_Y - GRIPPER_Y_HOME - grip_offset_y
#           = 0.9 - 0.592 - 0.148 = 0.160 m
ARM_REACH_TABLE  =  0.160   # extension so fingertip_y = TABLE_Y  (zero-teleport pick)
ARM_REACH_CONV   =  0.460   # extension so slider_y=-1.052 → cube_y=-1.200=CONV_Y
ARM_HOME         =  0.0

# Lift joint values  (gripper_z ≈ 0.695 + lift_joint)
# CUBE_Z = table surface + cube half-size
# LIFT_PICK: gripper at cube centre height (arm retracted formula: z ≈ 0.695 + lift)
CUBE_Z         = TABLE_TOP_Z + CUBE_HALF          # 0.74 — cube centre on table
# LIFT_PICK is not a fixed constant: LOWER_PICK monitors the rubber-tip z position
# every step and snaps when tip_z reaches CUBE_Z, so no formula is needed.
LIFT_DROP      =  0.0                # slider at z≈0.60 m; cube (fingertips
                                     # ~0.15 m below slider) at z≈0.45 m,
                                     # safely above belt surface (0.31 m)
LIFT_HOME      = 0.35       # carry / travel height  (gripper ≈ 1.045 m)

# Gripper (joint_gripper_slide ctrl range: -0.005 … 0.04)
GRIP_OPEN   =  0.04
GRIP_CLOSED = -0.005
GRIP_WAIT   =  80    # physics steps to hold gripper command before continuing

# Kinematic attachment: cube is snapped to gripper within this radius.
# At EXTEND_PICK the gripper is ~0.07 m above the cube; with small x/y
# misalignment the 3D distance can reach ~0.20 m, so use 0.25 m.
SNAP_RADIUS = 0.25   # metres

# Belt force applied to cubes on the conveyor
BELT_FORCE  = 4.0

# x-coordinate at which cubes vanish off the far end of the belt.
# Belt half-length = 0.63 m  →  physical far edge at CONV_X + 0.63 = 2.63 m.
BELT_END_X  = CONV_X + 0.63

# Actuator indices (must match order in scene.xml → from arm actuators)
A_FWD, A_TURN, A_LIFT, A_ARM, A_WRIST, A_GRIP = 0, 1, 2, 3, 4, 5

# Grip reference: link_gripper_slider sits at the arm tip (reliable y/z position
# used for snap distance checks).  At snap time we compute the rubber_tip midpoint
# offset once and store it, so the cube tracks the actual fingertip position
# without any hardcoded directional guesses.
GRIPPER_BODY = "link_gripper_slider"

# Simulation speed: physics steps executed per viewer frame.
# 5 steps/frame ≈ 5× faster navigation than single-step with sleep.
SIM_STEPS_PER_FRAME = 5

# ── State IDs ─────────────────────────────────────────────────────────────────
(
    INIT,
    FIND_CUBE,
    NAV_TO_PICK,    # drive to cube x, work y
    FACE_PICK,      # rotate to face +x (theta = 0)
    OPEN_GRIP,      # open gripper before extending
    EXTEND_PICK,    # extend arm at HOME height (sweeps above cube)
    LOWER_PICK,     # lower lift to cube height; snap cube to gripper
    CLOSE_GRIP,     # close gripper fingers (visual)
    RAISE,          # raise cube above table
    RETRACT_PICK,   # retract arm (cube travels with gripper)
    NAV_TO_CONV,    # drive to conveyor x, work y
    FACE_DROP,      # rotate to face +x
    EXTEND_DROP,    # extend arm over conveyor
    LOWER_DROP,     # lower lift to drop height
    RELEASE,        # detach cube, open gripper
    RETRACT_DROP,   # retract arm
    HOME_LIFT,      # raise lift back to carry height
    DONE,
) = range(18)

STATE_NAMES = [
    "INIT", "FIND_CUBE",
    "NAV_TO_PICK", "FACE_PICK", "OPEN_GRIP",
    "EXTEND_PICK", "LOWER_PICK", "CLOSE_GRIP",
    "RAISE", "RETRACT_PICK",
    "NAV_TO_CONV", "FACE_DROP", "EXTEND_DROP", "LOWER_DROP",
    "RELEASE", "RETRACT_DROP", "HOME_LIFT",
    "DONE",
]

# ── Low-level helpers ─────────────────────────────────────────────────────────

def robot_pos(data):
    """Return robot base world position [x, y, z]."""
    return data.qpos[0:3].copy()


def robot_angle(data):
    """Return robot heading (rad) — yaw (rotation around world z).
    Uses the full quaternion yaw formula so it stays correct even after
    the base quaternion accumulates non-zero x/y components from physics.
    """
    q = data.qpos[3:7]          # w, x, y, z (freejoint)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def cube_pos(model, data, idx):
    """Return world position of cube_<idx> centre."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{idx}")
    return data.xpos[bid].copy()


def gripper_pos(model, data):
    """Return world position of the gripper-slider body (arm tip).
    This is the reliable position for snap detection and carry tracking.
    A fixed GRIP_Z_OFFSET is applied in _try_snap to shift the cube down
    to fingertip level.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
    return data.xpos[bid].copy()


def joint_val(model, data, name):
    """Return current qpos value for a named joint."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[model.jnt_qposadr[jid]])


def navigate(data, target_xy, threshold=0.06):
    """
    Point-and-drive 2-D navigation for differential drive.
    Turns to face the target, then drives forward with heading correction.
    Returns True when within threshold of target.
    """
    pos   = robot_pos(data)[:2]
    theta = robot_angle(data)
    dx, dy = target_xy[0] - pos[0], target_xy[1] - pos[1]
    dist   = np.hypot(dx, dy)

    if dist < threshold:
        data.ctrl[A_FWD]  = 0.0
        data.ctrl[A_TURN] = 0.0
        return True

    desired_theta = np.arctan2(dy, dx)
    herr = (desired_theta - theta + np.pi) % (2 * np.pi) - np.pi

    if abs(herr) > 0.25:          # large heading error — turn in place
        data.ctrl[A_FWD]  =  0.0
        data.ctrl[A_TURN] = -float(np.clip(3.0 * herr, -1.0, 1.0))
    else:                          # drive and steer
        data.ctrl[A_FWD]  = -float(np.clip(1.5 * dist, 0.0, 1.0))
        data.ctrl[A_TURN] = -float(np.clip(2.0 * herr, -1.0, 1.0))
    return False


def face(data, target_theta=0.0, threshold=0.04):
    """Rotate in place until heading matches target_theta. Returns True when done."""
    theta = robot_angle(data)
    herr  = (target_theta - theta + np.pi) % (2 * np.pi) - np.pi
    data.ctrl[A_FWD]  =  0.0
    data.ctrl[A_TURN] = -float(np.clip(3.0 * herr, -1.0, 1.0))
    return abs(herr) < threshold


def set_lift(model, data, target, tol=0.012):
    """Command lift to target (m). Returns True when within tol."""
    data.ctrl[A_LIFT] = float(np.clip(target, -0.5, 0.6))
    return abs(joint_val(model, data, "joint_lift") - target) < tol


def set_arm(model, data, target, tol=0.012):
    """Command total arm extension to target (m). Returns True when within tol."""
    data.ctrl[A_ARM] = float(np.clip(target, 0.0, 0.52))
    current = joint_val(model, data, "joint_arm_l0") * 4
    return abs(current - target) < tol


def apply_belt_force(model, data):
    """
    Push cubes resting on the conveyor belt in the +x direction.
    """
    for i in range(NUM_CUBES):
        cp = cube_pos(model, data, i)
        on_belt = (
            abs(cp[0] - CONV_X) < 0.62 and
            abs(cp[1] - CONV_Y) < 0.13 and   # belt half-width = 0.13 m
            abs(cp[2] - (CONV_TOP_Z + CUBE_HALF)) < 0.08
        )
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{i}")
        data.xfrc_applied[bid, 0] = BELT_FORCE if on_belt else 0.0


def set_cube_qpos(model, data, cube_idx, xyz, quat=(1, 0, 0, 0)):
    """Directly set a cube's position via its freejoint qpos."""
    bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{cube_idx}")
    jid  = model.body_jntadr[bid]
    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr    :qadr + 3] = xyz
    data.qpos[qadr + 3:qadr + 7] = quat
    # Zero velocity so the cube doesn't fly off
    dadr = model.jnt_dofadr[jid]
    data.qvel[dadr:dadr + 6] = 0.0


def cube_geom_id(model, cube_idx):
    """Return the geom id of the (single) geom on cube_<idx>."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{cube_idx}")
    return int(model.body_geomadr[bid])


def set_cube_collision(model, cube_idx, active: bool):
    """
    Enable (active=True) or disable (active=False) collision for a cube.
    While kinematically carried the cube is made a ghost (contype=0) so it
    does not create drag forces against the robot.  On release it is restored
    to contype=3/conaffinity=3 so it lands properly on the belt.
    """
    gid = cube_geom_id(model, cube_idx)
    if active:
        model.geom_contype[gid]     = 3
        model.geom_conaffinity[gid] = 3
    else:
        model.geom_contype[gid]     = 0
        model.geom_conaffinity[gid] = 0


# ── Scene initialisation ──────────────────────────────────────────────────────

def setup_scene(model, data):
    """
    Reset simulation, place robot at starting position,
    and scatter cubes randomly along the table (fixed y, random x).
    """
    mujoco.mj_resetData(model, data)

    # Robot: start left of table, at working y, facing +x.
    data.qpos[0] = TABLE_X - 1.2    # x — clear of table
    data.qpos[1] = ROBOT_WORK_Y
    data.qpos[2] = 0.0
    data.qpos[3] = 1.0              # quaternion w=1 → facing +x
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

    data.ctrl[A_LIFT]  = LIFT_HOME
    data.ctrl[A_ARM]   = ARM_HOME
    data.ctrl[A_GRIP]  = GRIP_OPEN
    data.ctrl[A_WRIST] = 0.0

    # Scatter cubes along table surface (random x, fixed y and z).
    # Spread across ±0.22 m so cubes stay well within table x half-size of 0.35 m.
    rng = np.random.default_rng(seed=42)
    xs = np.linspace(TABLE_X - 0.22, TABLE_X + 0.22, NUM_CUBES)
    xs += rng.uniform(-0.03, 0.03, NUM_CUBES)
    xs = np.clip(xs, TABLE_X - 0.28, TABLE_X + 0.28)

    for i in range(NUM_CUBES):
        set_cube_qpos(model, data, i, [float(xs[i]), TABLE_Y, CUBE_Z])

    mujoco.mj_forward(model, data)


# ── State machine ─────────────────────────────────────────────────────────────

class StateMachine:
    def __init__(self, model, data):
        self.model     = model
        self.data      = data
        self.state     = INIT
        self.cube_idx  = 0
        self.collected = set()
        self.timer     = 0
        self._prev     = -1
        # Kinematic grip state
        self.gripped_cube   = None   # index of cube being carried, or None
        self.grip_offset    = None   # offset from gripper body to cube centre (world)
        # Descent ghost: cube is frozen here during LOWER_PICK so the fingertips
        # can descend through it without physically knocking it away.
        self._pick_target_pos = None
        # Cubes that have slid off the far end of the belt and been hidden.
        self.disposed = set()

    def _transition(self, new_state):
        self.state = new_state
        self.timer = 0

    def _carry_cube(self):
        """If a cube is kinematically attached, update its position every step."""
        if self.gripped_cube is None:
            return
        gpos = gripper_pos(self.model, self.data)
        target_xyz = gpos + self.grip_offset
        set_cube_qpos(self.model, self.data, self.gripped_cube, target_xyz)

    def _check_belt_exit(self):
        """Sink cubes that have slid off the far end of the belt out of view.

        Only checks cubes that have already been delivered (self.collected) so
        we never accidentally vanish a cube that is still being carried or is
        still sitting on the table.
        """
        for i in self.collected:
            if i in self.disposed:
                continue
            cp = cube_pos(self.model, self.data, i)
            if cp[0] > BELT_END_X:
                # Move far below the floor and kill all velocity
                set_cube_qpos(self.model, self.data, i, [CONV_X, CONV_Y, -10.0])
                set_cube_collision(self.model, i, active=False)
                self.disposed.add(i)
                print(f"    ✓ cube_{i} exited belt — removed")

    def _try_snap(self):
        """Attach self.cube_idx to the gripper.

        Always targets the intended cube (self.cube_idx) so we never accidentally
        grab a neighbour.  grip_offset = rubber_tip_midpoint - slider, confirmed
        by a prior session to place the cube visually inside the fingers during
        carry.  set_cube_qpos teleports the cube to tip_mid — at snap time the
        fingertips are already at the cube (LOWER_PICK waits until tip_z ≈ cube_z
        and ARM_REACH_TABLE places tip_y ≈ cube_y), so the jump is ≤ a few mm.
        """
        gpos = gripper_pos(self.model, self.data)
        cp   = cube_pos(self.model, self.data, self.cube_idx)
        d    = float(np.linalg.norm(cp - gpos))
        if d >= SNAP_RADIUS:
            print(f"    ⚠ cube_{self.cube_idx} out of reach (d={d:.3f} m)")
            return False
        self.gripped_cube = self.cube_idx
        bid_l    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rubber_tip_left")
        bid_r    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rubber_tip_right")
        tip_mid  = (self.data.xpos[bid_l] + self.data.xpos[bid_r]) * 0.5
        self.grip_offset = tip_mid - gpos          # cube rides at fingertip midpoint
        set_cube_qpos(self.model, self.data, self.cube_idx, tip_mid)
        set_cube_collision(self.model, self.cube_idx, active=False)
        print(f"    → snapped cube_{self.cube_idx}  d={d:.3f} m  tip_z={tip_mid[2]:.3f}")
        return True

    def step(self):
        model, data = self.model, self.data

        apply_belt_force(model, data)
        self._check_belt_exit()
        self._carry_cube()

        if self.state != self._prev:
            label = ""
            if self.state not in (INIT, FIND_CUBE, DONE):
                label = f"  (cube {self.cube_idx})"
            print(f"  → {STATE_NAMES[self.state]}{label}")
            self._prev = self.state

        s = self.state

        # ── INIT ─────────────────────────────────────────────────────────────
        if s == INIT:
            data.ctrl[A_GRIP] = GRIP_OPEN
            lift_ok = set_lift(model, data, LIFT_HOME)
            arm_ok  = set_arm(model, data, ARM_HOME)
            if lift_ok and arm_ok:
                self._transition(FIND_CUBE)

        # ── FIND_CUBE ─────────────────────────────────────────────────────────
        elif s == FIND_CUBE:
            for i in range(NUM_CUBES):
                if i not in self.collected:
                    cp = cube_pos(model, data, i)
                    if cp[2] > 0.5:      # still on table (z > 0.5 m)
                        self.cube_idx = i
                        self._transition(NAV_TO_PICK)
                        return
            self._transition(DONE)

        # ── NAV_TO_PICK ───────────────────────────────────────────────────────
        elif s == NAV_TO_PICK:
            cpos   = cube_pos(model, data, self.cube_idx)
            target = np.array([cpos[0], ROBOT_WORK_Y])
            if navigate(data, target):
                self._transition(FACE_PICK)

        # ── FACE_PICK ─────────────────────────────────────────────────────────
        elif s == FACE_PICK:
            if face(data, target_theta=0.0):
                self._transition(OPEN_GRIP)

        # ── OPEN_GRIP ─────────────────────────────────────────────────────────
        elif s == OPEN_GRIP:
            data.ctrl[A_GRIP] = GRIP_OPEN
            self.timer += 1
            if self.timer >= 60:
                self._transition(EXTEND_PICK)

        # ── EXTEND_PICK — arm sweeps ABOVE cube (lift stays at HOME) ────────
        elif s == EXTEND_PICK:
            set_lift(model, data, LIFT_HOME)   # keep lift high while extending
            if set_arm(model, data, ARM_REACH_TABLE):
                self._transition(LOWER_PICK)   # descend to cube next

        # ── LOWER_PICK — ghost cube, descend, snap when tips reach cube height ───
        # Problem without ghosting: fingertips reach TABLE_Y (same y as cube) and
        # physically press on the cube top as the lift descends, pushing it upward
        # and preventing the snap condition from firing → first grab fails.
        # Fix: disable cube collision on entry and kinematically freeze it in place
        # so the gripper descends through the cube freely.  Snap fires cleanly when
        # tip_z reaches cube_z.  Collision is restored if snap somehow still fails.
        elif s == LOWER_PICK:
            # ── Entry: ghost the target cube once ─────────────────────────────
            if self._pick_target_pos is None:
                self._pick_target_pos = cube_pos(model, data, self.cube_idx).copy()
                set_cube_collision(model, self.cube_idx, active=False)
            # Hold cube frozen at table position (gripper descends through freely)
            set_cube_qpos(model, data, self.cube_idx, self._pick_target_pos)

            set_lift(model, data, 0.10)   # command well below cube; snap fires first
            self.timer += 1
            bid_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rubber_tip_left")
            bid_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rubber_tip_right")
            tip_z = (data.xpos[bid_l][2] + data.xpos[bid_r][2]) * 0.5
            cp    = self._pick_target_pos   # use frozen position, not physics pos
            if tip_z <= cp[2] + 0.015 or self.timer >= 800:
                if self.gripped_cube is None:
                    if not self._try_snap():
                        # Snap failed — restore cube so it can rest on table again
                        set_cube_collision(model, self.cube_idx, active=True)
                self._pick_target_pos = None   # reset for next pick
                self._transition(CLOSE_GRIP)

        # ── CLOSE_GRIP ────────────────────────────────────────────────────────
        elif s == CLOSE_GRIP:
            data.ctrl[A_GRIP] = GRIP_CLOSED
            set_lift(model, data, LIFT_HOME)   # start rising immediately — cancels the
                                               # 0.10 command from LOWER_PICK so the lift
                                               # doesn't continue past the cube
            self.timer += 1
            if self.timer >= GRIP_WAIT:
                self._transition(RAISE)

        # ── RAISE — lift cube above table surface ─────────────────────────────
        elif s == RAISE:
            if set_lift(model, data, LIFT_HOME):
                self._transition(RETRACT_PICK)

        # ── RETRACT_PICK — bring arm back (cube follows) ──────────────────────
        elif s == RETRACT_PICK:
            if set_arm(model, data, ARM_HOME):
                self._transition(NAV_TO_CONV)

        # ── NAV_TO_CONV ───────────────────────────────────────────────────────
        elif s == NAV_TO_CONV:
            target = np.array([CONV_X, ROBOT_WORK_Y])
            if navigate(data, target):
                self._transition(FACE_DROP)

        # ── FACE_DROP ─────────────────────────────────────────────────────────
        elif s == FACE_DROP:
            if face(data, target_theta=0.0):
                self._transition(EXTEND_DROP)

        # ── EXTEND_DROP — extend arm over conveyor ────────────────────────────
        elif s == EXTEND_DROP:
            set_lift(model, data, LIFT_HOME)
            if set_arm(model, data, ARM_REACH_CONV):
                self._transition(LOWER_DROP)

        # ── LOWER_DROP — lower to drop height (timeout safe) ─────────────────
        elif s == LOWER_DROP:
            done = set_lift(model, data, LIFT_DROP)
            self.timer += 1
            if done or self.timer >= 800:
                self._transition(RELEASE)

        # ── RELEASE — detach cube, open gripper ───────────────────────────────
        elif s == RELEASE:
            data.ctrl[A_GRIP] = GRIP_OPEN
            self.timer += 1
            if self.timer >= GRIP_WAIT:
                if self.gripped_cube is not None:
                    # Restore collision so cube lands properly on belt
                    set_cube_collision(self.model, self.gripped_cube, active=True)
                    self.collected.add(self.cube_idx)
                    self.gripped_cube = None
                    self.grip_offset  = None
                    print(f"    Cube {self.cube_idx} delivered "
                          f"({len(self.collected)}/{NUM_CUBES})")
                else:
                    # Snap never succeeded — don't mark as collected, retry later
                    print(f"    ⚠ No cube was gripped — will retry cube {self.cube_idx}")
                self._transition(RETRACT_DROP)

        # ── RETRACT_DROP ──────────────────────────────────────────────────────
        elif s == RETRACT_DROP:
            if set_arm(model, data, ARM_HOME):
                self._transition(HOME_LIFT)

        # ── HOME_LIFT ─────────────────────────────────────────────────────────
        elif s == HOME_LIFT:
            if set_lift(model, data, LIFT_HOME):
                self._transition(FIND_CUBE)

        # ── DONE ──────────────────────────────────────────────────────────────
        elif s == DONE:
            data.ctrl[A_FWD]  = 0.0
            data.ctrl[A_TURN] = 0.0
            if self._prev != DONE:
                print("\nAll cubes placed on conveyor. Simulation complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading model ...")
    model = mujoco.MjModel.from_xml_path("loader_scene_21582762.xml")
    data  = mujoco.MjData(model)

    print("Initialising scene ...")
    setup_scene(model, data)

    sm = StateMachine(model, data)

    print("Launching viewer  (close window to quit) ...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = -150.0
        viewer.cam.elevation =  -22.0
        viewer.cam.distance  =    4.5
        viewer.cam.lookat[:] = [0.8, -0.6, 0.4]

        while viewer.is_running():
            # Run multiple physics steps per viewer frame for faster navigation.
            for _ in range(SIM_STEPS_PER_FRAME):
                mujoco.mj_step(model, data)
                sm.step()

            viewer.sync()
            # Small yield so the OS can service the viewer window
            time.sleep(0.001)


if __name__ == "__main__":
    main()
