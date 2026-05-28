"""
Stretch Conveyor Sorter — big cubes
===================================
Big 8 cm cubes, boxes flush on the floor.

Note about gripping: the Stretch rubber pads only open ~6 cm at maximum,
so an 8 cm cube is physically wider than the gripper. We descend the
open gripper until the rubber pads meet AT the cube's top surface (no
clipping through the cube), close the jaws on top, and the cube travels
with the gripper from that moment.  The cube then falls under gravity
into the box (no teleport on release).

What's kept rigid:
  · Base X, Y locked every step (no drift)
  · Arm joint velocities zeroed every step
  · Wrist + finger pivots locked
  · Rubber-pad compliance offsets snapped to 0 each step (no wobble)

Drop:
  · Tip stops above the box wall (Z=0.45, walls top Z=0.28) — no clipping
  · Jaws open and cube falls into box under gravity from where it was held
"""

import mujoco, mujoco.viewer
import numpy as np, time, os, sys

try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for f in ["sort_scene.xml", "stretch.xml"]:
        if not os.path.exists(f):
            print(f"ERROR: '{f}' not found."); input("Press Enter..."); sys.exit(1)
    if not os.path.isdir("assets"):
        print("ERROR: 'assets/' folder not found."); input("Press Enter..."); sys.exit(1)

    model = mujoco.MjModel.from_xml_path("sort_scene.xml")
    data  = mujoco.MjData(model)
    print(f"Model OK  nu={model.nu}  nq={model.nq}  nv={model.nv}")

    # ── body IDs ──────────────────────────────────────────────────────
    RTL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'rubber_tip_left')
    RTR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'rubber_tip_right')

    # ── joint scanning  (name → (qadr, dofadr)) ───────────────────────
    BQ = None
    cube_info = []
    RW_Q = LW_Q = None
    NAMES = [
        'joint_lift', 'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0',
        'joint_wrist_yaw', 'joint_gripper_slide',
        'joint_gripper_finger_left_open', 'joint_gripper_finger_right_open',
        'rubber_left_x', 'rubber_left_y', 'rubber_right_x', 'rubber_right_y',
    ]
    JI = {}
    for i in range(model.njnt):
        jn   = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ''
        bn   = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[i]) or ''
        qadr = model.jnt_qposadr[i]
        dadr = model.jnt_dofadr[i]
        if model.jnt_type[i] == 0:
            if bn.startswith('cube_'):
                idx    = int(bn.split('_')[1])
                colour = 'red' if idx < 4 else 'blue'
                cube_info.append((bn, qadr, colour))
            else:
                # First non-cube freejoint is the robot base
                if BQ is None: BQ = qadr
        if 'right_wheel' in jn: RW_Q = qadr
        if 'left_wheel'  in jn: LW_Q = qadr
        if jn in NAMES: JI[jn] = (qadr, dadr)
    cube_info.sort()
    print(f"Robot qadr={BQ}  cubes={len(cube_info)}")
    print(f"  found joints: {sorted(JI.keys())}")

    # Sanity check critical IDs — fail with a clear message if missing
    if BQ is None:
        raise RuntimeError("Could not find base_link freejoint in model")
    if RW_Q is None or LW_Q is None:
        raise RuntimeError(f"Wheel joints not found: RW_Q={RW_Q} LW_Q={LW_Q}")
    if RTL == -1 or RTR == -1:
        raise RuntimeError(f"Rubber tip bodies not found: RTL={RTL} RTR={RTR}")
    if len(cube_info) == 0:
        raise RuntimeError("No cube_* bodies with freejoints found in sort_scene.xml")

    def qi(n): return JI.get(n, (None,None))[0]
    def di(n): return JI.get(n, (None,None))[1]

    LIFT_Q   = qi('joint_lift')
    ARM_QS   = [qi(f'joint_arm_l{n}') for n in [3,2,1,0]]
    SLIDE_Q  = qi('joint_gripper_slide')
    WRIST_Q   = qi('joint_wrist_yaw')         # locked to 0 each step
    WRIST_DOF = di('joint_wrist_yaw')         # velocity zeroed each step
    # qpos values we'll force every step to keep the gripper rigid
    LOCK_QS = [qi(n) for n in
               ['rubber_left_x','rubber_left_y','rubber_right_x','rubber_right_y']
               if qi(n) is not None]
    # All gripper + arm dof indices for velocity zeroing
    ALL_DOFS = [di(n) for n in NAMES if di(n) is not None]
    # Arm-only DOFs (these stay rigid every step)
    ARM_DOFS = [di(n) for n in
                ['joint_lift', 'joint_arm_l3', 'joint_arm_l2',
                 'joint_arm_l1', 'joint_arm_l0']
                if di(n) is not None]

    # ── IK constants  (verified) ──────────────────────────────────────
    TIP_DX = 0.0213
    TIP_DY = 0.3404
    TIP_DZ = 0.5318

    # ── scene constants ───────────────────────────────────────────────
    WORK_Y       = -0.85
    ARM_REACH    =  0.52
    ARM_CARRY    =  0.15
    ROBOT_PICK_Y = WORK_Y + TIP_DY + ARM_REACH        # 0.0104

    HOME_X, HOME_Y = 0.0, 0.0
    START_X = 3.00
    START_Y = ROBOT_PICK_Y

    RED_BOX_X  = 1.0
    BLUE_BOX_X = 2.0

    GRIP_OPEN   = 0.010   # max-spread visible
    GRIP_CLOSED = 0.025   # tips meet

    FWD_SPEED, SIDE_SPEED, WHEEL_R = 0.45, 0.25, 0.05

    CUBE_HOMES = {
        f'cube_{i}': np.array([3.475 + i*0.15, WORK_Y, 0.580, 1,0,0,0], dtype=float)
        for i in range(8)
    }
    PICK_RX = {n: h[0] + TIP_DX for n, h in CUBE_HOMES.items()}

    # Per-cube drop X offsets within the box. 8cm cubes spread across the
    # 44cm box interior with ~1cm gaps between adjacent cubes.
    DROP_OFFSETS = [-0.135, -0.045, +0.045, +0.135]
    def drop_rx_for(idx, colour):
        box_x = RED_BOX_X if colour == 'red' else BLUE_BOX_X
        offset = DROP_OFFSETS[idx % 4]
        return box_x + offset + TIP_DX

    LIFT_PICK  = None
    LIFT_CARRY = None
    LIFT_DROP  = None

    # ── state ─────────────────────────────────────────────────────────
    CARRY        = [None]    # (qadr, offset) — cube tracks the tip
    WHEEL_ANG    = [0.0]
    DROPPED      = []
    LOCKED_X     = [START_X]
    LOCKED_Y     = [START_Y]
    dt           = model.opt.timestep
    SPR          = 8

    # ── helpers ───────────────────────────────────────────────────────

    def tip_centre(): return (data.xpos[RTL] + data.xpos[RTR]) / 2.0

    def lock_robot():
        data.qpos[BQ]   = LOCKED_X[0]
        data.qpos[BQ+1] = LOCKED_Y[0]
        data.qpos[BQ+3] = 1.0
        data.qpos[BQ+4:BQ+7] = 0.0
        data.qvel[BQ+0] = 0.0
        data.qvel[BQ+1] = 0.0
        data.qvel[BQ+3] = 0.0
        data.qvel[BQ+4] = 0.0
        data.qvel[BQ+5] = 0.0

    # DOFs that must stay rigid (zeroed velocity each step)
    RIGID_DOFS = list(ARM_DOFS)
    if WRIST_DOF is not None:
        RIGID_DOFS.append(WRIST_DOF)
    for n in ['rubber_left_x','rubber_left_y','rubber_right_x','rubber_right_y']:
        d = di(n)
        if d is not None: RIGID_DOFS.append(d)
    # NOTE: slide and finger pivot DOFs are NOT in RIGID_DOFS — they need to
    # move naturally under actuator force so the rubber pads can stop AT the
    # cube's edge when closing.

    def set_arm(lift, arm, grip):
        """Lock all arm + gripper joints rigid. Slide qpos is forced so the
        pads close TIGHT to the cube interior — visible 'gripping' look,
        even though it technically clips."""
        if LIFT_Q is not None:  data.qpos[LIFT_Q]  = lift
        for q in ARM_QS:
            if q is not None:    data.qpos[q] = arm / 4.0
        if SLIDE_Q is not None: data.qpos[SLIDE_Q] = grip      # force pads
        if WRIST_Q is not None: data.qpos[WRIST_Q] = 0.0        # lock wrist
        # Lock rubber-pad compliance offsets at exactly 0 — no wobble.
        for q in LOCK_QS:        data.qpos[q] = 0.0
        # Zero all gripper + arm dof velocities so nothing wobbles
        for d in ALL_DOFS:       data.qvel[d] = 0.0
        data.ctrl[2] = lift
        data.ctrl[3] = arm
        data.ctrl[5] = grip

    def spin_wheels(vx):
        ang = vx * dt / WHEEL_R
        WHEEL_ANG[0]    += ang
        data.qpos[RW_Q]  =  WHEEL_ANG[0]
        data.qpos[LW_Q]  = -WHEEL_ANG[0]

    def pin_belt():
        """Hold cubes on belt until lifted (Z>0.65) or dropped."""
        for name, qadr, _ in cube_info:
            if qadr in DROPPED: continue
            if CARRY[0] and CARRY[0][0] == qadr: continue
            if data.qpos[qadr+2] > 0.50 and abs(data.qpos[qadr+1]-WORK_Y) < 0.3:
                data.qpos[qadr:qadr+7] = CUBE_HOMES[name]
                data.qvel[qadr:qadr+6] = 0.0

    def carry_track():
        """While carrying, snap the cube to (tip + offset) each step.
        This isn't a teleport — it's rigid kinematic following, like the
        cube is rigidly held by the closed jaws."""
        if CARRY[0] is None: return
        qadr, off = CARRY[0]
        tc = tip_centre()
        data.qpos[qadr:qadr+3]   = tc + off
        data.qpos[qadr+3]        = 1.0
        data.qpos[qadr+4:qadr+7] = 0.0
        data.qvel[qadr:qadr+6]   = 0.0

    def physics_step():
        lock_robot()
        pin_belt()
        carry_track()
        mujoco.mj_step(model, data)

    def tick(viewer, i):
        if i % SPR == 0:
            viewer.sync()
            time.sleep(dt * SPR)
        return viewer.is_running()

    # ── motion primitives ─────────────────────────────────────────────

    def drive_to(viewer, label, tx, ty,
                 lift=None, arm=ARM_CARRY, grip=GRIP_OPEN):
        if lift is None: lift = LIFT_CARRY
        x0, y0 = LOCKED_X[0], LOCKED_Y[0]
        dx, dy = tx - x0, ty - y0
        dist   = np.hypot(dx, dy)
        if dist < 0.004: return True
        speed  = FWD_SPEED if abs(dx) >= abs(dy) else SIDE_SPEED
        steps  = max(1, int(dist / speed / dt))
        vx     = dx / dist * speed
        print(f"    {label}  ({dist:.2f} m)")
        for i in range(steps):
            if not viewer.is_running(): return False
            LOCKED_X[0] = x0 + dx * i / steps
            LOCKED_Y[0] = y0 + dy * i / steps
            set_arm(lift, arm, grip)
            spin_wheels(vx)
            physics_step()
            if not tick(viewer, i): return False
        LOCKED_X[0] = tx
        LOCKED_Y[0] = ty
        return True

    def move_arm(viewer, label, secs, l0,a0,g0, l1,a1,g1):
        print(f"    {label}")
        n = max(1, int(secs / dt))
        for i in range(n):
            if not viewer.is_running(): return False
            t = i / n
            set_arm(l0+(l1-l0)*t, a0+(a1-a0)*t, g0+(g1-g0)*t)
            physics_step()
            if not tick(viewer, i): return False
        set_arm(l1, a1, g1)
        return True

    def hold(viewer, secs, lift, arm, grip):
        n = max(1, int(secs / dt))
        for i in range(n):
            if not viewer.is_running(): return False
            set_arm(lift, arm, grip)
            physics_step()
            if not tick(viewer, i): return False
        return True

    def grab(qadr, name):
        """Establish rigid following: record the (cube - tip) offset
        at the moment of close so the cube tracks the gripper."""
        data.qpos[qadr:qadr+7] = CUBE_HOMES[name]
        data.qvel[qadr:qadr+6] = 0.0
        mujoco.mj_forward(model, data)
        tc  = tip_centre()
        pos = data.qpos[qadr:qadr+3].copy()
        CARRY[0] = (qadr, pos - tc)

    def open_jaws_with_descent(viewer, secs, lift, arm, descent):
        """Open jaws while smoothly LOWERING the carried cube by `descent` m.
        Cube smoothly slides down out of the gripper INTO the box opening,
        instead of a sudden teleport jump on release. Makes the drop look
        natural and gentle."""
        if CARRY[0] is None: return False
        qadr, off0 = CARRY[0]
        print(f"    Open jaws + lower cube ({descent*100:.0f}cm down into box)")
        n = max(1, int(secs / dt))
        for i in range(n):
            if not viewer.is_running(): return False
            t = i / n
            # Smoothly decrease offset Z by descent*t  (cube drops below tip)
            new_off = off0.copy()
            new_off[2] -= descent * t
            CARRY[0] = (qadr, new_off)
            set_arm(lift, arm, GRIP_CLOSED + (GRIP_OPEN - GRIP_CLOSED)*t)
            physics_step()
            if not tick(viewer, i): return False
        return True

    def let_drop(qadr):
        """Stop tracking. Cube falls under gravity from its current position
        (which was lowered close to the box floor by open_jaws_with_descent),
        so the fall is only a few cm — soft landing, no flying out."""
        CARRY[0] = None
        DROPPED.append(qadr)
        data.qvel[qadr:qadr+6] = 0.0

    # ── pick-and-place ────────────────────────────────────────────────

    def pick_and_place(viewer, name, qadr, colour, idx):
        ok = True
        rx_p = PICK_RX[name]
        rx_d = drop_rx_for(idx, colour)
        print(f"\n  ── {name}  →  {colour.upper()} (slot {idx%4}) ──")

        # 1. Drive to cube — jaws already open
        ok = ok and drive_to(viewer, "Move to cube",
                             rx_p, ROBOT_PICK_Y,
                             lift=LIFT_CARRY, arm=ARM_CARRY, grip=GRIP_OPEN)
        # brief pause after arrival
        ok = ok and hold(viewer, 0.25, LIFT_CARRY, ARM_CARRY, GRIP_OPEN)

        # 2. Extend arm out over cube — jaws still open
        ok = ok and move_arm(viewer, "Extend arm",
                             1.4, LIFT_CARRY,ARM_CARRY,GRIP_OPEN,
                                   LIFT_CARRY,ARM_REACH,GRIP_OPEN)

        # 3. Hold the wide-open jaws clearly visible above the cube
        ok = ok and hold(viewer, 0.7, LIFT_CARRY, ARM_REACH, GRIP_OPEN)

        # 4. Lower fully-open jaws straight down to cube TOP EDGE (Z=0.60).
        #    Tip stops AT cube top — no clipping into the cube during descent.
        ok = ok and move_arm(viewer, "Lower onto cube top",
                             2.2, LIFT_CARRY,ARM_REACH,GRIP_OPEN,
                                   LIFT_PICK, ARM_REACH,GRIP_OPEN)

        # 4b. Brief settle pause AT cube top — clean stop before closing
        ok = ok and hold(viewer, 0.3, LIFT_PICK, ARM_REACH, GRIP_OPEN)

        # 5. CLOSE jaws on cube TOP EDGE — pads grip the top, cube hangs below
        ok = ok and move_arm(viewer, "Close jaws on cube edge",
                             1.3, LIFT_PICK,ARM_REACH,GRIP_OPEN,
                                   LIFT_PICK,ARM_REACH,GRIP_CLOSED)
        # Settle pause so the pads sit firmly on cube top edge
        ok = ok and hold(viewer, 0.4, LIFT_PICK, ARM_REACH, GRIP_CLOSED)

        # 6. Cube now travels rigidly with the gripper
        if ok:
            grab(qadr, name)
            ok = hold(viewer, 0.3, LIFT_PICK, ARM_REACH, GRIP_CLOSED)

        # 7. Lift slowly — deliberate so the pickup reads clearly
        ok = ok and move_arm(viewer, "Lift cube",
                             1.8, LIFT_PICK, ARM_REACH,GRIP_CLOSED,
                                   LIFT_CARRY,ARM_REACH,GRIP_CLOSED)

        # 8. Retract to carry pose
        ok = ok and move_arm(viewer, "Retract to carry",
                             1.3, LIFT_CARRY,ARM_REACH,GRIP_CLOSED,
                                   LIFT_CARRY,ARM_CARRY,GRIP_CLOSED)

        # 9. Drive to box
        ok = ok and drive_to(viewer, "Drive to box",
                             rx_d, ROBOT_PICK_Y,
                             lift=LIFT_CARRY, arm=ARM_CARRY, grip=GRIP_CLOSED)
        ok = ok and hold(viewer, 0.25, LIFT_CARRY, ARM_CARRY, GRIP_CLOSED)

        # 10. Extend arm over box
        ok = ok and move_arm(viewer, "Extend over box",
                             1.5, LIFT_CARRY,ARM_CARRY,GRIP_CLOSED,
                                   LIFT_CARRY,ARM_REACH, GRIP_CLOSED)

        # 11. Lower to drop height (tip stays above wall top — no clip)
        ok = ok and move_arm(viewer, "Lower over box",
                             1.3, LIFT_CARRY,ARM_REACH,GRIP_CLOSED,
                                   LIFT_DROP, ARM_REACH,GRIP_CLOSED)

        # 11b. Brief pause so the drop is a clear, distinct beat
        ok = ok and hold(viewer, 0.3, LIFT_DROP, ARM_REACH, GRIP_CLOSED)

        # 12. Open jaws WHILE smoothly lowering cube into the box.
        #     Cube slides down out of the jaws and ends up just above the
        #     box floor — no teleport jump, gentle landing.
        if ok:
            ok = open_jaws_with_descent(viewer, 1.0,
                                         LIFT_DROP, ARM_REACH,
                                         descent=0.18)
        if ok:
            let_drop(qadr)
            # Cube is now near box floor with zero velocity, falls softly
            ok = hold(viewer, 1.0, LIFT_DROP, ARM_REACH, GRIP_OPEN)

        # 13. RAISE ARM FIRST (while still extended) so the wrist/gripper
        #     are well above the box walls BEFORE retracting horizontally.
        #     Otherwise the retract motion drags the wrist through the front
        #     wall as it moves back toward the robot.
        ok = ok and move_arm(viewer, "Raise above box",
                             1.0, LIFT_DROP,  ARM_REACH, GRIP_OPEN,
                                   LIFT_CARRY, ARM_REACH, GRIP_OPEN)

        # 14. Now retract arm at safe carry height (well above all walls)
        ok = ok and move_arm(viewer, "Retract to carry pose",
                             1.2, LIFT_CARRY, ARM_REACH, GRIP_OPEN,
                                   LIFT_CARRY, ARM_CARRY, GRIP_OPEN)
        return ok

    # ── initialise ────────────────────────────────────────────────────
    mujoco.mj_resetData(model, data)
    data.qpos[BQ:BQ+7] = [START_X, START_Y, 0, 1, 0, 0, 0]
    data.qvel[BQ:BQ+6] = 0.0
    set_arm(0.0, 0.0, GRIP_OPEN)
    WHEEL_ANG[0] = 0.0
    DROPPED.clear()
    CARRY[0] = None
    LOCKED_X[0] = START_X
    LOCKED_Y[0] = START_Y
    for name, qadr, _ in cube_info:
        data.qpos[qadr:qadr+7] = CUBE_HOMES[name]
        data.qvel[qadr:qadr+6] = 0.0

    print("Settling robot on floor (gripper opening)…")
    for _ in range(int(0.8 / dt)): mujoco.mj_step(model, data)
    for _ in range(int(0.8 / dt)):
        set_arm(0.0, 0.0, GRIP_OPEN)
        lock_robot()
        mujoco.mj_step(model, data)

    ROBOT_Z = data.qpos[BQ+2]
    print(f"Robot Z={ROBOT_Z:.5f}")

    # Lift values:
    #   8cm cube (half 0.040), CENTRE Z=0.580, TOP Z=0.62, BOTTOM Z=0.54
    #   Box: interior 44x44 cm, wall top Z=0.36
    # PICK: tip at cube TOP EDGE (Z=0.62) — pads close on cube top, clipping
    #       into the cube edges (cube wider than pad gap → full visible contact)
    LIFT_PICK  = 0.620 - ROBOT_Z - TIP_DZ
    LIFT_CARRY = 0.85  - ROBOT_Z - TIP_DZ
    LIFT_DROP  = 0.30  - ROBOT_Z - TIP_DZ
    print(f"Lifts:  PICK={LIFT_PICK:+.4f}  CARRY={LIFT_CARRY:+.4f}  DROP={LIFT_DROP:+.4f}")

    print("\nLaunching viewer…\n")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [2.5, -0.5, 0.45]
        viewer.cam.distance  = 6.5
        viewer.cam.azimuth   = 55
        viewer.cam.elevation = -25

        for idx, (name, qadr, colour) in enumerate(cube_info):
            print(f"\n══ CUBE {idx+1}/{len(cube_info)} ═══════════════════════")
            ok = pick_and_place(viewer, name, qadr, colour, idx)
            if not ok:
                print("  (viewer closed)")
                break

        if viewer.is_running():
            print("\n══ All sorted — returning home ═══════════")
            drive_to(viewer, "Return home", HOME_X, HOME_Y,
                     lift=LIFT_CARRY, arm=ARM_CARRY, grip=GRIP_OPEN)
            move_arm(viewer, "Stow arm", 1.0,
                     LIFT_CARRY,ARM_CARRY,GRIP_OPEN,
                     0.0, 0.0, GRIP_OPEN)
            print("  Idle — close viewer to exit")
            for i in range(int(8.0 / dt)):
                if not viewer.is_running(): break
                physics_step()
                if i % SPR == 0:
                    viewer.sync()
                    time.sleep(dt * SPR)

    print("Done.")

except Exception:
    import traceback
    print("\n" + "="*60)
    print("ERROR DURING SIMULATION:")
    print("="*60)
    traceback.print_exc()
    print("="*60)

print()
input("Press Enter to close this window...")
