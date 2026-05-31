"""
---------------------------------------------------------------------------
Student name: Stephanie Chmielewski
Student ID: 21411598

Model used: Universal Robots UR5e (universal_robots_ur5e)

HOW TO RUN:
   1. Install dependencies in CMD:
       --> pip install mujoco numpy
   2. Clone MuJoCo Menagerie into the same folder as this script:
        git clone https://github.com/google-deepmind/mujoco_menagerie
   3. Run:
        python object_sorting_21411598.py
   4.  MuJoCo viewer window opens.
-----------------------------------------------------------------------------
"""
import mujoco
import mujoco.viewer
import numpy as np

UR5E_XML = "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
with open(UR5E_XML) as f:
    xml = f.read()

xml = xml.replace('meshdir="assets"',
                  'meshdir="mujoco_menagerie/universal_robots_ur5e/assets"')

# Wrap arm:
xml = xml.replace("<worldbody>",
    "<worldbody>\n  <body name='arm_mount' pos='0 0 0' euler='0 0 180'>", 1)
xml = xml.replace("</worldbody>", "  </body>\n</worldbody>", 1)
xml = xml.replace("<worldbody>", """
  <visual>
    <headlight diffuse="0.78 0.78 0.78" ambient="0.42 0.42 0.42"/>
    <rgba haze="0.80 0.80 0.86 1"/>
  </visual>
  <worldbody>""", 1)

scene = r"""
    <geom name="floor" type="plane" size="8 8 0.1"
          rgba="0.55 0.57 0.60 1" contype="1" conaffinity="1"/>

    <!-- conveyor slab -->
    <geom type="box" pos="0.72 0 0.07"  size="0.28 2.20 0.07"  rgba="0.20 0.20 0.22 1"/>
    <geom type="box" pos="0.72 0 0.155" size="0.28 2.20 0.020" rgba="0.16 0.16 0.18 1"/>
    <!-- side rails -->
    <geom type="box" pos="0.45 0 0.125" size="0.012 2.20 0.055" rgba="0.32 0.32 0.35 1"/>
    <geom type="box" pos="0.99 0 0.125" size="0.012 2.20 0.055" rgba="0.32 0.32 0.35 1"/>
    <!-- end rollers -->
    <geom type="cylinder" pos="0.72  2.20 0.10" size="0.28 0.04" euler="0 90 0" rgba="0.40 0.40 0.42 1"/>
    <geom type="cylinder" pos="0.72 -2.20 0.10" size="0.28 0.04" euler="0 90 0" rgba="0.40 0.40 0.42 1"/>

    <!-- scrolling yellow stripes — 6 evenly spaced -->
    <body name="s1" mocap="true" pos="0.72  1.50 0.178">
      <geom type="box" size="0.26 0.022 0.005" rgba="0.95 0.80 0.0 1" contype="0" conaffinity="0"/>
    </body>
    <body name="s2" mocap="true" pos="0.72  0.80 0.178">
      <geom type="box" size="0.26 0.022 0.005" rgba="0.95 0.80 0.0 1" contype="0" conaffinity="0"/>
    </body>
    <body name="s3" mocap="true" pos="0.72  0.10 0.178">
      <geom type="box" size="0.26 0.022 0.005" rgba="0.95 0.80 0.0 1" contype="0" conaffinity="0"/>
    </body>
    <body name="s4" mocap="true" pos="0.72 -0.60 0.178">
      <geom type="box" size="0.26 0.022 0.005" rgba="0.95 0.80 0.0 1" contype="0" conaffinity="0"/>
    </body>
    <body name="s5" mocap="true" pos="0.72 -1.30 0.178">
      <geom type="box" size="0.26 0.022 0.005" rgba="0.95 0.80 0.0 1" contype="0" conaffinity="0"/>
    </body>
    <body name="s6" mocap="true" pos="0.72 -2.00 0.178">
      <geom type="box" size="0.26 0.022 0.005" rgba="0.95 0.80 0.0 1" contype="0" conaffinity="0"/>
    </body>

    <!-- 1 red + 1 blue cube, well spaced on belt -->
    <body name="red_cube"   mocap="true" pos="0.72  0.55 0.242">
      <geom type="box" size="0.068 0.068 0.068" rgba="0.95 0.08 0.08 1"/>
    </body>
    <body name="blue_cube"  mocap="true" pos="0.72  1.40 0.242">
      <geom type="box" size="0.068 0.068 0.068" rgba="0.08 0.25 0.95 1"/>
    </body>

    <!-- RED BIN — left of arm, clearly off belt -->
    <geom type="box" pos="-0.10  0.62 0.07"  size="0.24 0.24 0.07"  rgba="0.92 0.06 0.06 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.10  0.86 0.18"  size="0.24 0.025 0.11" rgba="0.92 0.06 0.06 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.10  0.38 0.18"  size="0.24 0.025 0.11" rgba="0.92 0.06 0.06 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.34  0.62 0.18"  size="0.025 0.24 0.11" rgba="0.92 0.06 0.06 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.14   0.62 0.18"  size="0.025 0.24 0.11" rgba="0.92 0.06 0.06 1" contype="0" conaffinity="0"/>

    <!-- BLUE BIN — right of arm, clearly off belt -->
    <geom type="box" pos="-0.10 -0.62 0.07"  size="0.24 0.24 0.07"  rgba="0.08 0.22 0.92 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.10 -0.38 0.18"  size="0.24 0.025 0.11" rgba="0.08 0.22 0.92 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.10 -0.86 0.18"  size="0.24 0.025 0.11" rgba="0.08 0.22 0.92 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.34 -0.62 0.18"  size="0.025 0.24 0.11" rgba="0.08 0.22 0.92 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="0.14  -0.62 0.18"  size="0.025 0.24 0.11" rgba="0.08 0.22 0.92 1" contype="0" conaffinity="0"/>
"""
xml = xml.replace("</worldbody>", scene + "\n  </worldbody>")

model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)


hand_id = None
for name in ["wrist_3_link", "flange", "tool0", "ee_link"]:
    try:
        hand_id = model.body(name).id
        print(f"End effector: '{name}'")
        break
    except: continue
if hand_id is None:
    hand_id = model.nbody - 1

n_ctrl = 6
cube_names   = ["red_cube", "blue_cube"]
cube_colours = ["red", "blue"]
cube_ids     = [model.body(n).mocapid[0] for n in cube_names]
stripe_ids   = [model.body(n).mocapid[0] for n in ["s1","s2","s3","s4","s5","s6"]]
BELT = 0.00028          
YMIN, YMAX = -2.18, 2.18  

# ── Jacobian IK — computes exact joint angles at startup ──────────────────────
def compute_ik(target_xyz, q_seed, n_iter=4000, alpha=0.15):
    """Find ctrl (joint positions) that puts hand at target_xyz."""
    d = mujoco.MjData(model)
    q = q_seed.copy()

    for step in range(n_iter):
        
        for i in range(n_ctrl):
            jnt = model.actuator_trnid[i, 0]
            adr = model.jnt_qposadr[jnt]
            d.qpos[adr] = q[i]
        d.qvel[:] = 0
        mujoco.mj_forward(model, d)

        hand = d.xpos[hand_id].copy()
        err  = target_xyz - hand
        dist = np.linalg.norm(err)
        if dist < 0.008:
            break

        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, d, jacp, None, hand, hand_id)

        # Map nv columns → n_ctrl joints
        J = np.zeros((3, n_ctrl))
        for i in range(n_ctrl):
            jnt = model.actuator_trnid[i, 0]
            dof = model.jnt_dofadr[jnt]
            J[:, i] = jacp[:, dof]

        dq = alpha * J.T @ (err / dist)
        dq = np.clip(dq, -0.08, 0.08)
        q += dq

        # Joint limits
        for i in range(n_ctrl):
            jnt = model.actuator_trnid[i, 0]
            lo, hi = model.jnt_range[jnt]
            if lo < hi:
                q[i] = np.clip(q[i], lo, hi)

    return q, dist

# ── Compute all poses using IK ─────────────────────────────────────────────────
SEED = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])

print("Computing IK poses (takes ~5 seconds)...")

# Positions: arm base at (0,0,0), facing +x, conveyor at x=0.72
PICK_ABOVE = np.array([0.72,  0.00, 0.35])
PICK_DOWN  = np.array([0.72,  0.00, 0.242])

# cube lock offset = -0.10 in z
# bin floor top = z 0.14, cube half = 0.068
# cube center inside bin = 0.14 + 0.068 = 0.208
RED_ABOVE_POS  = np.array([-0.10,  0.62, 0.45])  # well above bin
RED_DROP_POS   = np.array([-0.10,  0.62, 0.31])  # hand here → cube at z=0.21 inside bin
BLUE_ABOVE_POS = np.array([-0.10, -0.62, 0.45])
BLUE_DROP_POS  = np.array([-0.10, -0.62, 0.31])

HOME_Q, _      = compute_ik(PICK_ABOVE, SEED)
print(f"  HOME done")
ABOVE_Q, d1   = compute_ik(PICK_ABOVE, HOME_Q.copy())
print(f"  ABOVE done  dist={d1:.4f}")
DOWN_Q,  d2   = compute_ik(PICK_DOWN,  ABOVE_Q.copy())
print(f"  DOWN done   dist={d2:.4f}")
# LIFT = same as ABOVE — arm lifts only to belt-clear height, then rotates directly
LIFT_Q         = ABOVE_Q.copy()
print(f"  LIFT = ABOVE (low lift)")

RED_OVER_Q, d3 = compute_ik(RED_ABOVE_POS,  ABOVE_Q.copy())
print(f"  RED_OVER    dist={d3:.4f}")
RED_DOWN_Q, d4 = compute_ik(RED_DROP_POS,   RED_OVER_Q.copy())
print(f"  RED_DOWN    dist={d4:.4f}")

BLU_OVER_Q, d5 = compute_ik(BLUE_ABOVE_POS, ABOVE_Q.copy())
print(f"  BLU_OVER    dist={d5:.4f}")
BLU_DOWN_Q, d6 = compute_ik(BLUE_DROP_POS,  BLU_OVER_Q.copy())
print(f"  BLU_DOWN    dist={d6:.4f}")

print("IK complete — starting simulation.")

# Helpers
def go(target, speed=0.003):
    data.ctrl[:n_ctrl] += (target - data.ctrl[:n_ctrl]) * speed

def here(target, tol=0.04):
    return np.allclose(data.ctrl[:n_ctrl], target, atol=tol)

def lock(cid):
    h = data.xpos[hand_id].copy()
    data.mocap_pos[cid] = h + np.array([0.0, 0.0, -0.10])

# State 
phase = "home"
active_cube = active_index = None
holding = False
placed  = set()
red_n = blue_n = 0
grip_timer = 0

data.ctrl[:n_ctrl] = HOME_Q.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # recycle cubes — reset when both placed
        if len(placed) == len(cube_ids):
            placed.clear()
            data.mocap_pos[cube_ids[0]] = np.array([0.72,  0.55, 0.242])
            data.mocap_pos[cube_ids[1]] = np.array([0.72,  1.40, 0.242])

        # scroll stripes
        for sid in stripe_ids:
            p = data.mocap_pos[sid].copy()
            p[1] -= BELT
            if p[1] < YMIN: p[1] = YMAX
            data.mocap_pos[sid] = p

        # belt — move cubes
        for i, cid in enumerate(cube_ids):
            if cid != active_cube and cid not in placed:
                p = data.mocap_pos[cid].copy()
                p[1] -= BELT
                if p[1] < YMIN: p[1] = YMAX
                data.mocap_pos[cid] = p

        if holding and active_cube is not None:
            lock(active_cube)

        # state machine 
        if phase == "home":
            go(HOME_Q)
            if here(HOME_Q, tol=0.08):
                for i, cid in enumerate(cube_ids):
                    if cid in placed: continue
                    cy = data.mocap_pos[cid][1]
                    if -0.10 < cy < 0.13:
                        active_cube, active_index = cid, i
                        holding = False
                        phase = "above"
                        break

        elif phase == "above":
            go(ABOVE_Q, speed=0.005)
            if here(ABOVE_Q): phase = "down"

        elif phase == "down":
            go(DOWN_Q, speed=0.002)
            if here(DOWN_Q, tol=0.04):
                grip_timer = 0
                phase = "grip"

        elif phase == "grip":
            go(DOWN_Q, speed=0.001)
            grip_timer += 1
            if grip_timer > 60:
                holding = True
                lock(active_cube)
                phase = "lift"

        elif phase == "lift":
            go(LIFT_Q, speed=0.004)
            if holding: lock(active_cube)
            if here(LIFT_Q): phase = "over"

        elif phase == "over":
            colour = cube_colours[active_index]
            target = RED_OVER_Q if colour == "red" else BLU_OVER_Q
            go(target, speed=0.006)
            if holding: lock(active_cube)
            if here(target): phase = "drop_down"

        elif phase == "drop_down":
            colour = cube_colours[active_index]
            target = RED_DOWN_Q if colour == "red" else BLU_DOWN_Q
            go(target, speed=0.004)
            if holding: lock(active_cube)
            if here(target, tol=0.04): phase = "release"

        elif phase == "release":
            colour = cube_colours[active_index]
            if colour == "red":
                data.mocap_pos[active_cube] = np.array([-0.10,  0.62, 0.21])
                red_n += 1
            else:
                data.mocap_pos[active_cube] = np.array([-0.10, -0.62, 0.21])
                blue_n += 1
            placed.add(active_cube)
            holding = False
            active_cube = active_index = None
            phase = "home"

        hand = data.xpos[hand_id]
        print(f"phase={phase:10s} "
              f"hand=({hand[0]:+.3f},{hand[1]:+.3f},{hand[2]:+.3f}) "
              f"R={red_n} B={blue_n}", end="\r")

        mujoco.mj_step(model, data)
        viewer.sync()









