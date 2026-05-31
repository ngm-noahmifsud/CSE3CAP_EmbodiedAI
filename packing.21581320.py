# ---------------------------------------------------------------------------
# Student name: Damon Tsimis
# Student ID: 21581320
# Model used: Franka Emika Panda (franka_emika_panda)
# HOW TO RUN:
#    1. Install dependencies in terminal:
#        --> pip install mujoco numpy
#    2. Clone MuJoCo Menagerie into the same folder as this script:
#         git clone https://github.com/google-deepmind/mujoco_menagerie
#    3. Place this file and packing.scene.xml into the following folder:
#         mujoco_menagerie/franka_emika_panda/
#    4. Run:
#         mjpython packing_21581320.py
#    5. MuJoCo viewer window opens and the simulation runs automatically.
# ---------------------------------------------------------------------------

import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path(
    "/Users/damon/Downloads/mujoco_menagerie-main/franka_emika_panda/packing_scene.xml"
)
data = mujoco.MjData(model)

J1 = model.joint("joint1").qposadr[0]
J2 = model.joint("joint2").qposadr[0]
J3 = model.joint("joint3").qposadr[0]
J4 = model.joint("joint4").qposadr[0]
J5 = model.joint("joint5").qposadr[0]
J6 = model.joint("joint6").qposadr[0]
J7 = model.joint("joint7").qposadr[0]
F1 = model.joint("finger_joint1").qposadr[0]
F2 = model.joint("finger_joint2").qposadr[0]
CJ = model.joint("cube1").qposadr[0]

OPEN   = 0.04
CLOSED = 0.003

def set_arm(pose):
    data.qpos[J1]=pose[0]; data.qpos[J2]=pose[1]
    data.qpos[J3]=pose[2]; data.qpos[J4]=pose[3]
    data.qpos[J5]=pose[4]; data.qpos[J6]=pose[5]
    data.qpos[J7]=pose[6]; data.qpos[F1]=pose[7]
    data.qpos[F2]=pose[7]
    mujoco.mj_forward(model, data)

def set_cube(x, y, z):
    data.qpos[CJ+0]=x; data.qpos[CJ+1]=y; data.qpos[CJ+2]=z
    data.qpos[CJ+3]=1.0; data.qpos[CJ+4]=0.0
    data.qpos[CJ+5]=0.0; data.qpos[CJ+6]=0.0
    dof = model.joint("cube1").dofadr[0]
    data.qvel[dof:dof+6] = 0
    mujoco.mj_forward(model, data)

def get_link7():
    return data.body("link7").xpos.copy()

def ease(t):
    return t * t * (3 - 2 * t)

def interp_free(viewer, pa, pb, ca, cb, frames):
    pa=np.array(pa,float); pb=np.array(pb,float)
    ca=np.array(ca,float); cb=np.array(cb,float)
    for i in range(frames):
        t = ease(i / max(frames-1,1))
        set_arm(pa + t*(pb-pa))
        c = ca + t*(cb-ca)
        set_cube(c[0], c[1], c[2])
        viewer.sync()
        time.sleep(0.016)

def interp_carry(viewer, pa, pb, cube_pos, frames):
    pa = np.array(pa, float)
    pb = np.array(pb, float)
    set_arm(pa)
    offset = np.array(cube_pos, float) - get_link7()
    for i in range(frames):
        t = ease(i / max(frames-1, 1))
        set_arm(pa + t*(pb-pa))
        pos = get_link7() + offset
        set_cube(pos[0], pos[1], pos[2])
        viewer.sync()
        time.sleep(0.016)
    set_arm(pb)
    final = get_link7() + offset
    set_cube(final[0], final[1], final[2])
    return list(final)

def hold_free(viewer, pose, cxyz, frames):
    for _ in range(frames):
        set_arm(pose)
        set_cube(cxyz[0], cxyz[1], cxyz[2])
        viewer.sync()
        time.sleep(0.016)

def hold_carry(viewer, pose, cube_pos, frames):
    set_arm(pose)
    offset = np.array(cube_pos, float) - get_link7()
    for _ in range(frames):
        set_arm(pose)
        pos = get_link7() + offset
        set_cube(pos[0], pos[1], pos[2])
        viewer.sync()
        time.sleep(0.016)
    set_arm(pose)
    return list(get_link7() + offset)

HOME    = [0.0,  0.0,  0.0, -1.5708, 0.0, 1.5708, -0.7854, OPEN]
HOVER   = [0.0,  0.35, 0.0, -1.10,   0.0, 1.5708, -0.7854, OPEN]
PICK    = [0.0,  0.45, 0.0, -1.10,   0.0, 1.5708, -0.7854, OPEN]
GRIP    = [0.0,  0.45, 0.0, -1.10,   0.0, 1.5708, -0.7854, CLOSED]
LIFT    = [0.0,  0.0,  0.0, -1.5708, 0.0, 1.5708, -0.7854, CLOSED]
MID     = [1.4,  0.0,  0.0, -1.5708, 0.0, 1.5708, -0.7854, CLOSED]
OVER    = [2.8,  0.0,  0.0, -1.5708, 0.0, 1.5708, -0.7854, CLOSED]
DROP    = [2.8,  0.30, 0.0, -1.30,   0.0, 1.5708, -0.7854, CLOSED]
RELE    = [2.8,  0.30, 0.0, -1.30,   0.0, 1.5708, -0.7854, OPEN]
RETRACT = [2.8,  0.0,  0.0, -1.5708, 0.0, 1.5708, -0.7854, OPEN]

C_START  = [0.68, 0.75, 0.44]
C_ARRIVE = [0.68, 0.0,  0.44]
C_IN_BOX = [-0.60, 0.215, 0.435]

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("=== PANDA PACKING SIMULATION ===")

    set_arm(HOME)
    set_cube(*C_START)
    hold_free(viewer, HOME, C_START, 30)

    print("STEP 1 - Conveyor...")
    interp_free(viewer, HOME, HOME, C_START, C_ARRIVE, 150)
    hold_free(viewer, HOME, C_ARRIVE, 20)

    print("STEP 2 - Hovering...")
    interp_free(viewer, HOME, HOVER, C_ARRIVE, C_ARRIVE, 80)
    hold_free(viewer, HOVER, C_ARRIVE, 15)

    print("STEP 3 - Lowering...")
    interp_free(viewer, HOVER, PICK, C_ARRIVE, C_ARRIVE, 80)
    hold_free(viewer, PICK, C_ARRIVE, 20)

    print("STEP 4 - Gripping - cube stays frozen...")
    for i in range(50):
        t = ease(i / 49.0)
        pose = list(np.array(PICK) + t*(np.array(GRIP)-np.array(PICK)))
        set_arm(pose)
        set_cube(C_ARRIVE[0], C_ARRIVE[1], C_ARRIVE[2])
        viewer.sync()
        time.sleep(0.016)
    hold_free(viewer, GRIP, C_ARRIVE, 15)

    print("STEP 5 - Lifting...")
    gripped = interp_carry(viewer, GRIP, LIFT, C_ARRIVE, 80)
    gripped = hold_carry(viewer, LIFT, gripped, 15)

    print("STEP 6 - Swinging to box...")
    gripped = interp_carry(viewer, LIFT, MID, gripped, 60)
    gripped = interp_carry(viewer, MID, OVER, gripped, 60)
    gripped = hold_carry(viewer, OVER, gripped, 15)

    print("STEP 7 - Lowering into box...")
    gripped = interp_carry(viewer, OVER, DROP, gripped, 70)
    gripped = hold_carry(viewer, DROP, gripped, 20)

    print("STEP 8 - Releasing...")
    drop_pos = list(gripped)
    for i in range(30):
        t = ease(i / 29.0)
        pose = list(np.array(DROP) + t*(np.array(RELE)-np.array(DROP)))
        set_arm(pose)
        set_cube(drop_pos[0], drop_pos[1], drop_pos[2])
        viewer.sync()
        time.sleep(0.016)

    for frame in range(35):
        set_arm(RELE)
        t = ease(frame / 34.0)
        fx = drop_pos[0] + t*(C_IN_BOX[0]-drop_pos[0])
        fy = drop_pos[1] + t*(C_IN_BOX[1]-drop_pos[1])
        fz = drop_pos[2] - t*(drop_pos[2]-C_IN_BOX[2])
        set_cube(fx, fy, fz)
        viewer.sync()
        time.sleep(0.016)

    set_cube(*C_IN_BOX)
    hold_free(viewer, RELE, C_IN_BOX, 20)
    print("  Cube placed!")

    print("STEP 9 - Returning home...")
    interp_free(viewer, RELE, RETRACT, C_IN_BOX, C_IN_BOX, 55)
    interp_free(viewer, RETRACT, HOME, C_IN_BOX, C_IN_BOX, 70)
    hold_free(viewer, HOME, C_IN_BOX, 30)

    print("=== COMPLETE ===")
    while viewer.is_running():
        set_arm(HOME)
        set_cube(*C_IN_BOX)
        viewer.sync()
        time.sleep(0.016)
