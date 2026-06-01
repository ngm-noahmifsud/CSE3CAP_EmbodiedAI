import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Run the following command to launch the demo with default parameters:
# python -u allegro_dynamic_grab_22068917.py --xml scene_22068917.xml --cube-size 0.023 --grip-scale 1.0 --object-z 0.072 --debug --realtime

# ---------------------------------------------------------------------
# XML generation
# ---------------------------------------------------------------------

def make_cube_scene(original_xml_path, cube_half_size=0.023):
    """
    Creates scene_loop_cube_generated.xml beside scene_22068917.xml.

    MuJoCo box size is half-extent:
        cube_half_size=0.023 -> cube side length 0.046 m
    """
    original_xml_path = Path(original_xml_path).resolve()

    tree = ET.parse(original_xml_path)
    root = tree.getroot()

    object_body = None
    for body in root.iter("body"):
        if body.attrib.get("name") == "object":
            object_body = body
            break

    if object_body is None:
        raise RuntimeError("Could not find body named 'object' in the XML.")

    object_body.set("pos", "0 0 0.072")

    geom = object_body.find("geom")
    if geom is None:
        geom = ET.SubElement(object_body, "geom")

    geom.set("name", "cube")
    geom.set("type", "box")
    geom.set("size", f"{cube_half_size} {cube_half_size} {cube_half_size}")
    geom.set("rgba", "0.2 0.9 0.35 1")
    geom.set("condim", "6")
    geom.set("priority", "1")
    geom.set("friction", "1.6 0.01 0.001")
    geom.set("density", "250")

    generated_path = original_xml_path.parent / "scene_loop_cube_generated.xml"
    tree.write(generated_path, encoding="utf-8", xml_declaration=False)

    return generated_path


# ---------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------

def mj_name(model, obj_type, obj_id):
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else ""


def get_body_id(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Could not find body '{body_name}'.")
    return body_id


def get_geom_id(model, geom_name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise RuntimeError(f"Could not find geom '{geom_name}'.")
    return geom_id


def get_object_freejoint(model, body_name="object"):
    body_id = get_body_id(model, body_name)

    for joint_id in range(model.njnt):
        if (
            model.jnt_bodyid[joint_id] == body_id
            and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        ):
            return joint_id

    raise RuntimeError(f"Body '{body_name}' does not have a freejoint.")


def set_object_pose(model, data, body_name="object", pos=(0.0, 0.0, 0.072)):
    """
    Reset cube pose.

    Freejoint qpos:
        x y z qw qx qy qz

    Freejoint qvel:
        vx vy vz wx wy wz
    """
    joint_id = get_object_freejoint(model, body_name)
    qadr = model.jnt_qposadr[joint_id]
    dadr = model.jnt_dofadr[joint_id]

    data.qpos[qadr:qadr + 7] = np.array([
        pos[0], pos[1], pos[2],
        1.0, 0.0, 0.0, 0.0,
    ])

    data.qvel[dadr:dadr + 6] = 0.0


def set_cube_size(model, cube_geom_id, cube_half_size):
    """
    Allows changing cube size from command line without editing XML manually.
    """
    model.geom_size[cube_geom_id, :] = np.array([
        cube_half_size,
        cube_half_size,
        cube_half_size,
    ])

    # Update bounding radius for collision broadphase.
    model.geom_rbound[cube_geom_id] = np.sqrt(3.0 * cube_half_size * cube_half_size)


# ---------------------------------------------------------------------
# Actuator helpers
# ---------------------------------------------------------------------

def actuator_joint_names(model):
    result = []

    for actuator_id in range(model.nu):
        actuator_name = mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_name = ""

        if model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT:
            joint_id = model.actuator_trnid[actuator_id, 0]
            joint_name = mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

        result.append((actuator_id, actuator_name, joint_name))

    return result


def group_allegro_actuators(model):
    groups = {
        "ff": [],
        "mf": [],
        "rf": [],
        "th": [],
        "other": [],
    }

    for actuator_id, actuator_name, joint_name in actuator_joint_names(model):
        label = f"{actuator_name} {joint_name}".lower()

        if "ff" in label or "index" in label:
            groups["ff"].append(actuator_id)
        elif "mf" in label or "middle" in label:
            groups["mf"].append(actuator_id)
        elif "rf" in label or "ring" in label:
            groups["rf"].append(actuator_id)
        elif "th" in label or "thumb" in label:
            groups["th"].append(actuator_id)
        else:
            groups["other"].append(actuator_id)

    return groups


def clamp_ctrl(model, ctrl):
    out = ctrl.copy()

    for actuator_id in range(model.nu):
        if model.actuator_ctrllimited[actuator_id]:
            lo, hi = model.actuator_ctrlrange[actuator_id]
            out[actuator_id] = np.clip(out[actuator_id], lo, hi)

    return out


def make_pose_ctrl(model, base_ctrl, groups, pose):
    ctrl = base_ctrl.copy()

    for group_name, offsets in pose.items():
        actuator_ids = groups.get(group_name, [])

        for k, actuator_id in enumerate(actuator_ids):
            if k < len(offsets):
                ctrl[actuator_id] = base_ctrl[actuator_id] + offsets[k]

    return clamp_ctrl(model, ctrl)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def interpolate_ctrl(ctrl_a, ctrl_b, alpha):
    alpha = smoothstep(alpha)
    return (1.0 - alpha) * ctrl_a + alpha * ctrl_b


def print_debug(model, groups):
    print("\nModel loaded.")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}")

    print("\nActuators:")
    for actuator_id, actuator_name, joint_name in actuator_joint_names(model):
        print(f"  {actuator_id:2d}: actuator='{actuator_name}', joint='{joint_name}'")

    print("\nDetected finger groups:")
    for name, ids in groups.items():
        print(f"  {name}: {ids}")

    print()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="scene_left.xml")

    # Main tuning variables.
    parser.add_argument(
        "--cube-size",
        type=float,
        default=0.023,
        help="Cube half-size in metres. Actual side length is cube-size * 2.",
    )
    parser.add_argument(
        "--grip-scale",
        type=float,
        default=1.0,
        help="Multiplies the hand closing amount. Try 0.5, 0.75, 1.0, 1.25.",
    )

    # Cube start pose tuning.
    parser.add_argument("--object-x", type=float, default=0.0)
    parser.add_argument("--object-y", type=float, default=0.0)
    parser.add_argument("--object-z", type=float, default=0.072)

    # Timing.
    parser.add_argument("--flat-time", type=float, default=1.0)
    parser.add_argument("--close-time", type=float, default=2.0)
    parser.add_argument("--hold-time", type=float, default=2.0)
    parser.add_argument("--release-time", type=float, default=2.0)
    parser.add_argument("--reset-pause", type=float, default=0.8)

    # Use -1 if the hand moves the wrong way.
    parser.add_argument("--close-sign", type=float, default=1.0)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    original_xml = Path(args.xml).resolve()
    if not original_xml.exists():
        raise FileNotFoundError(original_xml)

    generated_xml = make_cube_scene(
        original_xml,
        cube_half_size=args.cube_size,
    )

    print(f"Original XML:  {original_xml}", flush=True)
    print(f"Generated XML: {generated_xml}", flush=True)
    print("Loading MuJoCo model...", flush=True)

    model = mujoco.MjModel.from_xml_path(str(generated_xml))
    data = mujoco.MjData(model)

    cube_geom_id = get_geom_id(model, "cube")
    set_cube_size(model, cube_geom_id, args.cube_size)

    groups = group_allegro_actuators(model)

    if args.debug:
        print_debug(model, groups)
        print("Tuning values:")
        print(f"  cube half-size: {args.cube_size}")
        print(f"  cube side length: {2.0 * args.cube_size}")
        print(f"  grip scale: {args.grip_scale}")
        print(f"  cube start position: ({args.object_x}, {args.object_y}, {args.object_z})")
        print(f"  close sign: {args.close_sign}")
        print()

    # Default / flat hand control.
    # The hand resets to this every loop.
    base_ctrl = np.zeros(model.nu)

    for actuator_id in range(model.nu):
        if model.actuator_ctrllimited[actuator_id]:
            lo, hi = model.actuator_ctrlrange[actuator_id]
            base_ctrl[actuator_id] = 0.5 * (lo + hi)
        else:
            base_ctrl[actuator_id] = 0.0

    s = args.close_sign * args.grip_scale

    # This is the tunable base grasp shape.
    # grip-scale multiplies all these values.
    #
    # If the hand barely moves:
    #   increase --grip-scale
    #
    # If the hand crushes / clips through cube:
    #   decrease --grip-scale
    #
    # If it moves away from cube:
    #   use --close-sign -1
    grasp_pose = {
        "ff": [s * 0.06, s * 0.38, s * 0.50, s * 0.26],
        "mf": [s * 0.06, s * 0.40, s * 0.52, s * 0.26],
        "rf": [s * 0.04, s * 0.34, s * 0.44, s * 0.22],
        "th": [s * 0.18, s * 0.30, s * 0.38, s * 0.24],
    }

    grasp_ctrl = make_pose_ctrl(model, base_ctrl, groups, grasp_pose)

    cube_pos = (args.object_x, args.object_y, args.object_z)

    print("Launching viewer...", flush=True)
    print(
        "Loop: reset cube + flat hand -> close -> hold -> release -> reset",
        flush=True,
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        loop_start = time.time()
        last_wall = loop_start

        cycle_duration = (
            args.flat_time
            + args.close_time
            + args.hold_time
            + args.release_time
            + args.reset_pause
        )

        last_cycle_index = -1

        while viewer.is_running():
            now = time.time()
            elapsed = now - loop_start

            cycle_index = int(elapsed // cycle_duration)
            cycle_t = elapsed % cycle_duration

            # At the start of every new loop, reset cube + hand.
            if cycle_index != last_cycle_index:
                last_cycle_index = cycle_index

                mujoco.mj_resetData(model, data)
                set_cube_size(model, cube_geom_id, args.cube_size)
                set_object_pose(model, data, "object", cube_pos)
                mujoco.mj_forward(model, data)

                data.ctrl[:] = base_ctrl

                print(
                    f"\nCycle {cycle_index + 1} | "
                    f"cube-size={args.cube_size:.3f} half / "
                    f"{2.0 * args.cube_size:.3f} side | "
                    f"grip-scale={args.grip_scale:.2f}",
                    flush=True,
                )

            # 1. Flat/default hand.
            if cycle_t < args.flat_time:
                data.ctrl[:] = base_ctrl

            # 2. Gentle close.
            elif cycle_t < args.flat_time + args.close_time:
                alpha = (cycle_t - args.flat_time) / args.close_time
                data.ctrl[:] = interpolate_ctrl(base_ctrl, grasp_ctrl, alpha)

            # 3. Hold.
            elif cycle_t < args.flat_time + args.close_time + args.hold_time:
                data.ctrl[:] = grasp_ctrl

            # 4. Gentle release.
            elif cycle_t < (
                args.flat_time
                + args.close_time
                + args.hold_time
                + args.release_time
            ):
                alpha = (
                    cycle_t
                    - args.flat_time
                    - args.close_time
                    - args.hold_time
                ) / args.release_time

                data.ctrl[:] = interpolate_ctrl(grasp_ctrl, base_ctrl, alpha)

            # 5. Pause at default hand before reset.
            else:
                data.ctrl[:] = base_ctrl

            mujoco.mj_step(model, data)
            viewer.sync()

            if args.realtime:
                dt = model.opt.timestep
                wall_elapsed = time.time() - last_wall
                if wall_elapsed < dt:
                    time.sleep(dt - wall_elapsed)
                last_wall = time.time()


if __name__ == "__main__":
    main()