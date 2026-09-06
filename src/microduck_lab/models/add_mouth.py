#!/usr/bin/env python3
"""Give the head a working mouth joint.

The onshape export fuses the whole head into one rigid body: `jaw_soft` carries
the lens, the face, both head shells and the beak as geoms, and its only joint
is `head_roll`. That matches the policies -- every alpha network is 14 actions
with the mouth skipped -- but it means the beak physically cannot open, which is
wrong for anything that picks things up. On the real robot the mouth is the 15th
servo, driven by robotd (`duck-control/src/model.rs`: "neck, head, mouth"), not
by the policy.

This lifts the lower-jaw geoms into a child body on a hinge, and adds a position
actuator for it. Named `mouth`, so it does NOT match the `^(?!passive_).*`
actuator regexes by accident -- it is appended last, after the 14 servos, so
every existing joint and actuator index is unchanged and the policies still map
straight onto ctrl[0:14].

    python3 add_mouth.py robot_allcollisions.xml -o robot_allcollisions_mouth.xml
"""

import argparse
import xml.etree.ElementTree as ET

# Which meshes swing with the lower jaw. Everything else stays on the head:
# `soft_mouth_top` is the upper beak and `bottom_head_shell` is the fixed lower
# head shell, so neither belongs here.
JAW_MESHES = {"jaw", "jaw_soft"}

HEAD_BODY = "jaw_soft"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xml")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--pivot", type=float, nargs=3, default=[-0.0045, 0.0, -0.030],
                   help="Hinge position in the head body frame")
    p.add_argument("--axis", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    p.add_argument("--range", type=float, nargs=2, default=[0.0, 0.7])
    p.add_argument("--meshes", type=str, default=None,
                   help="Comma-separated mesh names to swing (default: jaw,jaw_soft)")
    args = p.parse_args()

    tree = ET.parse(args.xml)
    root = tree.getroot()

    head = None
    for body in root.iter("body"):
        if body.get("name") == HEAD_BODY:
            head = body
            break
    if head is None:
        raise SystemExit(f"no body named {HEAD_BODY} in {args.xml}")

    wanted = set(args.meshes.split(",")) if args.meshes else JAW_MESHES
    moving = [g for g in head.findall("geom") if g.get("mesh") in wanted]
    if not moving:
        raise SystemExit("found no jaw geoms to move")

    jaw = ET.SubElement(head, "body")
    jaw.set("name", "mouth_jaw")
    jaw.set("pos", " ".join(str(v) for v in args.pivot))
    # A hinge needs mass on the body or the solver has nothing to integrate.
    inertial = ET.SubElement(jaw, "inertial")
    inertial.set("pos", "0 0 0")
    inertial.set("mass", "0.02")
    inertial.set("diaginertia", "1e-5 1e-5 1e-5")
    joint = ET.SubElement(jaw, "joint")
    joint.set("name", "mouth")
    joint.set("type", "hinge")
    joint.set("axis", " ".join(str(v) for v in args.axis))
    joint.set("range", " ".join(str(v) for v in args.range))

    for g in moving:
        head.remove(g)
        # Geom positions were relative to the head; re-express against the pivot.
        pos = [float(v) for v in (g.get("pos") or "0 0 0").split()]
        g.set("pos", " ".join(str(pos[i] - args.pivot[i]) for i in range(3)))
        jaw.append(g)

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    act = ET.SubElement(actuator, "position")
    act.set("name", "mouth")
    act.set("joint", "mouth")
    act.set("kp", "20")
    act.set("ctrlrange", " ".join(str(v) for v in args.range))

    tree.write(args.out)
    print(f"wrote {args.out}: moved {len(moving)} geoms "
          f"({', '.join(sorted(g.get('mesh') for g in moving))}) onto joint 'mouth'")


if __name__ == "__main__":
    main()
