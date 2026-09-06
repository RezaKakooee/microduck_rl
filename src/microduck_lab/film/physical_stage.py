"""Physical film set: solid nest, contact ring, hollow eggs and release gates.

No mocap props, collision toggles, runtime body placement, or shell impulses.
The two eggs contain their ducklings from initialization onward.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from microduck_lab.film import stage

EGG_SPOTS = (np.array([-.047, .070]), np.array([.047, .070]))
EGG_RADIUS, EGG_HEIGHT = .034, .038
GATE_TOP = .016
EGG_START_Z = GATE_TOP + EGG_HEIGHT
INNER_FLOOR = -.024
MAT_TOP = .002
ALL = 127


def numbers(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def add_geom(body, name, kind, **attrs):
    common = dict(name=name, type=kind, contype='16', conaffinity=str(ALL),
                  solref='.012 1', solimp='.98 .999 .001', friction='1 .005 .0001')
    common.update({k: numbers(v) if isinstance(v, (tuple, list, np.ndarray)) else str(v)
                   for k, v in attrs.items()})
    return ET.SubElement(body, 'geom', common)


def shell_meshes(asset, body, label, top, offset):
    """Small convex wall panels preserve the actual hollow interior.

    A single convex hull of a concave shell would fill the entire egg. Each
    panel here is only 1.2 mm thick and serves as both visible and contact mesh.
    """
    angles = np.linspace(0 if top else np.arccos(.12), np.arccos(.12) if top else np.pi, 5)
    for row in range(4):
        for sector in range(16):
            points = []
            for inner in (False, True):
                a, c = EGG_RADIUS - .0012 * inner, EGG_HEIGHT - .0012 * inner
                for theta, phi in ((angles[row], sector * np.pi / 8),
                                   (angles[row], (sector + 1) * np.pi / 8),
                                   (angles[row + 1], sector * np.pi / 8),
                                   (angles[row + 1], (sector + 1) * np.pi / 8)):
                    r = a * np.sin(theta) * (1 - .16 * np.cos(theta))
                    points.append(np.array([r * np.cos(phi), r * np.sin(phi), c * np.cos(theta)]) - offset)
            vertices = np.unique(np.round(points, 10), axis=0)
            hull = ConvexHull(vertices)
            faces = hull.simplices.copy()
            for i, f in enumerate(faces):
                if np.dot(np.cross(vertices[f[1]] - vertices[f[0]], vertices[f[2]] - vertices[f[0]]), hull.equations[i, :3]) < 0:
                    faces[i] = f[::-1]
            meshname = f'{label}_panel_{row}_{sector}'
            ET.SubElement(asset, 'mesh', name=meshname, vertex=numbers(vertices.ravel()),
                          face=' '.join(str(i) for i in faces.ravel()))
            add_geom(body, meshname, 'mesh', mesh=meshname, rgba=(*stage.SHELL, 1), group=0)


def world_xml():
    root = ET.fromstring(stage.world_xml())
    root.set('model', 'duck_love_story_physical')
    for tag in ('equality', 'contact'):
        node = root.find(tag)
        if node is not None:
            root.remove(node)
    world, asset = root.find('worldbody'), root.find('asset')
    for child in list(world):
        if child.tag == 'body':
            world.remove(child)
        elif child.tag == 'geom':
            if child.get('name') == 'floor':
                child.set('contype', '16'); child.set('conaffinity', str(ALL))
                child.set('solref', '.01 1'); child.set('solimp', '.98 .999 .001')
                child.set('priority', '1')
                continue
            # Keep the garden physically solid and outside the acting area.
            xyz = np.fromstring(child.get('pos', child.get('fromto', '0 0 0')), sep=' ')[:3]
            if child.get('name') or (abs(xyz[0]) < .80 and -.55 < xyz[1] < 1.0):
                world.remove(child)
            else:
                child.set('contype', '16'); child.set('conaffinity', str(ALL))
    add_geom(world, 'nest_mat', 'cylinder', pos=(0, 0, MAT_TOP / 2), size=(.18, MAT_TOP / 2),
             rgba=(.44, .31, .17, 1))
    # Rear/sides are woven; the entrance facing the parents is actually open.
    for i, angle in enumerate(np.linspace(-.15, np.pi + .15, 22)):
        xy = .174 * np.array([np.cos(angle), np.sin(angle)])
        tangent = .024 * np.array([-np.sin(angle), np.cos(angle)])
        add_geom(world, f'nest_twig_{i}', 'capsule', size=(.003,),
                 fromto=(* (xy - tangent), .003, *(xy + tangent), .004), rgba=(.58,.42,.23,1))
    add_geom(world, 'ring_dais', 'cylinder', pos=(-.10,.76,.004), size=(.035,.004), rgba=(.65,.35,.42,1))
    ring = ET.SubElement(world, 'body', name='ring', pos='0 0 .1')
    ET.SubElement(ring, 'freejoint', name='ring_free')
    ET.SubElement(ring, 'inertial', pos='0 0 0', mass='.001', diaginertia='1e-7 1e-7 1e-7')
    for i in range(24):
        a, b = 2 * np.pi * np.array([i, i + 1]) / 24
        add_geom(ring, f'ring_segment_{i}', 'capsule', size=(.0024,),
                 fromto=(.0125*np.cos(a),.0125*np.sin(a),0,.0125*np.cos(b),.0125*np.sin(b),0),
                 rgba=(.95,.78,.22,1), friction='1.8 .005 .0001')
    add_geom(ring, 'ring_gem', 'sphere', pos=(0,.014,0), size=(.004,), rgba=(.55,.88,1,1))
    actuators = ET.SubElement(root, 'actuator')
    for i, xy in enumerate(EGG_SPOTS):
        gate = ET.SubElement(world, 'body', name=f'dispenser_gate_{i}', pos=numbers((*xy, GATE_TOP-.001)))
        ET.SubElement(gate, 'joint', name=f'egg_gate_{i}', type='slide', axis=f'{-1 if i==0 else 1} 0 0',
                      range='0 .085', damping='.15', armature='.002')
        add_geom(gate, f'gate_plate_{i}', 'box', size=(.039,.042,.001), mass='.015', rgba=(.30,.38,.44,1), priority='2', friction='.01 .0001 .00001')
        add_geom(gate, f'gate_carriage_{i}', 'box', pos=(0,.052,.0065),
                 size=(.012,.010,.002), mass='.004', rgba=(.38,.43,.48,1), priority='2', friction='.01 .0001 .00001')
        for j, ends in enumerate(((0,.042,0,0,.042,.010),
                                  (0,.042,.010,0,.052,.010),
                                  (0,.052,.010,0,.052,.006))):
            add_geom(gate, f'gate_link_{i}_{j}', 'capsule', fromto=ends,
                     size=(.0015,), mass='.001', rgba=(.38,.43,.48,1))
        ET.SubElement(actuators, 'position', name=f'gate_motor_{i}', joint=f'egg_gate_{i}',
                      kp='30', kv='1', ctrlrange='0 .085', forcerange='-2 2')
        base = ET.SubElement(world, 'body', name=f'egg{i}_base', pos=numbers((*xy, EGG_START_Z)))
        ET.SubElement(base, 'freejoint', name=f'egg{i}_free')
        ET.SubElement(base, 'inertial', pos='0 0 -.012', mass='.009', diaginertia='2e-6 2e-6 2e-6')
        shell_meshes(asset, base, f'egg{i}_bottom', False, np.zeros(3))
        add_geom(base, f'egg{i}_floor', 'cylinder', pos=(0,0,INNER_FLOOR-.001), size=(.027,.001), rgba=(.91,.84,.64,1))
        add_geom(base, f'egg{i}_sole', 'cylinder', pos=(0,0,-EGG_HEIGHT+.001), size=(.016,.001), rgba=(*stage.SHELL,1))
        pivot = np.array([0, EGG_RADIUS*.96, EGG_HEIGHT*.12])
        lid = ET.SubElement(base, 'body', name=f'egg{i}_lid', pos=numbers(pivot))
        ET.SubElement(lid, 'joint', name=f'egg_lid_{i}', type='hinge', axis='1 0 0',
                      range='-126.0507 0', damping='.00015', armature='.000001')
        ET.SubElement(lid, 'inertial', pos=numbers((0,-EGG_RADIUS*.65,.015)), mass='.003',
                      diaginertia='1e-6 1e-6 1e-6')
        shell_meshes(asset, lid, f'egg{i}_top', True, pivot)
        add_geom(base, f'egg{i}_hinge', 'capsule', size=(.003,),
                 fromto=(-.008,pivot[1],pivot[2],.008,pivot[1],pivot[2]), rgba=(.35,.38,.4,1))
        ET.SubElement(actuators, 'position', name=f'lid_motor_{i}', joint=f'egg_lid_{i}',
                      kp='.012', kv='.0003', ctrlrange='-2.2 0', forcerange='-.0025 .0025')
    # Open sleeve fingers prevent tipping as a low-friction support withdraws.
    for i, xy in enumerate(EGG_SPOTS):
        for j, angle in enumerate(np.linspace(0, 2*np.pi, 16, endpoint=False)):
            offset = .0365 * np.array([np.cos(angle), np.sin(angle)])
            add_geom(world, f'guide_{i}_{j}', 'capsule',
                     fromto=(*(xy+offset), .019, *(xy+offset), .043),
                     size=(.0015,), rgba=(.55,.59,.62,1), friction='.1 .001 .0001')
    # Visible fixed guide rails and actuator housings, outside the drop lanes.
    for side in (-1,1):
        add_geom(world, f'dispenser_post_{side}', 'box', pos=(side*.177,.122,.069), size=(.006,.008,.069), rgba=(.27,.32,.36,1))
        add_geom(world, f'dispenser_rail_{side}', 'box', pos=(side*.115,.122,.014), size=(.068,.004,.005), rgba=(.55,.59,.62,1))
    add_geom(world, 'dispenser_header', 'box', pos=(0,.122,.145), size=(.183,.008,.009), rgba=(.3,.46,.62,1))
    ET.SubElement(root, 'option', timestep='.005', integrator='implicitfast', iterations='100')
    return ET.tostring(root, encoding='unicode')


def build_model():
    spec = mujoco.MjSpec.from_string(world_xml())
    for prefix, (_, _, scale) in stage.CAST.items():
        duck = mujoco.MjSpec.from_file(stage.DUCK_XML)
        if scale != 1:
            stage.scale_spec(duck, scale)
        spec.attach(duck, prefix=prefix+'_', frame=spec.worldbody.add_frame())
    model = spec.compile()
    stage.paint(model)
    # Keep the CAD collision representation; all external objects and all
    # other ducks collide. Overlapping CAD parts in one robot retain the
    # original film's internal collision filtering.
    for g in range(model.ngeom):
        bodyname = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g]) or ''
        prefix = bodyname.split('_')[0]
        if prefix in stage.CAST:
            if model.geom_group[g] == 3:
                bit = 1 << list(stage.CAST).index(prefix)
                model.geom_contype[g] = bit
                model.geom_conaffinity[g] = ALL ^ bit
            else:
                model.geom_contype[g] = model.geom_conaffinity[g] = 0
        model.geom_margin[g] = 0
    for prefix, (_,_,scale) in stage.CAST.items():
        if scale < 1:
            stage.scale_actuation(model,prefix,scale)
    from bam.model import load_model
    torque = load_model(motor_name='xl330',model='m6').kt.value * 1.75
    for a in range(model.nu):
        name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a) or ''
        if name.startswith(('he_','she_')):
            model.actuator_forcerange[a]=(-torque,torque)
            model.actuator_forcelimited[a]=1
    return model, mujoco.MjData(model)
