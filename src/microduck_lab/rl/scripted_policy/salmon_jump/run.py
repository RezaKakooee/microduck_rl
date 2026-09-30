"""Filmed back-to-feet jump probes with unmodified BAM actuators.

A jump must leave the ground and finish standing on its feet. All setup is
before World.start(); afterwards only joint targets or the standing policy
act on the robot. Every physical candidate is filmed on a GPU node.
"""
import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import mujoco
import numpy as np

from microduck_lab.tasks.human_bridge.scripted_policy.world import World, SetDesign, SERVOS, CONTROL_DT
from microduck_lab.tasks.human_bridge.scripted_policy.runtime import brain

ROOT = Path(__file__).resolve().parent
OUTPUT = Path('videos/human_bridge/scripted_policy/salmon_jump')
INDEX = {name: i for i, name in enumerate(SERVOS)}


@dataclass
class Candidate:
    # Absolute symmetric sagittal angles: left hip/knee/ankle, neck/head.
    poses: tuple = ((-1.2, 1.2, .5, .8, -.7),
                    (.7, -.8, -.7, -1.0, .7),
                    (-.6, .9, -.3, .35, .35))
    durations: tuple = (.3, .16, .3)
    hold: float = .5
    start_pitch: float = -np.pi/2
    kind: str = 'jump'
    interpolation: str = 'smoothstep'
    recovery: str = 'auto'
    air_catch_s: float = 0.
    head_settle_s: float = 0.
    landing_hip_bias: float = 0.
    landing_ankle_bias: float = 0.
    landing_bias_s: float = .3
    catch_start_s: float | None = None

    def __post_init__(self):
        if self.kind not in ('jump', 'rest', 'stand'):
            raise ValueError('Unknown probe kind')
        shape = np.asarray(self.poses).shape
        if len(shape) != 2 or shape[1] != 5 or shape[0] < 2:
            raise ValueError('At least two five-angle poses required')
        if self.interpolation not in ('smoothstep', 'linear', 'step'):
            raise ValueError('Unknown interpolation')
        if self.recovery not in ('auto', 'after_script', 'disabled', 'airborne'):
            raise ValueError('Unknown recovery mode')
        if len(self.durations) != len(self.poses) or not np.isfinite(self.durations).all() or min(self.durations) < .04:
            raise ValueError('Finite segment durations >= .04 s required')
        if not np.isfinite([self.air_catch_s,self.head_settle_s]).all() or min(self.air_catch_s,self.head_settle_s) < 0.:
            raise ValueError('Finite nonnegative catch duration required')
        if not np.isfinite([self.landing_hip_bias,self.landing_ankle_bias,self.landing_bias_s]).all() or self.landing_bias_s <= 0.:
            raise ValueError('Finite landing corrections and positive fade time required')
        if self.catch_start_s is not None and (not np.isfinite(self.catch_start_s) or self.catch_start_s < .04):
            raise ValueError('Catch time must be >= .04 s after the settle period')
        if not np.isfinite(self.poses).all() or not np.isfinite([self.hold, self.start_pitch]).all() or self.hold < 0:
            raise ValueError('Non-finite or negative parameters')


def pose(home, values):
    q = home.copy()
    for joint, value in zip(('hip_pitch', 'knee', 'ankle'), values[:3]):
        q[INDEX['left_' + joint]] = value
        q[INDEX['right_' + joint]] = -value
    q[INDEX['neck_pitch']], q[INDEX['head_pitch']] = values[3:]
    return q


def target_at(candidate, home, seconds):
    previous = home
    elapsed = 0.
    for values, duration in zip(candidate.poses, candidate.durations):
        target = pose(home, values)
        if seconds < elapsed + duration:
            u = np.clip((seconds-elapsed)/duration, 0., 1.)
            blend = u*u*(3-2*u) if candidate.interpolation == 'smoothstep' else u
            if candidate.interpolation == 'step':
                blend = 1. if seconds >= 0. else 0.
            return previous + (target-previous)*blend
        previous, elapsed = target, elapsed+duration
    if seconds < elapsed + candidate.hold:
        return previous
    u = np.clip((seconds-elapsed-candidate.hold)/.3, 0., 1.)
    return previous + (home-previous)*(u*u*(3-2*u))


def reserve(folder=OUTPUT):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for n in range(1, 100000):
        stem = folder / f'salmon_try{n}'
        if any(stem.with_suffix(s).exists() for s in ('.json', '.mp4', '.npz')):
            continue
        try:
            fd = os.open(stem.with_suffix('.lock'), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        return stem
    raise RuntimeError('No free trial number')


def contact_state(duck):
    m, d = duck.model, duck.data
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    feet = set(m.site_bodyid[list(duck.foot_sites)])
    solid = set(duck.solid)
    foot_force = body_force = 0.
    force = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if floor not in (c.geom1, c.geom2):
            continue
        g = c.geom2 if c.geom1 == floor else c.geom1
        if g not in solid:
            continue
        mujoco.mj_contactForce(m, d, i, force)
        if m.geom_bodyid[g] in feet:
            foot_force += max(0., force[0])
        else:
            body_force += max(0., force[0])
    clearance = float(duck.points(duck.solid)[:, 2].min())
    return float(foot_force), float(body_force), clearance


def summarize(trace, kind='jump'):
    a = np.asarray(trace)
    # Columns: time, trunk xyz, up, forward_z, com_z, vz, foot_N, body_N,
    # minimum geometric clearance, max abs motor torque, recovery mode, COM vertical velocity.
    settled = a[(a[:, 0] >= 2.5) & (a[:, 0] <= 3.00001)]
    initial_valid = bool(len(settled) and np.max(np.abs(settled[:, 4])) < .35
                         and np.min(settled[:, 5]) > .8
                         and np.ptp(settled[:, 6]) < .003)
    active = a[a[:, 0] > 3.00001]
    air = ((active[:, 8]+active[:, 9] < .05) & (active[:, 10] > .002))
    longest = current = 0
    for i, airborne in enumerate(air):
        current = current+1 if airborne else 0
        longest = max(longest, current)
    good = ((active[:, 4] > .95) & (active[:, 3] > .09)
            & (active[:, 8] > 1.) & (active[:, 9] < .5))
    # Measure launch speed at contact loss, not later when clearance crosses
    # 2 mm: on a small hop that threshold can be reached near the apex.
    # Upright flight is allowed; a settled standing pause before it is not.
    free = (active[:,8]+active[:,9] < .05) & (active[:,10] > 0.)
    first_air = None
    flight_duration = 0.
    start = None
    stood = False
    standing_ticks = 0
    for i in range(len(active)+1):
        if i < len(active):
            standing_ticks = standing_ticks+1 if good[i] else 0
            stood = stood or standing_ticks >= 10
        if i < len(active) and free[i]:
            if start is None:
                start = i
                eligible = active[i,13] > .1 and not stood
        elif start is not None:
            qualified = np.any(air[start:i-1] & air[start+1:i])
            if eligible and qualified and first_air is None:
                first_air = start
                flight_duration = (i-start)*CONTROL_DT
            start = None
    last_bad = np.flatnonzero(~good)
    first = last_bad[-1]+1 if len(last_bad) else 0
    final_hold = float(active[-1, 0]-active[first, 0]+CONTROL_DT) if first < len(active) else 0.
    baseline = float(np.mean(settled[:, 6]))
    rise = float(active[:, 6].max()-baseline)
    airborne_before_landing = first_air is not None and first_air < first
    clean_landing = bool(first_air is not None and final_hold >= 2.
                         and np.max(active[first_air:,9]) < .5)
    success = bool(kind == 'jump' and initial_valid and longest*CONTROL_DT >= .04
                   and rise >= .025 and airborne_before_landing and clean_landing)
    return dict(judge_version='clean_landing_v3',success=success, valid_back_start=initial_valid,
                longest_airborne_s=longest*CONTROL_DT, com_rise_m=rise,
                final_standing_s=final_hold, final_up=float(a[-1,4]),
                final_height_m=float(a[-1,3]), max_up=float(active[:,4].max()),
                max_motor_torque_nm=float(a[:,11].max()),
                max_up_while_airborne=float(active[air,4].max()) if air.any() else None,
                peak_vertical_speed=float(active[:,13].max()),
                clean_landing=clean_landing,
                qualifying_airborne_s=flight_duration,
                takeoff_time_s=float(active[first_air,0]) if first_air is not None else None,
                max_clearance_m=float(active[:,10].max()),
                takeoff_com_vz=float(active[first_air,13]) if first_air is not None else None,
                settled_up_range=settled[:,4].tolist()[::5])


def run(candidate, seconds=8., width=640):
    if os.environ.get('MUJOCO_GL') != 'egl' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Physical attempts must run through our video.sbatch on a GPU node')
    import imageio.v2 as imageio
    if seconds <= 3.1:
        raise ValueError('Episode must include settling and active motion')
    source_files = list(ROOT.glob('*.py'))
    source_files += [Path(__import__('inspect').getfile(World)), Path(__import__('inspect').getfile(brain))]
    sources = {p: p.read_bytes() for p in source_files}
    stem = reserve()
    w = World(SetDesign([]), cast=('she',))
    duck = w.ducks['she']
    controller = brain(duck, stand_only=True)
    home = controller.default_pose.copy()
    pitch = 0. if candidate.kind == 'stand' else candidate.start_pitch
    duck.place((0.,0.,.4), pitch=pitch, q=home)
    position = duck.pos()
    position[2] += .0005-duck.points(duck.solid)[:,2].min()
    duck.place(position, pitch=pitch, q=home)
    w.start()
    renderer = mujoco.Renderer(w.model, height=width*9//16, width=width)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .62, 95., -18.
    writer = imageio.get_writer(str(stem.with_suffix('.mp4')), fps=25, codec='libx264',
                                macro_block_size=2, output_params=['-crf','25','-movflags',
                                '+frag_keyframe+empty_moov+default_base_moof'])
    rows, states, targets, velocities, torques = [], [], [], [], []
    recovery = False
    catch_until = 0.
    recovery_time = None
    head_ids = [INDEX['neck_pitch'],INDEX['head_pitch']]
    recovery_head = home[head_ids].copy()
    started = time.monotonic()
    error = None
    try:
        for tick in range(round(seconds/CONTROL_DT)):
            if candidate.kind == 'stand':
                controller.act()
            elif w.t < 3. or candidate.kind == 'rest':
                duck.hold(home)
            elif recovery:
                if w.t < catch_until:
                    duck.hold(home)
                else:
                    controller.act()
                if candidate.head_settle_s > 0. and recovery_time is not None:
                    u = np.clip((w.t-recovery_time)/candidate.head_settle_s,0.,1.)
                    target = duck.target.copy()
                    target[head_ids] = (1-u)*recovery_head + u*target[head_ids]
                    # Preserve untouched policy commands, including intentional
                    # PD target overshoot beyond joint travel.
                    duck.target = target
                    duck.motor.q_target[:] = target
                if recovery_time is not None and (candidate.landing_hip_bias or candidate.landing_ankle_bias):
                    blend = 1.-np.clip((w.t-recovery_time)/candidate.landing_bias_s,0.,1.)
                    for joint,bias in [('hip_pitch',candidate.landing_hip_bias),('ankle',candidate.landing_ankle_bias)]:
                        for side,sign in [('left',1.),('right',-1.)]:
                            j = INDEX[side+'_'+joint]
                            duck.target[j] += sign*bias*blend
                    duck.motor.q_target[:] = duck.target
            else:
                duck.hold(target_at(candidate,home,w.t-3.))
            w.step()
            mujoco.mj_forward(w.model,w.data)
            mujoco.mj_subtreeVel(w.model,w.data)
            feet, body, clearance = contact_state(duck)
            handoff = sum(candidate.durations)
            if candidate.recovery == 'after_script':
                handoff += candidate.hold + .3
            early_catch = candidate.catch_start_s is not None and w.t >= 3.+candidate.catch_start_s
            airborne_catch = (candidate.recovery == 'airborne' and feet+body < .05
                              and clearance > 0. and duck.up() > .5
                              and w.data.subtree_linvel[duck.trunk,2] > .1)
            if not recovery and candidate.kind == 'jump' and w.t > 3. and (early_catch or airborne_catch):
                recovery = True
                catch_until = w.t + candidate.air_catch_s
                recovery_time = w.t
                recovery_head = duck.q[head_ids].copy()
            if (not recovery and candidate.kind == 'jump' and candidate.recovery != 'disabled' and w.t > 3.+handoff
                    and duck.up() > .85 and feet > 1. and body < .5 and duck.pos()[2] > .085):
                recovery = True
                recovery_time = w.t
                recovery_head = duck.q[head_ids].copy()
            rows.append([w.t,*duck.pos(),duck.up(),duck.R()[2,0],duck.com()[2],
                         w.data.qvel[duck.dof+2],feet,body,clearance,
                         np.max(np.abs(duck.torque())),float(recovery),
                         w.data.subtree_linvel[duck.trunk,2]])
            targets.append(duck.target.copy())
            velocities.append(w.data.qvel.copy())
            torques.append(duck.torque().copy())
            if tick % 2 == 1:
                states.append(w.data.qpos.copy())
                camera.lookat[:] = [duck.pos()[0],duck.pos()[1],.14]
                renderer.update_scene(w.data,camera=camera)
                writer.append_data(renderer.render())
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        writer.close()
        renderer.close()
        np.savez_compressed(stem.with_suffix('.npz'), trace=rows,qpos=states,targets=targets,qvel=velocities,torques=torques)
        result = dict(candidate=asdict(candidate),error=error,video=str(stem.with_suffix('.mp4')),
                      wall_s=time.monotonic()-started,robot_mass_kg=duck.mass(),
                      motor_model='BAM XL330 M6, 7.4 V, unmodified limits',
                      sampling_hz=50)
        if len(rows) > 155:
            result.update(summarize(rows,candidate.kind))
        result['sources'] = {str(p):hashlib.sha256(data).hexdigest() for p, data in sources.items()}
        with zipfile.ZipFile(stem.with_suffix('.sources.zip'),'w',zipfile.ZIP_DEFLATED) as z:
            for p, data in sources.items():
                z.writestr(str(p.relative_to(Path.cwd())), data)
        stem.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(trial=stem.name,**{k:v for k,v in result.items() if k not in ('sources','candidate','settled_up_range')})),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--case',type=int)
    a=p.parse_args()
    cases=json.loads(a.config.read_text())
    if a.case is not None: cases=[cases[a.case]]
    for c in cases: run(Candidate(**c),a.seconds)


if __name__ == '__main__': main()
