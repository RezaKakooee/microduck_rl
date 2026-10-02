"""Replay one candidate, recording contacts at every 400 Hz physics step.

The observer reads MuJoCo's solved forces and pre-integration kinematics. It
never writes state, motor targets, model parameters or applied forces. The
normal runner still owns the 50 Hz controller and numbered video reservation.
"""
import argparse
import hashlib
import inspect
import json
import zipfile
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
from microduck_lab.tasks.salmon_jump import run as runner

BaseWorld = runner.World


class AuditedWorld(BaseWorld):
    latest = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_trace = []
        self.physics_qpos = []
        AuditedWorld.latest = self

    def step(self):
        original = mujoco.mj_step
        duck = self.ducks['she']

        def observe(model, data):
            t = float(data.time)
            self.physics_qpos.append(data.qpos.copy())
            original(model, data)
            # mj_step leaves solved contacts and geometry at the start of the
            # integration step. cvel is likewise the pre-integration velocity.
            mujoco.mj_subtreeVel(model, data)
            feet, body, clearance = runner.contact_state(duck)
            self.physics_trace.append([t, feet, body, clearance,
                                       duck.com()[2], data.subtree_linvel[duck.trunk,2],
                                       duck.up()])

        with patch.object(mujoco, 'mj_step', observe):
            return super().step()


def analyze(a, dt):
    free = (a[:,0]>3.) & (a[:,1]+a[:,2]<.05) & (a[:,3]>0.)
    flights = []
    start = None
    for i in range(len(a)+1):
        if i < len(a) and free[i]:
            if start is None:
                start = i
        elif start is not None:
            segment = a[start:i]
            if (i-start)*dt >= .01:
                flights.append(dict(start_s=float(a[start,0]),duration_s=(i-start)*dt,
                                    takeoff_com_vz=float(a[start,5]),
                                    max_clearance_m=float(segment[:,3].max()),
                                    max_com_rise_in_flight_m=float(segment[:,4].max()-a[start,4]),
                                    body_contact_after_N=float(a[i:,2].max()) if i<len(a) else None))
            start = None
    return flights


def slow_replay(world, a, qpos, stem):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    path = Path(str(stem)+'_slow.mp4')
    path.touch(exist_ok=False)
    data = mujoco.MjData(world.model)  # rendering copy; never advance dynamics
    renderer = mujoco.Renderer(world.model,height=540,width=960)
    camera = mujoco.MjvCamera()
    camera.distance,camera.azimuth,camera.elevation=.62,95.,-14.
    font=ImageFont.truetype('DejaVuSans.ttf',20)
    writer=imageio.get_writer(str(path),fps=25,codec='libx264',macro_block_size=2,
                              output_params=['-crf','20'])
    try:
        for i in range(0,len(a),4):  # 100 Hz saved states played at 25 fps
            if not 2.8 <= a[i,0] <= 5.8:
                continue
            data.qpos[:]=qpos[i]
            mujoco.mj_forward(world.model,data)
            camera.lookat[:]=[*data.xpos[world.ducks['she'].trunk,:2],.14]
            renderer.update_scene(data,camera=camera)
            frame=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,960,68),fill=(18,22,28))
            draw.text((15,8),f'0.25x recorded replay | simulation time {a[i,0]:.2f} s',font=font,fill='white')
            free=a[i,1]+a[i,2]<.05 and a[i,3]>0.
            label='NO GROUND CONTACT' if free else 'GROUND CONTACT'
            draw.text((15,36),f'{label} | minimum clearance {1000*a[i,3]:.1f} mm',font=font,
                      fill=(100,240,160) if free else (230,230,230))
            writer.append_data(np.asarray(frame))
    finally:
        writer.close();renderer.close()
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=9.)
    parser.add_argument('--slow-motion',action='store_true')
    args = parser.parse_args()
    original = json.loads(args.trial.read_text())
    candidate = runner.Candidate(**original['candidate'])
    base_path = Path(inspect.getfile(BaseWorld))
    base_source = base_path.read_bytes()
    with patch.object(runner,'World',AuditedWorld):
        result = runner.run(candidate,seconds=args.seconds,width=960)
    world = AuditedWorld.latest
    a = np.asarray(world.physics_trace)
    stem = Path(result['video']).with_suffix('')
    np.savez_compressed(str(stem)+'.physics.npz',trace=a,qpos=world.physics_qpos)
    with zipfile.ZipFile(stem.with_suffix('.sources.zip'),'a',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(str(base_path.relative_to(Path.cwd())),base_source)
    info = dict(source_trial=str(args.trial),video=result['video'],
                physics_hz=1/world.model.opt.timestep,
                columns=['time','foot_N','body_N','clearance_m','com_z','com_vz','up'],
                base_world_sha256=hashlib.sha256(base_source).hexdigest(),
                flights=analyze(a,world.model.opt.timestep))
    # Prove the observer did not alter the 50 Hz physical trajectory.
    old_path = args.trial.with_suffix('.npz')
    if old_path.exists():
        old=np.load(old_path)['trace'];new=np.load(stem.with_suffix('.npz'))['trace']
        if old.shape==new.shape:
            info['max_replay_trace_error']=float(np.abs(old-new).max())
    if args.slow_motion:
        info['slow_video']=slow_replay(world,a,world.physics_qpos,stem)
    Path(str(stem)+'.audit.json').write_text(json.dumps(info,indent=2)+'\n')
    print('PHYSICS_AUDIT',json.dumps(info),flush=True)


if __name__=='__main__':
    main()
