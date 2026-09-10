"""Phase-locked swing pumping using only the duck's servo targets.

Simulation expert, not a trained 61D robotd policy. The swing angle and angular
velocity must be observed. No motor, imposed motion or push drives the swing.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from microduck_lab import paths
from microduck_lab.tasks.swing.swing import Swing,DT


class PumpingExpert:
    """Phase-locked pumping with the head, the hips and the knees.

    The duck has two ways to pump, and they are not equally strong. Measured on
    this model:

    | joint      | CoM shift per rad | inertia change |
    |------------|-------------------|----------------|
    | neck_pitch | -28.1 mm          | 3.0% over +/-0.6 rad |
    | head_pitch | +11.8 mm          |                |
    | hip_pitch  | -10.7 mm          | 2.1% over +/-0.5 rad |
    | knee       |  +1.2 mm          | 4.0% over +/-0.6 rad |

    So the head and hips pump by moving the centre of mass. They act once per
    swing cycle. The knee changes the moment of inertia most of all.

    A 2x knee motion (parametric pumping, what a standing person does) was
    tried and it FAILS here: legs-only with `knee_harmonic=2` reached 0.1 deg,
    against 8.8 deg at `knee_harmonic=1`. Knee alone at 2x spent 37.6 J and
    reached 0.3 deg. So the knee must move once per swing, with the hip.

    The head is 251 g and the legs are 116 g, so the head has more authority per
    radian. `legs_only` holds the head still to measure the legs on their own.
    """

    def __init__(self,swing,target_degrees=22.,phase=-np.pi/2,hip=-.85,head=-.25,
                 neck=.45,knee=.65,knee_harmonic=1,knee_phase=0.,legs_only=False):
        self.base=swing.base.copy();self.ai=dict(swing.ai)
        self.omega=4.8*np.sqrt(.48/swing.length)
        self.target=np.deg2rad(target_degrees)
        self.phase=phase;self.hip=hip;self.head=head;self.neck=neck
        self.knee=knee;self.knee_harmonic=knee_harmonic;self.knee_phase=knee_phase
        if legs_only:self.neck=0.;self.head=0.
        self.centre=None

    def __call__(self,angle,velocity,time):
        # The swing hangs a little off vertical, because the duck's mass is not
        # centred over the seat. That angle moves whenever the seat changes, so
        # it is tracked here instead of being written in as a number. A fixed
        # 0.025 rad left the swing stuck at 0.5 deg once the seat was rebuilt.
        self.centre=angle if self.centre is None else self.centre+(angle-self.centre)*DT/4.
        target=self.base.copy()
        if time<=2:return target
        centered=angle-self.centre
        phase=np.arctan2(self.omega*centered,velocity)
        amplitude_sq=centered**2+(velocity/self.omega)**2
        gain=np.clip((self.target**2-amplitude_sq)/.0144,-.6,1.)
        ramp=min(1.,(time-2)/2)
        wave=float(gain*np.sin(phase+self.phase)*ramp)
        target[self.ai['neck_pitch']]+=self.neck*wave
        target[self.ai['head_pitch']]+=self.head*wave
        target[self.ai['left_hip_pitch']]+=self.hip*wave
        target[self.ai['right_hip_pitch']]-=self.hip*wave
        if self.knee:
            # A second harmonic tucks and extends the knee twice per swing.
            # `gain` stays signed, so the amplitude regulator still works.
            bend=float(gain*np.sin(self.knee_harmonic*phase+self.knee_phase)*ramp)
            target[self.ai['left_knee']]+=self.knee*bend
            target[self.ai['right_knee']]-=self.knee*bend
        return target


def speed_at(spec,t):
    """Playback speed at simulated time `t`.

    `spec` is a number for one speed throughout, or a list of (until, speed)
    pairs, so a clip can run real time at the start, race through the middle and
    slow down again for the end.
    """
    if not isinstance(spec,(list,tuple)):return float(spec)
    for until,value in spec:
        if t<until:return float(value)
    return float(spec[-1][1])


def _label(frame,text):
    """Write `text` on a rendered frame, so a sped-up clip says so on its face."""
    from PIL import Image,ImageDraw,ImageFont
    image=Image.fromarray(frame);draw=ImageDraw.Draw(image)
    size=max(20,image.height//22)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',size)
    except OSError:font=ImageFont.load_default()
    x=y=max(12,image.height//30)
    for dx,dy in ((-2,0),(2,0),(0,-2),(0,2)):        # dark outline, readable on sky or grass
        draw.text((x+dx,y+dy),text,font=font,fill=(20,28,36))
    draw.text((x,y),text,font=font,fill=(255,255,255))
    return np.asarray(image)


def evaluate(seconds=60.,active=True,video=None,trace_path=None,expert_kw=None,
             video_speed=1.,video_label=None,**environment):
    s=Swing(**environment);expert=PumpingExpert(s,**(expert_kw or {}))
    # Once the seat lets go the duck is falling, so it stops pumping and puts
    # its legs down to meet the ground. After it lands it tries to get up.
    drop=environment.get('dismantle_at');landed=None
    def pose(**kw):
        q=s.base.copy()
        for name,value in kw.items():
            q[s.ai[name]]=value
            other=name.replace('left','right')
            if other in s.ai and other!=name:q[s.ai[other]]=-value
        return q
    brace=pose(left_hip_pitch=-1.30,left_knee=-1.00)
    # Two poses, alternating: fold the legs in under the body, then drive them
    # down. The duck lands on its side or its face, so this may not work. It is
    # an attempt, not a trained get-up.
    tuck=pose(left_hip_pitch=-.35,left_knee=1.30,neck_pitch=-.30,head_pitch=-.30)
    drive=pose(left_hip_pitch=-1.50,left_knee=-1.45,neck_pitch=-.20,head_pitch=-.10)
    history=[];failure=None;writer=renderer=None
    # A frame is emitted every `speed/25` simulated seconds, so 25 fps of output
    # plays back at `speed`. The speed may change part way through the clip.
    next_frame=0.;widest=0.;fixed_label=video_label
    if video:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True,exist_ok=True)
        renderer=mujoco.Renderer(s.model,height=720,width=1280)
        # faststart moves the index to the front of the file. Without it the
        # index lands after 58 MB of frames, and any player that streams the
        # file - a browser, an editor preview - shows nothing.
        writer=imageio.get_writer(video,fps=25,quality=8,macro_block_size=8,
                                  output_params=['-movflags','+faststart'])
    try:
        for i in range(round(seconds/DT)):
            before=s.data.qpos.copy(),s.data.qvel.copy()
            angle,velocity=s.observation()
            target=expert(angle,velocity,s.t) if active else s.base
            if s.released is not None:
                if landed is None and s.data.xpos[s.trunk][2]<.12:landed=s.t
                if landed is None:target=brace
                else:target=tuck if (s.t-landed)%1.6<.7 else drive
            if not np.array_equal(before[0],s.data.qpos) or not np.array_equal(before[1],s.data.qvel):
                raise RuntimeError('Expert edited simulator state')
            s.step(target)
            angle,velocity=s.observation()
            history.append([s.t,angle,velocity,s.work,s.min_up,*s.data.qpos[s.qidx]])
            widest=max(widest,abs(angle))
            if renderer is not None and s.t>=next_frame-1e-9:
                now=speed_at(video_speed,s.t)
                next_frame=s.t+now/25.
                cam=mujoco.MjvCamera()
                close=min(np.clip((s.t-15)/4,0,1),np.clip((seconds-5-s.t)/4,0,1))
                # Pull back as the arc grows, or the duck leaves the frame. At
                # 50 deg the seat covers 0.74 m, which needs about 1.1 m of
                # camera distance, so 1.55 leaves a margin without going small.
                wide=float(np.clip(widest/.87,0,1))
                cam.lookat[:]=[0,0,.40-.035*close+.04*wide]
                cam.distance=1.10-.20*close+.45*wide;cam.azimuth=150;cam.elevation=-12
                if s.released is not None:
                    # Follow the duck down, then drop to a low side-on view. The
                    # duck lands right under the seat, so a high angle puts the
                    # seat on top of it and it reads as standing on its head.
                    near=float(np.clip((s.t-s.released)/1.5,0,1))
                    cam.lookat[:]=(1-near)*cam.lookat+near*(s.data.xpos[s.trunk]+np.array([0,0,.06]))
                    cam.distance=cam.distance*(1-near)+.80*near
                    cam.elevation=-12+8*near
                    cam.azimuth=150-40*near
                renderer.update_scene(s.data,camera=cam);frame=renderer.render()
                text=fixed_label if fixed_label else (('%g'%now)+'x speed' if now!=1 else 'real time')
                writer.append_data(_label(frame,text))
    except RuntimeError as exc:failure=str(exc)
    finally:
        if writer is not None:writer.close();renderer.close()
    a=np.array(history)
    cycles=[]
    if len(a)>2:
        extrema=np.where(np.diff(np.sign(a[:,2]))!=0)[0]+1
        for i,j in zip(extrema[:-1],extrema[1:]):
            if a[j,0]>seconds-20 and a[j,0]-a[i,0]>.3:
                cycles.append(float(np.rad2deg(abs(a[j,1]-a[i,1])/2)))
    tail=a[a[:,0]>max(2,seconds-20)] if len(a) else a
    amplitude=float(np.rad2deg(np.ptp(tail[:,1])/2)) if len(tail) else 0.
    if failure is None and active and not (len(cycles)>=10 and min(cycles)>17 and max(cycles)<28):
        failure='Sustained amplitude criterion not reached within the evaluation duration'
    passive_id=s.id('JOINT','passive_swing')
    passive_actuators=int(np.sum(s.model.actuator_trnid[:,0]==passive_id))
    ranges={}
    if len(a)>2:
        window=a[a[:,0]>max(2,seconds-20)]
        for k,name in enumerate(s.names):
            ranges[name.removeprefix('he_')]=float(np.rad2deg(np.ptp(window[:,5+k])))
    report=dict(success=failure is None,
                failure=failure,mode='pumping' if active else 'fixed_pose',seconds=round(s.t,3),environment=environment,
                last_20s_amplitude_deg=amplitude,half_cycle_amplitudes_deg=cycles,
                minimum_seated_upright=s.min_up,maximum_contact_penetration_mm=1000*s.max_penetration,
                maximum_servo_torque_nm=s.max_torque,servo_torque_limit_nm=s.torque_limit,
                net_servo_work_j=s.work,passive_swing_actuators=passive_actuators,root_welds=s.model.neq,mocap_bodies=s.model.nmocap,
                joint_range_last_20s_deg={k:round(v,1) for k,v in ranges.items() if v>=0.5},
                expert_settings=expert_kw or {},
                controller='phase feedback expert, BAM XL330 M6, 50 Hz',physics_timestep=s.model.opt.timestep)
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(trace_path,trace=a,columns=np.array(['time','angle','angular_velocity','servo_work','minimum_upright',*s.names]))
    return report



def suite():
    """A longer, common duration checks eventual amplitude at low battery."""
    from concurrent.futures import ProcessPoolExecutor
    cases=[dict(active=False),dict(),dict(initial_angle=3),dict(initial_angle=-3),
           dict(length=.42),dict(length=.56),dict(voltage=6.5),dict(damping=.004)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(_suite_case,cases))
    return dict(success=all(r['success'] for r in results),duration_per_case=90,cases=results)


def _suite_case(case):
    return evaluate(seconds=90,**case)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite',action='store_true')
    ap.add_argument('--seconds',type=float,default=60.)
    ap.add_argument('--video');ap.add_argument('--passive',action='store_true')
    ap.add_argument('--video-speed',type=float,default=1.,help='2.5 makes a 300 s run a 120 s file')
    ap.add_argument('--video-label',default=None,help='text on the frame; defaults to the speed')
    ap.add_argument('--dismantle-at',type=float,default=None,
                    help='seconds after which the seat drops off its mount and the duck falls')
    ap.add_argument('--target-degrees',type=float,default=None,
                    help='swing amplitude to aim for; above about 50 the duck cannot reach it')
    ap.add_argument('--report',default=paths.video('swing','swing_report.json'))
    ap.add_argument('--trace',default=paths.video('swing','swing_trace.npz'))
    ap.add_argument('--length',type=float,default=.48);ap.add_argument('--damping',type=float,default=.002)
    ap.add_argument('--voltage',type=float,default=7.4);ap.add_argument('--initial-angle',type=float,default=0.)
    args=ap.parse_args()
    kw=dict(target_degrees=args.target_degrees) if args.target_degrees else None
    report=suite() if args.suite else evaluate(seconds=args.seconds,active=not args.passive,video=args.video,trace_path=args.trace,
                    expert_kw=kw,video_speed=args.video_speed,video_label=args.video_label,
                                       length=args.length,damping=args.damping,voltage=args.voltage,initial_angle=args.initial_angle,
                    **({'dismantle_at':args.dismantle_at} if args.dismantle_at else {}))
    Path(args.report).parent.mkdir(parents=True,exist_ok=True);Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
