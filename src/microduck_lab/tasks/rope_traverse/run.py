"""Film real rope-contact trials; there are no robot welds or support forces."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import zipfile
import mujoco
import numpy as np
from microduck_lab.tasks.rope_traverse.scene import Scene,build
from microduck_lab.tasks.rope_traverse.fit import place,I,balance_about_rope,yaw_with_aligned_head

ROOT=Path(__file__).resolve().parent
OUTPUT=Path('videos/rope_traverse')


def reserve():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for n in range(1,100000):
        p=OUTPUT/f'rope_try{n}'
        if any(p.with_suffix(ext).exists() for ext in ('.mp4','.json','.npz')):continue
        try:
            with p.with_suffix('.lock').open('x'):pass
            return p
        except FileExistsError:continue
    raise RuntimeError('No unused trial number')


def contacts(w):
    m,d=w.model,w.data;k=w.ducks['she'];jaw=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'she_mouth_jaw')
    feet=[int(m.site_bodyid[s]) for s in k.foot_sites]
    loads=np.zeros(6);f=np.zeros(6)
    floor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    for i,c in enumerate(d.contact):
        if c.geom1 in w.rope_geoms:g=c.geom2
        elif c.geom2 in w.rope_geoms:g=c.geom1
        elif floor in (c.geom1,c.geom2) and (c.geom1 in k.solid or c.geom2 in k.solid):
            mujoco.mj_contactForce(m,d,i,f);loads[5]+=max(0.,f[0]);continue
        else:continue
        if g not in k.solid:continue
        body=int(m.geom_bodyid[g]);index=0 if body==k.head else 1 if body==jaw else 2+feet.index(body) if body in feet else 4
        mujoco.mj_contactForce(m,d,i,f);loads[index]+=max(0.,f[0])
    return loads


def command(home,t,mode,amplitude):
    q=home.copy();mouth=0.;phase='grip and hold'
    if mode=='traverse' and t>1.:
        phase_index=int((t-1.)/.6)%4
        u=((t-1.)%.6)/.6
        lift=np.sin(np.pi*u)*amplitude
        if phase_index==0:
            mouth=.5 if u<.65 else 0.;q[I['head_roll']]+=lift;phase='reach with mouth'
        elif phase_index==1:
            q[I['left_knee']]+=lift;q[I['left_hip_roll']]+=lift;phase='move left foot'
        elif phase_index==2:
            q[I['right_knee']]-=lift;q[I['right_hip_roll']]+=lift;phase='move right foot'
        else:phase='settle grips'
    return q,mouth,phase


def summarize(a,height):
    # time,trunk xyz,robot lowest z,head/jaw/left/right/other rope N,floor N,mouth torque
    supported=(a[:,0]>.5)&(a[:,4]>.05)&(a[:,3]<height-.035)&(a[:,10]<.5)&(a[:,9]<.5)
    grips=np.c_[a[:,6]>.1,a[:,7]>.1,a[:,8]>.1]
    gripping=supported&np.all(grips,axis=1)
    supported=supported&(grips.sum(axis=1)>=2)
    longest=current=0
    for good in gripping:
        current=current+1 if good else 0;longest=max(longest,current)
    support_longest=support_current=0
    for good in supported:
        support_current=support_current+1 if good else 0;support_longest=max(support_longest,support_current)
    after=a[a[:,0]>=1.]
    displacement=float(after[-1,1]-after[0,1]) if len(after) else 0.
    return dict(sustained_support_s=support_longest*.02,sustained_grip_s=longest*.02,hang_pass=longest*.02>=2.,
                displacement_x_m=displacement,peak_jaw_torque_nm=float(abs(a[:,11]).max()),
                floor_contact=bool(np.any(a[:,10]>.5)),
                traverse_pass=bool(support_longest*.02>=2. and displacement>=.15 and gripping[-1]))


def run(candidate,mode='hold',seconds=6.,pinch=0.,jaw_torque=.25):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('MUJOCO_GL')!='egl':
        raise RuntimeError('Use the owned video.sbatch on a GPU node')
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    config=Scene(jaw_torque=jaw_torque);w=build(config);k=w.ducks['she'];m,d=w.model,w.data
    q,pos=place(w,np.array(candidate['x']),bite=candidate.get('bite',[-.0175,0.,-.0725]),mouth_open=candidate.get('mouth_open',.5));mujoco.mj_forward(m,d)
    balance_angle=balance_about_rope(w) if candidate.get('balance_about_rope',False) else 0.
    yaw_setup=None
    if candidate.get("yaw_offset_deg",0):
        q,yaw_setup=yaw_with_aligned_head(w,np.deg2rad(candidate["yaw_offset_deg"]),candidate.get("bite",[-.0175,0.,-.0725]))
    robot=set(k.solid)
    penetration=max([max(0.,-c.dist) for c in d.contact if
                     (c.geom1 in w.rope_geoms and c.geom2 in robot) or
                     (c.geom2 in w.rope_geoms and c.geom1 in robot)]+[0.])
    stem=reserve();sources={p:p.read_bytes() for p in ROOT.glob('*.py')}
    import inspect
    for cls in (type(w),type(k)):
        p=Path(inspect.getfile(cls));sources[p]=p.read_bytes()
    renderer=mujoco.Renderer(m,height=540,width=960);camera=mujoco.MjvCamera()
    camera.distance=1.35;camera.azimuth=105;camera.elevation=-16;camera.lookat[:]=[0,0,.31]
    writer=imageio.get_writer(str(stem.with_suffix('.mp4')),fps=25,codec='libx264',macro_block_size=2)
    font=ImageFont.truetype('DejaVuSans.ttf',18)
    pressure_joint=candidate.get('pressure_joint','hip_roll')
    if pressure_joint not in ('hip_roll','ankle'):raise ValueError('Unsupported pressure joint')
    q[I['left_'+pressure_joint]]+=pinch;q[I['right_'+pressure_joint]]-=pinch
    w.start();rows=[];states=[];error=None
    warnings=[];old_warning=mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(warnings.append)
    try:
        for tick in range(round(seconds/.02)):
            target,mouth,phase=command(q,w.t,mode,.16);k.hold(target);d.ctrl[k.mouth]=mouth
            if penetration>.002:
                phase='INVALID INITIAL OVERLAP - no dynamics run'
            else:w.step()
            mujoco.mj_forward(m,d);loads=contacts(w)
            rows.append([w.t,*k.pos(),k.points(k.solid)[:,2].min(),*loads,d.actuator_force[k.mouth]])
            if tick%2==1:
                states.append(d.qpos.copy());renderer.update_scene(d,camera=camera)
                frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,960,80),fill=(18,22,28))
                draw.text((12,5),f'Rope prototype | {mode} | {phase}',font=font,fill='white')
                draw.text((12,29),f'Contact only | jaw limit {jaw_torque:.2f} Nm (provisional) | time {w.t:.2f}s',font=font,fill='white')
                draw.text((12,54),f'Jaw contact {loads[1]:.1f} N | feet {loads[2]:.1f}/{loads[3]:.1f} N | floor contact: {loads[5]>.5}',font=font,fill='white')
                writer.append_data(np.asarray(frame))
    except Exception as e:error=repr(e);raise
    finally:
        mujoco.set_mju_user_warning(old_warning)
        writer.close();renderer.close()
        np.savez_compressed(stem.with_suffix('.npz'),trace=rows,qpos=states)
        result=dict(candidate=candidate,yaw_setup=yaw_setup,balance_angle=balance_angle,scene=asdict(config),mode=mode,pinch=pinch,error=error,
                    initial_penetration_m=penetration,valid_initial_pose=penetration<=.002,
                    warning_messages=warnings,physics_valid=error is None and not warnings and penetration<=.002,
                    video=str(stem.with_suffix('.mp4')),jaw_model='bounded position servo; torque calibration provisional',
                    sources={str(p):hashlib.sha256(b).hexdigest() for p,b in sources.items()})
        if rows:result.update(summarize(np.array(rows),config.height))
        if not result['physics_valid']:result.update(hang_pass=False,traverse_pass=False)
        with zipfile.ZipFile(stem.with_suffix('.sources.zip'),'w',zipfile.ZIP_DEFLATED) as z:
            for p,b in sources.items():z.writestr(str(p.relative_to(Path.cwd())),b)
        stem.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','candidate')}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--poses',type=Path,required=True)
    p.add_argument('--count',type=int,default=3);p.add_argument('--mode',choices=['hold','traverse'],default='hold')
    p.add_argument('--seconds',type=float,default=6.)
    p.add_argument('--pinches',type=float,nargs='+',default=[0.,.08,-.08])
    p.add_argument('--jaw-torque',type=float,default=.25)
    args=p.parse_args()
    cases=json.loads(args.poses.read_text())[:args.count]
    for c in cases:
        for pinch in args.pinches:run(c,args.mode,args.seconds,pinch,args.jaw_torque)


if __name__=='__main__':main()
