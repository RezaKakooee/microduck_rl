"""A crawl gait: each leg joint swings as a sine, the two legs half a cycle apart.

The pose cycle the duck used after falling off the swing looked like crawling but
went nowhere: 83 mm out, then back, 1 mm/s net. These numbers were searched for
distance travelled instead.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from microduck_lab import paths
from microduck_lab.tasks.crawl.crawl import Crawl,DT

JOINTS=('hip_pitch','knee','hip_roll','ankle')      # each gets offset, amplitude, phase
NP=len(JOINTS)*3+3                                  # + frequency, neck, head


def gait(c,p,t):
    """Joint targets at time `t`. Left and right run half a cycle apart."""
    q=c.base.copy()
    f=p[-3]
    for k,name in enumerate(JOINTS):
        off,amp,ph=p[3*k:3*k+3]
        wave=amp*np.sin(2*np.pi*f*t+ph)
        q[c.ai['left_'+name]]=off+wave
        q[c.ai['right_'+name]]=-(off-wave)          # mirrored, and half a cycle later
    q[c.ai['neck_pitch']]=p[-2];q[c.ai['head_pitch']]=p[-1]
    return q


def _label(frame,text):
    """Write the distance on the frame. Grass alone does not show travel."""
    from PIL import Image,ImageDraw,ImageFont
    image=Image.fromarray(frame);draw=ImageDraw.Draw(image)
    size=max(20,image.height//22)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',size)
    except OSError:font=ImageFont.load_default()
    x=y=max(12,image.height//30)
    for dx,dy in ((-2,0),(2,0),(0,-2),(0,2)):
        draw.text((x+dx,y+dy),text,font=font,fill=(20,28,36))
    draw.text((x,y),text,font=font,fill=(255,255,255))
    return np.asarray(image)


def rollout(p,seconds=10.,voltage=7.4):
    c=Crawl(voltage=voltage)
    try:
        for _ in range(int(seconds/DT)):c.step(gait(c,p,c.t))
    except RuntimeError as exc:return c,str(exc)
    return c,None


def evaluate(params,seconds=20.,video=None,voltage=7.4):
    c=Crawl(voltage=voltage);writer=renderer=None
    if video:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True,exist_ok=True)
        renderer=mujoco.Renderer(c.model,height=720,width=1280)
        writer=imageio.get_writer(video,fps=25,quality=8,macro_block_size=8,
                                  output_params=['-movflags','+faststart'])
    failure=None;track=[]
    try:
        for i in range(int(seconds/DT)):
            c.step(gait(c,params,c.t))
            track.append([c.t,*c.data.xpos[c.trunk]])
            if renderer is not None and i%2==0:
                cam=mujoco.MjvCamera()
                # Side on, and following a little behind, so the duck crosses the
                # stripes instead of sitting still in the middle of the frame.
                cam.lookat[:]=c.data.xpos[c.trunk]+np.array([.10,0,.02])
                cam.distance=.75;cam.azimuth=90;cam.elevation=-8
                renderer.update_scene(c.data,camera=cam)
                far=c.travelled()[0]
                writer.append_data(_label(renderer.render(),'%.2f m'%far))
    except RuntimeError as exc:failure=str(exc)
    finally:
        if writer is not None:writer.close();renderer.close()
    far,dx,dy=c.travelled()
    a=np.array(track)
    return dict(success=failure is None and far>.05,failure=failure,
                seconds=seconds,travelled_mm=1000*far,forward_mm=1000*dx,sideways_mm=1000*dy,
                speed_mm_s=1000*far/seconds,
                maximum_servo_torque_nm=c.max_torque,servo_torque_limit_nm=c.torque_limit,
                net_servo_work_j=c.work,
                trunk_height_mm=dict(mean=1000*float(a[:,3].mean()),max=1000*float(a[:,3].max())),
                gait=dict(zip([f'{n}_{w}' for n in JOINTS for w in ('offset','amplitude','phase')]
                              +['frequency_hz','neck_pitch','head_pitch'],
                              [round(float(v),4) for v in params])),
                controller='open-loop sine gait, BAM XL330 M6, 50 Hz',
                physics_timestep=.002)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=20.)
    ap.add_argument('--video');ap.add_argument('--voltage',type=float,default=7.4)
    ap.add_argument('--params',default=str(paths.model('..','crawl_gait.npy')))
    ap.add_argument('--report',default=paths.video('crawl','crawl_report.json'))
    a=ap.parse_args()
    p=np.load(a.params)
    r=evaluate(p,seconds=a.seconds,video=a.video,voltage=a.voltage)
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    json.dump(r,open(a.report,'w'),indent=1);print(json.dumps(r,indent=1))


if __name__=='__main__':main()
