"""Paint a flower using measured mouth-brush contact, with a live blank canvas."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from microduck_lab import paths
from microduck_lab.tasks.painting.painting import Painting, DT, CANVAS_X, CANVAS_Z, HALF_W, HALF_H, COLORS, WELLS


def strokes():
    """Paths in canvas coordinates (horizontal y, vertical z), in metres."""
    center=np.array([0.,.325])
    stem=np.column_stack([.003*np.sin(np.linspace(0,np.pi,65)),np.linspace(.264,.325,65)])
    yield 'green','A little stem',stem
    for sign,z in [(-1,.279),(1,.290)]:
        u=np.linspace(0,1,90)
        # A pointed leaf: two curved sides joined at the stem and leaf tip.
        leaf=np.array([[.002,z]])+np.column_stack([sign*.024*np.sin(np.pi*u),.018*np.sin(np.pi*u)+.006*np.sin(2*np.pi*u)])
        yield 'green','And two leaves',leaf
    for k in range(8):
        angle=2*np.pi*k/8
        radial=np.array([np.cos(angle),np.sin(angle)])
        cross=np.array([-radial[1],radial[0]])
        t=np.linspace(np.pi,3*np.pi,120)
        p=center+(.017+.012*np.cos(t))[:,None]*radial+(.0065*np.sin(t))[:,None]*cross
        yield 'pink',f'Petal {k+1} of 8',p
    t=np.linspace(0,7*np.pi,220);r=np.linspace(.0002,.008,220)
    yield 'yellow','A sunny yellow heart',center+np.column_stack([r*np.cos(t),r*np.sin(t)])


def schedule(start):
    actions=[];at=start.copy();last_color=None
    def add(points,seconds,mode,label,color=None):
        nonlocal at
        points=np.asarray(points)
        actions.append(dict(points=points,seconds=seconds,mode=mode,label=label,color=color))
        at=points[-1].copy()
    def move(target,seconds,label):add([at,target],seconds,'move',label)
    move(start,2.,'A blank canvas. A tiny painter.')
    for color,label,p in strokes():
        if color!=last_color:
            move(np.array([.147,at[1],max(at[2],.275)]),.8,'Choosing '+color)
            well=WELLS[color]+[0,0,.003]
            above=well+[0,0,.027]
            move(above,1.2,'Choosing '+color)
            add([at,well],1.,'dip','A dip of '+color,color)
            add([well,well],1.2,'dip','A dip of '+color,color)
            move(above,.7,'Ready to paint')
            last_color=color
        first=np.array([CANVAS_X,*p[0]])
        hover=first.copy();hover[0]=.147
        move(np.array([.147,at[1],at[2]]),.65,'Lifting the brush')
        move(hover,.9,label)
        move(first,.75,label)
        add([first,first],.4,'draw',label,color)
        points=np.column_stack([np.full(len(p),CANVAS_X),p])
        length=np.linalg.norm(np.diff(points,axis=0),axis=1).sum()
        add(points,max(1.2,length/.012),'draw',label,color)
        move(np.array([.147,at[1],at[2]]),.65,'Lifting the brush')
    move(np.array([.147,-.035,.310]),1.5,'A flower, for you.')
    add([at,at],5.,'move','A flower, for you.')
    return actions


class Canvas:
    def __init__(self):
        self.image=Image.new('RGB',(768,768),(250,247,235))
        self.draw=ImageDraw.Draw(self.image)
        self.color=None;self.previous=None;self.samples=0;self.contacts={k:0 for k in COLORS}
        self.loaded=[];self.max_gap=0.

    def update(self,tip,action):
        if action['mode']=='dip':
            color=action['color']
            if np.linalg.norm(tip-(WELLS[color]+[0,0,.003]))<.014:
                if self.color!=color:self.loaded.append(color)
                self.color=color
        on=(abs(tip[0]-CANVAS_X)<.0035 and abs(tip[1])<HALF_W
            and abs(tip[2]-CANVAS_Z)<HALF_H)
        if not on or self.color is None:
            self.previous=None;return False
        point=((tip[1]+HALF_W)/(2*HALF_W)*767,
               (CANVAS_Z+HALF_H-tip[2])/(2*HALF_H)*767)
        rgb=tuple(round(255*v) for v in COLORS[self.color])
        radius=6
        if self.previous is not None:
            self.draw.line([self.previous,point],fill=rgb,width=radius*2)
        self.draw.ellipse((point[0]-radius,point[1]-radius,point[0]+radius,point[1]+radius),fill=rgb)
        self.previous=point;self.samples+=1;self.contacts[self.color]+=1
        self.max_gap=max(self.max_gap,float(abs(tip[0]-CANVAS_X)))
        return True

    def upload(self,s,renderer):
        m=s.model;tid=s.id('TEXTURE','paper');adr=m.tex_adr[tid]
        # MuJoCo texture row zero is the bottom edge of the canvas mesh.
        pixels=np.asarray(self.image)[::-1].copy().ravel()
        m.tex_data[adr:adr+len(pixels)]=pixels
        mujoco.mjr_uploadTexture(m,renderer._mjr_context,tid)


def font(size):
    try:return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
    except OSError:return ImageFont.load_default()


def frame(s,renderer,canvas,label):
    cam=mujoco.MjvCamera();cam.lookat[:]=[.085,0,.245]
    cam.distance=.67;cam.azimuth=55;cam.elevation=-18
    renderer.update_scene(s.data,camera=cam)
    im=Image.fromarray(renderer.render());d=ImageDraw.Draw(im)
    d.rounded_rectangle((26,24,526,98),radius=14,fill=(250,247,235))
    d.text((44,31),'THE LITTLE PAINTER',font=font(27),fill=(46,67,58))
    d.text((45,66),label,font=font(17),fill=(83,95,79))
    # Live view of this very canvas, starting blank, for readable small strokes.
    x,y,w,h=994,170,246,284
    d.rounded_rectangle((x-12,y-38,x+w+12,y+h+14),radius=12,fill=(250,247,235))
    d.text((x+6,y-30),'On the canvas',font=font(19),fill=(61,77,64))
    im.paste(canvas.image.resize((w,h),Image.Resampling.LANCZOS),(x,y))
    return np.asarray(im)


def evaluate(video=None,report=None,trace=None,still=None):
    s=Painting();canvas=Canvas();actions=schedule(s.data.site_xpos[s.tip])
    rows=[];stats=[];writer=renderer=None
    if video:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True,exist_ok=True)
        renderer=mujoco.Renderer(s.model,height=720,width=1280)
        writer=imageio.get_writer(str(video),fps=25,codec='libx264',quality=None,
                                  output_params=['-crf','24','-pix_fmt','yuv420p','-movflags','+faststart'])
    frame_id=0
    try:
        for a in actions:
            count=max(1,round(a['seconds']/DT));p=a['points']
            lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
            if lengths[-1]<1e-8:lengths=np.linspace(0,1,len(p))
            on=0;errors=[]
            for k in range(count):
                u=(k+1)/count
                # Smooth travel; constant-speed brush strokes.
                if a['mode']!='draw':u=u*u*(3-2*u)
                target=np.array([np.interp(u*lengths[-1],lengths,p[:,j]) for j in range(3)])
                s.step(s.command(target))
                tip=s.data.site_xpos[s.tip].copy()
                contact=canvas.update(tip,a);on+=int(contact)
                errors.append(float(np.linalg.norm(tip-target)))
                rows.append([s.t,*tip,*target,int(contact)])
                if renderer is not None and frame_id%2==1:
                    canvas.upload(s,renderer)
                    if canvas.color:
                        s.model.geom_rgba[s.id('GEOM','he_bristles'),:3]=COLORS[canvas.color]
                    writer.append_data(frame(s,renderer,canvas,a['label']))
                frame_id+=1
            stats.append(dict(label=a['label'],mode=a['mode'],color=a['color'],
                              seconds=count*DT,paint_samples=on,mean_error_mm=1000*float(np.mean(errors))))
            if a['mode'] in ('dip','draw'):
                print(json.dumps(stats[-1]),flush=True)
    finally:
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
    output=Path(still or paths.video('painting','duck_flower.png'))
    output.parent.mkdir(parents=True,exist_ok=True);canvas.image.save(output)
    result=dict(seconds=s.t,paint_samples=canvas.samples,paint_samples_by_color=canvas.contacts,
                colors_loaded=canvas.loaded,maximum_paint_tip_gap_mm=1000*canvas.max_gap,
                minimum_trunk_upright=s.min_up,maximum_servo_torque_nm=s.max_torque,
                servo_torque_limit_nm=s.torque_limit,root_drift_mm=1000*float(np.linalg.norm(s.data.xpos[s.trunk]-s.start)),
                strokes=stats,brush_grip='rigid attachment at mouth, placed before simulation',
                controller='scripted Cartesian paths, scratch-state IK, BAM XL330 M6 motors',
                paint='deposited only from measured brush tip within 3.5 mm of canvas')
    if report:Path(report).write_text(json.dumps(result,indent=2)+'\n')
    if trace:np.savez_compressed(trace,trajectory=rows)
    print(json.dumps({k:v for k,v in result.items() if k!='strokes'},indent=2),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video');ap.add_argument('--report',default=paths.video('painting','painting_report.json'))
    ap.add_argument('--trace',default=paths.video('painting','painting_trace.npz'))
    ap.add_argument('--still',default=paths.video('painting','duck_flower.png'))
    a=ap.parse_args();evaluate(**vars(a))

if __name__=='__main__':main()
