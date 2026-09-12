"""Fast and Little crawls through the same low, taut training net.

The existing gait parameters, motors and ground are reused unchanged. The net
is a static sagging rope grid with solid capsules, not a deformable cloth model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from microduck_lab import paths
from microduck_lab.tasks.crawl import crawl, expert, baby

ENTRY_X, EXIT_X = -.30, -2.10
HALF_WIDTH = .32
EDGE_HEIGHT, SAG = .172, .014
ROPE_RADIUS = .0014
TIMBER_RADIUS = .012
NET_LOWEST = EDGE_HEIGHT - SAG - .0019  # knots are slightly thicker than rope
GAITS = {'fast': (expert.gait, 'crawl_gait.npy', 20., 'Fast crawl'),
         'little': (baby.gait, 'crawl_baby_gait.npy', 20., 'Little crawl')}


def net_xml():
    """Wooden perimeter and a 25 mm grid of dark, knotted rope."""
    parts=[]
    def capsule(name,a,b,radius,rgba):
        points=' '.join(f'{v:.7f}' for v in [*a,*b])
        parts.append(f'<geom name="{name}" type="capsule" fromto="{points}" size="{radius}" rgba="{rgba}"/>')
    wood='.31 .26 .16 1'
    bark='.19 .17 .11 1'
    for x in (ENTRY_X,(ENTRY_X+EXIT_X)/2,EXIT_X):
        for y in (-HALF_WIDTH,HALF_WIDTH):
            key=f'{x}_{y}'
            capsule('net_post_'+key,(x,y,.012),(x,y,EDGE_HEIGHT+.010),.017,wood)
            parts.append(f'<geom name="net_post_cap_{key}" type="cylinder" pos="{x} {y} {EDGE_HEIGHT+.027}" size=".014 .0008" rgba=".48 .39 .23 1"/>')
        if x in (ENTRY_X,EXIT_X):
            capsule('net_end_'+str(x),(x,-HALF_WIDTH,EDGE_HEIGHT),(x,HALF_WIDTH,EDGE_HEIGHT),TIMBER_RADIUS,wood)
    for y in (-HALF_WIDTH,HALF_WIDTH):
        capsule('net_side_'+str(y),(EXIT_X,y,EDGE_HEIGHT),(ENTRY_X,y,EDGE_HEIGHT),TIMBER_RADIUS,wood)
        # A narrow visible grain line along each timber beam.
        capsule('net_grain_'+str(y),(EXIT_X,y,EDGE_HEIGHT+.011),(ENTRY_X,y,EDGE_HEIGHT+.011),.001,bark)
    xs=np.linspace(EXIT_X,ENTRY_X,round((ENTRY_X-EXIT_X)/.025)+1)
    ys=np.linspace(-HALF_WIDTH,HALF_WIDTH,27)
    def point(x,y):
        u=(x-EXIT_X)/(ENTRY_X-EXIT_X)
        v=(y+HALF_WIDTH)/(2*HALF_WIDTH)
        return (x,y,EDGE_HEIGHT-SAG*np.sin(np.pi*u)*np.sin(np.pi*v))
    rope='.13 .16 .085 1'
    for i,x in enumerate(xs):
        for j in range(len(ys)-1):
            capsule(f'net_cross_{i}_{j}',point(x,ys[j]),point(x,ys[j+1]),ROPE_RADIUS,rope)
    for j,y in enumerate(ys):
        for i in range(len(xs)-1):
            capsule(f'net_long_{i}_{j}',point(xs[i],y),point(xs[i+1],y),ROPE_RADIUS,rope)
    for i in range(0,len(xs),2):
        for j in range(0,len(ys),2):
            pos=' '.join(map(str,point(xs[i],ys[j])))
            parts.append(f'<geom name="net_knot_{i}_{j}" type="sphere" pos="{pos}" size=".0019" rgba="{rope}"/>')
    return '<body name="training_net">'+''.join(parts)+'</body>'


class NetCrawl(crawl.Crawl):
    def __init__(self):
        xml=crawl.world_xml().replace('</worldbody>',net_xml()+'</worldbody>')
        # Scope scene substitution to construction; keep the original Crawl file
        # and all of its BAM/contact setup unchanged. Construct on one thread.
        with patch.object(crawl,'world_xml',return_value=xml):
            super().__init__()
        m=self.model
        self.robot_geoms=np.array([g for g in range(m.ngeom)
            if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or '').startswith('he_')
            and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH])
        signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]).T
        boxes=m.geom_aabb[self.robot_geoms]
        self.corners=boxes[:,:3,None]+boxes[:,3:,None]*signs[None,:,:]
        net_body=self.id('BODY','training_net')
        self.net_geoms=set(np.flatnonzero(m.geom_bodyid==net_body).tolist())
        assert all(m.geom_contype[g] and m.geom_conaffinity[g] for g in self.net_geoms)

    def bounds(self):
        d=self.data;ids=self.robot_geoms
        xyz=d.geom_xpos[ids,:,None]+np.einsum('nij,njk->nik',d.geom_xmat[ids].reshape(-1,3,3),self.corners)
        return xyz.min(axis=(0,2)),xyz.max(axis=(0,2))


def evaluate(kind,video=None,report=None,trace=None):
    gait,param_file,seconds,label=GAITS[kind]
    steps=round(seconds/crawl.DT)
    p=np.load(paths.REPO/param_file)
    c=NetCrawl();m,d=c.model,c.data
    writer=renderer=None
    entered_at=cleared_at=None
    min_clearance=float('inf');contacts=0;rows=[]
    if video:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True,exist_ok=True)
        renderer=mujoco.Renderer(m,height=720,width=1280,max_geom=8000)
        writer=imageio.get_writer(str(video),fps=25,codec='libx264',quality=None,
            output_params=['-crf','23','-preset','medium','-movflags','+faststart'])
    try:
        for i in range(steps):
            c.step(gait(c,p,c.t))
            low,high=c.bounds()
            overlaps=low[0]<=ENTRY_X and high[0]>=EXIT_X
            if overlaps:
                if entered_at is None:entered_at=c.t
                min_clearance=min(min_clearance,NET_LOWEST-high[2])
                if low[1]<-HALF_WIDTH+TIMBER_RADIUS or high[1]>HALF_WIDTH-TIMBER_RADIUS:
                    raise RuntimeError('Duck left the corridor under the net')
            if high[0]<EXIT_X-TIMBER_RADIUS and cleared_at is None:cleared_at=c.t
            contacts+=sum((int(contact.geom1) in c.net_geoms) != (int(contact.geom2) in c.net_geoms)
                          for contact in d.contact)
            if d.xpos[c.trunk,2]>.2 or abs(d.xmat[c.trunk,8])>.7:
                raise RuntimeError('Duck stopped crawling')
            rows.append([c.t,*d.xpos[c.trunk],*low,*high])
            if renderer is not None and i%2==1:
                cam=mujoco.MjvCamera()
                cam.lookat[:]=[d.xpos[c.trunk,0]-.20,0,.085]
                cam.distance=1.35;cam.azimuth=105;cam.elevation=-12
                renderer.update_scene(d,camera=cam)
                mm=1000*c.travelled()[0]
                title=f'{label} | {c.t:4.1f} s | {mm:.0f} mm | {mm/c.t:.0f} mm/s'
                subtitle='Under the net | Stripes: 25 cm | real time'
                frame=expert._label(renderer.render(),title+'\n'+subtitle)
                writer.append_data(frame)
    finally:
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
    a=np.array(rows)
    final_clear=bool(a[-1,7]<EXIT_X-TIMBER_RADIUS)
    final_under=bool(a[-1,7]<ENTRY_X-TIMBER_RADIUS and a[-1,4]>EXIT_X+TIMBER_RADIUS)
    success=entered_at is not None and (final_clear if kind=='fast' else final_under)
    result=dict(gait=label,seconds=seconds,video_seconds=seconds,playback_speed=1.,success=success,
                final_whole_body_under_net=final_under,
                final_whole_body_clear=final_clear,
                entered_at_seconds=entered_at,whole_body_cleared_at_seconds=cleared_at,
                distance_mm=1000*c.travelled()[0],speed_mm_s=1000*c.travelled()[0]/seconds,
                trunk_height_mm=dict(mean=1000*float(a[:,3].mean()),max=1000*float(a[:,3].max())),
                maximum_full_body_height_bound_mm=1000*float(a[:,9].max()),
                minimum_conservative_rope_clearance_mm=1000*min_clearance,
                observed_net_contacts=contacts,net_length_mm=1000*(ENTRY_X-EXIT_X),
                net_width_mm=2000*HALF_WIDTH,net_lowest_rope_surface_mm=1000*NET_LOWEST,
                net_model='static sagging grid; every rope, knot and timber collides',
                gait_parameters_file=param_file,maximum_servo_torque_nm=c.max_torque,
                camera='same following camera rule for both gaits')
    if report:
        Path(report).parent.mkdir(parents=True,exist_ok=True)
        Path(report).write_text(json.dumps(result,indent=2)+'\n')
    if trace:np.savez_compressed(trace,trajectory=a)
    print(json.dumps(result,indent=2),flush=True)
    if not result['success']:raise RuntimeError(f'{label} did not reach its intended final position')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gait',choices=GAITS,required=True)
    ap.add_argument('--video');ap.add_argument('--report');ap.add_argument('--trace')
    a=ap.parse_args()
    evaluate(a.gait,a.video,a.report,a.trace)

if __name__=='__main__':main()
