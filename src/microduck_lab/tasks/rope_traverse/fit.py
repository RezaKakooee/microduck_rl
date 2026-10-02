"""Offline inverse kinematics for starting a contact-only grip test."""
import json
import argparse
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from microduck_lab.tasks.rope_traverse.scene import build,Scene
from microduck_lab.tasks.common.world import SERVOS

I={n:i for i,n in enumerate(SERVOS)}
BITE=np.array([-.013716432289324856,0.,-.08022895677252034])
MOUTH_OPEN=.25001156264105406
HOME=np.array([0,-.0873,-.4579,-.0049,.453,.3491,.3491,0,0,0,.0873,.4579,.0049,-.453])


def targets(x):
    q=HOME.copy()
    for name,value in zip(('hip_pitch','knee','ankle','hip_roll'),x[1:5]):
        q[I['left_'+name]]=value;q[I['right_'+name]]=-value
    for name,value in zip(('neck_pitch','head_pitch','head_yaw','head_roll'),x[5:]):q[I[name]]=value
    return q


def place(w,x,mouth_x=0.,bite=None,mouth_open=MOUTH_OPEN):
    duck=w.ducks['she'];q=targets(x)
    if w.running: raise RuntimeError('IK placement is setup-only')
    w.data.qpos[duck.adr:duck.adr+3]=[0,0,.4]
    w.data.qpos[duck.adr+3:duck.adr+7]=np.array([np.cos(x[0]/2),-np.sin(x[0]/2),np.sin(x[0]/2),np.cos(x[0]/2)])/np.sqrt(2)
    w.data.qpos[duck.qidx]=q
    jid=mujoco.mj_name2id(w.model,mujoco.mjtObj.mjOBJ_JOINT,'she_mouth')
    w.data.qpos[w.model.jnt_qposadr[jid]]=mouth_open
    mujoco.mj_kinematics(w.model,w.data)
    R=w.data.xmat[duck.head].reshape(3,3)
    bite=w.data.xpos[duck.head]+R@(BITE if bite is None else np.asarray(bite))
    c=w.rope_config;target=np.array([mouth_x,0,c.height-c.sag*(1-(2*mouth_x/c.span)**2)])
    position=duck.pos()+target-bite
    w.data.qpos[duck.adr:duck.adr+3]=position
    duck.hold(q)
    mujoco.mj_kinematics(w.model,w.data)
    return q,position



def balance_about_rope(w):
    """Initial placement only: rotate the entire robot about the rope axis."""
    if w.running:raise RuntimeError('Balancing placement is setup-only')
    duck=w.ducks['she'];mujoco.mj_comPos(w.model,w.data)
    pivot=np.array([0.,0.,w.rope_config.height-w.rope_config.sag])
    v=duck.com()-pivot;angle=np.arctan2(-v[1],-v[2])
    c,s=np.cos(angle),np.sin(angle);R=np.array([[1,0,0],[0,c,-s],[0,s,c]])
    w.data.qpos[duck.adr:duck.adr+3]=pivot+R@(duck.pos()-pivot)
    original=w.data.qpos[duck.adr+3:duck.adr+7].copy();quat=np.zeros(4)
    mujoco.mju_mulQuat(quat,np.array([np.cos(angle/2),np.sin(angle/2),0,0]),original)
    w.data.qpos[duck.adr+3:duck.adr+7]=quat
    mujoco.mj_forward(w.model,w.data)
    return float(angle)


def yaw_with_aligned_head(w, angle, bite=BITE):
    """Setup only: turn the body, then fit neck joints to retain mouth orientation."""
    if w.running:raise RuntimeError('Yaw placement is setup-only')
    duck=w.ducks['she'];m,d=w.model,w.data
    mujoco.mj_forward(m,d)
    desired=d.xmat[duck.head].reshape(3,3).copy()
    pivot=d.xpos[duck.head]+desired@np.asarray(bite)
    c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    d.qpos[duck.adr:duck.adr+3]=pivot+rotation@(duck.pos()-pivot)
    original=d.qpos[duck.adr+3:duck.adr+7].copy();quat=np.zeros(4)
    mujoco.mju_mulQuat(quat,np.array([np.cos(angle/2),0,0,np.sin(angle/2)]),original)
    d.qpos[duck.adr+3:duck.adr+7]=quat
    names=('neck_pitch','head_pitch','head_yaw','head_roll')
    indices=[duck.qidx[I[n]] for n in names]
    joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'she_'+n) for n in names]
    limits=m.jnt_range[joints]
    initial=d.qpos[indices].copy()
    def error(values):
        d.qpos[indices]=values;mujoco.mj_kinematics(m,d)
        return (d.xmat[duck.head].reshape(3,3)-desired).ravel()
    results=[least_squares(error,np.clip(initial+offset,limits[:,0],limits[:,1]),
               bounds=(limits[:,0],limits[:,1]),max_nfev=200) for offset in
               (np.zeros(4),np.array([0,0,1.57,0]),np.array([0,0,-1.57,0]))]
    best=min(results,key=lambda r:np.linalg.norm(r.fun));error(best.x)
    current=d.xpos[duck.head]+d.xmat[duck.head].reshape(3,3)@np.asarray(bite)
    d.qpos[duck.adr:duck.adr+3]+=pivot-current
    q=d.qpos[duck.qidx].copy();duck.hold(q);mujoco.mj_forward(m,d)
    return q,dict(yaw_offset_deg=float(np.rad2deg(angle)),head_joints=dict(zip(names,best.x.tolist())),
                  head_orientation_error=float(np.linalg.norm(best.fun)))


def residual(w,x,foot_x=.06,foot_y=0.):
    q,pos=place(w,x);duck=w.ducks['she'];c=w.rope_config
    height=c.height-c.sag*(1-(2*foot_x/c.span)**2)+c.radius+.002
    feet=np.array(duck.feet())
    error=(feet-np.array([[-foot_x,foot_y,height],[foot_x,foot_y,height]])).ravel()
    axis=w.data.xmat[duck.head].reshape(3,3)[:,1]
    # Mouth width lies along the rope; feet are on opposite sides of it.
    distances=[];contact_errors=[]
    for j,site in enumerate(duck.foot_sites):
        body=w.model.site_bodyid[site]
        geoms=[g for g in duck.solid if w.model.geom_bodyid[g]==body]
        center=(-foot_x if j==0 else foot_x)
        indices=range(max(0,int((center/c.span+.5)*c.segments)-2),min(c.segments,int((center/c.span+.5)*c.segments)+3))
        rope=[mujoco.mj_name2id(w.model,mujoco.mjtObj.mjOBJ_GEOM,f'rope_geom_{i}') for i in indices]
        best=(np.inf,None)
        for g in geoms:
            for r in rope:
                vector=np.zeros(6)
                distance=mujoco.mj_geomDistance(w.model,w.data,g,r,.2,vector)
                if distance<best[0]:best=(distance,vector)
        distances.append(best[0])
        point=best[1][:3]
        rope_z=c.height-c.sag*(1-(2*point[0]/c.span)**2)
        contact_errors.extend([point[1],point[2]-rope_z-c.radius-.00015])
    distances=np.array(distances)
    return np.r_[error*np.tile([1.,.4,.25],2), .07*(axis-np.array([-1.,0,0])),
                 .6*(pos[2]-(c.height-.06)), .8*(distances-.0002),3*np.minimum(distances,0.),1.5*np.array(contact_errors)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-poses',type=Path)
    parser.add_argument('--attempts',type=int,default=16)
    args=parser.parse_args()
    seeds=json.loads(args.seed_poses.read_text()) if args.seed_poses else []
    w=build();rng=np.random.default_rng(14)
    lo=np.array([-3.1,-1.5,-1.5,-1.5,-.38,-1.5,-1.5,-2.95,-.43])
    hi=np.array([0.,1.5,1.5,1.5,.38,1.,1.5,2.95,.43])
    results=[]
    for k in range(args.attempts):
        start=np.clip(seeds[k]['x'],lo,hi) if k<len(seeds) else rng.uniform(lo,hi)
        r=least_squares(lambda x:residual(w,x),start,bounds=(lo,hi),max_nfev=120,diff_step=1e-4)
        q,pos=place(w,r.x)
        mujoco.mj_forward(w.model,w.data)
        penetration=max([max(0.,-c.dist) for c in w.data.contact if (c.geom1 in w.rope_geoms and c.geom2 in w.ducks['she'].solid) or (c.geom2 in w.rope_geoms and c.geom1 in w.ducks['she'].solid)]+[0.])
        results.append(dict(bite=BITE.tolist(),mouth_open=MOUTH_OPEN,error=float(np.linalg.norm(r.fun)),x=r.x.tolist(),target=q.tolist(),position=pos.tolist(),initial_penetration_m=float(penetration)))
    results.sort(key=lambda r:r['error'])
    folder=Path('local_storage/hb_dev/scripted_policy/rope_traverse')
    n=1
    while (folder/f'fit_{n}.json').exists():n+=1
    p=folder/f'fit_{n}.json'
    with p.open('x') as f:json.dump(results,f,indent=2)
    print(json.dumps(results[:4],indent=2))


if __name__=='__main__':main()
