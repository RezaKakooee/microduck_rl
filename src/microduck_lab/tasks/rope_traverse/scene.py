"""Flexible rope with fixed endpoints; robot attachment is contact-only."""
from dataclasses import dataclass
import numpy as np
import mujoco
from microduck_lab.tasks.common.world import World, SetDesign, Box


@dataclass(frozen=True)
class Scene:
    span: float = .8
    height: float = .50
    radius: float = .006
    sag: float = .012
    segments: int = 32
    jaw_torque: float = .25  # provisional, explicitly bounded; not hardware calibration
    # Extra solver pass that removes MuJoCo's soft-contact friction creep. Without it
    # a gripping leg slides 3-11 mm along the rope under side loads far below mu*N,
    # which a real rope does not do (measured 2026-10-01, dev t35/t36).
    noslip: int = 10


def build(config=Scene(),robot=True):
    if config.segments<4 or min(config.span,config.height,config.radius,config.jaw_torque)<=0:
        raise ValueError('Positive dimensions and at least four segments required')
    x=np.linspace(-config.span/2,config.span/2,config.segments+1)
    z=config.height-config.sag*(1-(2*x/config.span)**2)
    points=np.c_[x,np.zeros_like(x),z]
    posts=[Box(f'post_{i}',(v-.025,-.04,0),(v+.025,.04,config.height+.05),rgba=(.30,.23,.18,1)) for i,v in enumerate((x[0],x[-1]))]

    def edit(spec):
        spec.option.timestep=.0005
        spec.option.iterations=150
        spec.option.noslip_iterations=config.noslip
        parent=spec.worldbody
        for i,delta in enumerate(np.diff(points,axis=0)):
            body=parent.add_body(name=f'rope_link_{i}',pos=points[0] if i==0 else points[i]-points[i-1])
            body.add_joint(name=f'passive_rope_{i}',type=mujoco.mjtJoint.mjJNT_BALL,damping=.001,stiffness=.0002)
            body.add_geom(name=f'rope_geom_{i}',type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=[0,0,0,*delta],size=[config.radius,0,0],density=500,
                          rgba=[.76,.59,.30,1],friction=[.8,.003,.0001])
            parent=body
        for i in [0,1,config.segments-2,config.segments-1]:
            spec.add_exclude(name=f'mount_clearance_{i}',bodyname1='world',bodyname2=f'rope_link_{i}')
        parent.add_site(name='rope_tip',pos=points[-1]-points[-2],size=[.002,0,0])
        spec.worldbody.add_site(name='rope_anchor',pos=points[-1],size=[.002,0,0])
        spec.add_equality(name='rope_end_anchor',type=mujoco.mjtEq.mjEQ_CONNECT,
                          objtype=mujoco.mjtObj.mjOBJ_SITE,name1='rope_tip',name2='rope_anchor',
                          solref=[.004,1.],solimp=[.99,.999,.001,.5,2.])
        if robot:
            for actuator in spec.actuators:
                if actuator.name=='she_mouth':
                    actuator.forcelimited=True
                    actuator.forcerange=[-config.jaw_torque,config.jaw_torque]
    world=RopeWorld(SetDesign(posts),cast=('she',) if robot else (),edit=edit)
    world.rope_config=config
    world.rope_geoms={mujoco.mj_name2id(world.model,mujoco.mjtObj.mjOBJ_GEOM,f'rope_geom_{i}') for i in range(config.segments)}
    return world


class RopeWorld(World):
    """Keep 50 Hz motor commands while resolving light rope links at 2 kHz."""
    def step(self):
        m,d=self.model,self.data
        q,v=d.qpos.copy(),d.qvel.copy();start=float(d.time)
        warning_before=d.warning.number.copy()
        for _ in range(round(.02/m.opt.timestep)):
            if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise RuntimeError('External force applied')
            for duck in self.ducks.values():duck.motor.update()
            mujoco.mj_step(m,d)
            if np.any(d.warning.number>warning_before):raise RuntimeError('MuJoCo warning: invalid physics trial')
        if abs(d.time-start-.02)>1e-8:raise RuntimeError('Simulator time reset: invalid physics trial')
        if self.guard is not None:
            for key,value in self.guard.items():
                if not np.array_equal(getattr(m,key),value):raise RuntimeError(f'Model field {key} changed')
        if not np.isfinite(d.qpos).all():raise RuntimeError('Non-finite state')
        self.t+=.02
        return q,v
