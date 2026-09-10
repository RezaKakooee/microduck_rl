"""Flat ground, free duck, BAM motors. Nothing to sit on and nothing to hold it.

The duck lies on its front and drives itself along with its legs. The set is a
plane, so every metre of travel comes from the servos and friction.
"""
from __future__ import annotations
import numpy as np
import mujoco
from bam.model import load_model
from bam.mujoco import MujocoController
from microduck_lab import paths
from microduck_lab.film import stage

DT=.02
# The parts that touch the ground must collide with the shape you can see.
# Group 3 leaves the thigh and the front of the belly out, which would let the
# duck sink into the grass.
GROUND=('he_trunk_base','he_upper_leg_left','he_upper_leg_right',
        'he_leg','he_leg_2','he_ankle_left','he_ankle_right')


def world_xml():
    return f'''<mujoco model="microduck_crawl"><compiler angle="radian"/>
    <option timestep=".002" integrator="implicitfast" iterations="100"/>
    <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/><headlight ambient=".28 .28 .28" diffuse=".55 .55 .55" specular=".15 .15 .15"/></visual>
    <asset><texture name="sky" type="skybox" builtin="gradient" rgb1=".48 .72 .91" rgb2=".92 .96 1" width="512" height="512"/>
    <texture name="grass" type="2d" builtin="checker" rgb1=".28 .43 .18" rgb2=".31 .46 .20" width="512" height="512"/>
    <material name="grass" texture="grass" texrepeat="16 16" reflectance="0"/></asset>
    <default><geom friction="1 .005 .0001" solref=".008 1" solimp=".98 .999 .001"/></default>
    <worldbody><light pos="1 -2 3" dir="-.3 .3 -1" directional="true" ambient=".2 .2 .2" diffuse=".8 .8 .8"/>
    <light pos="-1 1 2" diffuse=".3 .3 .3" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 .1" material="grass" rgba="1 1 1 1"/>
    '''+markers_xml()+'''
    </worldbody></mujoco>'''


def markers_xml(step=.25,count=22):
    """Stripes across the path every 25 cm, so travel is visible.

    On plain grass with a camera that follows the duck, nothing shows it moving.
    These are scenery: `mark_` geoms are set to no collision at all.
    """
    xml=''
    for i in range(count):
        x=-i*step
        wide=i%4==0                       # a longer, brighter one every metre
        xml+=f'<geom name="mark_{i}" type="box" pos="{x} 0 .001" size=".006 {.36 if wide else .22} .001" rgba="{".97 .83 .25 1" if wide else ".85 .88 .90 1"}"/>'
    return xml


class Crawl:
    def __init__(self,voltage=7.4,settle=1.0):
        spec=mujoco.MjSpec.from_string(world_xml())
        spec.attach(mujoco.MjSpec.from_file(paths.model('robot_allcollisions_mouth.xml')),
                    prefix='he_',frame=spec.worldbody.add_frame())
        bam=load_model(motor_name='xl330',model='m6')
        bam.actuator.kp=200;bam.actuator.vin=voltage;bam.actuator.max_current=1.75
        self.torque_limit=float(bam.kt.value*1.75)
        names=[]
        for a in spec.actuators:
            a.set_to_motor();a.forcelimited=True;a.forcerange=(-self.torque_limit,self.torque_limit)
            a.ctrllimited=False;names.append(a.name)
        for j in spec.joints:
            if j.name.startswith('he_') and j.type!=mujoco.mjtJoint.mjJNT_FREE:
                j.damping=np.zeros((3,1));j.frictionloss=0.
                j.solref_friction=(-5e4,-2e2);j.solimp_friction=(.99,.9999,.001,.5,2)
        self.model=spec.compile();m=self.model;stage.paint(m)
        for g in range(m.ngeom):
            n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
            if n.startswith('he_'):
                on=m.geom_group[g]==3 or (n in GROUND and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH)
                m.geom_contype[g]=1 if on else 0
                m.geom_conaffinity[g]=2 if on else 0
            elif (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '').startswith('mark_'):
                m.geom_contype[g]=0;m.geom_conaffinity[g]=0      # scenery only
            else:m.geom_contype[g]=2;m.geom_conaffinity[g]=3
            m.geom_margin[g]=0
        self.data=mujoco.MjData(m);d=self.data
        self.motor=MujocoController(bam,names,m,d,vin_drop_gain=.1,vin_min=6.)
        self.qidx=m.jnt_qposadr[m.actuator_trnid[:,0]]
        self.vidx=m.jnt_dofadr[m.actuator_trnid[:,0]]
        self.names=names
        self.ai={n.removeprefix('he_'):i for i,n in enumerate(names)}
        self.lo=m.jnt_range[m.actuator_trnid[:,0]][:,0]
        self.hi=m.jnt_range[m.actuator_trnid[:,0]][:,1]
        self.base=np.zeros(m.nu)
        for name,value in {'left_hip_pitch':-1.20,'left_knee':-.60,'left_ankle':0,
                           'right_hip_pitch':1.20,'right_knee':.60,'right_ankle':0,
                           'neck_pitch':-.20,'head_pitch':.10}.items():
            self.base[self.ai[name]]=value
        self.root=m.jnt_qposadr[self.id('JOINT','he_trunk_base_freejoint')]
        self.trunk=self.id('BODY','he_trunk_base')
        # Nose down, belly on the grass: the pose the duck lands in.
        d.qpos[self.qidx]=self.base
        d.qpos[self.root:self.root+3]=[0,0,.10]
        half=np.pi/4                              # pitched forward 90 deg
        d.qpos[self.root+3:self.root+7]=[np.cos(half),0,np.sin(half),0]
        self.motor.q_target[:]=self.base;mujoco.mj_forward(m,d)
        self.t=0.;self.max_torque=0.;self.work=0.
        self.guard={k:getattr(m,k).copy() for k in ('geom_contype','geom_conaffinity','body_gravcomp')}
        for _ in range(int(settle/DT)):self.step(self.base)   # let it settle first
        self.start=d.xpos[self.trunk].copy()
        self.t=0.;self.max_torque=0.;self.work=0.

    def id(self,kind,name):
        i=mujoco.mj_name2id(self.model,getattr(mujoco.mjtObj,'mjOBJ_'+kind),name)
        if i<0:raise KeyError(name)
        return i

    def step(self,target):
        m,d=self.model,self.data
        if not np.isfinite(target).all():raise RuntimeError('Nonfinite command')
        self.motor.q_target[:]=np.clip(target,self.lo,self.hi)
        for _ in range(10):
            if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise RuntimeError('External force present')
            self.motor.update();mujoco.mj_step(m,d)
            self.work+=float(np.dot(d.actuator_force,d.qvel[self.vidx]))*m.opt.timestep
            self.max_torque=max(self.max_torque,float(np.max(np.abs(d.actuator_force))))
        mujoco.mj_forward(m,d);self.t+=DT
        for k,v in self.guard.items():
            if not np.array_equal(getattr(m,k),v):raise RuntimeError('Changed '+k)
        if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite dynamics')

    def travelled(self):
        """Distance from where it started, in the ground plane."""
        d=self.data.xpos[self.trunk]-self.start
        return float(np.hypot(d[0],d[1])),float(d[0]),float(d[1])
