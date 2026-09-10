"""Free seated duck with BAM motors, a rigid mouth-held brush and a paint easel."""
from __future__ import annotations
import numpy as np
import mujoco
from bam.model import load_model
from bam.mujoco import MujocoController
from microduck_lab import paths
from microduck_lab.film import stage
from microduck_lab.tasks.swing.swing import floor_xml, shell_xml, SEATED

DT = .02
CANVAS_X = .165
CANVAS_Z = .315
HALF_W, HALF_H = .078, .090
COLORS = {'green': (.12,.52,.27), 'pink': (.91,.20,.43), 'yellow': (1.,.69,.07)}
WELLS = {name: np.array([.145, y, .244]) for name,y in zip(COLORS, [-.045, 0., .045])}


def world_xml():
    assets = '''<texture name="sky" type="skybox" builtin="gradient" rgb1=".55 .76 .89" rgb2=".96 .96 .88" width="512" height="512"/>
    <texture name="floor" type="2d" builtin="checker" rgb1=".62 .71 .54" rgb2=".66 .75 .58" width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="8 8"/>
    <texture name="paper" type="2d" builtin="flat" rgb1=".98 .97 .92" width="768" height="768"/>
    <material name="paper" texture="paper" specular="0" shininess="0"/>
    <mesh name="paper" inertia="shell" vertex="0 -.078 -.09  0 .078 -.09  0 .078 .09  0 -.078 .09" texcoord="0 0  1 0  1 1  0 1" face="0 2 1  0 3 2"/>
    '''
    scene = f'''<geom name="floor" type="plane" size="0 0 .1" material="floor"/>
    <body name="chair" pos="0 0 .14">{floor_xml(0)}{shell_xml(0)}</body>
    <geom name="canvas_back" type="box" pos="{CANVAS_X+.004} 0 {CANVAS_Z}" size=".004 {HALF_W+.006} {HALF_H+.006}" friction=".15 .005 .0001" rgba=".45 .25 .12 1"/>
    <geom name="paper" type="mesh" mesh="paper" pos="{CANVAS_X-.0001} 0 {CANVAS_Z}" material="paper" contype="0" conaffinity="0"/>
    <geom name="palette" type="box" pos=".145 0 .231" size=".030 .076 .006" rgba=".65 .41 .20 1"/>
    '''
    for side in [-1,1]:
        scene += f'<geom name="chair_side_{side}" type="box" pos="-.028 {side*.033} .181" size=".021 .004 .020" rgba=".58 .34 .17 1"/>'
    for y in [-.075,.075]:
        scene += f'<geom type="capsule" fromto=".19 {y} .018 .174 {y} .430" size=".005" rgba=".62 .37 .17 1"/>'
        scene += f'<geom type="capsule" fromto=".31 {y} .016 .18 {y} .34" size=".005" rgba=".62 .37 .17 1"/>'
    for x in [-.055,.035]:
        for y in [-.085,.085]:
            scene += f'<geom type="capsule" fromto="{x} {y} .012 {x} {y} .155" size=".007" rgba=".38 .23 .14 1"/>'
    for name,xyz in WELLS.items():
        x,y,z=xyz; rgb=' '.join(map(str,COLORS[name]))
        scene += f'<geom name="pot_{name}" type="cylinder" pos="{x} {y} {z-.006}" size=".016 .008" rgba=".90 .87 .78 1"/>'
        scene += f'<geom name="paint_{name}" type="cylinder" pos="{x} {y} {z+.002}" size=".012 .001" rgba="{rgb} 1"/>'
    return f'''<mujoco model="duck_painting"><compiler angle="radian"/>
    <option timestep=".002" integrator="implicitfast" iterations="100"/>
    <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/><headlight ambient=".35 .35 .35" diffuse=".55 .55 .55" specular=".1 .1 .1"/></visual>
    <asset>{assets}</asset><default><geom friction="1 .005 .0001" solref=".008 1" solimp=".98 .999 .001"/></default>
    <worldbody><light pos="-.5 -1 2" dir=".2 .4 -1" directional="true" diffuse=".8 .8 .8"/><light pos="0 1 1" diffuse=".4 .4 .4" castshadow="false"/>{scene}</worldbody></mujoco>'''


class Painting:
    def __init__(self):
        spec=mujoco.MjSpec.from_string(world_xml())
        robot=mujoco.MjSpec.from_file(paths.model('robot_allcollisions_mouth.xml'))
        head=robot.body('jaw_soft')
        brush=head.add_body(name='brush',pos=[-.00809334,0,-.0727383])
        # Fixed relative transform models a brush already clamped in the mouth.
        brush.add_geom(name='brush_handle',type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       fromto=[0,0,.015,0,0,-.080],size=[.0027,0,0],mass=.0025,rgba=[.70,.22,.10,1])
        brush.add_geom(name='brush_ferrule',type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       fromto=[0,0,-.078,0,0,-.092],size=[.0033,0,0],mass=.0006,rgba=[.72,.75,.77,1])
        brush.add_geom(name='bristles',type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       fromto=[0,0,-.092,0,0,-.104],size=[.0025,0,0],mass=.0002,rgba=[.34,.20,.10,1])
        brush.add_site(name='brush_tip',pos=[0,0,-.1065],size=[.001,0,0])
        spec.attach(robot,prefix='he_',frame=spec.worldbody.add_frame())
        bam=load_model(motor_name='xl330',model='m6')
        bam.actuator.kp=200;bam.actuator.vin=7.4;bam.actuator.max_current=1.75
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
            body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
            if body.startswith('he_'):
                on=m.geom_group[g]==3 or (body in SEATED and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH) or body=='he_brush'
                m.geom_contype[g]=1 if on else 0;m.geom_conaffinity[g]=2 if on else 0
            elif name=='paper':m.geom_contype[g]=0;m.geom_conaffinity[g]=0
            else:m.geom_contype[g]=2;m.geom_conaffinity[g]=1
            m.geom_margin[g]=0
            if body=='he_brush':m.geom_friction[g,0]=.15
        self.data=mujoco.MjData(m);d=self.data
        self.motor=MujocoController(bam,names,m,d,vin_drop_gain=.1,vin_min=6.)
        self.ai={n.removeprefix('he_'):i for i,n in enumerate(names)}
        self.qidx=m.jnt_qposadr[m.actuator_trnid[:,0]]
        self.vidx=m.jnt_dofadr[m.actuator_trnid[:,0]]
        self.lo=m.jnt_range[m.actuator_trnid[:,0],0];self.hi=m.jnt_range[m.actuator_trnid[:,0],1]
        self.trunk=self.id('BODY','he_trunk_base');self.tip=self.id('SITE','he_brush_tip')
        self.root=m.jnt_qposadr[self.id('JOINT','he_trunk_base_freejoint')]
        self.base=np.zeros(m.nu)
        for n,v in {'left_hip_pitch':-.9,'left_knee':.3,'right_hip_pitch':.9,'right_knee':-.3,
                    'neck_pitch':.5,'head_pitch':.55,'mouth':.08}.items():self.base[self.ai[n]]=v
        d.qpos[self.qidx]=self.base;d.qpos[self.root:self.root+3]=[0,0,.22]
        mujoco.mj_forward(m,d)
        self.t=0.;self.max_torque=0.;self.min_up=1.
        self.guard={k:getattr(m,k).copy() for k in ('geom_contype','geom_conaffinity','body_gravcomp')}
        for _ in range(150):self.step(self.base)
        self.t=0.;self.start=d.xpos[self.trunk].copy()
        self.arm=np.array([self.ai[n] for n in ('neck_pitch','head_pitch','head_yaw')])
        self.ikdata=mujoco.MjData(m);self.integral=np.zeros(3)
        self.desired=d.qpos[self.qidx].copy()

    def id(self,kind,name):
        i=mujoco.mj_name2id(self.model,getattr(mujoco.mjtObj,'mjOBJ_'+kind),name)
        if i<0:raise KeyError(name)
        return i

    def step(self,q):
        m,d=self.model,self.data
        self.motor.q_target[:]=np.clip(q,self.lo,self.hi)
        for _ in range(10):
            if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise RuntimeError('External force')
            self.motor.update();mujoco.mj_step(m,d)
            self.max_torque=max(self.max_torque,float(np.max(abs(d.actuator_force))))
        mujoco.mj_forward(m,d);self.t+=DT
        self.min_up=min(self.min_up,float(d.xmat[self.trunk,8]))
        if d.xmat[self.trunk,8]<.90:raise RuntimeError('Duck tipped out of chair')
        if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite dynamics')
        for k,v in self.guard.items():
            if not np.array_equal(getattr(m,k),v):raise RuntimeError('Changed '+k)

    def command(self,target):
        """IK in scratch data; only BAM targets are written to the live robot."""
        m,d=self.model,self.ikdata
        d.qpos[:]=self.data.qpos
        d.qpos[self.qidx[self.arm]]=self.desired[self.arm]
        J=np.zeros((3,m.nv))
        for _ in range(18):
            mujoco.mj_forward(m,d)
            err=target-d.site_xpos[self.tip]
            if np.linalg.norm(err)<.00005:break
            mujoco.mj_jacSite(m,d,J,None,self.tip)
            jac=J[:,self.vidx[self.arm]]
            delta=jac.T@np.linalg.solve(jac@jac.T+np.eye(3)*1e-6,err)
            ix=self.qidx[self.arm]
            d.qpos[ix]=np.clip(d.qpos[ix]+np.clip(delta,-.12,.12),self.lo[self.arm]+.03,self.hi[self.arm]-.03)
        self.desired=self.base.copy();self.desired[self.arm]=d.qpos[self.qidx[self.arm]]
        error=self.desired[self.arm]-self.data.qpos[self.qidx[self.arm]]
        self.integral=np.clip(self.integral+error*DT*4.,-.10,.10)
        q=self.desired.copy();q[self.arm]+=self.integral
        return q
