"""Physical swing: passive bearings, solid bucket seat, free robot, BAM motors.

The seat is built from the duck's measured shape, not guessed. All numbers below
are millimetres from the trunk origin, with x forward and z up.

The duck's legs hang beside its body, never closer to the middle than |y| = 29.
Its head never drops below z = +43. So the seat lives in the gap between them:
everything is inside |y| <= 26 and z <= +35, where only the trunk can reach.

The trunk's underside is a narrow ridge from x -47 to -21. Forward of that the
belly only starts at z = -5, so a flat plate under the whole body would not hold
it. The parts are placed against that ridge:

  seat plate   x -50..-12   under the ridge, carries the weight
  seat lip     x -18..-12   stops the round belly rolling forward
  back panel   x -57..-49   2 mm behind the trunk
  lap guard    x +38..+44   3 mm in front of the belly; it never touches

The frame reaches the ropes through two cross bars at z = +28. That is above the
legs, which stop at +17, and below the head, which stops at +43.

The plate above is only 38 x 52 mm and the duck hides it, so the swing looked
like it had no seat. Bucket walls at |y| 75..100, a wide back panel and a rear
floor behind x -44 fill the three spaces the duck never enters. The legs come
out through the gaps at |y| 26..75. None of these parts ever touches the duck.

Two earlier versions failed:

  1. A wide plate under everything. The shins landed on it first, so the duck's
     underside stayed 36 mm up and nothing was under it.
  2. A tall back panel. The head struck it while pumping, drove 10 mm in and
     levered the duck out of the seat after 13 s.
"""
from __future__ import annotations
import numpy as np
import mujoco
from bam.model import load_model
from bam.mujoco import MujocoController
from microduck_lab import paths
from microduck_lab.film import stage

DT=.02

# Group 3 is the robot's own collision set. It leaves out the parts that meet
# the seat: the visible thigh hangs 41 mm below any group-3 shape, and the
# visible belly reaches 49 mm further forward. The seat then touched a shape
# nobody can see while the duck sank into the plate. So these bodies collide
# with their visible meshes instead.
SEATED=('he_trunk_base','he_upper_leg_left','he_upper_leg_right',
        'he_leg','he_leg_2','he_ankle_left','he_ankle_right')


HINGE_X,HINGE_Z=-.057,.004      # the floor hinges at the top of its back edge


def floor_xml(length,hinged=False):
    """The part the duck sits on. Either fixed, or a flap that swings open.

    When hinged the positions are relative to the hinge, which sits at the back
    edge of the pan. Then a positive hinge angle drops the front of the floor.
    """
    x,z=(-HINGE_X,length-HINGE_Z) if hinged else (0.,0.)
    return f'''
    <geom name="seat" type="box" pos="{-.031+x} 0 {-length-.004+z}" size=".019 .026 .008" mass=".09" rgba=".74 .43 .17 1"/>
    <geom name="seat_lip" type="box" pos="{-.015+x} 0 {-length+.022+z}" size=".003 .026 .018" mass=".01" rgba=".74 .43 .17 1"/>
    <geom name="seat_rear" type="box" pos="{-.0505+x} 0 {-length-.004+z}" size=".0065 .070 .008" mass=".02" rgba=".74 .43 .17 1"/>
    <geom name="lap_guard" type="box" pos="{.041+x} 0 {-length+.024+z}" size=".003 .026 .020" mass=".012" rgba=".16 .25 .32 1"/>
    <geom name="guard_arm" type="box" pos="{.0145+x} 0 {-length-.004+z}" size=".0295 .026 .008" mass=".02" rgba=".74 .43 .17 1"/>'''


def shell_xml(length):
    """The rest of the bucket. This stays on the ropes when the floor opens."""
    xml=f'''
    <geom name="back" type="box" pos="-.053 0 {-length+.0435}" size=".004 .100 .0395" mass=".025" rgba=".74 .43 .17 1"/>
'''
    for side in (-1,1):
        # Stubs, not full-width bars. The middle stays open so the duck has a
        # clear way down when the floor lets go. Both ends sit outside the legs,
        # which never pass |y| = 71 mm, and outside the head, which stays inside
        # |y| = 46 mm.
        for name,x in (('back_cross',-.053),('guard_cross',.041)):
            xml+=f'<geom name="{name}_{side}" type="capsule" fromto="{x} {side*.072} {-length+.076} {x} {side*.105} {-length+.076}" size=".004" mass=".004" rgba=".16 .25 .32 1"/>'
        xml+=f'<geom name="bucket_{side}" type="box" pos="-.0065 {side*.0875} {-length+.026}" size=".0505 .0125 .022" mass=".03" rgba=".74 .43 .17 1"/>'
        xml+=f'<geom name="seat_rail_{side}" type="capsule" fromto="-.053 {side*.095} {-length+.076} .041 {side*.095} {-length+.076}" size=".004" mass=".007" rgba=".16 .25 .32 1"/>'
    return xml


def world_xml(length=.48,damping=.002,trapdoor=False):
    height=length+.30
    xml=f'''<mujoco model="microduck_swing"><compiler angle="radian"/>
    <option timestep=".002" integrator="implicitfast" iterations="100"/>
    <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/><headlight ambient=".28 .28 .28" diffuse=".55 .55 .55" specular=".15 .15 .15"/></visual>
    <asset><texture name="sky" type="skybox" builtin="gradient" rgb1=".48 .72 .91" rgb2=".92 .96 1" width="512" height="512"/>
    <texture name="grass" type="2d" builtin="checker" rgb1=".28 .43 .18" rgb2=".31 .46 .20" width="512" height="512"/>
    <material name="grass" texture="grass" texrepeat="16 16" reflectance="0"/></asset>
    <default><geom friction="1 .005 .0001" solref=".008 1" solimp=".98 .999 .001"/></default>
    <worldbody><light pos="1 -2 3" dir="-.3 .3 -1" directional="true" ambient=".2 .2 .2" diffuse=".8 .8 .8"/><light pos="-1 1 2" diffuse=".3 .3 .3" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 .1" material="grass" rgba="1 1 1 1"/>
    <geom name="crossbar" type="capsule" fromto="0 -.34 {height+.04} 0 .34 {height+.04}" size=".018" rgba=".04 .29 .48 1"/>
    <body name="swing" pos="0 0 {height}"><joint name="passive_swing" axis="0 1 0" damping="{damping}"/>
{shell_xml(length)}{'' if trapdoor else floor_xml(length)}
        '''
    for side in (-1,1):
        xml+=f'''<geom name="axle_{side}" type="capsule" fromto="0 {side*.101} 0 0 {side*.125} 0" size=".004" mass=".004" rgba=".45 .5 .55 1"/>
        <geom name="rod_{side}" type="capsule" fromto="0 {side*.105} 0 0 {side*.105} {-length}" size=".004" mass=".015" rgba=".48 .54 .60 1"/>'''
    if trapdoor:
        # The floor is a flap on a free hinge at the back of the bucket, held
        # shut by one equality. Release it and the duck's weight swings it open.
        xml+=f'<body name="trapdoor" pos="{HINGE_X} 0 {-length+HINGE_Z}"><joint name="trap_hinge" type="hinge" axis="0 1 0" pos="0 0 0" range="0 2.80" damping=".0002"/>{floor_xml(length,hinged=True)}</body>'
    xml+='</body>'
    for side in (-1,1):
        for front in (-1,1):
            xml+=f'<geom name="frame_{side}_{front}" type="capsule" fromto="0 {side*.30} {height+.04} {front*.33} {side*.35} .02" size=".015" rgba=".05 .34 .56 1"/>'
            xml+=f'<geom name="foot_{side}_{front}" type="box" pos="{front*.33} {side*.35} .012" size=".048 .032 .012" rgba=".08 .22 .29 1"/>'
        xml+=f'<geom name="brace_{side}" type="capsule" fromto="-.23 {side*.333} .25 .23 {side*.333} .25" size=".01" rgba=".06 .32 .50 1"/>'
    # The joint represents a bearing; its visible ring has an actual hole.
    for side in (-1,1):
        y=side*.118
        xml+=f'<geom name="bearing_mount_{side}" type="capsule" fromto="0 {y} {height+.011} 0 {y} {height+.027}" size=".004" rgba=".25 .28 .30 1"/>'
        for i in range(20):
            a,b=2*np.pi*np.array([i,i+1])/20
            xml+=f'<geom name="bearing_{side}_{i}" type="capsule" fromto="{.008*np.sin(a)} {y} {height+.008*np.cos(a)} {.008*np.sin(b)} {y} {height+.008*np.cos(b)}" size=".0025" rgba=".42 .46 .50 1"/>'
    xml+='</worldbody>'
    if trapdoor:
        xml+='<equality><joint name="trap_latch" joint1="trap_hinge" polycoef="0 0 0 0 0" solref=".002 1" solimp=".999 .99999 .001"/></equality>'
    xml+='</mujoco>'
    return xml


class Swing:
    def __init__(self,length=.48,damping=.002,voltage=7.4,initial_angle=0,
                 seated_until=None,dismantle_at=None):
        # `dismantle_at` puts the seat floor on a hinge, latched shut by one
        # equality. At that time the latch lets go, the duck's weight swings the
        # floor open, and the duck drops through while the seat stays on the
        # ropes. Without it the floor is fixed and there is no equality, so the
        # graded task still reports zero.
        self.dismantle_at=dismantle_at
        # The seated check is on for the whole run by default. It is dropped
        # once the seat has gone, since the duck is then meant to be falling.
        self.seated_until=np.inf if seated_until is None else seated_until
        if dismantle_at is not None and seated_until is None:self.seated_until=dismantle_at
        self.length=length
        spec=mujoco.MjSpec.from_string(world_xml(length,damping,dismantle_at is not None))
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
                on=m.geom_group[g]==3 or (n in SEATED and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH)
                m.geom_contype[g]=1 if on else 0
                m.geom_conaffinity[g]=2 if on else 0
            else:m.geom_contype[g]=2;m.geom_conaffinity[g]=3
            m.geom_margin[g]=0
        self.data=mujoco.MjData(m);d=self.data
        self.motor=MujocoController(bam,names,m,d,vin_drop_gain=.1,vin_min=6.)
        self.qidx=m.jnt_qposadr[m.actuator_trnid[:,0]]
        self.vidx=m.jnt_dofadr[m.actuator_trnid[:,0]]
        self.names=names
        self.ai={n.removeprefix('he_'):i for i,n in enumerate(names)}
        self.base=np.zeros(m.nu)
        for name,value in {'left_hip_pitch':-.9,'left_knee':.3,'left_ankle':0,
                           'right_hip_pitch':.9,'right_knee':-.3,'right_ankle':0,
                           'neck_pitch':.5,'head_pitch':.55}.items():
            self.base[self.ai[name]]=value
        self.root=m.jnt_qposadr[self.id('JOINT','he_trunk_base_freejoint')]
        self.swing_q=m.jnt_qposadr[self.id('JOINT','passive_swing')]
        self.swing_v=m.jnt_dofadr[self.id('JOINT','passive_swing')]
        self.trunk=self.id('BODY','he_trunk_base')
        self.seat=self.id('BODY','swing')
        # Initial placement only; the duck is a free body, with no seat weld.
        d.qpos[self.qidx]=self.base
        theta=np.deg2rad(initial_angle);d.qpos[self.swing_q]=theta
        R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        d.qpos[self.root+3:self.root+7]=[np.cos(theta/2),0,np.sin(theta/2),0]
        # Find the seat height by search, not by a constant. A fixed offset has
        # to be re-tuned whenever the seat moves, and getting it wrong starts the
        # duck inside the seat: an early version began with the thigh 8.6 mm in,
        # which the solver then had to push out.
        pivot=np.array([0,0,length+.3])
        low,high=-length-.02,-length+.20
        for _ in range(40):
            mid=.5*(low+high)
            d.qpos[self.root:self.root+3]=pivot+R@np.array([0,0,mid])
            mujoco.mj_forward(m,d)
            if self.robot_touches_set():low=mid
            else:high=mid
        d.qpos[self.root:self.root+3]=pivot+R@np.array([0,0,high+.0005])
        self.motor.q_target[:]=self.base;mujoco.mj_forward(m,d)
        # The guard measures drift from where the duck actually started. An
        # absolute box only fits one sitting pose, and trips at once on another.
        self.seat0=d.xmat[self.seat].reshape(3,3).T@(d.xpos[self.trunk]-d.xpos[self.seat])
        self.t=0.;self.work=0.;self.max_torque=0.;self.max_penetration=0.;self.min_up=1.
        self.last_speed=0.;self.released=None
        self.guard={k:getattr(m,k).copy() for k in ('geom_contype','geom_conaffinity','geom_rgba','body_gravcomp')}

    def robot_touches_set(self):
        """True when any robot collision geom touches the swing or the ground."""
        m,d=self.model,self.data
        for c in d.contact[:d.ncon]:
            for g in (c.geom1,c.geom2):
                name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
                if name.startswith('he_'):return True
        return False

    def id(self,kind,name):
        i=mujoco.mj_name2id(self.model,getattr(mujoco.mjtObj,'mjOBJ_'+kind),name)
        if i<0:raise KeyError(name)
        return i

    def observation(self):
        d=self.data
        return float(d.qpos[self.swing_q]),float(d.qvel[self.swing_v])

    def step(self,target):
        m,d=self.model,self.data
        if not np.isfinite(target).all():raise RuntimeError('Nonfinite command')
        self.motor.q_target[:]=target
        for _ in range(10):
            if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise RuntimeError('External force present')
            before=d.qpos.copy(),d.qvel.copy()
            self.motor.update()
            if not np.array_equal(before[0],d.qpos) or not np.array_equal(before[1],d.qvel):raise RuntimeError('Motor changed state')
            mujoco.mj_step(m,d)
            self.work+=float(np.dot(d.actuator_force,d.qvel[self.vidx]))*m.opt.timestep
            self.max_torque=max(self.max_torque,float(np.max(np.abs(d.actuator_force))))
            self.max_penetration=max(self.max_penetration,max((-c.dist for c in d.contact),default=0))
        if self.dismantle_at is not None and self.t>=self.dismantle_at and d.eq_active.any():
            # The flap must fold right back, not stop at 90 deg. Hanging straight
            # down it is a hook: at a 50 deg swing the duck caught on it every
            # time and never reached the ground.
            # Wait for the front of the arc, where the swing has just stopped and
            # started back. Releasing anywhere else throws the duck against the
            # backrest instead: of 11 points across one period, the 4 failures
            # were all at the back extreme or still moving forward.
            angle=float(d.qpos[self.swing_q]);speed=float(d.qvel[self.swing_v])
            if angle>0 and speed<=0<self.last_speed:
                d.eq_active[:]=0;self.released=self.t
            self.last_speed=speed
        mujoco.mj_forward(m,d);self.t+=DT
        relative=d.xmat[self.seat].reshape(3,3).T@d.xmat[self.trunk].reshape(3,3)
        self.min_up=min(self.min_up,float(relative[2,2]))
        for k,v in self.guard.items():
            if not np.array_equal(getattr(m,k),v):raise RuntimeError('Changed '+k)
        drift=d.xmat[self.seat].reshape(3,3).T@(d.xpos[self.trunk]-d.xpos[self.seat])-self.seat0
        if self.t<=self.seated_until and (
                relative[2,2]<.7 or abs(drift[1])>.06 or abs(drift[0])>.07 or abs(drift[2])>.06):
            raise RuntimeError('Duck left its seated posture')
        if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite dynamics')
