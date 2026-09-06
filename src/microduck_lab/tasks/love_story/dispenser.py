#!/usr/bin/env python3
"""A love story driven exclusively by MuJoCo dynamics after initialization.

Adults use the existing ONNX skills; chicks and visible nest mechanisms use
bounded joint servos. The ring is a solid gift on the ground. Eggs are loaded
before the story starts and released from a visible dispenser under gravity.
Run --dry for the same simulation without rendering; failures exit nonzero.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from microduck_lab.film import physical_stage as stage
from microduck_lab.tasks.love_story.original import Agent, Story as LegacyPoses, turn_cmd

DT = .02


class Story:
    def __init__(self, verbose=True):
        self.model, self.data = stage.build_model()
        m, d = self.model, self.data
        self.verbose = verbose
        self.she = Agent(m,d,'she',.16,.60,np.pi)
        self.he = Agent(m,d,'he',-.62,.60,0)
        self.adults = (self.she,self.he)
        self.kids = []
        for i,prefix in enumerate(('kidb','kidp')):
            kid = Agent(m,d,prefix,*stage.EGG_SPOTS[i],-np.pi/2,has_policy=False)
            kid.pose_target = LegacyPoses.curl_pose()
            d.qpos[kid.qpos_idx] = kid.pose_target
            d.qpos[kid.adr+2] = stage.EGG_START_Z+stage.INNER_FLOOR+.012
            self.kids.append(kid)
        self.actors = (*self.adults,*self.kids)
        self.gate_act = [self.id('ACTUATOR',f'gate_motor_{i}') for i in (0,1)]
        self.lid_act = [self.id('ACTUATOR',f'lid_motor_{i}') for i in (0,1)]
        self.gate_q = [m.jnt_qposadr[self.id('JOINT',f'egg_gate_{i}')] for i in (0,1)]
        self.lid_q = [m.jnt_qposadr[self.id('JOINT',f'egg_lid_{i}')] for i in (0,1)]
        self.egg_body = [self.id('BODY',f'egg{i}_base') for i in (0,1)]
        self.ring = self.id('BODY','ring')
        rq = m.jnt_qposadr[self.id('JOINT','ring_free')]
        d.qpos[rq:rq+3] = (-.10,.76,.012)
        d.qpos[rq+3:rq+7] = (1,0,0,0)
        mujoco.mj_forward(m,d)
        self.t = 0.
        self.phase_i = 0
        self.phase_t = 0.
        self.scratch = {}
        self.events = []
        self.failure = None
        self.min_up = np.ones(4)
        self.max_penetration = 0.
        self.worst_contact = None
        self.release_drop = np.zeros(2)
        self.nuzzle_contact = False
        self.head_kiss_contact = False
        self.camera_now = np.array([-.12,.52,.12,1.12,85.,-24.])
        self.immutable = {k:getattr(m,k).copy() for k in ('geom_contype','geom_conaffinity','geom_rgba','body_gravcomp')}
        self.phases = [
            ('Meet',20,self.meet), ('A bow and a yes',12,self.propose),
            ('A gentle nuzzle',20,self.nuzzle), ('Make room',4,self.separate),
            ('Walk around the nest',45,self.to_nest), ('Settle by the dispenser',12,self.sit),
            ('Release the first egg',6,lambda e:self.release(e,0)),
            ('Release the second egg',6,lambda e:self.release(e,1)),
            ('A kiss on her head',30,self.head_kiss),
            ('Keeping watch together',7,self.brood), ('Mother makes room',15,self.make_room),
            ('Father takes a turn',25,self.father), ('Two small surprises',10,self.hatch),
            ('Father comes to see',20,self.father_watches), ('Hello, little ones',9,self.finale),
        ]
        self.note(self.phases[0][0])

    def id(self,kind,name):
        i = mujoco.mj_name2id(self.model,getattr(mujoco.mjtObj,'mjOBJ_'+kind),name)
        if i<0:raise KeyError(name)
        return i

    def note(self,text):
        self.events.append({'time':round(self.t,2),'event':text})
        if self.verbose:print(f'{self.t:6.2f}s {text}',flush=True)

    def held(self,key,condition,seconds):
        if not condition:self.scratch.pop(key,None);return False
        self.scratch.setdefault(key,self.t)
        return self.t-self.scratch[key]>=seconds

    def meet(self,e):
        if 'arrived' not in self.scratch:
            if self.he.drive_to(self.she.xy(),stop_r=.07,ahead=.36):self.scratch['arrived']=self.t
        else:self.he.stop()
        self.she.look_at(self.data.xpos[self.he.head_body])
        return 'arrived' in self.scratch and self.t-self.scratch['arrived']>1 and np.linalg.norm(self.he.xy()-self.she.xy())<.45

    def propose(self,e):
        if 'bow' not in self.scratch:
            self.he.head();self.he.bow();self.scratch['bow']=True
        self.she.look_at(self.data.xpos[self.ring])
        if self.he.bowing:return False
        self.scratch.setdefault('up',self.t)
        u=self.t-self.scratch['up']
        self.she.head(pitch=.45*max(0,np.sin(2*np.pi*u)))
        self.he.look_at(self.data.xpos[self.she.head_body])
        if u>3:self.she.relax();return True
        return False

    def nuzzle(self,e):
        he,she=self.he,self.she
        if 'close' not in self.scratch:
            facing=she.face(float(np.arctan2(*(he.xy()-she.xy())[::-1])),tol=.15)
            reached=he.drive_to(she.xy(),stop_r=.03,ahead=.14)
            if facing and reached:self.scratch['close']=self.t
            return False
        he.stop();she.stop()
        touching=self.heads_touch(he,she)
        if touching:
            self.nuzzle_contact=True
            if 'touch' not in self.scratch:self.note('Head contact measured');self.scratch['touch']=self.t
        lam=self.scratch.get('lean',0.)
        if not touching:lam=min(1.,lam+DT/4)
        self.scratch['lean']=lam
        # Measured: with both leaning the same way the top shells meet first and
        # the beaks stay 29 mm apart. He goes low and down, she high and up, and
        # his jaw lands on her face -- the nearest thing to a kiss these heads
        # allow (the tips cannot pass the shells).
        he.lean(x=.02*lam,z=-.02*lam,pitch=-.30*lam,neck=0.,head_pitch=.70*lam)
        she.lean(x=.02*lam,z=.02*lam,pitch=.20*lam,neck=-.50*lam,head_pitch=-.50*lam)
        if 'touch' in self.scratch and self.t-self.scratch['touch']>2:
            for a in self.adults:a.relax()
            return True
        return False

    def heads_touch(self,a,b):
        return any((c.geom1 in a.head_geoms and c.geom2 in b.head_geoms) or
                   (c.geom2 in a.head_geoms and c.geom1 in b.head_geoms) for c in self.data.contact)

    def head_kiss(self,e):
        """He comes to her side while she sits with the eggs and rests his head
        on hers. Standing-policy body and head commands, leaned in until the
        contact is measured; every step of it is a servo target."""
        he,she=self.he,self.she
        head=self.data.xpos[she.head_body][:2]
        if 'at' not in self.scratch:
            # a waypoint west of her, then straight east along her side, so he
            # arrives facing her head; a direct drive from a skewed heading
            # oscillated at the mark and drifted into the nest
            if self.route(he,[(-.36,head[1]),(-.085,head[1])],'kiss_way'):
                self.scratch['at']=self.t;he.stop()
                self.note('At her side, %.0f mm from her head'%(1000*np.linalg.norm(he.xy()-head)))
            return False
        if self.t-self.scratch['at']<1.:return False
        touching=self.heads_touch(he,she)
        if touching:
            self.head_kiss_contact=True
            if 'touch' not in self.scratch:self.note('His head rests on hers');self.scratch['touch']=self.t
        lam=self.scratch.get('lean',0.)
        if not touching:lam=min(1.,lam+DT/2.5)
        self.scratch['lean']=lam
        # If the lean is at full stretch and still short, take a small step in
        # and lean again -- closed loop on the contact itself, three tries.
        creep=self.scratch.setdefault('creep',{'n':0,'until':-1.,'last':self.t})
        if self.t<creep['until']:
            he.vel(.3)
            return False
        if lam>=1. and not touching and 'touch' not in self.scratch and creep['n']<3 and self.t-creep['last']>3.:
            creep.update(n=creep['n']+1,until=self.t+.6,last=self.t)
            self.note('Short of her by a step -- creeping in')
            he.vel(.3)
            return False
        if he.policy.current_policy=='walking' and 'touch' not in self.scratch:he.stop()
        he.lean(x=.02*lam,z=-.03*lam,pitch=-.35*lam,neck=.20*lam,head_pitch=1.0*lam)
        she.head(pitch=.3)
        if 'touch' in self.scratch and self.t-self.scratch['touch']>2.5:
            he.relax();she.relax();return True
        return False

    def separate(self,e):
        self.he.vel(-.3)
        if e>2.5:self.he.stop();return True
        return False

    def route(self,actor,points,key):
        j=self.scratch.get(key,0)
        if j>=len(points):actor.stop();return True
        if actor.drive_to(points[j],stop_r=.055):self.scratch[key]=j+1
        return False

    def reverse_to(self,actor,target,tol=.04):
        delta=np.asarray(target)-actor.xy()
        if np.linalg.norm(delta)<tol:
            actor.stop();return True
        heading=float(np.arctan2(delta[1],delta[0])+np.pi)
        error=float(np.arctan2(np.sin(heading-actor.yaw()),np.cos(heading-actor.yaw())))
        if abs(error)>.25:actor.vel(wz=turn_cmd(error,0))
        else:actor.vel(-.3)
        return False

    def to_nest(self,e):
        a=self.route(self.she,[(.34,.44),(.34,-.23),(0,-.20)],'she_way')
        b=self.route(self.he,[(-.34,.39),(-.32,-.12)],'he_way')
        if b:self.he.look_at([0,.07,.065])
        if a:
            if 'faced' not in self.scratch:
                if self.she.face(-np.pi/2):self.scratch['faced']=True
                return False
            a=self.reverse_to(self.she,(0,-.085))
        return self.held('arrived',a and b,.6)

    def sit(self,e):
        self.she.stop();self.she.sit(True)
        self.he.look_at(self.data.xpos[self.she.head_body])
        return self.held('seated',self.she.z()<.085 and self.she.up()>.75,1.)

    def release(self,e,i):
        if 'z' not in self.scratch:self.scratch['z']=float(self.data.xpos[self.egg_body[i],2])
        self.data.ctrl[self.gate_act[i]]=.085*np.clip(e/1.5,0,1)
        self.release_drop[i]=self.scratch['z']-self.data.xpos[self.egg_body[i],2]
        return e>3 and self.data.qpos[self.gate_q[i]]>.08 and self.release_drop[i]>.010

    def brood(self,e):
        self.she.head(pitch=.15*np.sin(e*1.3),yaw=.25*np.sin(e*.8))
        self.he.look_at([0,.07,.065])
        return e>6

    def make_room(self,e):
        self.she.sit(False);self.she.relax()
        if e<2:return False
        return self.route(self.she,[(0,-.36),(.29,-.31)],'exit')

    def father(self,e):
        if self.she.face(float(np.arctan2(*((np.array([0,.07])-self.she.xy())[::-1])))):
            self.she.look_at([0,.07,.065])
        if 'at' not in self.scratch:
            if self.route(self.he,[(-.26,-.24),(0,-.20)],'entry'):self.scratch['at']=True
            return False
        if 'faced' not in self.scratch:
            if self.he.face(-np.pi/2):self.scratch['faced']=True
            return False
        if 'backed' not in self.scratch:
            if self.reverse_to(self.he,(0,-.085)):self.scratch['backed']=True
            return False
        self.he.sit(True)
        return self.held('seated',self.he.z()<.085 and self.he.up()>.75,3.)

    def hatch(self,e):
        for i in (0,1):
            self.data.ctrl[self.lid_act[i]]=-2*np.clip((e-i*2)/2.5,0,1)
            if self.data.qpos[self.lid_q[i]]< -1.5:
                # Uncurl through joint torques, while still supported by the shell.
                self.kids[i].pose_target[5]=.15
                self.kids[i].pose_target[6]=-.10
        self.she.look_at([0,.07,.065])
        return e>7 and all(self.data.qpos[q]<-1.7 for q in self.lid_q)

    def father_watches(self,e):
        self.he.sit(False);self.he.relax()
        if e<2:return False
        if not self.route(self.he,[(0,-.29),(-.23,-.23)],'exit'):return False
        angle=float(np.arctan2(*((np.array([0,.07])-self.he.xy())[::-1])))
        if self.he.face(angle):
            self.he.look_at([0,.07,.07]);return True
        return False

    def finale(self,e):
        for i,kid in enumerate(self.kids):
            kid.pose_target[6]=-.10+.10*np.sin(e*2+i)
            kid.pose_target[7]=.25*np.sin(e*1.5+i)
            kid.mouth_cmd=.20*max(0,np.sin(e*3+i))
        self.he.look_at([0,.07,.07])
        self.she.look_at([0,.07,.07])
        return e>8

    def control(self):
        """Only actuator controls may change simulator state during this call."""
        name,timeout,fn=self.phases[self.phase_i]
        done=fn(self.t-self.phase_t)
        if done:
            self.note('Completed: '+name)
            self.phase_i+=1;self.phase_t=self.t;self.scratch={}
            for a in self.adults:a.reset_progress()
            if self.phase_i<len(self.phases):self.note(self.phases[self.phase_i][0])
        elif self.t-self.phase_t>timeout:
            raise RuntimeError('Timed out: '+name)
        for actor in self.actors:actor.clock=self.t;actor.tick()

    def step(self):
        if self.phase_i>=len(self.phases):return False
        d,m=self.data,self.model
        q,v=d.qpos.copy(),d.qvel.copy()
        self.control()
        if not np.array_equal(q,d.qpos) or not np.array_equal(v,d.qvel):
            raise RuntimeError('Controller modified position or velocity')
        for key,value in self.immutable.items():
            if not np.array_equal(getattr(m,key),value):raise RuntimeError('Controller modified '+key)
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise RuntimeError('Unmodelled external force')
        for _ in range(4):
            mujoco.mj_step(m,d)
            for c in d.contact:
                if -c.dist>self.max_penetration:
                    self.max_penetration=-float(c.dist)
                    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
                    self.worst_contact=dict(time=round(self.t,3),geoms=names,penetration_mm=-1000*float(c.dist))
        mujoco.mj_forward(m,d)
        self.t+=DT
        self.min_up=np.minimum(self.min_up,[a.up() for a in self.actors])
        if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite simulation')
        if any(a.up()<.5 for a in self.adults):raise RuntimeError('An adult fell')
        if any(a.up()<.7 for a in self.kids):raise RuntimeError('A duckling tipped inside its capsule')
        return True

    def camera(self):
        p=min(self.phase_i,len(self.phases)-1)
        if p<4:
            center=(self.she.xy()+self.he.xy())*.5
            target=np.array([*center,.12,.85 if p in (1,2) else 1.12,85.,-22.])
        elif p==4:target=np.array([0,.15,.11,1.45,75.,-35.])
        elif p in (6,7,12):target=np.array([0,.065,.055,.46,235.,-65.])
        elif p==8:target=np.array([-.03,-.10,.13,.62,110.,-18.])
        elif p==13 and self.t-self.phase_t<3:target=np.array([0,.06,.055,.42,90.,-36.])
        else:target=np.array([0,-.07,.095,.95,75.,-27.])
        self.camera_now+=(target-self.camera_now)*(1-np.exp(-DT/.8))
        c=mujoco.MjvCamera();c.type=mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:]=self.camera_now[:3];c.distance,c.azimuth,c.elevation=self.camera_now[3:]
        return c

    def report(self):
        return dict(success=self.failure is None and self.phase_i==len(self.phases),failure=self.failure,
                    seconds=round(self.t,2),completed_phases=self.phase_i,total_phases=len(self.phases),
                    head_contact=self.nuzzle_contact,head_kiss_contact=self.head_kiss_contact,
                    release_drop_mm=(1000*self.release_drop).tolist(),
                    lid_angles_rad=self.data.qpos[self.lid_q].tolist(),minimum_upright=self.min_up.tolist(),
                    maximum_contact_penetration_mm=1000*self.max_penetration,
                    worst_contact=self.worst_contact,
                    mocap_bodies=self.model.nmocap,equality_constraints=self.model.neq,
                    actuator_only_guard=True,events=self.events)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry',action='store_true');ap.add_argument('--trace',action='store_true')
    ap.add_argument('--video');ap.add_argument('--report',default='videos/love_story/duck_love_story_physics.json')
    ap.add_argument('--seconds',type=float,default=240)
    ap.add_argument('--width',type=int,default=1280);ap.add_argument('--height',type=int,default=720)
    args=ap.parse_args();story=Story();writer=renderer=None
    if args.video:
        import imageio.v2 as imageio
        Path(args.video).parent.mkdir(parents=True,exist_ok=True)
        renderer=mujoco.Renderer(story.model,height=args.height,width=args.width)
        writer=imageio.get_writer(args.video,fps=25,quality=8,macro_block_size=8)
    tick=0
    try:
        while story.t<args.seconds and story.step():
            cam=story.camera()
            if writer is not None and tick%2==0:
                renderer.update_scene(story.data,camera=cam);writer.append_data(renderer.render())
            if args.trace and tick%50==0:
                print('TRACE',round(story.t,1),story.phase_i,[(a.prefix,np.round(a.xy(),3).tolist(),round(a.yaw(),2),round(a.z(),3)) for a in story.adults],flush=True)
            tick+=1
        if story.phase_i<len(story.phases):raise RuntimeError('Duration exhausted before story completion')
    except RuntimeError as exc:
        story.failure=str(exc);story.note('FAILED: '+str(exc))
    finally:
        if writer is not None:writer.close();renderer.close()
    report=story.report();Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2))
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
