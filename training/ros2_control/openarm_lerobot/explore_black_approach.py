#!/usr/bin/env python3
"""Read-only black-tape approach exploration. No publishers or action clients.

The output is a diagnostic report, deliberately NOT a replay plan.
Unmeasured supports/hook/tape and swept-volume checks remain blockers.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path
from slow_demo_replay import NAMES, KEYS, validate


def backoff_vector(q, distance):
    x,y,z,w=q
    if not all(math.isfinite(v) for v in (*q,distance)) or distance<=0:
        raise ValueError('Invalid backoff')
    if abs(sum(v*v for v in q)-1)>.0001:
        raise ValueError('Non-unit quaternion')
    # Hand +Z points toward TCP, per openarm_hand_arguments.xacro.
    return [-distance*2*(x*z+w*y),-distance*2*(y*z-w*x),
            -distance*(1-2*(x*x+y*y))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    data=json.loads(args.source.read_text())
    if data['keys']!=KEYS: raise ValueError('Unexpected joint order')
    validate(data['actions'],data['timestamps'])
    import rclpy
    from sensor_msgs.msg import JointState
    from moveit_msgs.msg import RobotState, Constraints, JointConstraint
    from moveit_msgs.srv import GetPlanningScene, GetPositionFK, GetPositionIK, GetStateValidity, GetMotionPlan
    from rosidl_runtime_py.convert import message_to_ordereddict
    rclpy.init(); node=rclpy.create_node('black_approach_readonly_exploration')
    readings={}; clients={}
    def cb(msg):
        readings.update({n:(v,time.monotonic()) for n,v in zip(msg.name,msg.position)})
    node.create_subscription(JointState,'/joint_states',cb,50)
    def call(kind,name,request):
        if name not in clients: clients[name]=node.create_client(kind,name)
        client=clients[name]
        if not client.wait_for_service(timeout_sec=3): raise RuntimeError(f'Missing service {name}')
        future=client.call_async(request)
        rclpy.spin_until_future_complete(node,future,timeout_sec=12)
        if not future.done() or future.result() is None: raise RuntimeError(f'Timeout {name}')
        return future.result()
    def state(q):
        s=RobotState(); s.is_diff=False
        s.joint_state.name=NAMES
        s.joint_state.position=[float(v)*(.03 if i in (7,15) else 1) for i,v in enumerate(q)]
        return s
    def fk(q):
        req=GetPositionFK.Request();req.header.frame_id='openarm_body_link0'
        req.fk_link_names=['openarm_left_hand'];req.robot_state=state(q)
        res=call(GetPositionFK,'/compute_fk',req)
        if res.error_code.val!=1 or len(res.pose_stamped)!=1: raise RuntimeError('FK failed')
        return res.pose_stamped[0]
    def valid(q):
        req=GetStateValidity.Request();req.robot_state=state(q)
        res=call(GetStateValidity,'/check_state_validity',req)
        if not res.valid: raise RuntimeError('Invalid state: '+str([(c.contact_body_1,c.contact_body_2) for c in res.contacts]))
    report={'purpose':'diagnostic_only_not_replay','execution_allowed':False,
            'physical_motion_sent':False,'source':str(args.source.resolve())}
    try:
        deadline=time.monotonic()+3
        while time.monotonic()<deadline: rclpy.spin_once(node,timeout_sec=.02)
        if any(n not in readings or time.monotonic()-readings[n][1]>.3 for n in NAMES):
            raise RuntimeError('Missing/stale actual feedback')
        actual=[readings[n][0]/(.03 if i in (7,15) else 1) for i,n in enumerate(NAMES)]
        if not all(math.isfinite(v) for v in actual): raise RuntimeError('Nonfinite feedback')
        report['actual']=actual
        req=GetPlanningScene.Request()
        req.components.components=req.components.WORLD_OBJECT_GEOMETRY | req.components.OCTOMAP
        scene=call(GetPlanningScene,'/get_planning_scene',req).scene
        report['world_objects']=[o.id for o in scene.world.collision_objects]
        report['octomap_bytes']=len(scene.world.octomap.octomap.data)
        # Hold the non-task right arm at the measured current pose for exploration.
        contact=list(actual);contact[:8]=data['actions'][0][:8];contact[7]=1.
        valid(contact)
        pose=fk(contact);o=pose.pose.orientation
        shift=backoff_vector((o.x,o.y,o.z,o.w),.08)
        report['backoff_m']=shift
        report['reference_contact_pose']=message_to_ordereddict(pose)
        candidates=[contact];max_fk_error=0.;max_angle=0.
        for k in range(1,17):
            target=copy.deepcopy(pose)
            target.pose.position.x+=shift[0]*k/16
            target.pose.position.y+=shift[1]*k/16
            target.pose.position.z+=shift[2]*k/16
            req=GetPositionIK.Request();r=req.ik_request
            r.group_name='left_arm';r.ik_link_name='openarm_left_hand'
            r.robot_state=state(candidates[-1]);r.pose_stamped=target
            r.avoid_collisions=True;r.timeout.nanosec=200000000
            res=call(GetPositionIK,'/compute_ik',req)
            if res.error_code.val!=1: raise RuntimeError(f'IK rejected at backoff {k*5} mm: {res.error_code.val}')
            values=dict(zip(res.solution.joint_state.name,res.solution.joint_state.position))
            q=list(contact)
            for i in range(7): q[i]=values[NAMES[i]]
            if not all(math.isfinite(v) for v in q): raise RuntimeError('Invalid IK')
            if max(abs(q[i]-candidates[-1][i]) for i in range(7))>.08: raise RuntimeError('IK branch jump')
            valid(q);got=fk(q).pose
            error=math.sqrt(sum((getattr(got.position,a)-getattr(target.pose.position,a))**2 for a in ('x','y','z')))
            dot=abs(sum(getattr(got.orientation,a)*getattr(o,a) for a in ('x','y','z','w')))
            angle=2*math.acos(min(1.,dot))
            if error>.0005 or angle>.005: raise RuntimeError('IK forward-check failed')
            max_fk_error=max(max_fk_error,error);max_angle=max(max_angle,angle)
            candidates.append(q)
        report.update(backoff_waypoints=candidates,backoff_waypoints_model_valid=True,
                      max_fk_position_error_m=max_fk_error,max_fk_angle_error_rad=max_angle)
        req=GetMotionPlan.Request();m=req.motion_plan_request
        m.group_name='left_arm';m.start_state=state(actual)
        m.num_planning_attempts=3;m.allowed_planning_time=5.
        m.max_velocity_scaling_factor=.1;m.max_acceleration_scaling_factor=.1
        c=Constraints()
        for i in range(7):
            j=JointConstraint();j.joint_name=NAMES[i];j.position=candidates[-1][i]
            j.tolerance_above=.002;j.tolerance_below=.002;j.weight=1.;c.joint_constraints.append(j)
        m.goal_constraints=[c]
        res=call(GetMotionPlan,'/plan_kinematic_path',req).motion_plan_response
        report['approach_planning_error_code']=res.error_code.val
        if res.error_code.val==1:
            report['model_only_approach']=message_to_ordereddict(res.trajectory)
        report['status']='explored_not_authorized_for_execution'
        report['blockers']=['Uncalibrated pegboard/hook/tape/support geometry; no certified obstacle clearance',
                            'Recorded grasp uses a different right-arm pose; full task and return transition not reconstructed',
                            'Swept-volume and real tracking-margin validation required before execution']
        print(json.dumps({k:v for k,v in report.items() if k not in ('model_only_approach','backoff_waypoints','reference_contact_pose','actual')},ensure_ascii=False,indent=2))
    except Exception as exc:
        report.update(status='exploration_rejected',error=str(exc));raise
    finally:
        with args.out.open('x') as f: json.dump(report,f,ensure_ascii=False,indent=2)
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__': main()
