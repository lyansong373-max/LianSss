#!/usr/bin/env python3
"""Read-only FK/IK candidate generation; never publishes robot commands."""
import argparse
import copy
import json
import math
from pathlib import Path
from slow_demo_replay import KEYS, NAMES, validate


def window_weight(t, window):
    """Quintic approach/exit; preserve source poses outside the time window."""
    if window is None:
        return 1.
    if len(window)!=4 or not all(math.isfinite(v) for v in window):
        raise ValueError('Window needs four finite timestamps')
    a,b,c,d=window
    if not 0<=a<b<=c<d:
        raise ValueError('Require 0 <= ramp start < full start <= full end < ramp end')
    if not math.isfinite(t):
        raise ValueError('Invalid timestamp')
    if t<=a or t>=d:
        return 0.
    if b<=t<=c:
        return 1.
    u=(t-a)/(b-a) if t<b else (d-t)/(d-c)
    return 10*u**3-15*u**4+6*u**5


def merge_right(original, values, previous=None, max_joint_offset=.15):
    if not math.isfinite(max_joint_offset) or not 0<max_joint_offset<=.3:
        raise ValueError('Invalid candidate joint-offset allowance')
    q=list(original)
    for i in range(8,15):
        value=values[NAMES[i]]
        if not math.isfinite(value) or abs(value-original[i])>max_joint_offset:
            raise ValueError(f'IK exceeds candidate joint-offset allowance {max_joint_offset} rad')
        q[i]=value
    if previous is not None and max(abs(q[i]-previous[i]) for i in range(8,15))>.08:
        raise ValueError('Discontinuous right-arm IK branch')
    return q


def lateral_direction(left, right, choice):
    if choice not in ('left','right'):
        raise ValueError('Invalid lateral direction')
    v=[r-l for l,r in zip(left,right)]
    if len(v)!=3 or not all(math.isfinite(x) for x in v):
        raise ValueError('Invalid arm-base anchors')
    norm=math.sqrt(sum(x*x for x in v))
    if norm<.01 or abs(v[2]/norm)>.1:
        raise ValueError('Cannot establish horizontal robot lateral direction')
    sign=-1. if choice=='left' else 1.
    return [sign*x/norm for x in v]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--offset-mm',type=float,required=True)
    parser.add_argument('--direction',choices=('right','left','down'),default='right')
    parser.add_argument('--window-s',type=float,nargs=4,
                        help='Source seconds: ramp start, full shift start/end, ramp end. Required for down.')
    parser.add_argument('--max-joint-offset-rad',type=float,default=.15,
                        help='Planning displacement from original, not a speed or tracking limit; maximum 0.3 rad')
    args=parser.parse_args()
    # Operator authorized planning a larger lateral avoidance. This changes
    # only the requested Cartesian offset, not joint/velocity/collision guards.
    if not math.isfinite(args.offset_mm) or not 0<args.offset_mm<=80:
        raise ValueError('Only offsets up to 80 mm are allowed')
    if args.out.exists():
        raise FileExistsError(args.out)
    data=json.loads(args.source.read_text())
    if data['keys']!=KEYS:
        raise ValueError('Joint order mismatch')
    validate(data['actions'],data['timestamps'])
    if args.direction=='down' and args.window_s is None:
        raise ValueError('Downward adjustment requires a bounded phase window')
    if args.window_s is not None:
        window_weight(0.,args.window_s)
        if args.window_s[-1]>data['timestamps'][-1]:
            raise ValueError('Offset window exceeds source recording')
    import rclpy
    from rclpy.node import Node
    from moveit_msgs.msg import RobotState
    from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetStateValidity
    rclpy.init()
    node=Node('right_offset_plan_only')
    clients={}
    for service,typ in (('/compute_fk',GetPositionFK),('/compute_ik',GetPositionIK),
                        ('/check_state_validity',GetStateValidity)):
        clients[service]=node.create_client(typ,service)
    def call(service,request):
        client=clients[service]
        if not client.wait_for_service(timeout_sec=3):
            raise RuntimeError(f'Service unavailable: {service}')
        future=client.call_async(request)
        rclpy.spin_until_future_complete(node,future,timeout_sec=4)
        if not future.done() or future.result() is None:
            raise RuntimeError(f'Service timeout: {service}')
        return future.result()
    def state(q):
        value=RobotState()
        value.is_diff=False
        value.joint_state.name=NAMES
        value.joint_state.position=[float(v)*(.03 if i in (7,15) else 1.) for i,v in enumerate(q)]
        return value
    def fk(q,links):
        req=GetPositionFK.Request()
        req.header.frame_id='openarm_body_link0'
        req.fk_link_names=links
        req.robot_state=state(q)
        response=call('/compute_fk',req)
        if response.error_code.val!=1 or len(response.pose_stamped)!=len(links):
            raise RuntimeError(f'FK failed: {response.error_code.val}')
        return response.pose_stamped
    try:
        anchors=fk(data['actions'][0],['openarm_left_link0','openarm_right_link0'])
        left,right=[p.pose.position for p in anchors]
        direction=lateral_direction((left.x,left.y,left.z),(right.x,right.y,right.z),
                                    'left' if args.direction=='left' else 'right')
        if args.direction=='down':
            # Body is mounted upright. Verify world +Z coincides with body +Z.
            world=fk(data['actions'][0],['world'])[0].pose.orientation
            if abs(world.x)>.001 or abs(world.y)>.001:
                raise ValueError('Body/world vertical axes are not aligned')
            direction=[0.,0.,-1.]
        shift=[v*args.offset_mm/1000 for v in direction]
        print('候选方向（机身基座坐标）:',direction,'最大偏移米:',shift,flush=True)
        result=[]
        max_error=0.
        max_angle=0.
        for index,original in enumerate(data['actions']):
            weight=window_weight(data['timestamps'][index],args.window_s)
            if weight==0.:
                q=merge_right(original,dict(zip(NAMES,original)),result[-1] if result else None,
                              args.max_joint_offset_rad)
                result.append(q)
                continue
            target=copy.deepcopy(fk(original,['openarm_right_hand'])[0])
            target.pose.position.x+=shift[0]*weight
            target.pose.position.y+=shift[1]*weight
            target.pose.position.z+=shift[2]*weight
            seed=list(original)
            if result:
                for i in range(8,15):
                    seed[i]+=result[-1][i]-data['actions'][index-1][i]
            req=GetPositionIK.Request()
            req.ik_request.group_name='right_arm'
            req.ik_request.ik_link_name='openarm_right_hand'
            req.ik_request.robot_state=state(seed)
            req.ik_request.pose_stamped=target
            req.ik_request.avoid_collisions=True
            req.ik_request.timeout.sec=0
            req.ik_request.timeout.nanosec=100000000
            response=call('/compute_ik',req)
            if response.error_code.val!=1:
                # Diagnose a rejected candidate only. This solution is never
                # appended or saved; execution collision checks stay enabled.
                diagnostic=copy.deepcopy(req)
                diagnostic.ik_request.avoid_collisions=False
                probe=call('/compute_ik',diagnostic)
                if probe.error_code.val==1:
                    check=GetStateValidity.Request()
                    check.robot_state=probe.solution
                    v=call('/check_state_validity',check)
                    print('Rejected candidate contacts:',
                          [(c.contact_body_1,c.contact_body_2,c.depth) for c in v.contacts],flush=True)
                raise RuntimeError(f'IK rejected frame {index}: {response.error_code.val}')
            solution=dict(zip(response.solution.joint_state.name,response.solution.joint_state.position))
            q=merge_right(original,solution,result[-1] if result else None,args.max_joint_offset_rad)
            check=GetStateValidity.Request()
            check.robot_state=state(q)
            validity=call('/check_state_validity',check)
            if not validity.valid:
                raise RuntimeError(f'Collision at frame {index}: '+str(
                    [(c.contact_body_1,c.contact_body_2) for c in validity.contacts]))
            actual=fk(q,['openarm_right_hand'])[0].pose
            xyz=lambda p:(p.x,p.y,p.z)
            error=math.sqrt(sum((a-b)**2 for a,b in zip(xyz(actual.position),xyz(target.pose.position))))
            quat=lambda q:(q.x,q.y,q.z,q.w)
            dot=abs(sum(a*b for a,b in zip(quat(actual.orientation),quat(target.pose.orientation))))
            angle=2*math.acos(min(1.,dot))
            if error>.0005 or angle>.005:
                raise RuntimeError(f'FK verification failed at frame {index}: {error}, {angle}')
            max_error=max(max_error,error)
            max_angle=max(max_angle,angle)
            result.append(q)
            if index%50==0:
                print(f'已验证 {index}/{len(data["actions"])} 个候选动作点；未发送运动',flush=True)
        validate(result,data['timestamps'])
        payload=dict(data)
        payload['actions']=result
        if 'right_offset_preparation' in data:
            payload['prior_offset_preparations']=list(data.get('prior_offset_preparations',[]))+[
                copy.deepcopy(data['right_offset_preparation'])]
        payload['right_offset_preparation']={
            'source_plan':str(args.source.resolve()),'offset_mm':args.offset_mm,
            'max_joint_offset_allowance_rad':args.max_joint_offset_rad,
            'max_joint_offset_rad':max(abs(a[i]-b[i]) for a,b in zip(result,data['actions']) for i in range(8,15)),
            'frame':'openarm_body_link0','direction':direction,
            'window_s':args.window_s,'direction_choice':args.direction,
            'left_arm_and_both_grippers_unchanged':True,
            'max_fk_position_error_m':max_error,'max_fk_orientation_error_rad':max_angle,
            'shifted_waypoints_model_valid':True,'full_transition_preflight_required':True,
            'physical_motion_sent':False,
            'note':'Body-frame translation; down requires upright body/world axes. Source outside window is unchanged. No camera extrinsic calibration or world-clearance guarantee.'}
        with args.out.open('x') as f:
            json.dump(payload,f,ensure_ascii=False)
        print('候选轨迹已保存（未运动）:',args.out,flush=True)
    finally:
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':
    main()
