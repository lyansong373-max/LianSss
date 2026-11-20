#!/usr/bin/env python3
"""One bounded left-joint microstep, never a loop of autonomous approach moves.

Requires the known last command log and matching stationary feedback. Retains
the held command reference rather than replacing it with sagging feedback.
Default is preflight only. No right-arm or gripper command interface is used.
"""
import argparse
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from slow_demo_replay import NAMES, ARMS, CameraRecording


def target_at(reference, elapsed):
    if len(reference)!=16 or not all(math.isfinite(v) for v in reference):
        raise ValueError('Invalid reference')
    if not math.isfinite(elapsed): raise ValueError('Invalid elapsed time')
    u=max(0.,min(1.,elapsed/3.))
    q=list(reference);q[0]-=.002*(10*u**3-15*u**4+6*u**5)
    return q


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-log',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    last=None
    with args.reference_log.open() as f:
        for line in f:
            if line.strip(): last=json.loads(line)
    if last is None: raise ValueError('Empty command log')
    reference=last['command'];expected=last['actual'];target_at(reference,0)
    args.out.mkdir(exist_ok=False)
    import rclpy
    from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String,Float64MultiArray
    from moveit_msgs.srv import GetStateValidity,GetPositionFK
    from controller_manager_msgs.srv import ListControllers
    rclpy.init();node=rclpy.create_node('left_one_microstep')
    values={};limits={};cameras=None;pub=None
    report={'motion_started':False,'status':'preflight','source_log':str(args.reference_log.resolve()),
            'delta_joint1_rad':-.002,'ramp_s':3.,'right_arm_commanded':False,'grippers_commanded':False}
    def spin(): rclpy.spin_once(node,timeout_sec=.002)
    def on_state(m): values.update({n:(v,time.monotonic()) for n,v in zip(m.name,m.position)})
    def on_desc(m):
        for j in ET.fromstring(m.data).findall('joint'):
            v=j.find('limit')
            if v is not None and v.get('lower') is not None:
                limits[j.get('name')]=(float(v.get('lower')),float(v.get('upper')))
    node.create_subscription(JointState,'/joint_states',on_state,50)
    node.create_subscription(String,'/robot_description',on_desc,QoSProfile(depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE))
    clients={}
    def call(kind,name,req):
        if name not in clients: clients[name]=node.create_client(kind,name)
        c=clients[name]
        if not c.wait_for_service(timeout_sec=3): raise RuntimeError('Service unavailable '+name)
        f=c.call_async(req);deadline=time.monotonic()+3
        while not f.done() and time.monotonic()<deadline: spin()
        if not f.done() or f.result() is None: raise RuntimeError('Service timeout '+name)
        return f.result()
    def feedback():
        if any(n not in values or time.monotonic()-values[n][1]>.3 for n in NAMES):
            raise RuntimeError('Feedback missing/stale >0.3s')
        q=[values[n][0]/(.03 if i in (7,15) else 1.) for i,n in enumerate(NAMES)]
        if not all(math.isfinite(x) for x in q): raise RuntimeError('Nonfinite feedback')
        return q
    def joint_state(req,q):
        req.robot_state.joint_state.name=NAMES
        req.robot_state.joint_state.position=[float(x)*(.03 if i in (7,15) else 1.) for i,x in enumerate(q)]
    def fk(q):
        req=GetPositionFK.Request();req.header.frame_id='openarm_body_link0'
        req.fk_link_names=['openarm_left_hand'];joint_state(req,q)
        r=call(GetPositionFK,'/compute_fk',req)
        if r.error_code.val!=1: raise RuntimeError('FK failed')
        pos=r.pose_stamped[0].pose.position
        return [pos.x,pos.y,pos.z]
    left='/left_forward_position_controller/commands'
    right='/right_forward_position_controller/commands'
    try:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline and (len(values)<16 or not limits): spin()
        initial=feedback();report['initial_actual']=initial;report['held_reference']=reference
        if max(abs(initial[i]-expected[i]) for i in ARMS)>.006:
            raise RuntimeError('Actual posture differs from known last-command log; no takeover')
        if node.count_publishers(left) or node.count_publishers(right) or node.count_publishers('/joint_states')!=1:
            raise RuntimeError('Competing publishers or non-unique feedback')
        controllers=call(ListControllers,'/controller_manager/list_controllers',ListControllers.Request())
        if not any(c.name=='left_forward_position_controller' and c.state=='active' for c in controllers.controller):
            raise RuntimeError('Left forward-position controller not active')
        stationary=[];deadline=time.monotonic()+1
        while time.monotonic()<deadline: spin();stationary.append(feedback())
        if max(max(q[i] for q in stationary)-min(q[i] for q in stationary) for i in ARMS)>.003:
            raise RuntimeError('Arm not stationary')
        checks=0
        for k in range(11):
            command=target_at(reference,3*k/10)
            predicted=list(initial);predicted[0]+=command[0]-reference[0]
            hybrid=list(command);hybrid[8:]=initial[8:]
            for q in (command,predicted,hybrid):
                for i,n in enumerate(NAMES):
                    v=q[i]*(.03 if i in (7,15) else 1.)
                    if n not in limits or not limits[n][0]-1e-8<=v<=limits[n][1]+1e-8:
                        raise RuntimeError('Joint limit: '+n)
                req=GetStateValidity.Request();joint_state(req,q)
                if not call(GetStateValidity,'/check_state_validity',req).valid:
                    raise RuntimeError('Invalid model state')
                checks+=1
        report['model_checks']=checks;report['initial_hand_xyz']=fk(initial)
        predicted=list(initial);predicted[0]-=.002
        report['predicted_hand_delta_m']=[b-a for a,b in zip(report['initial_hand_xyz'],fk(predicted))]
        cameras=CameraRecording(args.out);deadline=time.monotonic()+8
        while time.monotonic()<deadline and not cameras.healthy(): spin()
        if not cameras.healthy(): raise RuntimeError('Cameras unavailable')
        deadline=time.monotonic()+1
        while time.monotonic()<deadline: spin()
        if not args.execute:
            report['status']='preflight_passed';print(json.dumps(report,indent=2),flush=True);return
        print('预检通过。5秒后仅左关节1缓慢变化 -0.002rad；保持其他目标。现场急停保持可用。',flush=True)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            spin();q=feedback()
            if max(abs(q[i]-initial[i]) for i in ARMS)>.003: raise RuntimeError('Pose changed before motion')
            if not cameras.healthy(): raise RuntimeError('Camera lost')
        if node.count_publishers(left) or node.count_publishers(right): raise RuntimeError('Competing publisher')
        pub=node.create_publisher(Float64MultiArray,left,1)
        start=time.monotonic();last_tick=start;previous=list(reference)
        with (args.out/'commands_feedback.jsonl').open('x') as f:
            while True:
                spin();now=time.monotonic()
                if now-last_tick>.12: raise RuntimeError('Scheduling stall')
                if now-last_tick<.02: continue
                q=feedback()
                if max(abs(q[i]-initial[i]) for i in ARMS)>.008:
                    raise RuntimeError('Measured arm displacement >0.008rad; stop new targets')
                if not cameras.healthy(): raise RuntimeError('Camera stale')
                if node.count_publishers(left)!=1 or node.count_publishers(right): raise RuntimeError('Competing publisher')
                elapsed=now-start;target=target_at(reference,elapsed)
                if max(abs(target[i]-previous[i]) for i in range(7))>.0002:
                    raise RuntimeError('Unexpected command step')
                pub.publish(Float64MultiArray(data=target[:7]));report['motion_started']=True
                f.write(json.dumps({'monotonic':now,'elapsed':elapsed,'command':target,'actual':q,
                                    'published_joints':NAMES[:7]})+'\n')
                previous=target;last_tick=now
                if elapsed>=5: break
        final=feedback();report['final_actual']=final
        report['measured_hand_delta_m']=[b-a for a,b in zip(report['initial_hand_xyz'],fk(final))]
        report['status']='single_step_finished_no_automatic_next_step'
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as exc:
        report.update(status='stopped',error=str(exc))
        print('停止发送新目标；这不是硬件急停或断使能。原因：'+str(exc),flush=True)
        raise
    finally:
        if pub is not None: node.destroy_publisher(pub)
        if cameras is not None: cameras.close()
        with (args.out/'result.json').open('x') as f: json.dump(report,f,ensure_ascii=False,indent=2)
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__': main()
