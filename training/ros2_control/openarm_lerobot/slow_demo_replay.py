#!/usr/bin/env python3
"""Bounded joint replay, not visual servoing or a trained grasp policy.

Export with LeRobot Python; preflight/execute with ROS system Python.
Default mode is read-only preflight. Physical motion requires --execute.
No world collision guarantee; the demonstrated workspace must be restored.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

SIDES = ('left', 'right')
KEYS = [k for s in SIDES for k in
        ([f'{s}_joint{i}.pos' for i in range(1, 8)] + [f'{s}_gripper.pos'])]
NAMES = [k for s in SIDES for k in
         ([f'openarm_{s}_joint{i}' for i in range(1, 8)] + [f'openarm_{s}_finger_joint1'])]
ARMS = [i for i in range(16) if i not in (7, 15)]
CAMERAS = dict(camera1_global='10', camera2_left='9', camera3_right='3')
# Completion acceptance only, not a command limit or an encoder offset.
# Operator confirmed physically closed grippers; retain their raw feedback.
ZERO_GRIPPER_TOLERANCE = 0.10


def validate(actions, stamps):
    if len(actions) < 2 or len(actions) != len(stamps):
        raise ValueError('Need at least two matching samples')
    if not all(math.isfinite(t) for t in stamps) or stamps[0] != 0:
        raise ValueError('Invalid timestamps')
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError('Timestamps must strictly increase')
    for q in actions:
        if len(q) != 16 or not all(math.isfinite(x) for x in q):
            raise ValueError('Invalid action')
        if any(not 0 <= q[i] <= 1 for i in (7, 15)):
            raise ValueError('Gripper command outside [0,1]')


def validate_replay_start(current, first_action, start_at_curl=False):
    if len(current)!=16 or not all(math.isfinite(x) for x in current):
        raise ValueError('Invalid initial feedback')
    if start_at_curl:
        if len(first_action)!=16 or not all(math.isfinite(x) for x in first_action):
            raise ValueError('Invalid saved curl target')
        if max(abs(current[i]-first_action[i]) for i in ARMS)>.15:
            raise RuntimeError('Not near saved curl pose (0.15 rad); no motion sent')
    elif max(abs(current[i]) for i in ARMS)>.10:
        raise RuntimeError('Not at confirmed mechanical zero; no motion sent')


def timed_path(actions, stamps, current):
    """Preserve joint-space waypoints; slower locally for steep recorded steps."""
    validate(actions, stamps)
    if len(current) != 16 or not all(math.isfinite(v) for v in current):
        raise ValueError('Invalid initial feedback')
    initial = list(current)
    for i in (7, 15):
        initial[i] = max(0., min(1., initial[i]))
    bent = list(actions[0])
    for i in (7, 15):
        bent[i] = initial[i]
    duration = max(15., 1.875 * max(abs(bent[i]-initial[i]) for i in ARMS) / .12)
    points = [(0., initial, 'curl')]
    for j in range(1, math.ceil(duration * 50) + 1):
        u = min(1., j / math.ceil(duration * 50))
        s = 10*u**3 - 15*u**4 + 6*u**5
        points.append((u*duration, [a+(b-a)*s for a,b in zip(initial,bent)], 'curl'))
    points.append((duration + 6., list(actions[0]), 'open_grippers'))
    t = duration + 7.
    points.append((t, list(actions[0]), 'settle'))
    for j in range(1, len(actions)):
        prev, q = actions[j-1], actions[j]
        dt = max(4*(stamps[j]-stamps[j-1]),
                 max(abs(q[i]-prev[i]) for i in ARMS) / .18,
                 max(abs(q[i]-prev[i]) for i in (7, 15)) / .5)
        t += dt
        points.append((t, list(q), 'replay'))
    points.append((t+3., list(actions[-1]), 'release_hold'))
    return points


def interpolate(points, t, index):
    while index + 1 < len(points)-1 and points[index+1][0] <= t:
        index += 1
    a, b = points[index], points[index+1]
    u = max(0., min(1., (t-a[0])/(b[0]-a[0])))
    return [x+(y-x)*u for x,y in zip(a[1],b[1])], index, b[2]


def curl_only_path(actions, stamps, current):
    """Stop at the supplied known curl pose; never enter demonstrated grasp."""
    points=[p for p in timed_path(actions,stamps,current)
            if p[2] in ('curl','open_grippers','settle')]
    points.append((points[-1][0]+2.,list(points[-1][1]),'settle'))
    return points


def zero_path(current):
    """Slow joint-space return; retain gripper opening until arms reach zero.

    This is not motor encoder calibration and not an obstacle-aware planner.
    """
    if len(current)!=16 or not all(math.isfinite(v) for v in current):
        raise ValueError('Invalid initial feedback')
    initial=list(current)
    for i in (7,15):
        if not 0<=initial[i]<=1:
            raise ValueError('Initial gripper command outside [0,1]')
    duration=max(15.,1.875*max(abs(initial[i]) for i in ARMS)/.12)
    points=[(0.,initial,'return_zero')]
    steps=math.ceil(duration*50)
    for j in range(1,steps+1):
        u=j/steps
        weight=10*u**3-15*u**4+6*u**5
        q=[v*(1-weight) if i in ARMS else v for i,v in enumerate(initial)]
        points.append((u*duration,q,'return_zero'))
    arm_zero=points[-1][1]
    points.append((duration+2.,list(arm_zero),'zero_arm_settle'))
    for j in range(1,301):
        u=j/300
        weight=10*u**3-15*u**4+6*u**5
        q=[v*(1-weight) if i in (7,15) else 0. for i,v in enumerate(arm_zero)]
        points.append((duration+2.+u*6.,q,'close_at_zero'))
    points.append((duration+13.,[0.]*16,'zero_settle'))
    return points


def zero_pose_payload(samples):
    if len(samples)<20:
        raise ValueError('Insufficient settled zero feedback')
    for q in samples:
        if len(q)!=16 or not all(math.isfinite(v) for v in q):
            raise ValueError('Invalid zero feedback')
        if max(abs(q[i]) for i in ARMS)>.02 or max(abs(q[i]) for i in (7,15))>ZERO_GRIPPER_TOLERANCE:
            raise ValueError('Zero not reached within arm/gripper tolerances')
    spread=[max(q[i] for q in samples)-min(q[i] for q in samples) for i in range(16)]
    if max(spread[i] for i in ARMS)>.01 or max(spread[i] for i in (7,15))>.03:
        raise ValueError('Zero feedback not settled')
    means=[sum(q[i] for q in samples)/len(samples) for i in range(16)]
    return dict(schema_version=1,robot_type='openarm_ros2',label='standard_mechanical_zero',
                saved_at=datetime.now(timezone.utc).isoformat(),
                positions=dict.fromkeys(KEYS,0.),
                measured_positions=dict(zip(KEYS,means)),
                measured_ros_positions={n:means[i]*(.03 if i in (7,15) else 1.)
                                        for i,n in enumerate(NAMES)},
                measured_peak_to_peak=dict(zip(KEYS,spread)),sample_count=len(samples),
                gripper_scale_m=.03,arm_tolerance_rad=.02,
                gripper_tolerance_normalized=ZERO_GRIPPER_TOLERANCE,
                encoder_zero_rewritten=False,
                note='positions are commanded zero; measured fields retain real feedback, not offsets to apply.')


def validate_curl_completion(actual, reference, settled_samples, tolerance=.08):
    """Check the demonstrated *measured* start pose, not its action target."""
    if len(actual)!=16 or len(reference)!=16 or not all(math.isfinite(v) for v in actual+reference):
        raise ValueError('Invalid curl feedback/reference')
    if len(settled_samples)<20 or any(len(q)!=16 or not all(math.isfinite(v) for v in q)
                                      for q in settled_samples):
        raise RuntimeError('Curl feedback did not settle for a full second')
    spread=max(max(q[i] for q in settled_samples)-min(q[i] for q in settled_samples)
               for i in ARMS)
    if spread>.02:
        raise RuntimeError(f'Curl feedback still moving: spread {spread:.3f} rad')
    errors={NAMES[i]:abs(actual[i]-reference[i]) for i in ARMS}
    worst=max(errors,key=errors.get)
    if errors[worst]>tolerance:
        raise RuntimeError(f'Curl differs from demonstration observation: '
                           f'{worst} {errors[worst]:.3f} rad > {tolerance:g} rad')
    return dict(final_reference_max_error_rad=errors[worst],
                final_reference_worst_joint=worst,final_settle_spread_rad=spread)


def select_export_rows(rows, episode, end_s, fps):
    if not math.isfinite(end_s) or end_s <= 0 or fps <= 0:
        raise ValueError('Invalid export duration or fps')
    expected = round(end_s * fps)
    if abs(expected - end_s * fps) > 1e-6:
        raise ValueError('Export end must align with a frame boundary')
    rows = sorted((r for r in rows if r['episode_index'] == episode and
                   0 <= r['timestamp'] < end_s), key=lambda r:r['timestamp'])
    if len(rows) != expected or any(abs(r['timestamp'] - i/fps) > .001
                                    for i,r in enumerate(rows)):
        raise ValueError('Missing, duplicate or irregular demonstration frames')
    validate([r['action'] for r in rows], [r['timestamp'] for r in rows])
    return rows


def export(args):
    import pyarrow.parquet as pq
    info = json.loads((args.dataset/'meta/info.json').read_text())
    if info['features']['action']['names'] != KEYS:
        raise ValueError('Dataset action ordering mismatch')
    rows = []
    for file in sorted((args.dataset/'data').rglob('*.parquet')):
        rows.extend(pq.read_table(file, columns=['episode_index','timestamp','action']).to_pylist())
    rows = select_export_rows(rows, args.episode, args.end_s, info['fps'])
    actions, stamps = [r['action'] for r in rows], [r['timestamp'] for r in rows]
    validate(actions, stamps)
    payload = dict(keys=KEYS, source=str(args.dataset.resolve()),
                   episode_index=args.episode, end_exclusive_s=args.end_s,
                   actions=actions, timestamps=stamps)
    with args.plan.open('x') as f:
        json.dump(payload, f)
    print(f'Exported {len(rows)} unchanged command samples: {args.plan}', flush=True)


class CameraRecording:
    def __init__(self, output):
        self.stop = threading.Event()
        self.last = {}
        self.errors = []
        self.threads = []
        for name, port in CAMERAS.items():
            thread = threading.Thread(target=self.capture, args=(output,name,port), daemon=True)
            thread.start()
            self.threads.append(thread)

    def capture(self, output, name, port):
        import cv2
        cap = writer = None
        try:
            path = f'/dev/v4l/by-path/pci-0000:00:14.0-usb-0:{port}:1.0-video-index0'
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if not cap.isOpened():
                raise RuntimeError(f'{name}: cannot open')
            writer = cv2.VideoWriter(str(output/f'{name}.avi'),
                                     cv2.VideoWriter_fourcc(*'MJPG'), 30, (640,480))
            if not writer.isOpened():
                raise RuntimeError(f'{name}: cannot record')
            with (output/f'{name}_timestamps.jsonl').open('x') as log:
                frame_id = 0
                while not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame.shape[:2] != (480,640):
                        raise RuntimeError(f'{name}: missing/wrong-sized frame')
                    now = time.monotonic()
                    writer.write(frame)
                    log.write(json.dumps(dict(frame=frame_id,monotonic=now,wall=time.time()))+'\n')
                    if frame_id == 5:
                        cv2.imwrite(str(output/f'{name}_initial.jpg'),frame)
                    self.last[name] = now
                    frame_id += 1
        except Exception as e:
            self.errors.append(str(e))
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()

    def healthy(self):
        if self.errors:
            raise RuntimeError('; '.join(self.errors))
        return len(self.last)==3 and all(time.monotonic()-t < .7 for t in self.last.values())

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=2)


def run(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from rclpy.action import ActionClient
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String
    from control_msgs.action import GripperCommand
    from moveit_msgs.srv import GetStateValidity
    is_zero=args.mode=='zero'
    is_batch=args.mode=='batch'
    is_curl=args.mode=='curl'
    reference=None
    if not is_zero:
        data = json.loads(args.plan.read_text())
        if data['keys'] != KEYS:
            raise ValueError('Plan key ordering mismatch')
        validate(data['actions'], data['timestamps'])
        if is_curl and args.reference_observation is not None:
            payload=json.loads(args.reference_observation.read_text())
            if payload.get('keys')!=KEYS or payload.get('first_action')!=data['actions'][0]:
                raise ValueError('Curl reference does not match this plan')
            reference=payload.get('observation')
            if not isinstance(reference,list) or len(reference)!=16 or not all(
                    isinstance(v,(int,float)) and math.isfinite(v) for v in reference):
                raise ValueError('Invalid curl reference observation')
        if is_curl and getattr(args,'from_current',False) and reference is None:
            raise ValueError('--from-current requires --reference-observation; no motion sent')
    args.out.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = Node('slow_demo_replay')
    state, limits = {}, {}
    cameras = None
    display = None
    result = dict(status='preflight', motion_started=False, model_checks=0)
    last_goal = [None, None]
    futures = []
    def on_state(msg):
        now=time.monotonic()
        for name, pos in zip(msg.name,msg.position):
            state[name]=(pos,now)
    def on_description(msg):
        root=ET.fromstring(msg.data)
        for joint in root.findall('joint'):
            lim=joint.find('limit')
            if lim is not None and lim.get('lower') is not None:
                limits[joint.get('name')]=(float(lim.get('lower')),float(lim.get('upper')))
    node.create_subscription(JointState,'/joint_states',on_state,50)
    node.create_subscription(String,'/robot_description',on_description,
        QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE))
    def spin():
        rclpy.spin_once(node, timeout_sec=.002)
    def feedback():
        now=time.monotonic()
        if any(n not in state or now-state[n][1]>.3 for n in NAMES):
            raise RuntimeError('Actual joint feedback missing/stale >0.3 s')
        q=[state[n][0]/(.03 if i in (7,15) else 1) for i,n in enumerate(NAMES)]
        if not all(math.isfinite(x) for x in q):
            raise RuntimeError('Non-finite actual feedback')
        return q
    def await_future(future, timeout=3):
        deadline=time.monotonic()+timeout
        while not future.done() and time.monotonic()<deadline:
            spin()
        if not future.done():
            raise RuntimeError('ROS service/action timeout')
        return future.result()
    try:
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and (len(state)<16 or not limits):
            spin()
        current=feedback()
        if not is_zero:
            from_current = is_curl and getattr(args,'from_current',False)
            if not from_current:
                validate_replay_start(current,data['actions'][0],
                                      is_batch and getattr(args,'start_at_curl',False))
            result['start_mode']=('current_feedback' if from_current else
                                  'near_saved_curl' if is_batch and getattr(args,'start_at_curl',False) else
                                  'mechanical_zero')
        topics=[f'/{s}_forward_position_controller/commands' for s in SIDES]
        if any(node.count_publishers(t) for t in topics):
            raise RuntimeError('Competing arm command publisher')
        if node.count_publishers('/joint_states') != 1:
            raise RuntimeError('Require exactly one actual joint-state publisher')
        # Encoder feedback near a mechanical stop can be slightly negative.
        # Only the initial COMMAND is projected into URDF bounds. Keep raw
        # observations; reject larger offsets, never relax trajectory limits.
        initial_command=list(current)
        for i,n in enumerate(NAMES):
            if n not in limits:
                raise RuntimeError(f'Missing URDF limit: {n}')
            scale=.03 if i in (7,15) else 1.
            p=current[i]*scale
            lo,hi=limits[n]
            tolerance=.001 if i in (7,15) else .01
            if not lo-tolerance <= p <= hi+tolerance:
                raise RuntimeError(f'Initial feedback outside near-zero tolerance: {n}={p}')
            initial_command[i]=max(lo,min(hi,p))/scale
        for i in (7,15):
            initial_command[i]=min(1.,initial_command[i])
        points=(zero_path(initial_command) if is_zero else
                timed_path(data['actions'],data['timestamps'],initial_command))
        if is_curl:
            points=curl_only_path(data['actions'],data['timestamps'],initial_command)
            result['preparation_only']=True
        if is_batch:
            from batch_demo_record import batch_path, BatchDisplay
            points,windows=batch_path(data['actions'],data['timestamps'],initial_command,args.count)
            result['episode_windows']=windows
        result.update(initial=current,duration_s=points[-1][0],points=len(points))
        service=node.create_client(GetStateValidity,'/check_state_validity')
        if not service.wait_for_service(timeout_sec=5):
            raise RuntimeError('Self-collision checking service unavailable')
        # Every waypoint plus subdivisions <=0.01 rad / 0.001 m finger joint.
        samples=[current]
        for a,b in zip(points,points[1:]):
            subdivisions=max(1,math.ceil(max(abs(x-y)*(.03 if i in (7,15) else 1)/
                       (.001 if i in (7,15) else .01) for i,(x,y) in enumerate(zip(a[1],b[1])))))
            samples.extend([[x+(y-x)*k/subdivisions for x,y in zip(a[1],b[1])]
                            for k in range(1,subdivisions+1)])
        for j,q in enumerate(samples):
            positions=[v*(.03 if i in (7,15) else 1) for i,v in enumerate(q)]
            invalid = []
            for i,(n,p) in enumerate(zip(NAMES,positions)):
                tolerance=(.001 if i in (7,15) else .01) if j==0 else 1e-8
                if n not in limits or not limits[n][0]-tolerance<=p<=limits[n][1]+tolerance:
                    invalid.append((n,p,limits.get(n)))
            if invalid:
                raise RuntimeError(f'Joint limit violation at sample {j}: {invalid}')
            req=GetStateValidity.Request()
            req.robot_state.joint_state.name=NAMES
            req.robot_state.joint_state.position=positions
            req.robot_state.is_diff=False
            response=await_future(service.call_async(req))
            result['model_checks']+=1
            if not response.valid:
                contacts=[(c.contact_body_1,c.contact_body_2) for c in response.contacts]
                raise RuntimeError(f'Model collision/invalid state at sample {j}: {contacts}')
            if j%250==0:
                print(f'模型检查 {j}/{len(samples)}（不包含未建模托架）',flush=True)
        print(f'轨迹检查通过，计划用时 {points[-1][0]:.1f} 秒；未发送运动目标。',flush=True)
        cameras=CameraRecording(args.out)
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and not cameras.healthy():
            spin()
        if not cameras.healthy():
            raise RuntimeError('Three camera streams not ready')
        # Capture enough frames for initial snapshots, even in preflight mode.
        until=time.monotonic()+1
        while time.monotonic()<until:
            spin()
        if not args.execute:
            result['status']='preflight_passed'
            return
        if is_batch:
            display=BatchDisplay(points,args.count)
        publishers=[node.create_publisher(Float64MultiArray,t,1) for t in topics]
        clients=[ActionClient(node,GripperCommand,f'/{s}_gripper_controller/gripper_cmd') for s in SIDES]
        if not all(c.wait_for_server(timeout_sec=3) for c in clients):
            raise RuntimeError('Gripper server unavailable')
        print(('三路录像已开始。5 秒后仅弯举、张爪并停住；不执行抓取或自动接近。' if is_curl else
               '三路录像已开始。5 秒后准备弯举；按窗口提示等待 15 秒，每次抓取录制 20 秒。' if is_batch else
               '三路录像已开始。5 秒后缓慢回机械零位，到位后闭合夹爪；不重写电机零点。'
               if is_zero else '三路录像已开始。5 秒后慢速弯举、张爪、按示范抓取放入；现场急停保持可用。'),flush=True)
        until=time.monotonic()+5
        while time.monotonic()<until:
            spin()
            feedback()
            if display:
                display.update('curl',0)
            if not cameras.healthy():
                raise RuntimeError('Camera lost before motion')
        current2=feedback()
        if max(abs(current2[i]-current[i]) for i in ARMS)>.025:
            raise RuntimeError('Initial pose changed after preflight')
        if any(node.count_publishers(t)!=1 for t in topics):
            raise RuntimeError('Competing arm publisher before motion')
        start=time.monotonic()
        result['motion_start_monotonic']=start
        prev_time=start
        previous=list(points[0][1])
        index=0
        phase=None
        settled_samples=[]
        result['motion_started']=True
        with (args.out/'commands_feedback.jsonl').open('x') as log:
            while True:
                spin()
                now=time.monotonic()
                if now-prev_time>.12:
                    raise RuntimeError('Control scheduling stall >0.12 s')
                if now-prev_time < .02:
                    continue
                elapsed=now-start
                actual=feedback()
                if not cameras.healthy():
                    raise RuntimeError('Camera stream stale during motion')
                if max(abs(actual[i]-previous[i]) for i in ARMS)>.20:
                    raise RuntimeError('Arm tracking error >0.20 rad; no further targets')
                if any(node.count_publishers(t)!=1 for t in topics):
                    raise RuntimeError('Competing command publisher during motion')
                for fut,sent_at in list(futures):
                    if fut.done():
                        if not fut.result().accepted:
                            raise RuntimeError('Gripper goal rejected')
                        futures.remove((fut,sent_at))
                    elif now-sent_at>1:
                        raise RuntimeError('Gripper goal acknowledgement timeout')
                target,index,new_phase=interpolate(points,elapsed,index)
                if display:
                    display.update(new_phase,elapsed)
                if is_batch:
                    for window in windows:
                        if elapsed>=window['end_s']:
                            window['completed']=True
                    # Validate each return before the operator's reset window.
                    if new_phase.startswith('wait:') and max(abs(actual[i]-target[i]) for i in ARMS)>.15:
                        raise RuntimeError('Curl position not reached before reset wait')
                if new_phase != phase:
                    phase=new_phase
                    print(f'{elapsed:.1f}s 阶段：{phase}',flush=True)
                if max(abs(target[i]-previous[i]) for i in ARMS)>.025:
                    raise RuntimeError('Unexpected command step >0.025 rad')
                for s,pub in enumerate(publishers):
                    pub.publish(Float64MultiArray(data=target[s*8:s*8+7]))
                    grip=target[s*8+7]
                    if (last_goal[s] is None or abs(grip-last_goal[s])>=.005 or
                            (grip in (0.,1.) and grip!=last_goal[s])):
                        goal=GripperCommand.Goal()
                        goal.command.position=grip*.03
                        goal.command.max_effort=10.
                        futures.append((clients[s].send_goal_async(goal),now))
                        last_goal[s]=grip
                log.write(json.dumps(dict(monotonic=now,elapsed=elapsed,phase=phase,
                                          command=target,actual=actual))+'\n')
                previous=target
                prev_time=now
                if (is_zero or is_curl and reference is not None) and elapsed>=points[-1][0]-1.:
                    settled_samples.append(actual)
                if elapsed>=points[-1][0]:
                    if is_curl and reference is not None:
                        result.update(final_actual=actual,final_reference=reference)
                        result.update(validate_curl_completion(actual,reference,settled_samples))
                    elif max(abs(actual[i]-target[i]) for i in ARMS)>.08:
                        raise RuntimeError('Final arm target not reached')
                    break
        if is_batch:
            result['status']='batch_completed_success_labels_pending'
            print(f'{args.count} 次抓取回放及回弯举结束，电机保持使能。待裁出每段 20 秒录像。',flush=True)
        elif is_curl:
            result['status']='curl_preparation_completed'
            print('弯举、张爪准备结束；不自动接近，不执行抓取。电机保持使能。',flush=True)
        elif is_zero:
            payload=zero_pose_payload(settled_samples)
            with (args.out/'zero_pose.json').open('x') as file:
                json.dump(payload,file,ensure_ascii=False,indent=2)
            result.update(status='zero_reached_and_saved',zero_pose=str((args.out/'zero_pose.json').resolve()))
            print(f'已回零并保存实测信息：{args.out / "zero_pose.json"}。电机保持使能，未重写电机零点。',flush=True)
        else:
            result['status']='replay_completed_visual_success_unverified'
            print('慢速回放结束；不回零、不关闭电机。是否抓取成功需核对录像。',flush=True)
    except BaseException as e:
        result.update(status='stopped',error=str(e))
        if not result['motion_started']:
            print(f'预检未通过，未发送本次运动目标：{e}。请检查驱动和反馈状态；这不是硬件急停。',flush=True)
        else:
            print(f'停止发送新目标：{e}。不能据此保证真机已停止或仍保持姿态；异常时使用现场硬件急停。',flush=True)
        raise
    finally:
        if cameras:
            cameras.close()
        if display:
            display.close()
        (args.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    ex=sub.add_parser('export')
    ex.add_argument('--dataset',type=Path,required=True)
    ex.add_argument('--plan',type=Path,required=True)
    ex.add_argument('--episode',type=int,default=0)
    ex.add_argument('--end-s',type=float,default=11.)
    replay=sub.add_parser('run')
    replay.add_argument('--plan',type=Path,required=True)
    replay.add_argument('--out',type=Path,required=True)
    replay.add_argument('--execute',action='store_true')
    curl=sub.add_parser('curl',help='Known curl preparation only; no approach or grasp playback')
    curl.add_argument('--plan',type=Path,required=True)
    curl.add_argument('--out',type=Path,required=True)
    curl.add_argument('--execute',action='store_true')
    curl.add_argument('--from-current',action='store_true',
                      help='For supervised task switching only: interpolate from live feedback; still check limits, self-collision, cameras and publisher conflicts')
    curl.add_argument('--reference-observation',type=Path,
                      help='Saved first actual observation from the exact demonstration; required with --from-current')
    zero=sub.add_parser('zero',help='Return to existing mechanical zero, then save target and measured feedback')
    zero.add_argument('--out',type=Path,required=True)
    zero.add_argument('--execute',action='store_true')
    batch=sub.add_parser('batch',help='20 s grasp clips, slow curl return, 15 s manual reset')
    batch.add_argument('--plan',type=Path,required=True)
    batch.add_argument('--out',type=Path,required=True)
    batch.add_argument('--count',type=int,choices=range(1,11),default=10)
    batch.add_argument('--execute',action='store_true')
    batch.add_argument('--start-at-curl',action='store_true',
                       help='Require current arms within 0.15 rad of saved curl; still check and slowly approach it before recording')
    args=parser.parse_args()
    (export if args.mode=='export' else run)(args)


if __name__=='__main__':
    main()
