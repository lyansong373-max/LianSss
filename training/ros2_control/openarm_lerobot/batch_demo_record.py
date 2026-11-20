"""Timing/UI and postprocessing for repeated real-time demonstration replay."""
import argparse
import bisect
import json
import math
from pathlib import Path
import subprocess
from slow_demo_replay import ARMS, CAMERAS, timed_path


def fit_replay_times(actions, durations, budget=18.):
    """Keep every waypoint; allocate time without raising existing speed bounds."""
    minimum=[max(max(abs(b[i]-a[i])/.499 for i in ARMS),
                 max(abs(b[i]-a[i])/1.499 for i in (7,15)))
             for a,b in zip(actions,actions[1:])]
    if len(minimum)!=len(durations) or any(dt<=0 for dt in durations):
        raise ValueError('Invalid replay intervals')
    spare=budget-sum(minimum)
    if spare<=0:
        raise ValueError('20 s clip is infeasible within existing speed bounds')
    total=sum(durations)
    return [lo+spare*dt/total for lo,dt in zip(minimum,durations)]


def batch_path(actions, stamps, current, count=10):
    if not isinstance(count,int) or not 1<=count<=10:
        raise ValueError('Count must be 1–10')
    base=timed_path(actions,stamps,current)
    points=[p for p in base if p[2] in ('curl','open_grippers','settle')]
    origin=points[-1][0]
    replay=[p for p in base if p[2]=='replay']
    duration=replay[-1][0]-origin
    scale=18./duration
    previous_times=[origin]+[p[0] for p in replay[:-1]]
    intervals=[(p[0]-t)*scale for p,t in zip(replay,previous_times)]
    # Keep the original screwdriver timing if valid. Otherwise allocate slack
    # locally instead of increasing the speed limits or dropping waypoints.
    if any(max(abs(b[i]-a[i])/dt for i in ARMS)>.5 or
           max(abs(b[i]-a[i])/dt for i in (7,15))>1.5
           for a,b,dt in zip(actions,actions[1:],intervals)):
        intervals=fit_replay_times(actions,intervals)
    elapsed=origin
    adjusted=[]
    for p,dt in zip(replay,intervals):
        elapsed+=dt
        adjusted.append((elapsed,p[1],p[2]))
    replay=adjusted
    scale=1.
    # Explicit batch bounds, not relaxation of tracking or positional limits.
    prev=(origin,actions[0],'')
    for point in replay:
        dt=(point[0]-prev[0])*scale
        if max(abs(point[1][i]-prev[1][i])/dt for i in ARMS)>.5:
            raise ValueError('20 s clip would require arm target speed >0.5 rad/s')
        if max(abs(point[1][i]-prev[1][i])/dt for i in (7,15))>1.5:
            raise ValueError('20 s clip would require gripper speed >1.5 normalized/s')
        prev=point
    t=origin+15.
    points.append((t,list(actions[0]),'wait:00'))
    windows=[]
    for episode in range(1,count+1):
        began=t
        for point in replay:
            points.append((began+(point[0]-origin)*scale,list(point[1]),f'episode:{episode:02d}'))
        t=began+20.
        points.append((t,list(actions[-1]),f'episode:{episode:02d}'))
        windows.append(dict(episode=episode,start_s=began,end_s=t,duration_s=20.))
        # Return only after release. Grippers stay open; no reverse grasp path.
        duration_return=max(6.,1.875*max(abs(actions[-1][i]-actions[0][i]) for i in ARMS)/.12)
        steps=math.ceil(duration_return*50)
        for j in range(1,steps+1):
            u=j/steps
            s=10*u**3-15*u**4+6*u**5
            q=[a+(b-a)*s for a,b in zip(actions[-1],actions[0])]
            points.append((t+u*duration_return,q,f'return:{episode:02d}'))
        t+=duration_return+2.
        points.append((t,list(actions[0]),f'return:{episode:02d}'))
        windows[-1]['return_end_s']=t
        if episode<count:
            t+=15.
            points.append((t,list(actions[0]),f'wait:{episode:02d}'))
            windows[-1]['reset_wait_end_s']=t
    return points,windows


class BatchDisplay:
    """Visible countdown; abort requests checked before every outgoing target."""
    def __init__(self, points, count):
        import tkinter as tk
        self.root=tk.Tk()
        self.root.title('OpenArm 录制倒计时 — 现场硬件急停保持可用')
        self.root.geometry('720x300+80+60')
        self.root.attributes('-topmost',True)
        self.stopped=False
        self.last_phase=None
        self.last_second=None
        self.count=count
        self.ends={p[2]:p[0] for p in points}
        self.label=tk.Label(self.root,text='预检已通过，准备开始',font=('Sans',24),wraplength=700)
        self.label.pack(fill='both',expand=True)
        self.timer=tk.Label(self.root,text='',font=('Sans',42))
        self.timer.pack()
        self.warning=tk.Label(self.root,text='仅绿色等待阶段可放回物品；下一次动作会自动开始。',font=('Sans',14))
        self.warning.pack(pady=5)
        tk.Button(self.root,text='停止本批次（停止新目标，不是硬件急停）',
                  command=self.abort,font=('Sans',14),bg='#ffc4c4').pack(pady=8)
        self.root.protocol('WM_DELETE_WINDOW',self.abort)
        self.root.update()

    def abort(self):
        self.stopped=True

    def update(self,phase,elapsed):
        self.root.update()
        if self.stopped:
            raise KeyboardInterrupt('Operator stopped batch from countdown window')
        left=max(0,math.ceil(self.ends.get(phase,0)-elapsed))
        waiting=phase.startswith('wait:')
        if phase.startswith('episode:'):
            title=f'第 {int(phase.split(":")[1])}/{self.count} 次抓取 · 请勿进入'
        elif phase.startswith('return:'):
            title='正在恢复弯举姿态 · 请勿放回物品'
        elif waiting:
            title='保持示范起始姿态 · 可放回物品，结束前退出'
        else:
            title='准备弯举、张爪 · 请勿进入'
        self.label.config(text=title,bg='#b5edba' if waiting else '#ffc1b8')
        self.timer.config(text=f'{left} 秒')
        if phase!=self.last_phase or (waiting and left<=5 and left!=self.last_second):
            self.root.bell()
        self.last_phase,self.last_second=phase,left

    def close(self):
        self.root.destroy()


def render(root):
    """Cut actual captured frames, never speed up video to disguise slow motion."""
    summary=json.loads((root/'result.json').read_text())
    windows=summary['episode_windows']
    start=summary['motion_start_monotonic']
    timestamps={name:[json.loads(line)['monotonic'] for line in
                     (root/f'{name}_timestamps.jsonl').read_text().splitlines()]
                for name in CAMERAS}
    log=[json.loads(line) for line in (root/'commands_feedback.jsonl').read_text().splitlines()]
    for window in windows:
        if not window.get('completed'):
            continue
        out=root/f'episode_{window["episode"]:02d}'
        out.mkdir(exist_ok=False)
        begin=start+window['start_s']
        end=start+window['end_s']
        offsets={}
        for name,ts in timestamps.items():
            index=bisect.bisect_left(ts,begin)
            if index+600>len(ts) or ts[index+599]>end+.10:
                raise RuntimeError(f'{name}: insufficient real 20 s footage')
            if max(b-a for a,b in zip(ts[index:index+599],ts[index+1:index+600]))>.1:
                raise RuntimeError(f'{name}: frame gap in clip')
            offsets[name]=dict(start_frame=index,first_monotonic=ts[index],last_monotonic=ts[index+599])
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(index/30),
                '-i',str(root/f'{name}.avi'),'-frames:v','600','-r','30','-c:v','libx264',
                '-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',
                str(out/f'{name}.mp4')],check=True)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=nb_frames,duration','-of','json',str(out/f'{name}.mp4')]))['streams'][0]
            if int(probe['nb_frames'])!=600 or abs(float(probe['duration'])-20)>.001:
                raise RuntimeError('Rendered clip is not 600 frames / 20 s')
        args=['ffmpeg','-hide_banner','-loglevel','error']
        for name in ('camera2_left','camera1_global','camera3_right'):
            args+=['-i',str(out/f'{name}.mp4')]
        filters=';'.join(f'[{i}:v]scale=480:360,drawtext=text={label}:x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5[v{i}]'
                         for i,label in enumerate(('LEFT_WRIST','GLOBAL','RIGHT_WRIST')))
        filters+=';[v0][v1][v2]hstack=inputs=3[v]'
        subprocess.run(args+['-filter_complex',filters,'-map','[v]','-c:v','libx264','-preset','veryfast',
                             '-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'three_views.mp4')],check=True)
        with (out/'commands_feedback.jsonl').open('x') as f:
            for sample in log:
                if begin<=sample['monotonic']<end:
                    f.write(json.dumps(sample)+'\n')
        (out/'metadata.json').write_text(json.dumps(dict(**window,camera_frame_offsets=offsets,
            success=None,training_dataset_added=False,hardware_synchronized=False),indent=2))
        print(f'已生成第 {window["episode"]} 段：三路各 600 帧 / 20 秒及拼接视频',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    render(parser.parse_args().root)
