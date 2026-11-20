import math
import unittest
from slow_demo_replay import ARMS, interpolate, timed_path, validate, zero_path, zero_pose_payload, select_export_rows, validate_replay_start, curl_only_path, validate_curl_completion


class ReplayTests(unittest.TestCase):
    def test_curl_only_never_enters_grasp(self):
        a,t=self.data()
        path=curl_only_path(a,t,[0.]*16)
        self.assertEqual(path[-1][1],a[0])
        self.assertEqual(set(p[2] for p in path),{'curl','open_grippers','settle'})
        self.assertTrue(all(q[0]==a[0][0] for _,q,_ in path))
        self.assertTrue(all(q[7]==0 and q[15]==0 for _,q,p in path if p=='curl'))
        self.assertTrue(all(y[0]>x[0] for x,y in zip(path,path[1:])))

    def test_curl_completion_uses_recorded_feedback_not_action_target(self):
        reference=[0.]*16
        reference[3],reference[11]=1.2503,1.1774
        actual=list(reference)
        actual[3],actual[11]=1.2381,1.1389
        action_target=list(reference)
        action_target[3],action_target[11]=1.3510,1.2313
        self.assertGreater(max(abs(actual[i]-action_target[i]) for i in ARMS),.08)
        result=validate_curl_completion(actual,reference,[actual]*40)
        self.assertLess(result['final_reference_max_error_rad'],.04)
        self.assertEqual(result['final_reference_worst_joint'],'openarm_right_joint4')
        bad=list(actual)
        bad[3]=reference[3]-.081
        with self.assertRaisesRegex(RuntimeError,'demonstration observation'):
            validate_curl_completion(bad,reference,[bad]*40)
        moving=list(actual)
        moving[3]+=.03
        with self.assertRaisesRegex(RuntimeError,'still moving'):
            validate_curl_completion(actual,reference,[actual,moving]*20)
        with self.assertRaisesRegex(RuntimeError,'full second'):
            validate_curl_completion(actual,reference,[actual]*19)

    def test_explicit_curl_start_gate(self):
        first=[.8]*16
        current=[.7]*16
        validate_replay_start(current,first,True)
        with self.assertRaises(RuntimeError): validate_replay_start(current,first)
        with self.assertRaises(RuntimeError): validate_replay_start([.6]*16,first,True)
        validate_replay_start([0.]*16,first)
        with self.assertRaises(ValueError): validate_replay_start([float('nan')]*16,first,True)

    def test_export_window(self):
        rows=[dict(episode_index=0,timestamp=i/30,action=[0.]*16) for i in range(965)]
        for end, count in ((11.,330),(13.5,405)):
            selected=select_export_rows(rows,0,end,30)
            self.assertEqual(len(selected),count)
            self.assertEqual(selected[-1]['timestamp'],(count-1)/30)
        for bad in (rows[1:],rows[:50]+rows[51:],rows+[rows[10]]):
            with self.assertRaises(ValueError): select_export_rows(bad,0,13.5,30)
        with self.assertRaises(ValueError): select_export_rows(rows,1,13.5,30)

    def data(self):
        a = [0.] * 16
        a[3], a[11], a[7], a[15] = 1.2, 1.1, 1., 1.
        b = list(a)
        b[0], b[15] = .07, .05
        c = list(b)
        c[15] = 1.
        return [a,b,c], [0.,1/30,2/30]

    def test_reject_bad_input(self):
        a,t=self.data()
        for bad in (float('nan'),float('inf'),1.1,-.01):
            q=[list(row) for row in a]
            q[0][7]=bad
            with self.assertRaises(ValueError): validate(q,t)
        with self.assertRaises(ValueError): validate(a,[0.,0.,1.])
        with self.assertRaises(ValueError): validate(a,[0.,1.])

    def test_waypoints_and_bounds(self):
        a,t=self.data()
        path=timed_path(a,t,[0.]*16)
        self.assertEqual([q for _,q,p in path if p=='replay'],a[1:])
        self.assertEqual(path[-1][1],a[-1])
        for left,right in zip(path,path[1:]):
            dt=right[0]-left[0]
            self.assertGreater(dt,0)
            for i in ARMS:
                self.assertLessEqual(abs(right[1][i]-left[1][i])/dt,.18001)
            for i in (7,15):
                self.assertLessEqual(abs(right[1][i]-left[1][i])/dt,.50001)
        self.assertTrue(all(q[7]==0 and q[15]==0 for _,q,p in path if p=='curl'))

    def test_interpolation_no_overshoot(self):
        a,t=self.data()
        path=timed_path(a,t,[0.]*16)
        index=0
        for step in range(math.ceil(path[-1][0]*50)+2):
            q,index,_=interpolate(path,step/50,index)
            lo,hi=path[index][1],path[index+1][1]
            self.assertTrue(all(min(x,y)-1e-9 <= z <= max(x,y)+1e-9
                                for x,y,z in zip(lo,hi,q)))
        self.assertEqual(q,path[-1][1])

    def test_zero_motion(self):
        q=self.data()[0][0]
        path=zero_path(q)
        self.assertEqual(path[0][1],q)
        self.assertEqual(path[-1][1],[0.]*16)
        for t,row,phase in path:
            if phase=='return_zero':
                self.assertEqual([row[7],row[15]],[1.,1.])
            else:
                self.assertTrue(all(row[i]==0 for i in ARMS))
        for a,b in zip(path,path[1:]):
            dt=b[0]-a[0]
            self.assertGreater(dt,0)
            self.assertLessEqual(max(abs(b[1][i]-a[1][i])/dt for i in ARMS),.120001)
            self.assertLessEqual(max(abs(b[1][i]-a[1][i])/dt for i in (7,15)),.312501)

    def test_save_zero_requires_settled_feedback(self):
        q=[.005]*16
        q[7],q[15]=.048,-.003
        payload=zero_pose_payload([q]*40)
        self.assertFalse(payload['encoder_zero_rewritten'])
        self.assertTrue(all(v==0 for v in payload['positions'].values()))
        self.assertAlmostEqual(payload['measured_positions']['right_gripper.pos'],-.003)
        for bad in ([q],[[.03]*16]*40,[[float('nan')]*16]*40):
            with self.assertRaises(ValueError): zero_pose_payload(bad)

    def test_closed_gripper_acceptance_boundary_and_stability(self):
        for index in (7,15):
            for value in (.09323176894164972, .10, -.10):
                q=[0.]*16
                q[index]=value
                payload=zero_pose_payload([q]*40)
                self.assertEqual(payload['gripper_tolerance_normalized'], .10)
                self.assertEqual(payload['measured_positions'][
                    'left_gripper.pos' if index==7 else 'right_gripper.pos'],
                    sum([value]*40)/40)
            for value in (.100001, -.100001, math.inf, math.nan):
                q=[0.]*16
                q[index]=value
                with self.assertRaises(ValueError): zero_pose_payload([q]*40)
            low,high=[0.]*16,[0.]*16
            low[index],high[index]=.05,.09
            with self.assertRaisesRegex(ValueError, 'not settled'):
                zero_pose_payload([low,high]*20)
        q=[0.]*16
        q[0]=.020001
        with self.assertRaises(ValueError): zero_pose_payload([q]*40)


if __name__=='__main__': unittest.main()
