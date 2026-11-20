import json
from pathlib import Path
import unittest
from batch_demo_record import batch_path, fit_replay_times
from slow_demo_replay import ARMS

DEMONSTRATIONS = Path(__file__).resolve().parents[3] / 'data/demonstrations'


class BatchTests(unittest.TestCase):
    def test_clear_tape_timing(self):
        p=json.loads((DEMONSTRATIONS/'clear_tape/replay_0_to_13p5s.json').read_text())
        points,windows=batch_path(p['actions'],p['timestamps'],[0.]*16)
        self.assertEqual(len(windows),10)
        for w in windows:
            self.assertAlmostEqual(w['end_s']-w['start_s'],20.)
            if w['episode']<10:
                self.assertAlmostEqual(w['reset_wait_end_s']-w['return_end_s'],15.)
        for a,b in zip(points,points[1:]):
            dt=b[0]-a[0]
            self.assertGreater(dt,0)
            self.assertLessEqual(max(abs(b[1][i]-a[1][i])/dt for i in ARMS),.500001)
            self.assertLessEqual(max(abs(b[1][i]-a[1][i])/dt for i in (7,15)),1.500001)
        for n in range(1,11):
            rows=[q for _,q,s in points if s==f'episode:{n:02d}']
            self.assertEqual(rows[:-1],p['actions'][1:])
        with self.assertRaises(ValueError):
            fit_replay_times([[0.]*16,[20.]*16],[1.])

    def test_real_plan_timing_and_bounds(self):
        p=json.loads((DEMONSTRATIONS/'screwdriver/replay_0_to_11s.json').read_text())
        points,windows=batch_path(p['actions'],p['timestamps'],[0.]*16)
        self.assertEqual(len(windows),10)
        for w in windows:
            self.assertAlmostEqual(w['end_s']-w['start_s'],20)
            if w['episode']<10:
                self.assertAlmostEqual(w['reset_wait_end_s']-w['return_end_s'],15)
        for a,b in zip(points,points[1:]):
            dt=b[0]-a[0]
            self.assertGreater(dt,0)
            limit=.5 if b[2].startswith('episode:') else .120001
            self.assertLessEqual(max(abs(a[1][i]-b[1][i])/dt for i in ARMS),limit)
            if b[2].startswith('wait:'):
                self.assertEqual(a[1],b[1])
        for n in range(1,11):
            rows=[q for _,q,s in points if s==f'episode:{n:02d}']
            self.assertEqual(rows[:-1],p['actions'][1:])
            self.assertEqual(rows[-1],p['actions'][-1])
        self.assertEqual(points[-1][1],p['actions'][0])

    def test_count_rejected(self):
        for count in (0,11,-1,2.5):
            with self.assertRaises(ValueError): batch_path([],[],[],count)


if __name__=='__main__': unittest.main()
