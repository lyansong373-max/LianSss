import unittest
from prepare_right_offset import merge_right, window_weight, lateral_direction
from slow_demo_replay import NAMES


class OffsetTests(unittest.TestCase):
    def test_lateral_directions_are_opposite(self):
        a=(0,.031,.698); b=(0,-.031,.698)
        self.assertEqual(lateral_direction(a,b,'left'),[0.,1.,0.])
        self.assertEqual(lateral_direction(a,b,'right'),[0.,-1.,0.])
        for bad in (a,(0,.031,1.)):
            with self.assertRaises(ValueError): lateral_direction(a,bad,'left')

    def test_window_preserves_start_and_deposit(self):
        w=[.5,3.,6.,9.]
        for t in (0.,.5,9.,12.):
            self.assertEqual(window_weight(t,w),0.)
        for t in (3.,5.3,6.):
            self.assertEqual(window_weight(t,w),1.)
        for t in (1.,2.,7.,8.):
            self.assertTrue(0<window_weight(t,w)<1)
        self.assertEqual(window_weight(2.,None),1.)

    def test_invalid_window(self):
        for w in ([0,0,2,3],[0,2,1,3],[-1,1,2,3],[0,1,2,float('nan')]):
            with self.assertRaises(ValueError): window_weight(1.,w)

    def test_only_right_joints_change(self):
        a=[.2]*16
        q=merge_right(a,dict.fromkeys(NAMES,.21))
        self.assertEqual(q[:8],a[:8])
        self.assertEqual(q[15],a[15])
        self.assertEqual(q[8:15],[.21]*7)
        self.assertEqual(a,[.2]*16)

    def test_reject_large_nonfinite_and_branch_jump(self):
        for value in (.151,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                merge_right([0.]*16,dict.fromkeys(NAMES,value))
        with self.assertRaises(ValueError):
            merge_right([0.]*16,dict.fromkeys(NAMES,.09),[0.]*16)

    def test_explicit_larger_planning_displacement(self):
        q=merge_right([0.]*16,dict.fromkeys(NAMES,.2),max_joint_offset=.3)
        self.assertEqual(q[8:15],[.2]*7)
        for bound in (.31,float('nan'),0):
            with self.assertRaises(ValueError):
                merge_right([0.]*16,dict.fromkeys(NAMES,.2),max_joint_offset=bound)
        with self.assertRaises(ValueError):
            merge_right([0.]*16,dict.fromkeys(NAMES,.2),[0.]*16,max_joint_offset=.3)


if __name__=='__main__':unittest.main()
