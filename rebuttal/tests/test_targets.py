import unittest
import numpy as np
import targets


def packet():
    n=40;p=np.zeros((n,162),np.float32)
    for i in range(n):
        t=i*.03;c,s=np.cos(t),np.sin(t);h=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        p[i,:9]=h.ravel();p[i,9:12]=[i*.01,0,0]
        for ps,rs,ls in [(12,18,36),(15,27,99)]:
            p[i,ps:ps+3]=[.4,.2+i*.003,1];p[i,rs:rs+9]=np.eye(3).ravel()
            local=np.zeros((21,3));local[:,1]=np.arange(21)*targets.SCALE/9
            p[i,ls:ls+63]=local.ravel()
    return dict(pose153=p,camera_valid=np.ones(n,bool),head_valid=np.ones(n,bool),hand_valid=np.ones((n,2),bool))


class TargetTests(unittest.TestCase):
    def test_future_head_and_current_head_roundtrip(self):
        p=packet()
        for kind in ['full','no_explicit_head','current_head','no_scale']:
            s,a,sm,am=targets.encode(p,np.array([0,4]),kind)
            self.assertEqual(s.shape,(2,156));self.assertEqual(a.shape,(2,32,153))
            xyz=targets.decode(s,a,kind,np)
            for side,ps,ls in [(0,12,36),(1,15,99)]:
                future=p['pose153'][np.array([0,4])[:,None]+np.arange(1,33)]
                expected=future[...,ps:ps+3,None].swapaxes(-1,-2)+future[...,ls:ls+63].reshape(2,32,21,3)
                np.testing.assert_allclose(xyz[:,:,side],expected,atol=2e-6)

    def test_no_scale_keeps_per_frame_scale_variation(self):
        p=packet();p['pose153'][2:,36:99]*=2
        _,a,_,_=targets.encode(p,np.array([0]),'full')
        _,b,_,_=targets.encode(p,np.array([0]),'no_scale')
        np.testing.assert_allclose(a[:,0,27:90],a[:,1,27:90],atol=1e-6)
        np.testing.assert_allclose(b[:,0,27:90]*2,b[:,1,27:90],atol=1e-6)

    def test_current_head_encoding_does_not_reuse_future_head_coordinates(self):
        p=packet();s,a,_,_=targets.encode(p,np.array([0]),'full')
        ss,b,_,_=targets.encode(p,np.array([0]),'current_head')
        np.testing.assert_allclose(s,ss);self.assertGreater(np.max(abs(a[...,9:27]-b[...,9:27])),.01)
        np.testing.assert_allclose(targets.decode(s,a,'full',np),targets.decode(ss,b,'current_head',np),atol=2e-6)

    def test_no_explicit_head_keeps_geometry_validity(self):
        p=packet();_,_,_,am=targets.encode(p,np.array([0]),'no_explicit_head')
        self.assertTrue(am[...,:9].all());self.assertTrue(np.all(targets.action_weights('no_explicit_head')[:9]==0))
        self.assertTrue(np.all(targets.action_weights('full')[:9]>0))

    def test_missing_hand_and_head_are_masked(self):
        p=packet();p['hand_valid'][1,0]=False;p['head_valid'][2]=False
        _,a,_,am=targets.encode(p,np.array([0]),'full')
        self.assertFalse(am[0,0,9:18].any());self.assertFalse(am[0,0,27:90].any())
        self.assertFalse(am[0,1,:9].any());self.assertFalse(am[0,1,9:27].any())
        self.assertTrue(np.all(a[~am]==0))

    def test_window_cannot_cross_episode_boundary(self):
        with self.assertRaises(ValueError):targets.encode(packet(),np.array([9]),'full')

    def test_all_component_ablations_use_identical_channel_support(self):
        p=packet();p['head_valid'][5]=False;p['hand_valid'][8,1]=False
        _,_,sm,am=targets.encode(p,np.array([0]),'full')
        for kind in ['current_head','no_scale','no_explicit_head','no_geometry']:
            _,_,ss,aa=targets.encode(p,np.array([0]),kind)
            np.testing.assert_array_equal(sm,ss);np.testing.assert_array_equal(am,aa)


if __name__=='__main__':unittest.main()
