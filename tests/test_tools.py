import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from generate_synthetic import generate
from prepare_pair import preprocess, load_cloud
from kitti_relative_pose import relative_lidar, load_calibration, load_poses
from evaluate import pose_error
class ToolsTest(unittest.TestCase):
    def test_synthetic(self):
        p,q,T=generate(17);a=p@T[:3,:3].T+T[:3,3]
        d=((a[:,None,:]-q[None,:,:])**2).sum(-1)
        self.assertLess(np.min(d,axis=1).max(),1e-20)
    def test_coordinate_direction(self):
        a=np.deg2rad(31);C=np.eye(4);C[:3,:3]=[[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]];C[:3,3]=[.2,-.4,1.1]
        A=np.eye(4);A[:3,3]=[2,3,4];B=np.eye(4);B[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];B[:3,3]=[-1,3,2]
        poses=np.stack([A,B]);T=relative_lidar(poses,C,1,0);p=np.array([1.2,2.3,3.4,1])
        np.testing.assert_allclose(A@C@T@p,B@C@p,atol=1e-12)
        np.testing.assert_allclose(relative_lidar(poses,C,0,1)@T,np.eye(4),atol=1e-12)
    def test_bin_and_voxel(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scan.bin';a=np.array([[1,2,3,.8],[4,5,6,.1]],dtype='<f4');a.tofile(path)
            np.testing.assert_allclose(load_cloud(path),a[:,:3])
        x=np.array([[0,0,0],[.1,0,0],[2,0,0],[0,2,0]])
        y=preprocess(x,1,100,0,100,1);self.assertEqual(len(y),3)
    def test_pose_error(self):
        T=np.eye(4);T[0,3]=2.5;te,re=pose_error(T,np.eye(4));self.assertAlmostEqual(te,2.5);self.assertAlmostEqual(re,0)
    def test_calibration_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'calib.txt';p.write_text('Tr: 1 0 0 1 0 1 0 2 0 0 1 3\n')
            C=load_calibration(p);np.testing.assert_allclose(C[:3,3],[1,2,3])
if __name__=='__main__':unittest.main()
