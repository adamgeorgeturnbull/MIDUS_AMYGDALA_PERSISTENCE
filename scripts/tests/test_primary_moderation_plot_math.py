"""Synthetic-only validation: never loads MIDUS inputs."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('plots',root/'scripts/visualization/primary_moderation_simple_slopes.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Checks(unittest.TestCase):
    def test_full_covariance_and_intercept(self):
        beta=np.array([2.,-.4,.2,.3,.5])
        a=np.array([[1.,0,0,0,0],[.2,.5,0,0,0],[.1,.1,.4,0,0],[.1,.1,.1,.3,0],[.1,0,0,0,.2]])
        cov=a@a.T
        x=np.array([[1.,0.,0.,0.,3.],[1.,2.,-1.,-2.,3.]])
        y,se,lo,hi=m.linear_summary(x,beta,cov)
        np.testing.assert_allclose(y,[3.5,1.9])
        np.testing.assert_allclose(se**2,[row@cov@row for row in x])
        self.assertGreater(se[0],0) # mean predictor still has uncertainty
        self.assertTrue((lo<y).all() and (hi>y).all())
    def test_slope_is_prediction_difference(self):
        beta=np.array([2.,-.4,.2,.3,.5]);cov=np.eye(5)*.1
        low=np.array([1,0,2,0,3]);high=np.array([1,1,2,2,3])
        slope=m.linear_summary(high-low,beta,cov)[0][0]
        self.assertAlmostEqual(slope,-.4+.3*2)
    def test_invalid_covariance_rejected(self):
        with self.assertRaises(ValueError):
            m.linear_summary([1,0],[1,2],[[1,0],[0,-1]])

if __name__=='__main__':unittest.main()
