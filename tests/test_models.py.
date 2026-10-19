import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from backtest import outcome_probs,score,elo_features
from audit_registry import audit

class ResearchTests(unittest.TestCase):
    def test_probs(self):
        for h,a in ((.2,.3),(1.4,1.1),(3.,.7),(7.,7.)):
            for rho in (-.1,0,.02):
                p,u,b=outcome_probs(h,a,rho)
                self.assertAlmostEqual(p.sum(),1,places=9)
                self.assertTrue(np.all(p>=0))
                self.assertTrue(0<=u<=1 and 0<=b<=1)
    def test_symmetry(self):
        p,_,_=outcome_probs(1.2,1.2,0)
        self.assertAlmostEqual(p[0],p[2],places=9)
    def test_scoring(self):
        p=np.array([[.7,.2,.1],[.2,.6,.2],[.1,.2,.7]])
        s=score(np.array(['H','D','A']),p)
        self.assertAlmostEqual(s['accuracy'],1.)
        self.assertGreater(s['multiclass_brier'],0)
    def test_same_day_no_leakage(self):
        from datetime import date
        d=date(2024,1,1)
        games=[dict(date=d,league='E0',home='A',away='B',result='H',hg=7,ag=0),
               dict(date=d,league='E0',home='A',away='C',result='A',hg=0,ag=3)]
        features=elo_features(games)
        self.assertEqual(features[0,0],features[1,0])
        self.assertEqual(features[0,3],features[1,3])
    def test_registry(self):
        r=audit()
        self.assertEqual(r['total'],12)
        self.assertEqual(r['pending_selected'],5)
        self.assertEqual(r['settled_selected'],0)

if __name__=='__main__': unittest.main()
