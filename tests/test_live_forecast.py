import datetime as dt
import unittest
from unittest.mock import patch
import live_forecast as lf

class LiveForecastTests(unittest.TestCase):
    def test_id_stable(self):
        m=dict(date=dt.date(2026,10,20),league='E0',home='A',away='B')
        self.assertEqual(lf.match_id(m),'2026-10-20|E0|A|B')
    def test_settlement_only_past(self):
        row={'forecast_id':'2026-10-20|E0|A|B','result':''}
        m=dict(date=dt.date(2026,10,20),league='E0',home='A',away='B',result='H',hg=2,ag=1)
        self.assertEqual(lf.settle([row],[m],dt.date(2026,10,20)),0)
        self.assertEqual(lf.settle([row],[m],dt.date(2026,10,21)),1)
        self.assertEqual(row['result'],'H')
    def test_no_overwrite(self):
        row={'forecast_id':'2026-10-20|E0|A|B','result':'D'}
        m=dict(date=dt.date(2026,10,20),league='E0',home='A',away='B',result='H',hg=2,ag=1)
        self.assertEqual(lf.settle([row],[m],dt.date(2026,10,21)),0)
        self.assertEqual(row['result'],'D')

if __name__=='__main__':unittest.main()
