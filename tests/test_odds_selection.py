import datetime as dt
import unittest
from odds_selection import select

NOW=dt.datetime(2026,10,19,12,tzinfo=dt.timezone.utc)

def fixture(n,p=.72):
    return {'forecast_id':f'2026-10-20|E0|Home{n}|Away{n}','match_date':'2026-10-20',
            'league':'E0','home':f'Home{n}','away':f'Away{n}','result':'',
            'p_home':str(p),'p_draw':'.2','p_away':'.08','p_over_2_5':'.65',
            'p_under_2_5':'.35','p_btts_yes':'.62','p_btts_no':'.38'}

def quote(n,odds='1.6',time='2026-10-19T11:00:00+00:00',market='H'):
    return {'forecast_id':f'2026-10-20|E0|Home{n}|Away{n}',
            'market':market,'decimal_odds':odds,'captured_utc':time,'bookmaker':'test'}

class OddsSelectionTests(unittest.TestCase):
    def test_four_qualified_legs(self):
        r=select([fixture(i) for i in range(4)],[quote(i) for i in range(4)],now=NOW)
        self.assertEqual(r['status'],'CANDIDATES_ONLY')
        self.assertEqual(r['number_of_legs'],4)
        self.assertAlmostEqual(r['combined_decimal_odds'],6.554,places=3)
    def test_no_odds_no_coupon(self):
        r=select([fixture(1)],[],now=NOW)
        self.assertEqual(r['number_of_legs'],0)
    def test_stale_quotes_rejected(self):
        r=select([fixture(1)],[quote(1,time='2026-10-17T10:00:00+00:00')],now=NOW)
        self.assertEqual(r['rejected']['stale'],1)
    def test_bad_value_rejected(self):
        r=select([fixture(1,p=.60)],[quote(1)],now=NOW)
        self.assertEqual(r['number_of_legs'],0)
    def test_same_match_only_once(self):
        r=select([fixture(1)],[quote(1),quote(1,market='OVER_2_5')],now=NOW)
        self.assertEqual(r['number_of_legs'],1)
    def test_today_rejected_without_kickoff(self):
        row=fixture(1);row['match_date']='2026-10-19'
        r=select([row],[quote(1)],now=NOW)
        self.assertEqual(r['number_of_legs'],0)

if __name__=='__main__':unittest.main()
