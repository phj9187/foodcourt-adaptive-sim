import json
from pathlib import Path
import tempfile
import unittest
from kiosk_adaptive.dashboard import comparison, chart, snapshot, render


class DashboardTests(unittest.TestCase):
    def test_negative_improvement_and_signed_baseline(self):
        def pair(b,a):return dict(baseline_score=b,ai_score=a,customer_delta=0.)
        self.assertAlmostEqual(comparison([pair(.5,.4)])['percent'],-20.)
        self.assertAlmostEqual(comparison([pair(.5,.6)])['percent'],20.)
        for baseline in (0.,-1.,1e-10):
            result=comparison([pair(baseline,baseline-.1)])
            self.assertIsNone(result['percent']);self.assertAlmostEqual(result['delta'],-.1)
        # Percentage of paired means, not mean of unstable per-day percentages.
        self.assertAlmostEqual(comparison([pair(1.,2.),pair(3.,3.)])['percent'],25.)

    def test_waiting_corrupt_files_and_legacy_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.assertIn('실험 시작 대기',render(snapshot(root)))
            objective=dict(id='test',taste=.4,affordability=.3,waiting=.3,
                           customer_share=.5,profit_share=.5,power=1.)
            (root/'manifest.json').write_text(json.dumps(dict(objectives=[objective],config=dict(replicas=1,training_blocks=4))))
            job=root/'test'/'rep_0';job.mkdir(parents=True)
            (job/'live.json').write_text('{partial')
            self.assertEqual(snapshot(root)['jobs'][0]['phase'],'waiting')
            (job/'learning_history.json').write_text(json.dumps([dict(block=2,validation_delta=-.1)]))
            state=snapshot(root);self.assertEqual(state['jobs'][0]['latest']['delta'],-.1)
            page=render(state,refresh=True)
            self.assertIn('-0.100',page);self.assertIn('3000',page)
            self.assertIn('미개입 점수가 0 이하',page)

    def test_charts_keep_negative_values_and_escape_names(self):
        points=[dict(block=0,percent=-20.,delta=-.1),dict(block=2,percent=10.,delta=.05)]
        svg=chart(points,'percent','<title>','%')
        self.assertIn('-20.000%',svg);self.assertIn('&lt;title&gt;',svg)
        self.assertIn('circle',svg)


if __name__=='__main__':unittest.main()
