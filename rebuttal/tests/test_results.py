import unittest
import results

class ResultTests(unittest.TestCase):
    def test_equal_task_average_after_equal_color_average(self):
        rows=[]
        for task,group,rate,n in [('Coffee','',.62,50),('Pouring','',.82,50),('Can Sort','red',.92,50),('Can Sort','blue',.96,50)]:
            for i in range(n):rows.append(dict(method='egoweave',train_seed=42,suite='dmg_b',task=task,group=group,rollout_seed=i,success=i<int(rate*n),checkpoint_step=5000))
        report=results.summarize(rows);self.assertAlmostEqual(report[0]['mean_success'],.7933333333333)
        self.assertIsNone(report[0]['training_seed_std'])
    def test_wrong_checkpoint_is_rejected(self):
        rows=[dict(method='egoweave',train_seed=42,suite='dmg_b',task='Coffee',group='',rollout_seed=0,success=True,checkpoint_step=7713)]
        with self.assertRaises(ValueError):results.summarize(rows)
    def test_duplicate_rollout_is_rejected(self):
        x=dict(method='egoweave',train_seed=42,suite='dmg_b',task='Coffee',group='',rollout_seed=0,success=True,checkpoint_step=5000)
        with self.assertRaises(ValueError):results.summarize([x,x])

if __name__=='__main__':unittest.main()
