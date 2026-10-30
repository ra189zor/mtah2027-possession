"""Guard the broadened benchmark against leakage and accidental label rewriting."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from possession.robustness import absorb_short_opposition,contextual_corrupt,excluded_pairs,nested_training_matches
from possession.inputs import array_digest


class RobustnessTests(unittest.TestCase):
    def test_absorbs_only_short_opposition_inside_alive(self):
        y=np.array(list('H'*100+'A'*74+'H'*100+'D'*5+'A'*75+'H'*100+'A'*100+'X'))
        p=np.ones(len(y),dtype=np.int8)
        result,info=absorb_short_opposition(y,p)
        self.assertEqual(info['absorbed_spells'],1)
        self.assertEqual(info['changed_frames'],74)
        np.testing.assert_array_equal(result[:274],'H')
        np.testing.assert_array_equal(result[274:],y[274:])

    def test_sensitivity_respects_halves_dead_unknown_and_idempotence(self):
        y=np.array(list('HHAAHHDDHHAAHHXXHHAAHH'))
        p=np.array([1]*4+[2]*(len(y)-4),dtype=np.int8)
        result,_=absorb_short_opposition(y,p)
        np.testing.assert_array_equal(result[:6],y[:6])
        np.testing.assert_array_equal(result[np.isin(y,['D','X'])],y[np.isin(y,['D','X'])])
        again,info=absorb_short_opposition(result,p)
        np.testing.assert_array_equal(again,result);self.assertEqual(info['absorbed_spells'],0)

    def test_every_nested_training_fold_excludes_both_test_matches(self):
        matches=list('ABCDEFG');pairs=excluded_pairs(matches)
        self.assertEqual(len(pairs),21)
        for outer in matches:
            inner=[pair for pair in pairs if outer in pair]
            self.assertEqual(len(inner),6)
            for pair in inner:
                training=nested_training_matches(matches,pair)
                self.assertEqual(len(training),5)
                self.assertFalse(set(training)&set(pair))
        with self.assertRaises(ValueError):nested_training_matches(matches,['Z'])

    def test_outer_reference_scores_cannot_change_its_inner_selection(self):
        import contextlib
        import io
        import tempfile
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
        from run_nested_validation import select_inner,write,read
        from possession.decoder import candidates
        matches=list('ABCDEFG')
        cfg={'matches':matches,'inner_conditions':['clean'],'seeds':[17]}
        decoder={'modes':['fixed'],'stop_restart_costs':[[8,8]],'owner_switch_costs':[8]}
        profiles=candidates(decoder)
        temporary_root=Path(__file__).resolve().parents[2]/'outputs'/'test_temporary'
        temporary_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary,contextlib.redirect_stdout(io.StringIO()):
            output=Path(temporary)
            for pair in excluded_pairs(matches):
                folder=output/'inner'/'_'.join(pair);folder.mkdir(parents=True)
                scores=[{'held_out':held,'condition':'clean','seed':17,'candidate':p['id'],
                         'spell_f1_mean':.8 if p['stop'] else .1,'macro_f1':.8}
                        for held in pair for p in profiles]
                write(folder/'report.json',{'excluded':list(pair),'training_matches':nested_training_matches(matches,pair),'scores':scores})
            before=select_inner(output,cfg,decoder,{})['folds']['A']['selected']
            for pair in excluded_pairs(matches):
                if 'A' not in pair:continue
                path=output/'inner'/'_'.join(pair)/'report.json';report=read(path)
                for row in report['scores']:
                    if row['held_out']=='A':row['spell_f1_mean']=1. if row['candidate']=='frame_map' else 0.
                write(path,report)
            after=select_inner(output,cfg,decoder,{})['folds']['A']['selected']
            self.assertEqual(before,after)
            path=output/'inner'/'A_B'/'report.json';report=read(path)
            report['training_matches'].append('A');write(path,report)
            with self.assertRaises(ValueError):select_inner(output,cfg,decoder,{})

    def test_contextual_noise_is_repeatable_label_free_and_keeps_membership(self):
        import json
        root=Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'configs/nested_validation.json').read_text())['contextual']
        base=json.loads((root/'configs/validation.json').read_text())['regimes']['moderate']
        n=500;players=np.zeros((n,4,2),dtype=np.float32)
        players[:,0]=[10,20];players[:,1]=[20,20];players[:,2]=[30,20];players[:,3]=[40,20]
        inputs={'players_xy':players,'ball_xy':np.column_stack([np.arange(n)*.7%105,np.full(n,20)]).astype(np.float32),
                'team_of_slot':np.array([0,0,1,1]),'period':np.r_[np.ones(249),0,np.full(250,2)].astype(np.int8),
                'pitch_m':np.array([105,68]),'fps':np.array(25)}
        # Force each kind of burst so this also exercises wrong-team coordinate swapping.
        cfg=cfg|{'extra_gap_event_mean_s':.1,'ghost_event_mean_s':.1,'wrong_team_event_mean_s':.1}
        original=array_digest(inputs)
        a,ma,aa=contextual_corrupt(inputs,base,cfg,17)
        b,mb,ab=contextual_corrupt(inputs,base,cfg,17)
        self.assertEqual(array_digest(a),array_digest(b));self.assertEqual(aa,ab)
        self.assertEqual(array_digest(inputs),original)
        self.assertEqual(set(a),set(inputs))
        np.testing.assert_array_equal(a['team_of_slot'],inputs['team_of_slot'])
        self.assertFalse(any(mask[249] for mask in ma.values()))
        self.assertTrue(all(ma[k].any() for k in ['extra_dropout','ghost','wrong_team']))
        with self.assertRaises(ValueError):contextual_corrupt(inputs|{'labels':np.full(n,'H')},base,cfg,17)


if __name__=='__main__':unittest.main()
