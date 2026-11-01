import json
from pathlib import Path
import platform
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from possession.inference import (read_tracking_npz,validate_tracking,validate_prediction,load_bundle,
    runtime_hashes,settings_digest,PACKAGES,write_prediction)
from possession.inputs import file_digest,array_digest
from importlib.metadata import version


class InferenceTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[2]/'outputs/test_temporary';parent.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.temporary.cleanup)
        self.folder=Path(self.temporary.name)

    def inputs(self):
        return {'players_xy':np.zeros((12,4,2),dtype=np.float32),'ball_xy':np.zeros((12,2),dtype=np.float32),
            'team_of_slot':np.array([0,0,1,1],dtype=np.int8),'period':np.array([1]*5+[0]+[2]*6,dtype=np.int8),
            'pitch_m':np.array([105,68],dtype=np.float32),'fps':np.array(25)}

    def manifest(self,model_hash,settings):
        return {'schema_version':1,'status':'complete','model_file':'models.joblib','model_sha256':model_hash,
            'runtime_sha256':runtime_hashes(),'python_version':platform.python_version(),
            'packages':{name:version(name) for name in PACKAGES},'settings':settings,
            'settings_sha256':settings_digest(settings),'training_matches':['example']}

    def test_npz_requires_only_tracking_and_no_sibling_reference(self):
        inputs=self.inputs();before=array_digest(inputs);path=self.folder/'only_input.npz'
        np.savez_compressed(path,**inputs)
        loaded=read_tracking_npz(path)
        self.assertEqual(array_digest(loaded),before)
        self.assertFalse((self.folder/'targets.npz').exists())
        np.savez_compressed(self.folder/'extra.npz',**inputs,labels=np.full(12,'H'))
        with self.assertRaisesRegex(ValueError,'six model input fields'):read_tracking_npz(self.folder/'extra.npz')

    def test_input_units_missingness_and_temporal_guards(self):
        inputs=self.inputs();inputs['ball_xy'][0]=np.nan;validate_tracking(inputs)
        with self.assertRaises(ValueError):validate_tracking(inputs|{'fps':np.array(25.5)})
        with self.assertRaises(ValueError):validate_tracking(inputs|{'pitch_m':np.array([0,68])})
        repeated=np.array([1]*5+[0]+[1]*6,dtype=np.int8)
        with self.assertRaises(ValueError):validate_tracking(inputs|{'period':repeated})
        bad=inputs['ball_xy'].copy();bad[1,0]=np.inf
        with self.assertRaises(ValueError):validate_tracking(inputs|{'ball_xy':bad})

    def test_corrupt_weights_rejected_before_deserialization(self):
        (self.folder/'models.joblib').write_bytes(b'not a model')
        manifest=self.manifest('0'*64,{})
        (self.folder/'manifest.json').write_text(json.dumps(manifest))
        with patch('possession.inference.joblib.load',side_effect=AssertionError('must not deserialize')):
            with self.assertRaisesRegex(ValueError,'checksum'):load_bundle(self.folder)

    def test_settings_tampering_and_runtime_drift_are_rejected(self):
        settings={'name':'original'}
        joblib.dump({'settings_sha256':settings_digest(settings)},self.folder/'models.joblib')
        manifest=self.manifest(file_digest(self.folder/'models.joblib'),{'name':'changed'})
        (self.folder/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'settings changed'):load_bundle(self.folder)
        manifest['runtime_sha256']['solution/predict.py']='0'*64
        (self.folder/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'Runtime source changed'):load_bundle(self.folder)

    def test_csv_contract_overwrite_protection_and_no_scoring(self):
        bundle_folder=self.folder/'bundle';bundle_folder.mkdir();(bundle_folder/'manifest.json').write_text('{}')
        bundle={'folder':bundle_folder,'manifest':{'kind':'held_out_verification','model_sha256':'example','training_matches':['training']}}
        labels=np.array(['H','X','A']);period=np.array([1,0,2])
        result={'labels':labels,'input_fingerprint':'input','timing':{}}
        path=self.folder/'prediction.csv'
        report=write_prediction(path,result,bundle,'baseline',{'kind':'test'},period)
        self.assertEqual(path.read_bytes(),b'frame,label\n0,H\n1,X\n2,A\n')
        self.assertFalse(report['reference_arrays_read']);self.assertFalse(report['scored'])
        self.assertTrue(report['official_format_valid'])
        before=file_digest(path)
        with self.assertRaises(FileExistsError):write_prediction(path,result,bundle,'baseline',{},period)
        self.assertEqual(file_digest(path),before)
        with self.assertRaises(ValueError):validate_prediction(np.array(['H','D','A']),period)
        with self.assertRaises(ValueError):validate_prediction(np.array(['X','X','A']),period)


if __name__=='__main__':unittest.main()
