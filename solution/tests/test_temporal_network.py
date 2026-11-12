"""TCN context isolation, masked preprocessing, learning and chunked inference."""
from pathlib import Path
import importlib.util
import sys
import unittest
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'src'), str(Path(__file__).resolve().parents[1]/'tools')]
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from possession.temporal_network import (PossessionTCN, fit_normalizer, normalize, sampled_indices,
        upsample_probabilities, boundary_targets, losses, infer_sampled, combine_probabilities)


@unittest.skipUnless(HAS_TORCH, 'TCN tests require the isolated optional PyTorch environment')
class TemporalNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return {'input_features': 3, 'hidden_channels': 8, 'kernel_size': 3,
                'dilations': [1, 2], 'dropout': 0., 'normalization_clip': 8.,
                'inference_core_steps': 7, 'boundary_loss_weight': .2,
                'boundary_class_weights': [1, 4, 4, 4]}


    def test_normalization_uses_only_supplied_training_values_and_tracks_missingness(self):
        train = np.array([[1., 2., np.nan], [3., 4., np.nan]])
        norm = fit_normalizer([train])
        np.testing.assert_array_equal(norm['mean'], [2, 3, 0])
        x = normalize(np.array([[1000., np.nan, 7.]]), norm)
        np.testing.assert_array_equal(x, [[8, 0, 7, 1, 0, 1]])
        np.testing.assert_array_equal(norm['count'], [2, 2, 0])
        with self.assertRaises(ValueError): fit_normalizer([np.array([[np.inf]])])

    def test_sampling_and_interpolation_reset_at_half_edges(self):
        period = np.array([1, 1, 1, 0, 2, 2, 2])
        ids = sampled_indices(period, 2)
        np.testing.assert_array_equal(ids, [0, 2, 4, 6])
        values = np.array([[.1, .2], [.3, .4], [.8, .9], [.6, .7]])
        full = upsample_probabilities(values, ids, period)
        np.testing.assert_allclose(full[1], [.2, .3])
        np.testing.assert_allclose(full[4], [.8, .9])
        self.assertTrue(np.isnan(full[3]).all())

    def test_chunked_inference_matches_whole_half(self):
        torch.manual_seed(1117); cfg = self.config(); model = PossessionTCN(cfg)
        rng = np.random.default_rng(1117); x = rng.normal(size=(40, 3)).astype(np.float32)
        x[4:7, 0] = np.nan
        period = np.r_[np.ones(17), np.zeros(1), np.full(22, 2)]
        norm = fit_normalizer([x])
        chunked = infer_sampled(model, x, period, norm, cfg)
        whole = infer_sampled(model, x, period, norm, cfg | {'inference_core_steps': 1000})
        np.testing.assert_allclose(chunked, whole, atol=2e-6, equal_nan=True)
        altered = x.copy(); altered[18:] = 100
        other = infer_sampled(model, altered, period, norm, cfg)
        np.testing.assert_array_equal(chunked[:17], other[:17])


    def test_auxiliary_targets_use_next_state_and_do_not_cross_halves(self):
        labels = np.array([0, 0, 1, 1, 2, -1, 0, 0, 0])
        period = np.array([1]*5+[0]+[2]*3)
        y = boundary_targets(labels, period, 0)
        np.testing.assert_array_equal(y, [0, 0, 2, 0, 3, -1, 0, 0, 0])

    def test_joint_blend_preserves_probability_mass_and_endpoints(self):
        h, d, nh, nd = (np.array(x) for x in ([.8, .2], [.1, .7], [.3, .9], [.8, .2]))
        result = combine_probabilities(h, d, nh, nd, np.ones(2), .5)
        mh, md = result['tcn_blend']
        np.testing.assert_allclose((1-md)*mh, .5*((1-d)*h+(1-nd)*nh))
        a, b = combine_probabilities(h, d, nh, nd, np.ones(2), 0)['tcn_blend']
        np.testing.assert_allclose(a, h); np.testing.assert_array_equal(b, d)

    def test_masked_loss_ignores_padded_targets(self):
        cfg = self.config()
        state = torch.zeros(1, 2, 4, requires_grad=True)
        boundary = torch.zeros(1, 4, 4, requires_grad=True)
        labels = torch.tensor([[0, 1, 2, -1]])
        loss, _ = losses(state, boundary, labels, torch.ones(1, 4), torch.tensor([[0, 0, 3, -1]]), cfg)
        loss.backward()
        np.testing.assert_array_equal(state.grad[:, :, -1].numpy(), 0)
        np.testing.assert_array_equal(boundary.grad[:, :, -1].numpy(), 0)
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_can_learn_a_simple_sequence(self):
        torch.manual_seed(1117); cfg = self.config(); model = PossessionTCN(cfg)
        labels = torch.tensor([[0]*15+[1]*15+[2]*15])
        physical = torch.nn.functional.one_hot(labels, 3).float().transpose(1, 2)
        x = torch.cat([physical, torch.ones_like(physical)], dim=1)
        boundary_labels = torch.zeros_like(labels)
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        history = []
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            state, boundary = model(x)
            loss, _ = losses(state, boundary, labels, torch.ones_like(labels).float(), boundary_labels, cfg)
            history.append(float(loss.detach())); loss.backward(); optimizer.step()
        self.assertLess(history[-1], history[0]*.15)
        self.assertTrue(all(np.isfinite(history)))


if __name__ == '__main__': unittest.main()
