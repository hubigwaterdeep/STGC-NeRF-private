"""Executable CUDA contracts; all data are synthetic, no scene fitting."""
import types
import unittest
import torch

from Modi01.field import HybridTemporalField, RepresentationConfig, parameter_count
from Modi01.hash_encoding import SpatialSchedule, SelectedHash, LocalTemporalHash, lagrange_weights
from Modi01.model import STGCNeRFModi01
from best_core.scene_field import AnchoredSplineHighOrderGeometryResidualField
from best_core.hash_field import HashGridT

CUDA = torch.cuda.is_available()


def seed():
    torch.manual_seed(20260918)
    torch.cuda.manual_seed_all(20260918)


@unittest.skipUnless(CUDA, 'tiny-cuda-nn requires CUDA')
class RepresentationTests(unittest.TestCase):
    def tearDown(self):
        torch.cuda.empty_cache()

    def test_spatial_lattice_and_original_hash_interpolation(self):
        seed()
        schedule = SpatialSchedule()
        levels = (0, 3, 4, 7)
        ref = HashGridT(log2_hashmap_size=13).cuda()
        local = LocalTemporalHash(schedule, levels, 13).cuda()
        entries = 2**13 * 4
        with torch.no_grad():
            for old, new in zip(ref.hash_t, local.hash_t):
                for level, grid in zip(levels, new.grids):
                    grid.params.copy_(old.params[level*entries:(level+1)*entries])
        x = torch.rand(257, 2, device='cuda') * 1.2 - .1
        x[:3] = torch.tensor([[0, 0], [1, 1], [.5, .5]], device='cuda')
        worst = 0.
        for value in [0., 1., .37] + [i/7 for i in range(8)]:
            t = torch.tensor(value, device='cuda')
            expected = ref(x, t)[:, levels].float()
            actual = local(x, t).float()
            worst = max(worst, (actual-expected).abs().max().item())
            torch.testing.assert_close(actual, expected, rtol=.02, atol=3e-6)
        for level, ratio in zip(levels, local.hash_t[0].coordinate_ratios):
            desired = x * schedule.scale(level) + .5
            lattice = x * ratio * (schedule.resolution(level) - 1) + .5
            torch.testing.assert_close(lattice, desired, rtol=2e-7, atol=.004)
        print('selected/original forward maximum absolute difference:', worst)
        many = torch.linspace(0, 1, len(x), device='cuda')
        batched = local(x, many)
        individual = torch.cat([local(x[i:i+1], many[i]) for i in range(len(x))])
        torch.testing.assert_close(batched, individual.float(), atol=3e-7, rtol=.02)
        for invalid in (-.01, 1.01, float('nan'), torch.zeros(3, device='cuda')):
            with self.assertRaises(ValueError):
                local(x, invalid)

    def test_all_modal_matches_old_forward_and_gradients(self):
        seed()
        old = AnchoredSplineHighOrderGeometryResidualField().cuda().requires_grad_(True)
        seed()
        new = HybridTemporalField(RepresentationConfig(8)).cuda()
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], atol=0, rtol=0)
        x = torch.rand(79, 3, device='cuda')
        for progress in (0., 1.):
            old.set_training_progress(high_order=progress)
            new.set_training_progress(high_order=progress)
            for value in (0., .37, 1.):
                t = x.new_tensor(value)
                a, b = old.query(x, t), new.query(x, t)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        for field in (old, new):
            field.query(x, x.new_tensor(.37)).float().square().mean().backward()
        new_params = dict(new.named_parameters())
        for name, p in old.named_parameters():
            other = new_params[name]
            self.assertEqual(p.grad is None, other.grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, other.grad, atol=2e-6, rtol=1e-4)

    def test_hybrid_keeps_shared_initialization_and_modal_prefix(self):
        seed()
        old = HybridTemporalField(RepresentationConfig(8)).cuda()
        seed()
        new = HybridTemporalField().cuda()
        old.set_training_progress(high_order=0)
        new.set_training_progress(high_order=0)
        for name in ('static_planes', 'modal_axes', 'basis', 'hash_basis', 'fusion', 'flow_net', 'geometry_residual_basis'):
            a, b = getattr(old, name).state_dict(), getattr(new, name).state_dict()
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        x = torch.rand(64, 3, device='cuda')
        for value in (0., .37, 1.):
            a = old._hash_dynamic(x, x.new_tensor(value)).view(-1, 3, 8)
            b = new._hash_dynamic(x, x.new_tensor(value)).view(-1, 3, 8)
            torch.testing.assert_close(a[..., :4], b[..., :4], atol=0, rtol=0)
        self.assertEqual(new.hash_parameter_count(), old.hash_parameter_count())
        self.assertEqual(new.n_output_dims, 120)

    def test_parameter_groups_and_all_selected_paths_receive_gradients(self):
        for mode in ('tied', 'untied', 'neural'):
            seed()
            field = HybridTemporalField(RepresentationConfig(coefficients=mode)).cuda()
            field.set_training_progress(high_order=1.)
            groups = field.parameter_groups(.01)
            ids = [id(p) for g in groups for p in g['params']]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in field.parameters() if p.requires_grad})
            self.assertEqual(field.budget_report().total, parameter_count(field))
            x = torch.rand(128, 3, device='cuda')
            times = torch.linspace(.01, .99, len(x), device='cuda')
            field._hash_dynamic(x, times).float().sum().backward()
            names = ('modal_hashes.', 'local_temporal_hashes.', 'high_coefficient_hashes.', 'neural_coefficients.')
            for name, p in field.named_parameters():
                if name.startswith(names):
                    self.assertIsNotNone(p.grad, (mode, name))
                    self.assertTrue(torch.isfinite(p.grad).all(), (mode, name))
                    self.assertGreater(float(p.grad.abs().max()), 0., (mode, name))
            if mode != 'neural':
                self.assertEqual(field.hash_parameter_count(), field.reference_hash_parameters)
            else:
                self.assertLessEqual(field.hash_parameter_count(), field.reference_hash_parameters)
            if mode == 'untied':
                for lo, hi in zip(field.modal_hashes, field.high_coefficient_hashes):
                    self.assertNotEqual(lo.params.data_ptr(), hi.params.data_ptr())
            del field

    def test_flow_neighbor_semantics_and_no_grad_policy(self):
        field = HybridTemporalField().cuda()
        field.set_training_progress(high_order=1)
        x = torch.rand(39, 3, device='cuda')
        original = field._hash_dynamic
        for value, expected_times, expected_grad in ((0., [0., .02], [True, False]),
                (.4, [.4, .42, .38], [True, False, False]),
                (1., [1., .98], [True, False])):
            calls = []
            def record(this, xyz, t, context=None):
                out = original(xyz, t, context)
                calls.append((xyz.detach().clone(), float(t), torch.is_grad_enabled(), out))
                return out
            field._hash_dynamic = types.MethodType(record, field)
            result = field.query_features(x, x.new_tensor(value))
            self.assertEqual([c[2] for c in calls], expected_grad)
            for c, t in zip(calls, expected_times):
                self.assertAlmostEqual(c[1], t, places=6)
            flow = field.flow_net(field._xt(x, x.new_tensor(value)))
            next_value = calls[0][3] if value == 1 else calls[1][3]
            previous_value = calls[0][3] if value == 0 else calls[-1][3]
            torch.testing.assert_close(result.hash_dynamic, .5*calls[0][3]+.25*(next_value+previous_value))
            if value < 1:
                torch.testing.assert_close(calls[1][0], x+flow[:, :3])
            if value > 0:
                torch.testing.assert_close(calls[-1][0], x+flow[:, 3:])
        field._hash_dynamic = original

    def test_checkpoint_contract_and_renderer_backward(self):
        seed()
        model = STGCNeRFModi01(near_lidar=.01, far_lidar=.5).cuda()
        state = model.state_dict()
        model.load_state_dict(state)
        invalid = dict(state)
        invalid['scene_field._extra_state'] = dict(state['scene_field._extra_state'], version=9)
        with self.assertRaises(ValueError):
            model.load_state_dict(invalid, strict=False)
        ids = [id(p) for g in model.get_params(.01) for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
        origins = torch.zeros(1, 32, 3, device='cuda')
        directions = torch.randn_like(origins)
        directions /= directions.norm(dim=-1, keepdim=True)
        out = model.render(origins, directions, torch.tensor([[.37]], device='cuda'),
                           num_steps=16, perturb=False)
        loss = out['depth_lidar'].float().mean() + out['image_lidar'].float().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(not p.requires_grad for p in model.unet.parameters()))
        self.assertIsNotNone(model.scene_field.modal_hashes[0].params.grad)
        self.assertIsNotNone(model.scene_field.flow_net.grid_enc.params.grad)

    def test_empty_and_single_modal_splits(self):
        x = torch.rand(17, 3, device='cuda')
        for modal in (0, 1, 7):
            field = HybridTemporalField(RepresentationConfig(modal)).cuda()
            for progress in (0., 1.):
                field.set_training_progress(high_order=progress)
                output = field.query(x, x.new_tensor(.37))
                self.assertEqual(output.shape, (17, 120))
                self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(field.hash_parameter_count(), field.reference_hash_parameters)

    def test_wrapper_preserves_shared_heads_and_rng(self):
        seed()
        old = STGCNeRFModi01(RepresentationConfig(8)).cuda()
        a = torch.rand(10, device='cuda')
        seed()
        new = STGCNeRFModi01().cuda()
        b = torch.rand(10, device='cuda')
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        for name in ('sigma_net', 'intensity_net', 'raydrop_net', 'unet'):
            left, right = getattr(old, name).state_dict(), getattr(new, name).state_dict()
            for key in left:
                torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
        xyz = torch.rand(13, 3, device='cuda') * 2 - 1
        t = xyz.new_tensor(.37)
        decomposed = new.scene_field.query_decomposition((xyz+1)/2, t)
        from best_core.activation import trunc_exp
        expected_full = new.sigma_net(decomposed.full)
        expected_base = new.sigma_net(decomposed.base)
        density = new.density(xyz, t)
        torch.testing.assert_close(density['sigma'], trunc_exp(expected_full[:, 0]))
        torch.testing.assert_close(density['geo_feat'], expected_base[:, 1:])

    def test_neural_role_domain_and_interpolate_then_decode(self):
        field = HybridTemporalField(RepresentationConfig(coefficients='neural')).cuda()
        net = field.neural_coefficients
        xy = torch.rand(21, 2, device='cuda', requires_grad=True)
        calls = []
        handle = net.decoder.register_forward_pre_hook(lambda m, args: calls.append(args[0]))
        output = net(xy, 1)
        handle.remove()
        latent = net.latents[1](xy).float()
        torch.testing.assert_close(calls[0][..., :4], latent, atol=0, rtol=0)
        torch.testing.assert_close(net.decoder(calls[0]), output, atol=0, rtol=0)
        self.assertEqual(output.shape, (21, 4, 8))
        output.sum().backward()
        self.assertIsNotNone(xy.grad)
        self.assertEqual(net.latents[1].grids[0].n_input_dims, 2)


class AlgebraTests(unittest.TestCase):
    def test_lagrange_endpoints_and_partition(self):
        weights = lagrange_weights(torch.linspace(0, 1, 51, dtype=torch.float64))
        torch.testing.assert_close(weights.sum(-1), torch.ones(51, dtype=torch.float64))
        torch.testing.assert_close(lagrange_weights(torch.linspace(0, 1, 4, dtype=torch.float64)), torch.eye(4, dtype=torch.float64))

    def test_config_validation(self):
        for config in (RepresentationConfig(-1), RepresentationConfig(9),
                       RepresentationConfig(0, 'neural'), RepresentationConfig(coefficients='unknown')):
            with self.assertRaises(ValueError):
                config.validate(8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
