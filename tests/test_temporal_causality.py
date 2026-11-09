"""CPU-only regression checks; no datasets, checkpoints, or training runs are written."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import copy
from pathlib import Path
import sys
import unittest

import torch

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.TCSD import TCSD
from models.modules import ProductMergeLayer


class Scorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TCSD(
            neighbor_sampler=None, n_layers=2, n_heads=2, hidden_size=16, hidden_dropout_prob=0.0, layer_norm_eps=1e-12,
            initializer_range=0.02, n_nodes=20, max_seq_length=3, dst_min_idx=1, device="cpu", emb_dropout_prob=0.0,
        )
        self.predictor = ProductMergeLayer(hidden_size=16, num_output_layer=2, dropout=0.0)

    def forward(self, **batch):
        return self.predictor(*self.backbone(**batch))


def make_model():
    torch.manual_seed(0)
    return Scorer().eval()


def inputs():
    result = dict(
        src_neighb_seq=torch.tensor([[2, 3, 4]]).repeat(3, 1),
        neighbors_interact_times=torch.tensor([[7., 8., 9.], [17., 18., 19.], [17., 18., 19.]]),
        cur_times=torch.tensor([10., 20., 20.]),
        test_dst=torch.tensor([[2, 4, 6]]).repeat(3, 1),
        dst_last_update_times=torch.tensor([[8., 9., 8.], [18., 19., 18.], [18., 19., 18.]]),
        dst_neighb_seq=torch.tensor([[[2, 3, 4], [3, 4, 5], [2, 5, 6]]]).repeat(3, 1, 1),
        dst_neighb_interact_times=torch.tensor([[[6., 7., 8.], [7., 8., 9.], [6., 7., 8.]]]).repeat(3, 1, 1),
    )
    result["dst_neighb_interact_times"][1:] += 10.
    return result


class TemporalCausalityTests(unittest.TestCase):
    def test_masked_lower_median_and_empty_history(self):
        diffs = torch.tensor([[1., 9., 3., 0.], [8., 2., 6., 4.], [.1, .5, 99., 99.], [99., 99., 99., 99.]])
        mask = torch.tensor([[False, False, False, True], [False]*4, [False, False, True, True], [True]*4])
        scale = TCSD._per_query_time_scale(diffs, mask)
        torch.testing.assert_close(scale, torch.tensor([[3.], [4.], [1.], [1.]]), rtol=0, atol=0)

    def test_matches_independent_row_reference(self):
        generator = torch.Generator().manual_seed(42)
        for dtype in [torch.float32, torch.float64]:
            diffs = torch.rand(128, 30, generator=generator, dtype=dtype) * 100
            mask = torch.rand(128, 30, generator=generator) < .4
            mask[0] = True
            expected = torch.stack([row[~pad].median().clamp(min=1.) if (~pad).any() else row.new_tensor(1.) for row, pad in zip(diffs, mask)])[:, None]
            actual = TCSD._per_query_time_scale(diffs, mask)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            changed = diffs.masked_fill(mask, 1e10)
            torch.testing.assert_close(TCSD._per_query_time_scale(changed, mask), expected, rtol=0, atol=0)

    def test_future_query_histories_do_not_change_earlier_query(self):
        model = make_model()
        before = inputs()
        after = copy.deepcopy(before)
        after["neighbors_interact_times"][1:] = torch.tensor([19.7, 19.8, 19.9])
        self.assertTrue(all(torch.equal(before[key][0], after[key][0]) for key in before))
        features = []
        hook = model.backbone.structural_dynamics.register_forward_pre_hook(lambda module, values: features.append(values[0][0].detach().clone()))
        with torch.no_grad():
            first = model(**before)[0]
            second = model(**after)[0]
        hook.remove()
        torch.testing.assert_close(features[0], features[1], rtol=0, atol=0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_batch_partition_and_row_order(self):
        model = make_model()
        batch = inputs()
        batch["neighbors_interact_times"][1:] = torch.tensor([19.7, 19.8, 19.9])
        order = torch.tensor([2, 0, 1])
        with torch.no_grad():
            combined = model(**batch)
            separate = torch.cat([model(**{key: value[i:i+1] for key, value in batch.items()}) for i in range(3)])
            shuffled = model(**{key: value[order] for key, value in batch.items()})
        # Different matrix batch shapes may introduce ordinary floating-point roundoff.
        torch.testing.assert_close(combined, separate, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(combined[order], shuffled, rtol=1e-6, atol=1e-7)

    def test_candidate_permutation(self):
        model = make_model()
        batch = inputs()
        changed = copy.deepcopy(batch)
        order = torch.tensor([2, 0, 1])
        for key in ["test_dst", "dst_last_update_times", "dst_neighb_seq", "dst_neighb_interact_times"]:
            changed[key] = changed[key][:, order]
        with torch.no_grad():
            original = model(**batch)
            permuted = model(**changed)
        torch.testing.assert_close(original[:, order], permuted, rtol=1e-6, atol=1e-7)

    def test_public_loss_and_backward_are_finite_with_padding(self):
        model = make_model().train()
        model.backbone.neighbor_drop_rate = 0.0
        batch = inputs()
        batch["src_neighb_seq"][1, 2] = 0
        batch["src_neighb_seq"][2] = 0
        batch["dst_neighb_seq"][2] = 0
        scores = model(**batch).sigmoid().flatten()
        labels = torch.zeros_like(scores)
        labels[::3] = 1.0
        loss = torch.nn.functional.binary_cross_entropy(scores, labels)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(scores).all())
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))

    def test_initialization(self):
        model = make_model()
        gate = model.backbone.structural_dynamics.gate_mlp
        self.assertTrue(torch.equal(gate[-2].bias, torch.full_like(gate[-2].bias, 2.0)))
        # the encoder is not re-initialized with the small normal weights of the other layers
        self.assertGreater(gate[0].bias.abs().sum().item(), 0.0)
        self.assertEqual(model.predictor.score_proj[1].bias.abs().sum().item(), 0.0)

    def test_checkpoint_roundtrip_and_public_name(self):
        original = make_model()
        restored = make_model()
        self.assertEqual(original.backbone.__class__.__name__, "TCSD")
        restored.load_state_dict(original.state_dict(), strict=True)
        single = {key: value[:1] for key, value in inputs().items()}
        with torch.no_grad():
            torch.testing.assert_close(original(**single), restored(**single), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
