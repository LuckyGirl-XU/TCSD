"""CPU-only tests for public configuration and portable data paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.load_configs import get_link_prediction_args
from tgb.linkproppred.dataset import LinkPropPredDataset


class ReleaseTests(unittest.TestCase):
    def test_all_dataset_presets(self):
        plan = json.loads((ROOT / 'configs' / 'datasets.json').read_text())
        self.assertEqual(len(plan['datasets']), 14)
        for name, record in plan['datasets'].items():
            args = get_link_prediction_args(argv=['--dataset_name', name, '--gpu', '-1', '--load_best_configs'])
            self.assertEqual(args.model_name, 'TCSD')
            self.assertEqual(args.dataset_name, name)
            self.assertEqual(args.device, 'cpu')
            self.assertEqual(args.batch_size, 256)
            self.assertEqual(args.learning_rate, .0003)
            self.assertEqual(args.num_neighbors, record['num_neighbors'])
            self.assertEqual(args.hidden_dropout, record['hidden_dropout'])

    def test_presets_are_opt_in(self):
        args = get_link_prediction_args(argv=['--dataset_name', 'uci', '--gpu', '-1'])
        self.assertEqual((args.batch_size, args.learning_rate, args.num_neighbors), (200, .0001, 20))
        self.assertFalse(args.use_mrr_val)

    def test_cli_overrides_and_alias(self):
        args = get_link_prediction_args(argv=['--dataset_name', 'GoogleLocal', '--gpu', '-1', '--load_best_configs', '--seed', '2', '--num_neighbors', '7', '--no-use_mrr_val', '--no-use_lr_scheduler'])
        self.assertEqual((args.seed, args.num_neighbors), (2, 7))
        self.assertFalse(args.use_mrr_val)
        self.assertFalse(args.use_lr_scheduler)
        args = get_link_prediction_args(argv=['--dataset_name', 'tgbl-flight', '--gpu', '-1', '--load_best_configs'])
        self.assertEqual(args.dataset_name, 'tgbl_flight')
        self.assertTrue(args.use_mrr_val)
        self.assertEqual((args.num_epochs, args.patience), (100, 20))

    def test_dataset_without_preset_uses_the_default_configuration(self):
        plan = json.loads((ROOT / 'configs' / 'datasets.json').read_text())
        args = get_link_prediction_args(argv=['--dataset_name', 'my_graph', '--gpu', '-1', '--load_best_configs', '--hidden_dropout', '0.3'])
        self.assertEqual(args.dataset_name, 'my_graph')
        self.assertEqual(args.num_neighbors, plan['default']['num_neighbors'])
        self.assertEqual(args.hidden_dropout, .3)

    def test_dataset_loaders(self):
        from utils.DataLoader import TGB_DATA_NAME, TGB_Seq_DATA_NAME
        plan = json.loads((ROOT / 'configs' / 'datasets.json').read_text())
        expected = {'GoogleLocal': 'tgb-seq', 'YouTube': 'tgb-seq', 'Flickr': 'tgb-seq', 'Taobao': 'tgb-seq', 'WikiLink': 'tgb-seq',
                    'tgbl-coin': 'tgb', 'tgbl-comment': 'tgb'}
        for name in plan['datasets']:
            loader = 'tgb-seq' if name in TGB_Seq_DATA_NAME else 'tgb' if name in TGB_DATA_NAME else 'csv'
            self.assertEqual(loader, expected.get(name, 'csv'), name)

    def test_dry_config_exits_without_data(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as exc:
            get_link_prediction_args(argv=['--dataset_name', 'mooc', '--gpu', '-1', '--load_best_configs', '--print_config'])
        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(json.loads(buf.getvalue())['model_name'], 'TCSD')

    def test_evaluation_requires_checkpoint_request(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            get_link_prediction_args(argv=['--dataset_name', 'mooc', '--gpu', '-1', '--num_epochs', '0'])
        args = get_link_prediction_args(argv=['--dataset_name', 'mooc', '--gpu', '-1', '--num_epochs', '0', '--load_pretrained'])
        self.assertTrue(args.load_pretrained)

    def test_tgb_root_is_not_prefixed_by_package_path(self):
        # Stop before any downloads or processing, and inspect the resolved root.
        class StopBeforeDownload(Exception):
            pass
        seen = []
        def capture(dataset):
            seen.append(dataset.root)
            raise StopBeforeDownload()
        with patch.object(LinkPropPredDataset, 'download', capture):
            for root in ['./data', '/tmp/tcsd_data_path_test']:
                with self.assertRaises(StopBeforeDownload):
                    LinkPropPredDataset(name='tgbl-coin', root=root)
                self.assertEqual(seen[-1], str(Path(root).expanduser().resolve() / 'tgbl_coin'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
