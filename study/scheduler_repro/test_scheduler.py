"""CPU scheduling regressions. Run in a standalone process, without package init."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
# nanovllm.__init__ imports ModelRunner and compiles CUDA. Skip that entry point,
# but load the actual sequence, block manager, config and scheduler modules.
package = ModuleType('nanovllm')
package.__path__ = [str(ROOT / 'nanovllm')]
sys.modules['nanovllm'] = package

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def scheduler(gpu_blocks=16, max_swap_skips=2):
    # Scheduler only needs these scalar fields; no model/tokenizer construction.
    return Scheduler(SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=4096,
                     eos=-1, num_kvcache_blocks=gpu_blocks,
                     kvcache_block_size=256, num_cpu_blocks=16,
                     max_swap_skips=max_swap_skips))


def add_running(s, seed, prompt_length=2):
    seq = Sequence(list(range(seed, seed + prompt_length)))
    s.block_manager.allocate(seq)
    # Represent completed prefill followed by its sampled token. No GPU KV data.
    seq.append_token(seed + prompt_length)
    seq.status = SequenceStatus.RUNNING
    s.running.append(seq)
    return seq


def move_to_cpu(s, seq):
    s.running.remove(seq)
    mappings = s.preempt(seq)
    assert seq.status == SequenceStatus.SWAPPED
    assert mappings and not seq.block_table


class SchedulingRegressions(unittest.TestCase):
    def test_control_four_running_requests(self):
        s = scheduler()
        expected = [add_running(s, i * 1000) for i in range(4)]
        seqs, prefill, incoming, outgoing = s.schedule()
        self.assertFalse(prefill)
        self.assertEqual(seqs, expected)
        self.assertEqual((incoming, outgoing), ([], []))

    def test_control_one_swapped_request(self):
        s = scheduler()
        seq = add_running(s, 100)
        move_to_cpu(s, seq)
        cpu_ids = list(seq.cpu_block_table)
        seqs, prefill, incoming, outgoing = s.schedule()
        self.assertEqual(seqs, [seq])
        self.assertFalse(prefill)
        self.assertFalse(s.swapped)
        self.assertEqual(incoming, list(zip(cpu_ids, seq.block_table)))
        self.assertEqual(outgoing, [])

    def test_four_swap_ins_should_produce_four_decode_requests(self):
        s = scheduler()
        expected = [add_running(s, i * 1000) for i in range(4)]
        for seq in expected:
            move_to_cpu(s, seq)
        self.assertGreaterEqual(len(s.block_manager.free_block_ids), 4)
        self.assertTrue(all(len(seq) % 256 != 1 for seq in expected))
        seqs, prefill, incoming, outgoing = s.schedule()
        self.assertFalse(prefill)
        self.assertEqual({seq.seq_id for seq in seqs}, {seq.seq_id for seq in expected})
        self.assertEqual(len(seqs), 4)
        self.assertFalse(s.swapped)
        self.assertEqual(len(incoming), 4)
        self.assertFalse(outgoing)

    def test_two_swap_ins_should_not_reduce_decode_capacity(self):
        s = scheduler()
        expected = [add_running(s, i * 1000) for i in range(4)]
        for seq in expected[:2]:
            move_to_cpu(s, seq)
        seqs, prefill, incoming, outgoing = s.schedule()
        self.assertFalse(prefill)
        self.assertEqual(len(seqs), 4, f'Actual decode batch has {len(seqs)} requests')
        self.assertEqual(len({seq.seq_id for seq in seqs}), 4)
        self.assertEqual(len(incoming), 2)
        self.assertFalse(outgoing)

    def test_swap_in_requires_space_for_next_append(self):
        s = scheduler(gpu_blocks=2)
        seq = add_running(s, 1000, prompt_length=512)
        move_to_cpu(s, seq)
        bm = s.block_manager
        self.assertEqual(len(seq), 513)
        self.assertEqual(len(seq.cpu_block_table), 2)
        self.assertFalse(bm.can_swap_in(seq))

    def test_multiple_swap_ins_reserve_append_space(self):
        s = scheduler(gpu_blocks=5)
        first = add_running(s, 1000, prompt_length=512)
        second = add_running(s, 2000, prompt_length=512)
        move_to_cpu(s, first)
        move_to_cpu(s, second)

        seqs, prefill, incoming, outgoing = s.schedule()

        self.assertFalse(prefill)
        self.assertEqual(seqs, [second])
        self.assertEqual(len(incoming), 2)
        self.assertFalse(outgoing)
        self.assertEqual(list(s.swapped), [first])
        self.assertEqual(len(s.block_manager.free_block_ids), 2)

    def test_blocked_head_allows_first_fitting_request_to_bypass(self):
        s = scheduler(gpu_blocks=5)
        candidate = add_running(s, 1000, prompt_length=256)
        move_to_cpu(s, candidate)
        head = add_running(s, 2000, prompt_length=1024)
        move_to_cpu(s, head)
        blocker = add_running(s, 4000, prompt_length=257)

        seqs, prefill, incoming, outgoing = s.schedule()

        self.assertFalse(prefill)
        self.assertEqual(seqs, [candidate, blocker])
        self.assertEqual(len(incoming), 1)
        self.assertFalse(outgoing)
        self.assertEqual(list(s.swapped), [head])
        self.assertEqual(head.swap_skip_count, 1)

    def test_head_at_skip_limit_blocks_later_request(self):
        s = scheduler(gpu_blocks=5)
        candidate = add_running(s, 1000, prompt_length=256)
        move_to_cpu(s, candidate)
        head = add_running(s, 2000, prompt_length=1024)
        move_to_cpu(s, head)
        blocker = add_running(s, 4000, prompt_length=257)
        head.swap_skip_count = 2

        seqs, prefill, incoming, outgoing = s.schedule()

        self.assertFalse(prefill)
        self.assertEqual(seqs, [blocker])
        self.assertFalse(incoming)
        self.assertFalse(outgoing)
        self.assertEqual(list(s.swapped), [head, candidate])
        self.assertEqual(head.swap_skip_count, 2)

    def test_only_one_request_bypasses_head_per_round(self):
        s = scheduler(gpu_blocks=6)
        second = add_running(s, 1000, prompt_length=255)
        move_to_cpu(s, second)
        first = add_running(s, 2000, prompt_length=255)
        move_to_cpu(s, first)
        head = add_running(s, 3000, prompt_length=1024)
        move_to_cpu(s, head)
        blocker = add_running(s, 5000, prompt_length=513)

        seqs, _, incoming, outgoing = s.schedule()

        self.assertIn(first, seqs)
        self.assertNotIn(second, seqs)
        self.assertEqual(len(incoming), 1)
        self.assertFalse(outgoing)
        self.assertEqual(list(s.swapped), [head, second])
        self.assertEqual(head.swap_skip_count, 1)

    def test_preempt_resets_previous_skip_count(self):
        s = scheduler()
        seq = add_running(s, 1000)
        seq.swap_skip_count = 2

        move_to_cpu(s, seq)

        self.assertEqual(seq.swap_skip_count, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
