"""Tests for the asynchronous KV block state machine."""

import unittest

from pending_block_manager import (
    BlockState,
    FakeEvent,
    OperationState,
    PendingBlockManager,
    RequestState,
)


class PendingBlockTests(unittest.TestCase):
    def test_swap_out_blocks_stay_unavailable_until_commit(self):
        manager = PendingBlockManager(6, 6)
        manager.add_running(1, [1, 2])
        event = FakeEvent()
        operation = manager.reserve_swap_out(1, [3, 4], event)

        self.assertEqual(manager.gpu_states[1], BlockState.PENDING_FREE)
        self.assertEqual(manager.cpu_states[3], BlockState.PENDING_WRITE)
        self.assertEqual(manager.num_free_gpu_blocks, 4)
        self.assertEqual(manager.num_free_cpu_blocks, 4)
        self.assertEqual(manager.reclaim_completed(), [])

        event.completed = True
        self.assertEqual(manager.reclaim_completed(), [operation])
        self.assertEqual(manager.gpu_states[1], BlockState.FREE)
        self.assertEqual(manager.cpu_states[3], BlockState.USED)
        self.assertEqual(manager.request_states[1], RequestState.SWAPPED)
        self.assertNotIn(operation, manager.pending)

    def test_swap_in_blocks_stay_unavailable_until_commit(self):
        manager = PendingBlockManager(6, 6)
        manager.add_swapped(1, [2, 3])
        event = FakeEvent()
        operation = manager.reserve_swap_in(1, [4, 5], event)

        self.assertEqual(manager.cpu_states[2], BlockState.PENDING_FREE)
        self.assertEqual(manager.gpu_states[4], BlockState.PENDING_WRITE)
        self.assertEqual(manager.request_states[1], RequestState.SWAPPING_IN)

        event.completed = True
        manager.reclaim_completed()
        self.assertEqual(operation.state, OperationState.COMMITTED)
        self.assertEqual(manager.cpu_states[2], BlockState.FREE)
        self.assertEqual(manager.gpu_states[4], BlockState.USED)
        self.assertEqual(manager.request_states[1], RequestState.RUNNING)

    def test_completed_operations_are_reclaimed_out_of_order(self):
        manager = PendingBlockManager(8, 8)
        manager.add_running(1, [1, 2])
        manager.add_running(2, [3, 4])
        first_event = FakeEvent()
        second_event = FakeEvent()
        first = manager.reserve_swap_out(1, [1, 2], first_event)
        second = manager.reserve_swap_out(2, [3, 4], second_event)

        second_event.completed = True
        self.assertEqual(manager.reclaim_completed(), [second])
        self.assertEqual(first.state, OperationState.PENDING)
        self.assertEqual(manager.gpu_states[1], BlockState.PENDING_FREE)
        self.assertEqual(manager.gpu_states[3], BlockState.FREE)

    def test_abort_swap_out_restores_running_request(self):
        manager = PendingBlockManager(4, 4)
        manager.add_running(1, [0, 1])
        operation = manager.reserve_swap_out(1, [2, 3], FakeEvent())

        manager.abort(operation)

        self.assertEqual(manager.request_states[1], RequestState.RUNNING)
        self.assertEqual(manager.gpu_states[:2], [BlockState.USED] * 2)
        self.assertEqual(manager.cpu_states[2:], [BlockState.FREE] * 2)
        self.assertNotIn(operation, manager.pending)

    def test_abort_swap_in_restores_swapped_request(self):
        manager = PendingBlockManager(4, 4)
        manager.add_swapped(1, [0, 1])
        operation = manager.reserve_swap_in(1, [2, 3], FakeEvent())

        manager.abort(operation)

        self.assertEqual(manager.request_states[1], RequestState.SWAPPED)
        self.assertEqual(manager.cpu_states[:2], [BlockState.USED] * 2)
        self.assertEqual(manager.gpu_states[2:], [BlockState.FREE] * 2)

    def test_operation_cannot_finish_twice(self):
        manager = PendingBlockManager(4, 4)
        manager.add_running(1, [0])
        operation = manager.reserve_swap_out(1, [0], FakeEvent())
        manager.commit(operation)

        with self.assertRaisesRegex(ValueError, "already finished"):
            manager.commit(operation)
        with self.assertRaisesRegex(ValueError, "already finished"):
            manager.abort(operation)

    def test_pending_blocks_cannot_be_reserved_again(self):
        manager = PendingBlockManager(4, 4)
        manager.add_running(1, [0])
        manager.reserve_swap_out(1, [0], FakeEvent())

        with self.assertRaisesRegex(ValueError, "not FREE"):
            manager.add_running(2, [0])
        with self.assertRaisesRegex(ValueError, "not FREE"):
            manager.add_swapped(2, [0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
