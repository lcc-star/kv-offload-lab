"""Reference state machine for asynchronous KV block transfers."""

from dataclasses import dataclass
from enum import Enum, auto


class BlockState(Enum):
    FREE = auto()
    USED = auto()
    PENDING_FREE = auto()
    PENDING_WRITE = auto()


class RequestState(Enum):
    RUNNING = auto()
    SWAPPING_OUT = auto()
    SWAPPED = auto()
    SWAPPING_IN = auto()


class Direction(Enum):
    OUT = auto()
    IN = auto()


class OperationState(Enum):
    PENDING = auto()
    COMMITTED = auto()
    ABORTED = auto()


class FakeEvent:
    def __init__(self):
        self.completed = False

    def query(self):
        return self.completed


@dataclass(eq=False)
class PendingSwap:
    seq_id: int
    direction: Direction
    gpu_block_ids: tuple[int, ...]
    cpu_block_ids: tuple[int, ...]
    event: FakeEvent
    state: OperationState = OperationState.PENDING


class PendingBlockManager:
    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int):
        self.gpu_states = [BlockState.FREE] * num_gpu_blocks
        self.cpu_states = [BlockState.FREE] * num_cpu_blocks
        self.gpu_owners = [None] * num_gpu_blocks
        self.cpu_owners = [None] * num_cpu_blocks
        self.request_states = {}
        self.pending = []

    @property
    def num_free_gpu_blocks(self):
        return self.gpu_states.count(BlockState.FREE)

    @property
    def num_free_cpu_blocks(self):
        return self.cpu_states.count(BlockState.FREE)

    def add_running(self, seq_id: int, gpu_block_ids):
        self._require_new_request(seq_id)
        self._require_blocks(
            self.gpu_states, self.gpu_owners, gpu_block_ids, BlockState.FREE
        )
        self._set_blocks(
            self.gpu_states,
            self.gpu_owners,
            gpu_block_ids,
            BlockState.USED,
            seq_id,
        )
        self.request_states[seq_id] = RequestState.RUNNING
        self.assert_invariants()

    def add_swapped(self, seq_id: int, cpu_block_ids):
        self._require_new_request(seq_id)
        self._require_blocks(
            self.cpu_states, self.cpu_owners, cpu_block_ids, BlockState.FREE
        )
        self._set_blocks(
            self.cpu_states,
            self.cpu_owners,
            cpu_block_ids,
            BlockState.USED,
            seq_id,
        )
        self.request_states[seq_id] = RequestState.SWAPPED
        self.assert_invariants()

    def reserve_swap_out(self, seq_id: int, cpu_block_ids, event: FakeEvent):
        self._require_request_state(seq_id, RequestState.RUNNING)
        gpu_block_ids = self._owned_blocks(
            self.gpu_states, self.gpu_owners, seq_id, BlockState.USED
        )
        self._require_same_count(gpu_block_ids, cpu_block_ids)
        self._require_blocks(
            self.cpu_states, self.cpu_owners, cpu_block_ids, BlockState.FREE
        )
        self._set_blocks(
            self.gpu_states,
            self.gpu_owners,
            gpu_block_ids,
            BlockState.PENDING_FREE,
            seq_id,
        )
        self._set_blocks(
            self.cpu_states,
            self.cpu_owners,
            cpu_block_ids,
            BlockState.PENDING_WRITE,
            seq_id,
        )
        self.request_states[seq_id] = RequestState.SWAPPING_OUT
        operation = PendingSwap(
            seq_id,
            Direction.OUT,
            tuple(gpu_block_ids),
            tuple(cpu_block_ids),
            event,
        )
        self.pending.append(operation)
        self.assert_invariants()
        return operation

    def reserve_swap_in(self, seq_id: int, gpu_block_ids, event: FakeEvent):
        self._require_request_state(seq_id, RequestState.SWAPPED)
        cpu_block_ids = self._owned_blocks(
            self.cpu_states, self.cpu_owners, seq_id, BlockState.USED
        )
        self._require_same_count(cpu_block_ids, gpu_block_ids)
        self._require_blocks(
            self.gpu_states, self.gpu_owners, gpu_block_ids, BlockState.FREE
        )
        self._set_blocks(
            self.cpu_states,
            self.cpu_owners,
            cpu_block_ids,
            BlockState.PENDING_FREE,
            seq_id,
        )
        self._set_blocks(
            self.gpu_states,
            self.gpu_owners,
            gpu_block_ids,
            BlockState.PENDING_WRITE,
            seq_id,
        )
        self.request_states[seq_id] = RequestState.SWAPPING_IN
        operation = PendingSwap(
            seq_id,
            Direction.IN,
            tuple(gpu_block_ids),
            tuple(cpu_block_ids),
            event,
        )
        self.pending.append(operation)
        self.assert_invariants()
        return operation

    def commit(self, operation: PendingSwap):
        self._require_pending(operation)
        if operation.direction == Direction.OUT:
            self._set_blocks(
                self.gpu_states,
                self.gpu_owners,
                operation.gpu_block_ids,
                BlockState.FREE,
                None,
            )
            self._set_blocks(
                self.cpu_states,
                self.cpu_owners,
                operation.cpu_block_ids,
                BlockState.USED,
                operation.seq_id,
            )
            self.request_states[operation.seq_id] = RequestState.SWAPPED
        else:
            self._set_blocks(
                self.cpu_states,
                self.cpu_owners,
                operation.cpu_block_ids,
                BlockState.FREE,
                None,
            )
            self._set_blocks(
                self.gpu_states,
                self.gpu_owners,
                operation.gpu_block_ids,
                BlockState.USED,
                operation.seq_id,
            )
            self.request_states[operation.seq_id] = RequestState.RUNNING
        operation.state = OperationState.COMMITTED
        self.pending.remove(operation)
        self.assert_invariants()

    def abort(self, operation: PendingSwap):
        self._require_pending(operation)
        if operation.direction == Direction.OUT:
            self._set_blocks(
                self.gpu_states,
                self.gpu_owners,
                operation.gpu_block_ids,
                BlockState.USED,
                operation.seq_id,
            )
            self._set_blocks(
                self.cpu_states,
                self.cpu_owners,
                operation.cpu_block_ids,
                BlockState.FREE,
                None,
            )
            self.request_states[operation.seq_id] = RequestState.RUNNING
        else:
            self._set_blocks(
                self.cpu_states,
                self.cpu_owners,
                operation.cpu_block_ids,
                BlockState.USED,
                operation.seq_id,
            )
            self._set_blocks(
                self.gpu_states,
                self.gpu_owners,
                operation.gpu_block_ids,
                BlockState.FREE,
                None,
            )
            self.request_states[operation.seq_id] = RequestState.SWAPPED
        operation.state = OperationState.ABORTED
        self.pending.remove(operation)
        self.assert_invariants()

    def reclaim_completed(self):
        completed = [operation for operation in self.pending
                     if operation.event.query()]
        for operation in completed:
            self.commit(operation)
        return completed

    def assert_invariants(self):
        self._assert_pool(self.gpu_states, self.gpu_owners)
        self._assert_pool(self.cpu_states, self.cpu_owners)
        for operation in self.pending:
            assert operation.state == OperationState.PENDING
            expected = (
                RequestState.SWAPPING_OUT
                if operation.direction == Direction.OUT
                else RequestState.SWAPPING_IN
            )
            assert self.request_states[operation.seq_id] == expected
            gpu_state = (BlockState.PENDING_FREE
                         if operation.direction == Direction.OUT
                         else BlockState.PENDING_WRITE)
            cpu_state = (BlockState.PENDING_WRITE
                         if operation.direction == Direction.OUT
                         else BlockState.PENDING_FREE)
            for block_id in operation.gpu_block_ids:
                assert self.gpu_states[block_id] == gpu_state
                assert self.gpu_owners[block_id] == operation.seq_id
            for block_id in operation.cpu_block_ids:
                assert self.cpu_states[block_id] == cpu_state
                assert self.cpu_owners[block_id] == operation.seq_id

    @staticmethod
    def _assert_pool(states, owners):
        assert len(states) == len(owners)
        for state, owner in zip(states, owners):
            assert (state == BlockState.FREE) == (owner is None)

    def _require_new_request(self, seq_id):
        if seq_id in self.request_states:
            raise ValueError(f"request {seq_id} already exists")

    def _require_request_state(self, seq_id, expected):
        if self.request_states.get(seq_id) != expected:
            raise ValueError(f"request {seq_id} must be {expected.name}")

    @staticmethod
    def _require_same_count(source_ids, destination_ids):
        if len(source_ids) != len(destination_ids):
            raise ValueError("source and destination block counts must match")

    @staticmethod
    def _require_blocks(states, owners, block_ids, expected):
        block_ids = tuple(block_ids)
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("block IDs must be unique")
        for block_id in block_ids:
            if not 0 <= block_id < len(states):
                raise ValueError(f"block {block_id} is out of range")
            if states[block_id] != expected or owners[block_id] is not None:
                raise ValueError(f"block {block_id} is not {expected.name}")

    @staticmethod
    def _set_blocks(states, owners, block_ids, state, owner):
        for block_id in block_ids:
            states[block_id] = state
            owners[block_id] = owner

    @staticmethod
    def _owned_blocks(states, owners, seq_id, state):
        return [
            block_id
            for block_id, (block_state, owner) in enumerate(zip(states, owners))
            if block_state == state and owner == seq_id
        ]

    def _require_pending(self, operation):
        if operation.state != OperationState.PENDING or operation not in self.pending:
            raise ValueError("pending operation has already finished")
