from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_swap_skips = config.max_swap_skips
        self.async_swap = getattr(config, "async_swap", False)
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size, config.num_cpu_blocks)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.swapping_out: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()
        self.swapping_in: deque[Sequence] = deque()
        self._new_swap_out = []
        self._new_swap_in = []

    def is_finished(self):
        return not any((self.waiting, self.running, self.swapping_out,
                        self.swapped, self.swapping_in))

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool, list, list]:
        if self.async_swap:
            self.reclaim_completed_swaps()
        swap_in_mappings = []
        swap_out_mappings = []

        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True, swap_in_mappings, swap_out_mappings

        # swap-in: try to bring swapped sequences back to GPU
        num_swap_ins = 0
        reserved_running_blocks = 0
        if self.async_swap:
            reserved_running_blocks = sum(
                len(seq) % self.block_manager.block_size == 1
                and not seq.append_block_reserved
                for seq in self.running
            )
        while self.swapped and num_swap_ins < self.max_num_seqs:
            head = self.swapped[0]
            seq = head
            bypassed_head = False
            if not self.block_manager.can_swap_in(head, reserved_running_blocks):
                if head.swap_skip_count >= self.max_swap_skips:
                    break
                seq = next((candidate for candidate in list(self.swapped)[1:]
                            if self.block_manager.can_swap_in(
                                candidate, reserved_running_blocks)), None)
                if seq is None:
                    break
                head.swap_skip_count += 1
                bypassed_head = True
            mappings = (self.block_manager.reserve_swap_in(seq)
                        if self.async_swap else self.block_manager.swap_in(seq))
            swap_in_mappings.extend(mappings)
            seq.swap_skip_count = 0
            self.swapped.remove(seq)
            if self.async_swap:
                seq.status = SequenceStatus.SWAPPING_IN
                self.swapping_in.append(seq)
                self._new_swap_in.append(seq)
            else:
                seq.status = SequenceStatus.RUNNING
                self.running.appendleft(seq)
            num_swap_ins += 1
            if bypassed_head:
                break

        # decode
        if self.async_swap:
            remaining = len(self.running)
            deferred = []
            while remaining and num_seqs < self.max_num_seqs:
                seq = self.running.popleft()
                remaining -= 1
                if self.block_manager.can_append(seq):
                    num_seqs += 1
                    self.block_manager.may_append(seq)
                    scheduled_seqs.append(seq)
                elif remaining and self.running:
                    victim = self.running.pop()
                    remaining -= 1
                    swap_out_mappings.extend(self.preempt(victim))
                    deferred.append(seq)
                elif self.swapping_out:
                    deferred.append(seq)
                else:
                    swap_out_mappings.extend(self.preempt(seq))
            self.running.extend(deferred)
            self.running.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, False, swap_in_mappings, swap_out_mappings

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    swap_out_mappings.extend(self.preempt(self.running.pop()))
                else:
                    swap_out_mappings.extend(self.preempt(seq))
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False, swap_in_mappings, swap_out_mappings

    def preempt(self, seq: Sequence) -> list[tuple[int, int]]:
        seq.swap_skip_count = 0
        if self.block_manager.can_swap_out(seq):
            if self.async_swap:
                mappings = self.block_manager.reserve_swap_out(seq)
                seq.status = SequenceStatus.SWAPPING_OUT
                self.swapping_out.append(seq)
                self._new_swap_out.append(seq)
            else:
                mappings = self.block_manager.swap_out(seq)
                seq.status = SequenceStatus.SWAPPED
                self.swapped.appendleft(seq)
            return mappings
        # fallback: recompute
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        return []

    def bind_swap_events(self, swap_in_event, swap_out_event):
        if self._new_swap_in:
            assert swap_in_event is not None
            for seq in self._new_swap_in:
                seq.swap_event = swap_in_event
        if self._new_swap_out:
            assert swap_out_event is not None
            for seq in self._new_swap_out:
                seq.swap_event = swap_out_event
        self._new_swap_in.clear()
        self._new_swap_out.clear()

    def reclaim_completed_swaps(self):
        for seq in list(self.swapping_out):
            if seq.swap_event is not None and seq.swap_event.query():
                self.block_manager.commit_swap_out(seq)
                seq.swap_event = None
                seq.status = SequenceStatus.SWAPPED
                self.swapping_out.remove(seq)
                self.swapped.appendleft(seq)
        for seq in list(self.swapping_in):
            if seq.swap_event is not None and seq.swap_event.query():
                self.block_manager.commit_swap_in(seq)
                seq.swap_event = None
                seq.status = SequenceStatus.RUNNING
                self.swapping_in.remove(seq)
                self.running.appendleft(seq)

    def wait_for_pending_swap(self):
        for queue in (self.swapping_out, self.swapping_in):
            for seq in queue:
                if seq.swap_event is not None:
                    seq.swap_event.synchronize()
                    return

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
