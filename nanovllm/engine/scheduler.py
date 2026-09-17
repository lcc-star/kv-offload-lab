from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_swap_skips = config.max_swap_skips
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size, config.num_cpu_blocks)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running and not self.swapped

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool, list, list]:
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
        reserved_append_blocks = 0
        while self.swapped and num_swap_ins < self.max_num_seqs:
            head = self.swapped[0]
            seq = head
            bypassed_head = False
            if not self.block_manager.can_swap_in(head, reserved_append_blocks):
                if head.swap_skip_count >= self.max_swap_skips:
                    break
                seq = next((candidate for candidate in list(self.swapped)[1:]
                            if self.block_manager.can_swap_in(
                                candidate, reserved_append_blocks)), None)
                if seq is None:
                    break
                head.swap_skip_count += 1
                bypassed_head = True
            mappings = self.block_manager.swap_in(seq)
            swap_in_mappings.extend(mappings)
            seq.status = SequenceStatus.RUNNING
            seq.swap_skip_count = 0
            self.swapped.remove(seq)
            self.running.appendleft(seq)
            num_swap_ins += 1
            reserved_append_blocks += len(seq) % self.block_manager.block_size == 1
            if bypassed_head:
                break

        # decode
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
            mappings = self.block_manager.swap_out(seq)
            seq.status = SequenceStatus.SWAPPED
            self.swapped.appendleft(seq)
            return mappings
        # fallback: recompute
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        return []

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
