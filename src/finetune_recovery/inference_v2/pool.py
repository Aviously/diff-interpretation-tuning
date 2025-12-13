from __future__ import annotations

import asyncio
import multiprocessing as mp
from contextlib import asynccontextmanager
from multiprocessing.context import SpawnContext, SpawnProcess

import torch

from . import worker


class InferenceWorker:
    """
    One worker process + queues.
    Requests are serialized per worker; async via asyncio.to_thread.
    """

    def __init__(
        self,
        worker_id: str,
        base_hf_model_id: str,
        device: str,
        ctx: SpawnContext,
    ):
        self.worker_id = worker_id
        self.base_hf_model_id = base_hf_model_id
        self.device = device

        # Wether the worker is currently reserved
        self._reserved: bool = False

        self.in_q: mp.Queue = ctx.Queue()
        self.out_q: mp.Queue = ctx.Queue()

        self.proc: SpawnProcess = ctx.Process(
            target=worker.llm_worker,
            args=(worker_id, base_hf_model_id, device, self.in_q, self.out_q),
            daemon=True,
        )
        self.proc.start()

        self._ask_lock = asyncio.Lock()

    def is_reserved(self) -> bool:
        return self._reserved

    def reserve(self) -> None:
        assert not self._reserved, "worker already reserved"
        self._reserved = True

    def release(self) -> None:
        assert self._reserved, "worker not reserved"
        self._reserved = False

    def load_lora(self, lora_path: str, lora_idx: int) -> None:
        req = worker.LoadLoraWorkerRequest(
            lora_path=lora_path,
            lora_idx=lora_idx,
        )
        self.in_q.put(req.model_dump())

    async def ask(
        self,
        *,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        enable_lora: bool,
    ) -> str:
        """
        One in-flight request per worker (guarded by _ask_lock).
        Uses a thread to do blocking queue I/O.
        """
        async with self._ask_lock:
            req = worker.AskWorkerRequest(
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                enable_lora=enable_lora,
            )

            await asyncio.to_thread(self.in_q.put, req.model_dump())
            raw_resp = await asyncio.to_thread(self.out_q.get)

            resp = worker.WorkerResponse.model_validate(raw_resp)
            return resp.text

    def shutdown(self) -> None:
        self.proc.terminate()


class InferenceWorkerPool:
    """
    Minimal pool:
      - holds a list of Worker objects
      - allocate/free workers
      - manage LoRA indices
    """

    def __init__(
        self,
        base_hf_model_id: str,
        n_models_per_gpu: int,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")

        self.base_hf_model_id = base_hf_model_id
        self.n_models_per_gpu = n_models_per_gpu

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            raise RuntimeError("No CUDA devices found")

        self.num_workers = self.num_gpus * self.n_models_per_gpu
        self._worker_semaphore = asyncio.Semaphore(self.num_workers)

        self._ctx = mp.get_context("spawn")
        self._workers = [
            InferenceWorker(
                worker_id=f"g{gpu_idx}-{model_idx}",
                base_hf_model_id=base_hf_model_id,
                device=f"cuda:{gpu_idx}",
                ctx=self._ctx,
            )
            for gpu_idx in range(self.num_gpus)
            for model_idx in range(self.n_models_per_gpu)
        ]

    @asynccontextmanager
    async def get_worker(self, lora_path_and_idx: tuple[str, int] | None):
        """
        Async context manager that reserves a worker (optionally loading a LoRA)
        and automatically releases it when the context is exited.
        """
        await self._worker_semaphore.acquire()

        # Search for a free worker
        free_worker = None
        for w in self._workers:
            if not w.is_reserved():
                free_worker = w
                w.reserve()
                break

        assert free_worker is not None, "no free inference workers"

        try:
            if lora_path_and_idx is not None:
                free_worker.load_lora(
                    lora_path=lora_path_and_idx[0],
                    lora_idx=lora_path_and_idx[1],
                )

            yield free_worker

        finally:
            free_worker.release()
            self._worker_semaphore.release()

    def shutdown(self) -> None:
        for w in self._workers:
            w.shutdown()

    def __del__(self) -> None:
        self.shutdown()
