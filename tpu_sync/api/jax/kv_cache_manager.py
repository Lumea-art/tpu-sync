# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-performance JAX KV Cache Manager (repurposed as TransferEngine)."""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax

from tpu_sync.frameworks.jax import _tpu_raiden_jax as _impl

_DEFAULT_MAX_STAGING_BLOCKS = 64


def _resolve_hbm_staging_enabled(explicit_flag: Optional[bool]) -> bool:
  """Resolves whether on-device HBM staging is enabled for D2H and H2D."""
  if explicit_flag is not None:
    return explicit_flag
  env_val = os.environ.get("RAIDEN_ENABLE_HBM_STAGING", "").strip().lower()
  return env_val in ("1", "true", "on")


def _next_power_of_two(n: int) -> int:
  """Returns the smallest power of two >= max(1, n)."""
  return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _decompose_power_of_two(n: int) -> List[int]:
  """Decomposes a positive integer into descending power-of-two bucket sizes."""
  buckets: List[int] = []
  rem = n
  while rem > 0:
    highest = 1 << (rem.bit_length() - 1)
    buckets.append(highest)
    rem -= highest
  return buckets


def _resolve_max_staging_blocks(
    explicit_max: Optional[int], num_device_blocks: int
) -> int:
  """Resolves power-of-two HBM staging window capacity in blocks."""
  if explicit_max is not None:
    if explicit_max < 1:
      raise ValueError(
          f"max_staging_blocks must be >= 1, got {explicit_max}"
      )
    return _next_power_of_two(explicit_max)
  env_val = os.environ.get("RAIDEN_HBM_STAGING_MAX_BLOCKS", "").strip()
  if env_val:
    parsed = int(env_val)
    if parsed < 1:
      raise ValueError(
          f"RAIDEN_HBM_STAGING_MAX_BLOCKS must be >= 1, got {parsed}"
      )
    return _next_power_of_two(parsed)
  return min(
      _DEFAULT_MAX_STAGING_BLOCKS, _next_power_of_two(max(1, num_device_blocks))
  )


class _StagedCopyFuture:
  """Wraps RaidenFutures while keeping staged device arrays alive until done."""

  def __init__(self, futures: Sequence[Any], keep_alive: Any = None):
    self._futures = list(futures)
    self._keep_alive = keep_alive

  def Await(self) -> None:  # pylint: disable=invalid-name
    try:
      for fut in self._futures:
        fut.Await()
    finally:
      self._keep_alive = None

  def wait(self) -> None:
    self.Await()

  def IsReady(self) -> bool:  # pylint: disable=invalid-name
    ready = all(fut.IsReady() for fut in self._futures)
    if ready:
      self._keep_alive = None
    return ready

  def is_ready(self) -> bool:
    return self.IsReady()


def _build_hbm_gather_fn(packed_sharding: Any) -> Callable[..., Any]:
  """Builds a bucketed JIT gather packing `[bucket_blocks, num_layers, ...]`."""

  def _gather(bucket_size: int, caches: Sequence[Any], idx: Any) -> Any:
    del bucket_size
    return jax.numpy.stack(
        [jax.numpy.take(layer, idx, axis=0) for layer in caches], axis=1
    )

  if packed_sharding is not None:
    return jax.jit(
        _gather, static_argnums=(0,), out_shardings=packed_sharding
    )
  return jax.jit(_gather, static_argnums=(0,))


def _build_hbm_scatter_fn(caches_shardings: Any) -> Callable[..., List[Any]]:
  """Builds a bucketed JIT scatter updating donated `caches` from staging."""

  def _scatter(
      bucket_size: int,
      caches: List[Any],
      staging_buf: Any,
      start_offset: Any,
      dst_block_ids: Any,
  ) -> List[Any]:
    staged_slice = jax.lax.dynamic_slice_in_dim(
        staging_buf, start_offset, bucket_size, axis=0
    )
    return [
        layer_cache.at[dst_block_ids].set(staged_slice[:, l_idx])
        for l_idx, layer_cache in enumerate(caches)
    ]

  if caches_shardings is not None:
    return jax.jit(
        _scatter,
        static_argnums=(0,),
        donate_argnums=(1,),
        out_shardings=caches_shardings,
    )
  return jax.jit(_scatter, static_argnums=(0,), donate_argnums=(1,))


class KVCacheManager:
  """Wrapper around compiled C++ TransferEngine.

  This class has been repurposed to expose the TransferEngine API interface
  for distributed disaggregated KV-cache movement on JAX TPUs.
  """

  def __init__(
      self,
      kv_caches: List[Any],
      local_control_port: int,
      max_blocks: Optional[int] = None,
      num_slots: Optional[int] = None,
      timeout_s: float = 120.0,
      unsafe_skip_buffer_lock: bool = True,
      host_blocks_to_allocate: Optional[int] = None,
      parallelism: int = 4,
      node_id: int = 0,
      raiden_worker_port: int = 0,
      raiden_controller_address: Optional[str] = None,
      worker_id: Optional[str] = None,
      enable_shm: bool = False,
      backend_configs: Optional[Sequence[Any]] = None,
      enable_hbm_staging: Optional[bool] = None,
      max_staging_blocks: Optional[int] = None,
  ):
    """Instantiates the TransferEngine-based KVCacheManager.

    Args:
      kv_caches: List of device-placed contiguous Tensors representing the
        sharded KV caches.
      local_control_port: TCP socket server port for control plane coordination.
      max_blocks: Maximum number of blocks per staging slot.
      num_slots: Number of transfer slots to allocate.
      timeout_s: Timeout in seconds for transfer operations.
      unsafe_skip_buffer_lock: Skip dynamic safety locking.
      host_blocks_to_allocate: Legacy/unified total blocks to allocate in host
        pool.
      parallelism: Number of parallel network copies per layer.
      node_id: This worker's index among the workers of its store: the
        workers registering with one controller must cover the dense range
        [0, expected_worker_count) (checked when the store has a global
        registry). Peer stores pair transfers on it -- worker k of one store
        reads from worker k of another -- so it must name the same KV shard
        in every store: the rank within the engine (TP x PCP x PP), not a
        rank counted across data-parallel replicas.
      raiden_worker_port: Optional port for WorkerService gRPC server.
      raiden_controller_address: Optional address of central RaidenController.
        If provided, the WorkerService gRPC server is enabled.
      worker_id: Optional identifier for this worker.
      enable_shm: Opt this manager's host buffers into the shared-memory
        segments named by RAIDEN_SHM_KEY. The env var supplies the segment
        namespace; this flag supplies the per-manager decision. Managers whose
        host buffers are transient staging must leave it off.
      backend_configs: Optional backend configurations (e.g. persistent
        storage). Pass the same BackendConfig objects given to the store's
        secondary_backend_configs. The posix backend derives its storage
        topology from `parallelism`: declare at least one axis (`tp`, `pcp`,
        `pp`) with `<axis>_size`, using the same axes and sizes as the store,
        and set `<axis>_rank` to this worker's rank in [0, size). Duplicate or
        missing ranks cause wrong or missed recalls. An invalid config aborts
        the process (LOG(FATAL)) at registration.
      enable_hbm_staging: Optional flag enabling on-device HBM gather/scatter
        staging for fragmented D2H and H2D copies (defaults to
        `RAIDEN_ENABLE_HBM_STAGING`).
      max_staging_blocks: Optional power-of-two cap on blocks staged per bulk
        DMA window when `enable_hbm_staging` is active (defaults to
        `RAIDEN_HBM_STAGING_MAX_BLOCKS` or 64).
    """
    self._kv_caches: List[Any] = (
        kv_caches if isinstance(kv_caches, list) else list(kv_caches)
    )
    self._unsafe_skip_buffer_lock: bool = unsafe_skip_buffer_lock
    self._enable_hbm_staging: bool = _resolve_hbm_staging_enabled(
        enable_hbm_staging
    )
    self._num_layers: int = len(self._kv_caches)
    self._num_device_blocks: int = 0
    self._block_tail_shape: Tuple[int, ...] = ()
    self._layer_dtype: Any = None
    self._max_staging_blocks: int = 0
    self._packed_sharding: Any = None
    self._index_sharding: Any = None
    self._gather_bucket_fn: Optional[Callable[..., Any]] = None
    self._scatter_bucket_fn: Optional[Callable[..., List[Any]]] = None
    self._h2d_staging_buf: Any = None

    if self._enable_hbm_staging:
      if not self._kv_caches:
        raise ValueError("enable_hbm_staging requires non-empty kv_caches.")
      first = self._kv_caches[0]
      if len(first.shape) < 2:
        raise ValueError(
            f"kv_caches[0] must be at least 2D, got shape={first.shape}"
        )
      for idx, arr in enumerate(self._kv_caches):
        if int(arr.shape[0]) < 1:
          raise ValueError(
              f"kv_caches[{idx}] must have >= 1 block, got shape={arr.shape}"
          )
        if arr.shape[1:] != first.shape[1:] or arr.dtype != first.dtype:
          raise ValueError(
              "enable_hbm_staging requires uniform per-block shape and dtype"
              f" across layers; layer 0 has ({first.shape[1:]}, {first.dtype}),"
              f" layer {idx} has ({arr.shape[1:]}, {arr.dtype})"
          )
      self._block_tail_shape = tuple(int(d) for d in first.shape[1:])
      self._layer_dtype = first.dtype
      self._num_device_blocks = min(
          int(arr.shape[0]) for arr in self._kv_caches
      )
      self._max_staging_blocks = _resolve_max_staging_blocks(
          max_staging_blocks, self._num_device_blocks
      )
      sharding = getattr(first, "sharding", None)
      if isinstance(sharding, jax.sharding.NamedSharding):
        if len(sharding.spec) >= 1 and sharding.spec[0] is not None:
          raise ValueError(
              "enable_hbm_staging requires axis 0 (blocks) to be unsharded;"
              f" got PartitionSpec={sharding.spec}"
          )
        mem_kind = getattr(sharding, "memory_kind", None)
        self._packed_sharding = jax.sharding.NamedSharding(
            sharding.mesh,
            jax.sharding.PartitionSpec(None, None, *sharding.spec[1:]),
            memory_kind=mem_kind,
        )
        self._index_sharding = jax.sharding.NamedSharding(
            sharding.mesh,
            jax.sharding.PartitionSpec(),
            memory_kind=mem_kind,
        )
        caches_shardings = [
            getattr(arr, "sharding", sharding) for arr in self._kv_caches
        ]
      else:
        self._packed_sharding = sharding
        self._index_sharding = sharding
        caches_shardings = None
      self._gather_bucket_fn = _build_hbm_gather_fn(self._packed_sharding)
      self._scatter_bucket_fn = _build_hbm_scatter_fn(caches_shardings)

    if host_blocks_to_allocate is not None:
      self._impl = _impl.KVCacheManager(
          self._kv_caches,
          local_control_port if local_control_port > 0 else None,
          host_blocks_to_allocate,
          unsafe_skip_buffer_lock,
          parallelism,
          raiden_worker_port,
          raiden_controller_address,
          worker_id,
          # Pass node_id so ReadRemote can always match src<->dst workers by
          # node_id (see the non-host_blocks branch, which already forwards it).
          node_id=node_id,
          enable_shm=enable_shm,
      )
    else:
      if max_blocks is None or num_slots is None:
        raise ValueError(
            "Must specify either (max_blocks, num_slots) or"
            " host_blocks_to_allocate."
        )
      self._impl = _impl.KVCacheManager(
          kv_caches=self._kv_caches,
          node_id=node_id,
          local_control_port=local_control_port,
          max_blocks=max_blocks,
          num_slots=num_slots,
          timeout_s=timeout_s,
          unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
          parallelism=parallelism,
          raiden_worker_port=raiden_worker_port,
          raiden_controller_address=raiden_controller_address,
          worker_id=worker_id,
          enable_shm=enable_shm,
      )
    if backend_configs:
      self._impl.register_kv_backends(list(backend_configs))

  @property
  def kv_caches(self) -> List[Any]:
    """Returns the latest per-layer JAX KV cache arrays."""
    return self._kv_caches

  def _place_index_array(self, values: Any) -> Any:
    """Places a small int32 index tensor on the replicated mesh sharding."""
    return jax.numpy.asarray(
        values, dtype=jax.numpy.int32, device=self._index_sharding
    )

  def _resolve_active_caches(
      self, kv_caches: Optional[Sequence[Any]]
  ) -> List[Any]:
    """Validates and records the caller's live KV cache arrays if supplied."""
    if kv_caches is not None:
      if len(kv_caches) != self._num_layers:
        raise ValueError(
            f"Expected {self._num_layers} KV cache layers, got {len(kv_caches)}"
        )
      if self._enable_hbm_staging:
        for idx, arr in enumerate(kv_caches):
          if (
              len(arr.shape) < 2
              or int(arr.shape[0]) < 1
              or tuple(arr.shape[1:]) != self._block_tail_shape
              or arr.dtype != self._layer_dtype
          ):
            raise ValueError(
                f"kv_caches[{idx}] has ({arr.shape}, {arr.dtype}), expected"
                f" block tail ({self._block_tail_shape}, {self._layer_dtype})"
            )
        self._num_device_blocks = min(int(arr.shape[0]) for arr in kv_caches)
      self._kv_caches = (
          kv_caches if isinstance(kv_caches, list) else list(kv_caches)
      )
    return self._kv_caches

  def _ensure_h2d_staging_buf(self) -> Any:
    """Lazily allocates and returns the reusable HBM staging buffer for H2D."""
    if self._h2d_staging_buf is None:
      staging_shape = (
          self._max_staging_blocks,
          self._num_layers,
          *self._block_tail_shape,
      )
      if self._packed_sharding is not None:
        alloc_fn = jax.jit(
            lambda: jax.numpy.zeros(staging_shape, dtype=self._layer_dtype),
            out_shardings=self._packed_sharding,
        )
      else:
        alloc_fn = jax.jit(
            lambda: jax.numpy.zeros(staging_shape, dtype=self._layer_dtype)
        )
      self._h2d_staging_buf = alloc_fn()
      jax.block_until_ready(self._h2d_staging_buf)
    return self._h2d_staging_buf

  def _staged_d2h_blocks(
      self,
      src_blocks: List[int],
      dst_blocks: List[int],
      active_caches: List[Any],
  ) -> _StagedCopyFuture:
    """Gathers `src_blocks` across layers in bucketed windows and DMAs D2H."""
    assert self._gather_bucket_fn is not None
    futures: List[Any] = []
    keep_alives: List[Any] = []
    for win_start in range(0, len(src_blocks), self._max_staging_blocks):
      win_src = src_blocks[win_start : win_start + self._max_staging_blocks]
      win_dst = dst_blocks[win_start : win_start + self._max_staging_blocks]
      win_len = len(win_src)
      bucket_size = _next_power_of_two(win_len)
      if win_len < bucket_size:
        padded_src = win_src + [win_src[-1]] * (bucket_size - win_len)
      else:
        padded_src = win_src
      idx_arr = self._place_index_array(padded_src)
      packed_arr = self._gather_bucket_fn(bucket_size, active_caches, idx_arr)
      jax.block_until_ready(packed_arr)
      fut = self._impl.d2h_packed(
          packed_arr,
          win_dst,
          [1] * win_len,
          self._unsafe_skip_buffer_lock,
      )
      futures.append(fut)
      keep_alives.append(packed_arr)
    return _StagedCopyFuture(futures, keep_alives)

  def get_raiden_worker_port(self) -> int:
    """Returns the gRPC server port if running, or 0."""
    return self._impl.get_raiden_worker_port()

  @property
  def is_listener_active(self) -> bool:
    """Returns True if the worker gRPC service listener is active."""
    return self.get_raiden_worker_port() > 0

  def get_local_endpoints(self) -> List[Dict[str, Any]]:
    """Returns the active Raiden endpoint descriptors."""
    return self._impl.get_local_endpoints()

  def register_read(self, req_id: str, uuid: int, block_ids: List[int]) -> bool:
    """Producer node notifies the registry/peer that blocks are ready for read.

    Args:
      req_id: The request ID of the transfer operation.
      uuid: The UUID of the request.
      block_ids: The list of block IDs to be read.

    Returns:
      True if a transfer is indeed needed; False if there is nothing to be
      transferred.

    Raises:
      ValueError: If `block_ids` contains duplicate block IDs.
      RuntimeError: If `uuid` already has an active registered read.
    """
    return bool(self._impl.notify_for_read(req_id, uuid, block_ids))

  def start_read(
      self,
      req_id: str,
      uuid: int,
      remote_endpoint: Union[str, List[Dict[str, Any]]],
      remote_block_ids: List[int],
      local_block_ids: List[int],
      parallelism: int = 1,
  ) -> None:
    """Consumer node initiates an asynchronous pull of blocks from a remote peer."""
    self._impl.start_read(
        req_id,
        uuid,
        remote_endpoint,
        remote_block_ids,
        local_block_ids,
        parallelism,
    )

  def poll_stats(self) -> Tuple[List[str], List[str], List[str]]:
    """Polls the status of all active background transfer operations.

    Returns:
      A tuple of (done_sending, done_recving, failed_recving) lists of request
      IDs.
    """
    return self._impl.complete_read()

  def d2h(
      self,
      src_offsets: List[int],
      dst_offsets: List[int],
      copy_sizes: List[int] | None = None,
      *,
      kv_caches: Optional[Sequence[Any]] = None,
  ) -> Any:
    """Device-to-Host (D2H) copy transfer.

    Args:
      src_offsets: Source block offsets.
      dst_offsets: Destination block offsets.
      copy_sizes: Optional number of contiguous blocks to copy per segment
        (defaults to 1 block per segment).
      kv_caches: Optional live per-layer JAX KV cache arrays (needed when the
        engine replaces `kv_caches` via buffer donation each step).

    Returns:
      A future representing the asynchronous copy transfer operation.
    """
    if copy_sizes is None:
      copy_sizes = [1] * len(src_offsets)
    if len(src_offsets) != len(dst_offsets) or len(src_offsets) != len(
        copy_sizes
    ):
      raise ValueError(
          "src_offsets, dst_offsets, and copy_sizes must have equal length:"
          f" got {len(src_offsets)}, {len(dst_offsets)}, {len(copy_sizes)}"
      )
    active_caches = self._resolve_active_caches(kv_caches)
    if (
        self._enable_hbm_staging
        and self._gather_bucket_fn is not None
        and src_offsets
    ):
      src_blocks: List[int] = []
      dst_blocks: List[int] = []
      for src_off, dst_off, sz in zip(src_offsets, dst_offsets, copy_sizes):
        if src_off < 0 or sz < 0 or src_off + sz > self._num_device_blocks:
          raise ValueError(
              f"D2H range [{src_off}, {src_off + sz}) out of bounds for"
              f" {self._num_device_blocks} device blocks"
          )
        if dst_off < 0:
          raise ValueError(f"Negative destination offset: {dst_off}")
        src_blocks.extend(range(src_off, src_off + sz))
        dst_blocks.extend(range(dst_off, dst_off + sz))
      if src_blocks:
        return self._staged_d2h_blocks(src_blocks, dst_blocks, active_caches)
    return self._impl.d2h(src_offsets, dst_offsets, copy_sizes)

  def h2d(
      self,
      src_offsets: List[int],
      dst_offsets: List[int],
      copy_sizes: List[int] | None = None,
      *,
      kv_caches: Optional[Sequence[Any]] = None,
  ) -> Any:
    """Host-to-Device (H2D) copy transfer.

     When `enable_hbm_staging` is active, gathers host blocks into a single bulk
    H2D DMA into the reusable HBM staging buffer, then scatters into `kv_caches`
    on device (updating `self.kv_caches` and mutating `kv_caches` in place if it
    is a `list`).

    Args:
      src_offsets: Source block offsets on host.
      dst_offsets: Destination block offsets on device.
      copy_sizes: Optional number of contiguous blocks to copy per segment
        (defaults to 1 block per segment).
      kv_caches: Optional live per-layer JAX KV cache arrays. When provided as a
        `list`, its elements are updated in place with the donated output arrays
        (also accessible via `self.kv_caches`).

    Returns:
      A future representing the copy transfer operation.
    """
    if copy_sizes is None:
      copy_sizes = [1] * len(src_offsets)
    if len(src_offsets) != len(dst_offsets) or len(src_offsets) != len(
        copy_sizes
    ):
      raise ValueError(
          "src_offsets, dst_offsets, and copy_sizes must have equal length:"
          f" got {len(src_offsets)}, {len(dst_offsets)}, {len(copy_sizes)}"
      )
    caller_list = (
        kv_caches
        if isinstance(kv_caches, list)
        else (self._kv_caches if kv_caches is None else None)
    )
    active_caches = self._resolve_active_caches(kv_caches)
    if (
        self._enable_hbm_staging
        and self._scatter_bucket_fn is not None
        and src_offsets
    ):
      src_blocks: List[int] = []
      dst_blocks: List[int] = []
      for src_off, dst_off, sz in zip(src_offsets, dst_offsets, copy_sizes):
        if dst_off < 0 or sz < 0 or dst_off + sz > self._num_device_blocks:
          raise ValueError(
              f"H2D range [{dst_off}, {dst_off + sz}) out of bounds for"
              f" {self._num_device_blocks} device blocks"
          )
        if src_off < 0:
          raise ValueError(f"Negative source offset: {src_off}")
        src_blocks.extend(range(src_off, src_off + sz))
        dst_blocks.extend(range(dst_off, dst_off + sz))
      if dst_blocks:
        staging_buf = self._ensure_h2d_staging_buf()
        cur_caches = list(active_caches)
        last_fut = None
        try:
          for win_start in range(0, len(dst_blocks), self._max_staging_blocks):
            win_src = src_blocks[
                win_start : win_start + self._max_staging_blocks
            ]
            win_dst = dst_blocks[
                win_start : win_start + self._max_staging_blocks
            ]
            win_len = len(win_dst)
            last_fut = self._impl.h2d_packed(
                staging_buf,
                win_src,
                [1] * win_len,
                self._unsafe_skip_buffer_lock,
            )
            last_fut.Await()
            offset = 0
            for b_size in _decompose_power_of_two(win_len):
              bucket_dst = self._place_index_array(
                  win_dst[offset : offset + b_size]
              )
              start_idx = self._place_index_array(offset)
              cur_caches = self._scatter_bucket_fn(
                  b_size, cur_caches, staging_buf, start_idx, bucket_dst
              )
              offset += b_size
            jax.block_until_ready(cur_caches)
        finally:
          if caller_list is not None:
            caller_list[:] = cur_caches
            self._kv_caches = caller_list
          else:
            self._kv_caches = cur_caches
        return _StagedCopyFuture([last_fut] if last_fut is not None else [])
    return self._impl.h2d(src_offsets, dst_offsets, copy_sizes)

  # =========================================================================
  # EXPERIMENTAL PHYSICAL CACHE MANAGEMENT APIs
  # The following APIs are experimental, expose physical cache internals,
  # and are subject to change in future releases.
  # =========================================================================

  def d2h_auto_allocate(
      self,
      src_offsets: List[int],
      copy_sizes: Optional[List[int]] = None,
      *,
      kv_caches: Optional[Sequence[Any]] = None,
  ) -> Tuple[List[int], Any]:
    """[EXPERIMENTAL] Device-to-Host (D2H) copy transfer with automatic host block allocation.

    WARNING: This API is experimental and subject to change in future releases.

    Args:
      src_offsets: Source block offsets on device.
      copy_sizes: Optional number of contiguous blocks to copy per segment.
      kv_caches: Optional live per-layer JAX KV cache arrays (needed when the
        engine replaces `kv_caches` via buffer donation each step).

    Returns:
      A tuple of (allocated_physical_chunk_ids, copy_future).
    """
    if copy_sizes is None:
      copy_sizes = [1] * len(src_offsets)
    if len(src_offsets) != len(copy_sizes):
      raise ValueError(
          "src_offsets and copy_sizes must have equal length:"
          f" got {len(src_offsets)}, {len(copy_sizes)}"
      )
    active_caches = self._resolve_active_caches(kv_caches)
    if self._enable_hbm_staging and self._gather_bucket_fn is not None:
      src_blocks: List[int] = []
      for src_off, sz in zip(src_offsets, copy_sizes):
        if src_off < 0 or sz < 0 or src_off + sz > self._num_device_blocks:
          raise ValueError(
              f"D2H range [{src_off}, {src_off + sz}) out of bounds for"
              f" {self._num_device_blocks} device blocks"
          )
        src_blocks.extend(range(src_off, src_off + sz))
      if not src_blocks:
        return [], self._impl.d2h([], [], [])
      allocated_chunks = list(self._impl.allocate_blocks(len(src_blocks)))
      try:
        fut = self._staged_d2h_blocks(
            src_blocks, allocated_chunks, active_caches
        )
      except Exception:
        self._impl.deallocate_blocks(allocated_chunks)
        raise
      return allocated_chunks, fut
    return self._impl.d2h_auto_allocate(src_offsets, copy_sizes)

  def unlock_blocks(self, block_ids: List[int]) -> None:
    """[EXPERIMENTAL] Unlocks the specified physical staging blocks on host.

    WARNING: This API is experimental and subject to change in future releases.

    Args:
      block_ids: List of physical chunk/block IDs to unlock.
    """
    self._impl.unlock_blocks(block_ids)

  def dump_metrics_to_string(self) -> str:
    """[EXPERIMENTAL] Dumps the metrics collector telemetry as a JSON string.

    WARNING: This API is experimental and subject to change in future releases.

    Returns:
      A JSON string representing the collected telemetry metrics.
    """
    return self._impl.dump_metrics_to_string()
