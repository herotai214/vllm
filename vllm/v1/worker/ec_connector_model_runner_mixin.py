# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define EC connector functionality mixin for model runners.
"""

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.logger import init_logger
from vllm.v1.outputs import ECConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# Defined as a EC connector functionality mixin for ModelRunner (GPU, TPU)
class ECConnectorModelRunnerMixin:
    @staticmethod
    def maybe_save_ec_to_connector(
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
    ):
        if not has_ec_transfer():
            logger.debug("Not have ec transfer please check")
            return
        connector = get_ec_transfer()
        connector.save_caches(encoder_cache=encoder_cache, mm_hash=mm_hash)

    @staticmethod
    def get_finished_ec_transfers(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[set[str] | None, set[str] | None]:
        if has_ec_transfer():
            return get_ec_transfer().get_finished(scheduler_output.finished_req_ids)
        return None, None

    @staticmethod
    def maybe_get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> AbstractContextManager[ECConnectorOutput | None]:
        return (
            ECConnectorModelRunnerMixin._get_ec_connector_output(
                scheduler_output, encoder_cache, **kwargs
            )
            if has_ec_transfer()
            else nullcontext()
        )

    @staticmethod
    def maybe_wait_for_ec_load():
        """Kept for API compatibility; wait_for_load is now called inside the
        context manager before yielding, so this is a no-op."""
        pass

    # This context manager must be used within an active forward context.
    # It encapsulates the entire EC connector lifecycle within execute_model.
    #
    # Design note on failure recovery
    # --------------------------------
    # Both synchronous (start_load_caches return value) and asynchronous
    # (Mooncake RDMA; detected after wait_for_load) failures are collected
    # into output.failed_mm_hashes BEFORE the yield.  This means the model
    # runner can immediately rescue failed loads by adding the affected
    # (req_id, input_id) pairs back into scheduler_output.scheduled_encoder_inputs,
    # so _execute_mm_encoder computes them locally in the same step.
    # No scheduler involvement or rescheduling is required.
    @staticmethod
    @contextmanager
    def _get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> Generator[ECConnectorOutput, None, None]:
        output = ECConnectorOutput()

        ec_connector = get_ec_transfer()
        assert isinstance(ec_connector, ECConnectorBase)
        assert scheduler_output.ec_connector_metadata is not None
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

        if not ec_connector.is_producer:
            # Fire loads (async connectors: starts background transfers;
            # sync connectors: completes immediately).
            output.failed_mm_hashes = ec_connector.start_load_caches(
                encoder_cache, **kwargs
            )
            # Block until all async transfers settle, then collect failures.
            # Doing this before the yield ensures all failures are visible to
            # the model runner before _execute_mm_encoder is called.
            ec_connector.wait_for_load()
            output.failed_mm_hashes |= ec_connector.get_failed_loads()

        try:
            yield output
        finally:
            output.finished_sending, output.finished_recving = (
                ec_connector.get_finished(scheduler_output.finished_req_ids)
            )
            ec_connector.maybe_update_remote_cache_state(encoder_cache)
            ec_connector.clear_connector_metadata()
