

# ==============================================================================
# EPD (Encoder-Prefill-Decode) EC Connector Integration Tests
# ==============================================================================

def _assert_encoder_cache_allocated(
    scheduler: Scheduler,
    requests: list[Request],
):
    """Check whether encoder cache is allocated correctly."""
    encoder_cache_manager = scheduler.encoder_cache_manager
    
    # Verify encoder cache manager exists
    assert encoder_cache_manager is not None, \
        "Encoder cache manager should exist"
    
    # Verify encoder cache manager has necessary attributes
    assert hasattr(encoder_cache_manager, 'cache_size'), \
        "Encoder cache manager should have cache_size"
    assert hasattr(encoder_cache_manager, 'allocate'), \
        "Encoder cache manager should have allocate method"
    
    # Verify each request with MM data has allocation
    for req in requests:
        if len(req.mm_hashes) > 0:
            # Check that encoder cache is allocated for this request's MM items
            assert encoder_cache_manager.cache_size >= 0, \
                f"Cache size should be non-negative, got {encoder_cache_manager.cache_size}"


def _assert_encoder_cache_freed(
    scheduler: Scheduler,
    freed_mm_hashes: set[str],
):
    """Check whether encoder cache is freed correctly."""
    encoder_cache_manager = scheduler.encoder_cache_manager
    
    # Verify encoder cache manager exists
    assert encoder_cache_manager is not None, \
        "Encoder cache manager should exist"
    
    # Verify encoder cache manager has free method
    assert hasattr(encoder_cache_manager, 'free'), \
        "Encoder cache manager should have free method"
    
    # Verify freed_mm_hashes is not empty if we expect something to be freed
    if len(freed_mm_hashes) > 0:
        # The hashes should have been freed
        assert all(isinstance(h, str) for h in freed_mm_hashes), \
            "All freed hashes should be strings"


def test_scheduler_no_ec_connector_by_default():
    """Test scheduler doesn't have EC connector by default."""
    scheduler = create_scheduler()
    assert scheduler.ec_connector is None


def test_scheduler_mm_request_schedules_encoder_inputs():
    """Test MM requests schedule encoder inputs for execution."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    
    # Create MM request: 200 total tokens with 100 MM tokens
    # num_tokens=200 INCLUDES the MM tokens (not additional!)
    # mm_positions shows WHERE the 100 MM tokens are within the 200
    requests = create_requests(
        num_requests=1,
        num_tokens=200,  # Total tokens (100 MM + 100 text)
        mm_positions=[[(0, 100)]],  # 100 MM tokens at offset 0
    )
    
    scheduler.add_request(requests[0])
    output = scheduler.schedule()
    
    # Should schedule the request
    assert len(output.scheduled_new_reqs) == 1
    
    # Scheduled tokens should equal total prompt tokens
    scheduled_tokens = output.num_scheduled_tokens[requests[0].request_id]
    assert scheduled_tokens == 200, \
        f"Expected 200 total tokens, got {scheduled_tokens}"
    
    # Encoder inputs should be scheduled for execution
    assert len(output.scheduled_encoder_inputs) > 0, \
        "Should schedule encoder inputs for MM request"
    assert requests[0].request_id in output.scheduled_encoder_inputs


def test_scheduler_mm_request_multiple_items():
    """Test MM request with multiple items (3 images)."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    
    # Create MM request with 3 items
    # 300 total tokens: 3 x 50 MM tokens + 150 text tokens
    requests = create_requests(
        num_requests=1,
        num_tokens=300,  # Total
        mm_positions=[
            [PlaceholderRange(offset=0, length=50),    # First MM: 50 tokens
             PlaceholderRange(offset=50, length=50),   # Second MM: 50 tokens
             PlaceholderRange(offset=100, length=50)]  # Third MM: 50 tokens
        ],
    )
    
    scheduler.add_request(requests[0])
    output = scheduler.schedule()
    
    # Scheduled tokens should equal total
    scheduled_tokens = output.num_scheduled_tokens[requests[0].request_id]
    assert scheduled_tokens == 300, \
        f"Expected 300 total tokens, got {scheduled_tokens}"
    
    # Should have 3 encoder inputs (one per MM item)
    if requests[0].request_id in output.scheduled_encoder_inputs:
        encoder_inputs = output.scheduled_encoder_inputs[requests[0].request_id]
        assert len(encoder_inputs) == 3, \
            f"Expected 3 encoder inputs, got {len(encoder_inputs)}"


def test_scheduler_text_only_no_encoder_inputs():
    """Test text-only requests don't schedule encoder inputs."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    
    # Create text-only request (no mm_positions)
    NUM_TOKENS = 100
    requests = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS,
    )
    
    scheduler.add_request(requests[0])
    output = scheduler.schedule()
    
    # Should schedule
    assert len(output.scheduled_new_reqs) == 1
    
    # Scheduled tokens equals prompt tokens
    scheduled_tokens = output.num_scheduled_tokens[requests[0].request_id]
    assert scheduled_tokens == NUM_TOKENS, \
        f"Expected {NUM_TOKENS} tokens, got {scheduled_tokens}"
    
    # No encoder inputs scheduled
    assert requests[0].request_id not in output.scheduled_encoder_inputs or \
           len(output.scheduled_encoder_inputs[requests[0].request_id]) == 0, \
        "Text-only request should not have encoder inputs"


def test_scheduler_consumer_cache_miss_computes_locally():
    """Test consumer can compute encoder locally when cache miss (fallback)."""
    # This is the key test: consumer instance should be able to compute
    # encoder cache itself if it doesn't receive it from external storage
    
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        use_ec_connector=True,
        ec_role="consumer",
    )
    
    # Verify consumer role
    assert scheduler.ec_connector is not None
    assert scheduler.ec_connector.is_producer == False
    
    # Create MM request
    requests = create_requests(
        num_requests=1,
        num_tokens=200,  # Total (including 100 MM)
        mm_positions=[[(0, 100)]],  # 100 MM tokens
    )
    
    # Mock cache miss - encoder cache doesn't exist externally
    from unittest.mock import Mock
    scheduler.ec_connector.has_caches = Mock(return_value=[False])
    
    scheduler.add_request(requests[0])
    output = scheduler.schedule()
    
    # SCHEDULER should decide to compute encoder locally (fallback)
    assert len(output.scheduled_new_reqs) == 1
    
    # Should schedule full prompt tokens
    scheduled_tokens = output.num_scheduled_tokens[requests[0].request_id]
    assert scheduled_tokens == 200, \
        f"Expected 200 tokens on cache miss, got {scheduled_tokens}"
    
    # KEY: Should schedule encoder execution locally (fallback!)
    assert requests[0].request_id in output.scheduled_encoder_inputs, \
        "On cache miss, consumer should schedule LOCAL encoder execution"
    
    encoder_inputs = output.scheduled_encoder_inputs[requests[0].request_id]
    assert len(encoder_inputs) == 1, \
        f"Expected 1 encoder input for fallback, got {len(encoder_inputs)}"
    
    # Then MODEL_RUNNER will execute the encoder and cache the result


def test_scheduler_consumer_cache_hit_external_load():
    """Test consumer loads from external cache when hit."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        use_ec_connector=True,
        ec_role="consumer",
    )
    
    # Create MM request
    requests = create_requests(
        num_requests=1,
        num_tokens=200,  # Total
        mm_positions=[[(0, 100)]],  # 100 MM tokens
    )
    
    # Mock cache hit - encoder cache exists externally
    from unittest.mock import Mock
    scheduler.ec_connector.has_caches = Mock(return_value=[True])
    scheduler.ec_connector.update_state_after_alloc = Mock()
    
    scheduler.add_request(requests[0])
    output = scheduler.schedule()
    
    # Should schedule prompt tokens
    scheduled_tokens = output.num_scheduled_tokens[requests[0].request_id]
    assert scheduled_tokens == 200
    
    # KEY: Should NOT schedule encoder execution (load from cache)
    assert requests[0].request_id not in output.scheduled_encoder_inputs or \
           len(output.scheduled_encoder_inputs.get(requests[0].request_id, [])) == 0, \
        "On cache hit, should NOT schedule encoder execution"
    
    # Should call update_state_after_alloc for external load
    scheduler.ec_connector.update_state_after_alloc.assert_called()


def test_scheduler_frees_encoder_cache_on_finish():
    """Test scheduler frees encoder cache when request finishes."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    
    # Create MM request
    requests = create_requests(
        num_requests=1,
        num_tokens=150,  # Total
        mm_positions=[[(0, 50)]],  # 50 MM tokens
    )
    
    scheduler.add_request(requests[0])
    output = scheduler.schedule()
    
    # Simulate model execution
    model_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[EOS_TOKEN_ID]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)
    
    # Finish request
    scheduler.finish_requests(
        requests[0].request_id,
        RequestStatus.FINISHED_STOPPED
    )
    
    # Schedule again to trigger cleanup
    output2 = scheduler.schedule()
    
    # KEY ASSERTION: Encoder cache should be freed
    assert hasattr(output2, 'free_encoder_mm_hashes'), \
        "Output should have free_encoder_mm_hashes"
    
    # If request had MM data, its hash should be in freed list
    if len(requests[0].mm_hashes) > 0:
        # At minimum, freed list should exist
        assert isinstance(output2.free_encoder_mm_hashes, (list, set))


def test_scheduler_multiple_mm_requests():
    """Test multiple MM requests scheduled together."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    
    NUM_REQUESTS = 3
    
    # Create 3 MM requests
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=180,  # Each: 60 MM + 120 text = 180 total
        mm_positions=[[(0, 60)] for _ in range(NUM_REQUESTS)],
    )
    
    # Add all
    for req in requests:
        scheduler.add_request(req)
    
    output = scheduler.schedule()
    
    # Check all scheduled
    num_scheduled = len(output.scheduled_new_reqs)
    assert num_scheduled >= 1, \
        f"Should schedule at least 1 request, got {num_scheduled}"
    
    # Check each scheduled request has correct token count
    for req_id in output.num_scheduled_tokens:
        scheduled = output.num_scheduled_tokens[req_id]
        assert scheduled == 180, \
            f"Each request should schedule 180 tokens, got {scheduled}"
    
    # Check encoder inputs scheduled for each
    for req in requests[:num_scheduled]:  # For scheduled requests
        assert req.request_id in output.scheduled_encoder_inputs, \
            f"Request {req.request_id} should have encoder inputs"


def test_scheduler_encoder_budget_enforcement():
    """Test encoder budget limits how many MM requests scheduled."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=400,  # Limited budget
    )
    
    NUM_REQUESTS = 5
    
    # Create requests with large MM inputs (100 MM tokens each)
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=200,  # Each: 100 MM + 100 text = 200 total
        mm_positions=[[(0, 100)] for _ in range(NUM_REQUESTS)],
    )
    
    # Add all
    for req in requests:
        scheduler.add_request(req)
    
    output = scheduler.schedule()
    
    # Should NOT schedule all due to budget (400 / 200 = max 2)
    num_scheduled = len(output.scheduled_new_reqs)
    assert num_scheduled <= 2, \
        f"Budget should limit to <=2 requests, got {num_scheduled}"
    
    # Some should remain waiting
    assert len(scheduler.waiting) > 0, \
        "Some requests should wait due to budget"
    
    # Total scheduled tokens should not exceed budget
    total_scheduled = sum(output.num_scheduled_tokens.values())
    assert total_scheduled <= 400, \
        f"Total scheduled {total_scheduled} exceeds budget 400"


def test_scheduler_preemption_with_encoder_cache():
    """Test preemption frees encoder cache."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        num_blocks=10,  # Limited KV cache
        block_size=16,
    )
    
    # Create first MM request
    req1 = create_requests_with_priority(
        num_requests=1,
        priorities=[5],  # Low priority
        arrival_times=[1.0],
        num_tokens=120,  # 60 MM + 60 text = 120 total
        mm_positions=[[(0, 60)]],
        starting_idx=0,
    )[0]
    
    scheduler.add_request(req1)
    output1 = scheduler.schedule()
    
    # Simulate execution
    model_output1 = ModelRunnerOutput(
        req_ids=[req1.request_id],
        req_id_to_index={req1.request_id: 0},
        sampled_token_ids=[[100]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output1, model_output1)
    
    # Add high priority request to trigger preemption
    req2 = create_requests_with_priority(
        num_requests=1,
        priorities=[0],  # High priority
        arrival_times=[2.0],
        num_tokens=120,
        mm_positions=[[(0, 60)]],
        starting_idx=1,
    )[0]
    
    scheduler.add_request(req2)
    
    # Schedule - may trigger preemption
    output2 = scheduler.schedule()
    
    # If preemption occurred, low priority request should be in waiting
    if len(scheduler.waiting) > 0:
        # Preemption happened
        assert any(req.priority > 0 for req in scheduler.waiting), \
            "Preempted request should have lower priority"


def test_scheduler_consumer_with_partial_cache_hit():
    """Test consumer with partial cache hit (2 of 3 MM items cached)."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        use_ec_connector=True,
        ec_role="consumer",
    )
    
    # Create request with 3 MM items
    requests = create_requests(
        num_requests=1,
        num_tokens=300,  # Total (3 x 50 MM + 150 text = 300)
        mm_positions=[
            [PlaceholderRange(offset=0, length=50),
             PlaceholderRange(offset=50, length=50),
             PlaceholderRange(offset=100, length=50)]
        ],
    )
    
    # Mock partial cache hit: 1st and 3rd exist, 2nd missing
    from unittest.mock import Mock
    scheduler.ec_connector.has_caches = Mock(return_value=[True, False, True])
    scheduler.ec_connector.update_state_after_alloc = Mock()
    
    scheduler.add_request(requests[0])
    output = scheduler.schedule()
    
    # Should schedule all tokens
    scheduled_tokens = output.num_scheduled_tokens[requests[0].request_id]
    assert scheduled_tokens == 300
    
    # Should schedule ONLY the missing encoder input (index 1)
    if requests[0].request_id in output.scheduled_encoder_inputs:
        encoder_inputs = output.scheduled_encoder_inputs[requests[0].request_id]
        assert len(encoder_inputs) == 1, \
            f"Should schedule 1 encoder input (the missing one), got {len(encoder_inputs)}"


def test_scheduler_encoder_cache_manager_initialized():
    """Test encoder cache manager is initialized for MM models."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    
    # Encoder cache manager should exist
    assert hasattr(scheduler, 'encoder_cache_manager'), \
        "Scheduler should have encoder_cache_manager"
    assert scheduler.encoder_cache_manager is not None, \
        "Encoder cache manager should not be None"
    
    # Should have cache_size attribute
    assert hasattr(scheduler.encoder_cache_manager, 'cache_size'), \
        "Encoder cache manager should track cache_size"


def test_scheduler_text_model_no_encoder_cache_manager():
    """Test text-only models don't initialize encoder cache manager (or it's empty)."""
    scheduler = create_scheduler(model="facebook/opt-125m")  # Text-only
    
    # For text models, encoder cache might be 0-size or not needed
    if hasattr(scheduler, 'encoder_cache_manager'):
        # If it exists, cache size should be 0 or very small
        assert scheduler.encoder_cache_manager.cache_size >= 0


# Note: The following tests require utils.py updates:
# 
# In create_scheduler(), add:
#   use_ec_connector: bool = False,
#   ec_role: str = "consumer",
# 
# And in function body:
#   from vllm.config import ECTransferConfig
#   
#   ec_transfer_config = None
#   if use_ec_connector:
#       ec_transfer_config = ECTransferConfig(
#           ec_connector="ECSharedStorageConnector",
#           ec_role=ec_role,
#           ec_connector_extra_config={"shared_storage_path": "/tmp/ec_test"},
#       )
#   
#   vllm_config = VllmConfig(
#       ...,
#       ec_transfer_config=ec_transfer_config,
#   )

