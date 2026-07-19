from mri_read.agent.context import AgentContext, PipelineError


def test_agent_context_defaults():
    ctx = AgentContext()
    assert ctx.manifest is None
    assert ctx.last_result is None


def test_pipeline_error_carries_context_and_message():
    ctx = AgentContext(manifest={"series": []})
    err = PipelineError("vision engine failed: timed out", ctx)

    assert str(err) == "vision engine failed: timed out"
    assert err.ctx is ctx
    assert err.ctx.manifest == {"series": []}


def test_pipeline_error_chains_original_exception():
    ctx = AgentContext()
    original = TimeoutError("timed out")
    try:
        try:
            raise original
        except TimeoutError as e:
            raise PipelineError("wrapped", ctx) from e
    except PipelineError as err:
        assert err.__cause__ is original
