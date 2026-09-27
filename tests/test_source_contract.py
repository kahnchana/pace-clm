from pace_clm.stream_gemma_adapter import validate_promoted_contract


def test_promoted_source_contract_is_present():
    observed = validate_promoted_contract("../stream-gemma")
    assert observed["model_name"] == "Qwen/Qwen3.5-35B-A3B"
    assert observed["model_revision"] == "59d61f3ce65a6d9863b86d2e96597125219dc754"
