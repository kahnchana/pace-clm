from pace_clm.stream_gemma_adapter import build_causal_messages


def test_message_builder_excludes_future_and_limits_window():
    images = [object() for _ in range(40)]
    messages = build_causal_messages(
        stream_gemma_root="../stream-gemma",
        question="What happened?",
        system_prompt="system",
        images=images,
        emitted=["</Silence>"] * 39,
        current_index=35,
        frame_times=[float(index) for index in range(40)],
    )
    included = [
        item["image"]
        for message in messages
        for item in message["content"]
        if item["type"] == "image"
    ]
    assert included == images[4:36]
    assert images[36] not in included
    assert len(included) == 32
