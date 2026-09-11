from embodied3.brain import parse_turn, as_schema_json, Move


def test_three_channels_with_fences_and_trailing_comma():
    t = parse_turn('```json\n{"emotion": "glad, a bit wary", "move": {"description": "a slow smile", "hold": "turn",}, "reply": "hi",}\n```')
    assert t.parsed and t.emotion == "glad, a bit wary" and t.move.description == "a slow smile" and t.reply == "hi"
    assert as_schema_json(t).startswith('{"emotion"') and '"move"' in as_schema_json(t)
    assert '"move"' not in as_schema_json(t, reply_only=True)


def test_plain_text_is_a_reply_without_emotion_or_move():
    t = parse_turn("Just a sentence.")
    assert not t.parsed and t.reply == "Just a sentence." and t.emotion == "" and not t.move


def test_move_shapes_are_lenient():
    assert parse_turn('{"move": "eyes go wide", "reply": "x"}').move.description == "eyes go wide"
    assert not parse_turn('{"emotion": "quiet", "move": {"description": ""}, "reply": "x"}').move
    assert not parse_turn('{"emotion": "quiet", "reply": null}').move
    m = parse_turn('{"move": {"description": "wave", "elements": "cat, hat", "hold": "forever"}, "reply": "x"}').move
    assert m.elements == ["cat", "hat"] and m.hold == "turn"
    assert parse_turn('{"perform": {"description": "old shape"}, "reply": "x"}').move.description == "old shape"


def test_history_keeps_only_the_last_emotion_and_move():
    from embodied3.brain import Brain, EchoBackend
    b = Brain(EchoBackend())
    for t in ("thank you", "why?", "there is a ghost"):
        b.turn(t, "mood line", "body line")
    assistant = [m["content"] for m in b.context_history() if m["role"] == "assistant"]
    assert all(c.startswith('{"reply"') for c in assistant[:-1]) and '"move"' not in assistant[0]
    assert assistant[-1].startswith('{"emotion"') and "hands come up" in assistant[-1]
