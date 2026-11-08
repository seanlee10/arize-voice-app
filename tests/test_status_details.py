import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grok_compat import normalize_grok_event, normalize_status_details


def _evt(t="response.created", sd="unimplemented"):
    return {"type": t, "event_id": "e1", "response": {"id": "r1", "status": "in_progress", "status_details": sd, "output": []}}


def test_string_status_details_normalized():
    for t in ("response.created", "response.done"):
        assert normalize_status_details(_evt(t))["response"]["status_details"] is None


def test_dict_status_details_untouched():
    e = _evt(sd={"type": "completed"})
    assert normalize_status_details(e) is e


def test_non_response_event_untouched():
    e = {"type": "ping"}
    assert normalize_status_details(e) is e


def test_grok_path_still_normalizes():
    assert normalize_grok_event(_evt())["response"]["status_details"] is None


def test_validates_with_sdk_adapter():
    from agents.realtime.openai_realtime import get_server_event_type_adapter
    e = normalize_status_details(_evt())
    e["response"].update({"object": "realtime.response"})
    get_server_event_type_adapter().validate_python(e)
