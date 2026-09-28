from __future__ import annotations

from chatbot.conversation import ConversationStore, build_prompt, estimate_tokens
from chatbot.llm import ChatMessage


def user(text: str) -> ChatMessage:
    return {"role": "user", "content": text}


def bot(text: str) -> ChatMessage:
    return {"role": "assistant", "content": text}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_estimate_tokens_is_monotonic_and_includes_overhead() -> None:
    assert estimate_tokens("") == 4
    assert estimate_tokens("abcd") == 5
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


def test_prompt_contains_system_history_and_new_message_in_order() -> None:
    plan = build_prompt("sys", [user("hi"), bot("hello")], "next", token_budget=1000)
    assert [m["role"] for m in plan.messages] == ["system", "user", "assistant", "user"]
    assert plan.messages[0]["content"] == "sys"
    assert plan.messages[-1]["content"] == "next"
    assert plan.dropped_messages == 0


def test_oldest_history_is_dropped_to_fit_budget() -> None:
    history = [user("a" * 400), bot("b" * 400), user("c" * 40), bot("d" * 40)]
    plan = build_prompt("sys", history, "new", token_budget=100)
    contents = [m["content"] for m in plan.messages]
    assert contents == ["sys", "c" * 40, "d" * 40, "new"]
    assert plan.dropped_messages == 2
    assert plan.estimated_tokens <= 100


def test_trimmed_window_never_starts_with_an_assistant_turn() -> None:
    history = [user("u" * 400), bot("short"), user("q"), bot("r")]
    # Budget fits "short" but not the user turn before it.
    plan = build_prompt("sys", history, "new", token_budget=40)
    assert plan.messages[1]["role"] == "user"


def test_system_and_user_message_are_kept_even_when_over_budget() -> None:
    plan = build_prompt("sys", [user("old"), bot("old")], "x" * 2000, token_budget=10)
    assert [m["role"] for m in plan.messages] == ["system", "user"]


def test_store_appends_and_isolates_sessions() -> None:
    store = ConversationStore(ttl_seconds=60, max_sessions=10, max_messages=10)
    store.append("a", user("hi"), bot("hello"))
    store.append("b", user("other"))
    assert store.history("a") == [user("hi"), bot("hello")]
    assert store.history("b") == [user("other")]
    assert store.history("missing") == []


def test_history_returns_a_copy() -> None:
    store = ConversationStore(ttl_seconds=60, max_sessions=10, max_messages=10)
    store.append("a", user("hi"))
    store.history("a").append(user("mutated"))
    assert store.history("a") == [user("hi")]


def test_store_caps_messages_per_session() -> None:
    store = ConversationStore(ttl_seconds=60, max_sessions=10, max_messages=3)
    for i in range(5):
        store.append("a", user(str(i)))
    assert [m["content"] for m in store.history("a")] == ["2", "3", "4"]


def test_store_expires_idle_sessions() -> None:
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=60, max_sessions=10, max_messages=10, clock=clock)
    store.append("a", user("hi"))
    clock.now += 30
    store.append("b", user("hi"))
    clock.now += 45  # "a" idle 75s, "b" idle 45s
    assert store.history("a") == []
    assert store.history("b") == [user("hi")]
    assert len(store) == 1


def test_store_evicts_least_recently_used_session() -> None:
    store = ConversationStore(ttl_seconds=60, max_sessions=2, max_messages=10)
    store.append("a", user("1"))
    store.append("b", user("2"))
    store.append("a", user("3"))  # "a" becomes most recent
    store.append("c", user("4"))  # evicts "b"
    assert store.history("b") == []
    assert len(store.history("a")) == 2
    assert len(store) == 2


def test_clear_removes_only_that_session() -> None:
    store = ConversationStore(ttl_seconds=60, max_sessions=10, max_messages=10)
    store.append("a", user("hi"))
    store.append("b", user("hi"))
    store.clear("a")
    store.clear("never-existed")
    assert store.history("a") == []
    assert store.history("b") == [user("hi")]
