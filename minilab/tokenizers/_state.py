from minilab.checks import require


def require_tokenizer_state(state, tokenizer_name: str, expected_type: str, fields: tuple[str, ...]) -> None:
    """Validate the saved JSON contract before a tokenizer trusts it."""
    expected = {"type", *fields}
    require(type(state) is dict, f"{tokenizer_name} tokenizer state must be a JSON object")
    require(set(state) == expected, (
        f"{tokenizer_name} tokenizer state fields must be exactly: {', '.join(sorted(expected))}"
    ))
    require(state["type"] == expected_type, f"{tokenizer_name} tokenizer state has wrong type")
    for field in fields:
        require(type(state[field]) is dict, f"{tokenizer_name} tokenizer state field '{field}' must be an object")


def require_id_map(ids, tokenizer_name: str) -> None:
    require(all(type(idx) is int and idx >= 0 for idx in ids), (
        f"{tokenizer_name} tokenizer state ids must be non-negative integers"
    ))
    require(len(set(ids)) == len(ids), f"{tokenizer_name} tokenizer state ids must be unique")
    require(sorted(ids) == list(range(len(ids))), (
        f"{tokenizer_name} tokenizer state ids must be contiguous from 0"
    ))
