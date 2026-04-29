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
