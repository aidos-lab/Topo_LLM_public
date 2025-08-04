"""Get default prompts for masked and causal language modeling."""


def get_default_mlm_prompts(
    mask_token: str,
) -> list[str]:
    """Get default masked language model prompts."""
    prompts: list[str] = [
        f"I am looking for a {mask_token}.",
        f"I am looking for a {mask_token}, can you help me?",
        f"Can you find me a {mask_token}?",
        f"I would like a {mask_token} hotel in the center of town, please.",
        f"{mask_token} is a cheap restaurant in the south of town.",
        f"The train should go to {mask_token}.",
        f"No, it should be {mask_token}, look again!",
        f"{mask_token} is a city in the south of England.",
        f"{mask_token} is the best city in the world.",
        f"I would like to invest in {mask_token}.",
        f"What is the best {mask_token} in town?",
        f"Can you recommend a good {mask_token}?",
    ]

    return prompts


def get_default_clm_prompts() -> list[str]:
    """Get default causal language model prompts."""
    prompts: list[str] = [
        "I am looking for a",
        "I am looking for a ",  # with space at the end
        "Can you find me a",
        "Can you find me a ",  # with space at the end
        "I would like a hotel in the",
        "Nandos is a",
        "The train should go to",
        "No, it should be",
        "No, you did not understand me! I want",
        "I'm so excited to visit",
        "Cambridge is",
        "I would like to invest in",
        "What is the best stock to buy tomorrow?",
        "What is the best",
        "Can you recommend a good",
        "The Eiffel tower is located in",
        "Eigenspaces corresponding to distinct eigenvalues are",
        "Eigenspaces corresponding to distinct eigenvalues are always linearly independent. Proof:",
        "Proof:",
    ]

    return prompts
