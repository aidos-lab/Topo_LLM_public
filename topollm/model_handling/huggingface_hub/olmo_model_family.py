"""Script for loading OLMo model family from Hugging Face Hub."""

import pprint

import torch
from huggingface_hub import list_repo_refs
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    """Load and test OLMo model family from Hugging Face Hub."""
    repo_id = "allenai/OLMo-2-0425-1B"

    # revision = "stage1-step300-tokens1B"
    # revision = "stage1-step140000-tokens294B"
    # revision = "stage1-step1907359-tokens4001B"
    revision = "stage2-ingredient3-step23000-tokens49B"

    out = list_repo_refs(
        repo_id=repo_id,
    )
    branches = [b.name for b in out.branches]

    print(  # noqa: T201 - we want this script to print
        f"{repo_id} has {len(branches)} branches:\n{pprint.pformat(branches)}",
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        revision=revision,
    )

    print(  # noqa: T201 - we want this script to print
        f"Loaded model {model.__class__.__name__} from {repo_id} with revision {revision}",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="allenai/OLMo-2-0425-1B",
        revision=revision,
    )

    message: list[str] = [
        # "Language modeling is ",
    ]
    inputs = tokenizer(
        message,
        return_tensors="pt",
        return_token_type_ids=False,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(  # noqa: T201 - we want this script to print
        f"Using device: {device}",
    )

    # Optional verifying cuda
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    print(  # noqa: T201 - we want this script to print
        f"Calling model.generate with inputs: {inputs}",
    )
    response = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    print(  # noqa: T201 - we want this script to print
        tokenizer.batch_decode(
            response,
            skip_special_tokens=True,
        )[0],
    )


if __name__ == "__main__":
    main()
