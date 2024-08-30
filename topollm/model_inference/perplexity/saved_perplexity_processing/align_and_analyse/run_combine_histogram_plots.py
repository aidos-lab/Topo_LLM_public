# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import re

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH


def get_checkpoint(path: str) -> int | None:
    """Extract the checkpoint value from the directory path using regex."""
    match = re.search(r"ckpt-(\d+)", path)
    return int(match.group(1)) if match else None


def find_histograms(
    base_dir: os.PathLike,
) -> tuple[list[str], dict[str, list[tuple[int, str]]]]:
    """Traverse the directory structure to find relevant histogram files.

    Args:
        base_dir: The base directory to search for histogram files.

    Returns:
        A tuple containing:
        - A list of paths to base model histograms.
        - A dictionary mapping model names to lists of tuples, each containing a checkpoint number and histogram file path.
    """
    base_histograms = []
    finetuned_histograms = {}

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "histograms_manual_scale.pdf":
                checkpoint = get_checkpoint(root)

                # Check if it's the base model directory
                if "model-roberta-base_task-masked_lm" in root and "ckpt-" not in root:
                    base_histograms.append(os.path.join(root, file))
                    print(f"Base model histogram found: {os.path.join(root, file)}")  # Debug log

                # Check for finetuned models by matching a specific naming pattern
                elif checkpoint is not None:
                    # Extract the finetuned model name
                    model_name_match = re.search(
                        r"model-(model-roberta-base_task-masked_lm_[^_]+)",
                        root,
                    )
                    if model_name_match:
                        model_name = model_name_match.group(1)
                        if model_name not in finetuned_histograms:
                            finetuned_histograms[model_name] = []
                        finetuned_histograms[model_name].append(
                            (checkpoint, os.path.join(root, file)),
                        )
                        print(
                            f"Finetuned model histogram found: {model_name} - {os.path.join(root, file)}"
                        )  # Debug log

    # Sort the histograms by checkpoint values for each finetuned model
    for model, files in finetuned_histograms.items():
        finetuned_histograms[model] = sorted(files, key=lambda x: x[0])

    return base_histograms, finetuned_histograms


def create_title_pdf(
    title: str,
    output_path: str,
):
    """Create a simple PDF with the title using reportlab."""

    # TODO: Fix the problem that this string is too long and not properly displayed in the PDF

    c = canvas.Canvas(
        output_path,
        pagesize=(800, 100),
    )
    c.setFont("Helvetica", 12)
    c.drawString(20, 10, title)  # Draw text near the top of the page
    c.showPage()
    c.save()


def add_title_page(
    pdf_writer: PdfWriter,
    title: str,
):
    """Create a title page and add it to the PdfWriter."""
    title_pdf_path = "temp_title.pdf"
    create_title_pdf(
        title,
        title_pdf_path,
    )

    # TODO: Find a way to remove the temp page after this script is run

    with open(title_pdf_path, "rb") as temp_pdf:
        reader = PdfReader(temp_pdf)
        pdf_writer.add_page(reader.pages[0])


def combine_histograms_for_model(
    base_histograms: list[str],
    finetuned_histograms: list[tuple[int, str]],
    model_name: str,
    output_file: os.PathLike,
):
    """Combine histograms into a single PDF file, including headers, for a specific model."""
    pdf_writer = PdfWriter()

    # Add histograms from the base model
    for base_histogram in base_histograms:
        title = f"Model: Base Model, Checkpoint: N/A\nPath: {base_histogram}"
        add_title_page(pdf_writer, title)
        add_pdf_to_writer(base_histogram, pdf_writer)

    # Add histograms for the specific finetuned model
    for checkpoint, histogram_path in finetuned_histograms:
        title = f"Model: {model_name}, Checkpoint: {checkpoint}\nPath: {histogram_path}"
        add_title_page(pdf_writer, title)
        add_pdf_to_writer(histogram_path, pdf_writer)

    # Write the combined PDF
    with open(output_file, "wb") as f_out:
        pdf_writer.write(f_out)


def add_pdf_to_writer(pdf_path: str, pdf_writer: PdfWriter):
    """Add all pages of a PDF to the PdfWriter object."""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        pdf_writer.add_page(page)


def main() -> None:
    """Run main function to find and plot joint histograms."""
    # Set the base directory to the root of your structure
    base_directory = pathlib.Path(
        TOPO_LLM_REPOSITORY_BASE_PATH,
        "data",
        "analysis",
        "aligned_and_analyzed",
        "twonn",
    )

    # Find all folders in the "twonn" directory starting with "data-"
    data_directories = (
        dir_path for dir_path in base_directory.iterdir() if dir_path.is_dir() and dir_path.name.startswith("data-")
    )

    # Process each folder
    for data_directory in data_directories:
        print(f"Processing directory: {data_directory}")

        # Find relevant histograms in the current data directory
        base_histograms, finetuned_histograms = find_histograms(data_directory)

        # Combine and save the histograms for each finetuning model
        for model_name, histograms in finetuned_histograms.items():
            output_file_name = f"{model_name}_combined_histograms.pdf"
            output_file_path = data_directory / "combined_histograms" / output_file_name
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            combine_histograms_for_model(
                base_histograms,
                histograms,
                model_name,
                output_file_path,
            )


if __name__ == "__main__":
    main()
