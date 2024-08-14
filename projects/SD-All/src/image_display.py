import os
import logging
import matplotlib.pyplot as plt
import torch
import PIL
from io import BytesIO
from model_loader import load_model

DEFAULT_SEED = 0
DEFAULT_INFERENCE_STEP = 20
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_GUIDANCE_SCALE = 7.0
DEFAULT_REF_IMAGE_FILEPATH = (
    "data/samples/hurricane-matthew_00000066_2_pre_disaster.png"
)
DEFAULT_OUTPUT_FOLDER = "out/ablation_all_in_one"


def display_ablation_steps_t2i(
    models,
    seed=DEFAULT_SEED,
    prompt=None,
    inference_steps=DEFAULT_INFERENCE_STEP,
    height=DEFAULT_HEIGHT,
    width=DEFAULT_WIDTH,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    output_folder=DEFAULT_OUTPUT_FOLDER,
):
    # Constants
    image_width, image_height = width, height  # Using provided dimensions
    dpi = 100  # Dots per inch
    n_rows = len(models)
    n_cols = len(inference_steps) + 1

    # Calculate figure size in inches to maintain original image size
    fig_width = n_cols * (image_width / dpi)
    fig_height = n_rows * (image_height / dpi) + 2  # Add extra space for description text


    # Create a figure with specified size
    fig, axes = plt.subplots(
        n_rows+1,
        n_cols,
        figsize=(fig_width, fig_height),
        dpi=dpi,
        gridspec_kw={'height_ratios': [image_height / dpi] * n_rows + [2]}
    )

    for i, model in enumerate(models):
        model_name = model["name"]
        model_path = model["path"]
        pipe = load_model(model_name, model_path, model_type="text-to-image")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        axes[i, 0].text(
            0.5,
            0.5,
            model_name,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=36,
        )
        axes[i, 0].axis("off")

        if pipe is not None:
            for j, steps in enumerate(inference_steps):
                try:

                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=steps,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]

                    axes[i, j + 1].imshow(image)
                    axes[i, j + 1].axis("off")
                    if i == 0:
                        axes[i, j + 1].set_title(f"{steps} steps", fontsize=36)

                except Exception as e:
                    logging.error(
                        f"Error generating image with {model_name} and {steps} inference steps: {e}"
                    )
                    axes[i, j + 1].set_visible(False)
    # Remove axes for the last row (used for text description)
    for ax in axes[-1, :]:
        ax.axis("off")
    # Add text description below the figures
    description_text=f"prompt:{prompt}"
    fig.text(0.5, 0.01, description_text, wrap=True, horizontalalignment='center', fontsize=48)

    result_image_path = os.path.join(output_folder, "ablation_experiment_results.png")
    plt.savefig(result_image_path, bbox_inches='tight')
    logging.info(f"Combined image saved to {result_image_path}")
    plt.show()


def display_ablation_steps_i2i(
    models,
    seed=DEFAULT_SEED,
    ref_image_filepath=DEFAULT_REF_IMAGE_FILEPATH,
    prompt=None,
    inference_steps=DEFAULT_INFERENCE_STEP,
    height=DEFAULT_HEIGHT,
    width=DEFAULT_WIDTH,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    output_folder=DEFAULT_OUTPUT_FOLDER,
):
    # Constants
    image_width, image_height = width, height  # Using provided dimensions
    dpi = 100  # Dots per inch
    n_rows = len(models)
    n_cols = len(inference_steps) + 2  # Additional column for model names

    # Calculate figure size in inches to maintain original image size
    fig_width = n_cols * (image_width / dpi)
    fig_height = n_rows * (image_height / dpi) + 2

    # Create a figure with specified size
    fig, axes = plt.subplots(
        n_rows+1,
        n_cols,
        figsize=(fig_width, fig_height),
        dpi=dpi,
        gridspec_kw={'height_ratios': [image_height / dpi] * n_rows + [2]}
    )

    for i, model in enumerate(models):
        model_name = model["name"]
        model_path = model["path"]
        pipe = load_model(model_name, model_path, model_type="image-to-image")
        ref_image = PIL.Image.open(ref_image_filepath).convert("RGB")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Display model name in the first column
        axes[i, 0].text(
            0.5,
            0.5,
            model_name,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=36,
        )
        axes[i, 0].axis("off")

        # Display ref_image in the second column
        axes[i, 1].imshow(ref_image)
        axes[i, 1].axis("off")

        if i == 0:
            axes[i, 1].set_title("Reference", fontsize=36)

        for j, steps in enumerate(inference_steps):
            try:

                image = pipe(
                    image=ref_image,
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
                axes[i, j + 2].imshow(image)
                axes[i, j + 2].axis("off")

                if i == 0:
                    axes[i, j + 2].set_title(f"{steps} steps", fontsize=36)

            except Exception as e:
                logging.error(
                    f"Error generating image with {model_name} and {steps} inference steps: {e}"
                )
                axes[i, j + 2].set_visible(False)

    # Remove axes for the last row (used for text description)
    for ax in axes[-1, :]:
        ax.axis("off")
    # Add text description below the figures
    description_text=f"prompt:{prompt}"
    fig.text(0.5, 0.01, description_text, wrap=True, horizontalalignment='center', fontsize=48)

    result_image_path = os.path.join(output_folder, "ablation_experiment_results.png")
    plt.savefig(result_image_path)
    logging.info(f"Combined image saved to {result_image_path}")
    plt.show()


def display_ablation_guidance_t2i(
    models,
    seed=DEFAULT_SEED,
    ref_image_filepath=DEFAULT_REF_IMAGE_FILEPATH,
    prompt=None,
    inference_steps=DEFAULT_INFERENCE_STEP,
    height=DEFAULT_HEIGHT,
    width=DEFAULT_WIDTH,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    output_folder=DEFAULT_OUTPUT_FOLDER,
):
    # Constants
    image_width, image_height = width, height
    dpi = 100
    n_rows = len(models)
    n_cols = len(guidance_scale) + 1
    fig_width = n_cols * (image_width / dpi)
    fig_height = n_rows * (image_height / dpi)

    # Create a figure with specified size
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        dpi=dpi,
    )

    for i, model in enumerate(models):
        model_name = model["name"]
        model_path = model["path"]
        pipe = load_model(model_name, model_path, model_type="text-to-image")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        axes[i, 0].text(
            0.5,
            0.5,
            model_name,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=36,
        )
        axes[i, 0].axis("off")

        if pipe is not None:
            for j, scale in enumerate(guidance_scale):
                try:
                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=inference_steps,
                        height=height,
                        width=width,
                        guidance_scale=scale,
                        generator=generator,
                    ).images[0]

                    axes[i, j + 1].imshow(image)
                    axes[i, j + 1].axis("off")
                    if i == 0:
                        axes[i, j + 1].set_title(f"guidance_scale:{scale}", fontsize=36)

                except Exception as e:
                    logging.error(
                        f"Error generating image with {model_name} and {scale} guidance_scale: {e}"
                    )
                    axes[i, j + 1].set_visible(False)

    result_image_path = os.path.join(output_folder, "ablation_experiment_results.png")
    plt.savefig(result_image_path)
    logging.info(f"Combined image saved to {result_image_path}")
    plt.show()


def display_ablation_guidance_i2i(
    models,
    seed=DEFAULT_SEED,
    ref_image_filepath=DEFAULT_REF_IMAGE_FILEPATH,
    prompt=None,
    inference_steps=DEFAULT_INFERENCE_STEP,
    height=DEFAULT_HEIGHT,
    width=DEFAULT_WIDTH,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    output_folder=DEFAULT_OUTPUT_FOLDER,
):
    # Constants
    image_width, image_height = width, height  # Using provided dimensions
    dpi = 100  # Dots per inch
    n_rows = len(models)
    n_cols = len(guidance_scale) + 2  # Additional column for model names

    # Calculate figure size in inches to maintain original image size
    fig_width = n_cols * (image_width / dpi)
    fig_height = n_rows * (image_height / dpi)

    # Create a figure with specified size
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        dpi=dpi,
        # gridspec_kw={"width_ratios": [0.2, 1] + [1] * len(guidance_scale)},
    )

    for i, model in enumerate(models):
        model_name = model["name"]
        model_path = model["path"]
        pipe = load_model(model_name, model_path, model_type="image-to-image")
        ref_image = PIL.Image.open(ref_image_filepath).convert("RGB")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Display model name in the first column
        axes[i, 0].text(
            0.5,
            0.5,
            model_name,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=36,
        )
        axes[i, 0].axis("off")

        # Display ref_image in the second column
        axes[i, 1].imshow(ref_image)
        axes[i, 1].axis("off")

        if i == 0:
            axes[i, i].set_title("Reference")

        for j, scale in enumerate(guidance_scale):
            try:

                image = pipe(
                    image=ref_image,
                    prompt=prompt,
                    num_inference_steps=inference_steps,
                    guidance_scale=scale,
                    generator=generator,
                ).images[0]
                axes[i, j + 2].imshow(image)
                axes[i, j + 2].axis("off")

                if i == 0:
                    axes[i, j + 2].set_title(f"Guidance Scale:{scale}", fontsize=36)

            except Exception as e:
                logging.error(
                    f"Error generating image with {model_name} and {scale} guidance scale: {e}"
                )
                axes[i, j + 2].set_visible(False)

    result_image_path = os.path.join(output_folder, "ablation_experiment_results.png")
    plt.savefig(result_image_path)
    logging.info(f"Combined image saved to {result_image_path}")
    plt.show()


def display_ablation_size_t2i(
    models,
    seed=DEFAULT_SEED,
    ref_image_filepath=DEFAULT_REF_IMAGE_FILEPATH,
    prompt=None,
    inference_steps=DEFAULT_INFERENCE_STEP,
    height=DEFAULT_HEIGHT,
    width=DEFAULT_WIDTH,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    output_folder=DEFAULT_OUTPUT_FOLDER,
):
    pass


def display_ablation_size_i2i(
    models,
    seed=DEFAULT_SEED,
    ref_image_filepath=DEFAULT_REF_IMAGE_FILEPATH,
    prompt=None,
    inference_steps=DEFAULT_INFERENCE_STEP,
    height=DEFAULT_HEIGHT,
    width=DEFAULT_WIDTH,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    output_folder=DEFAULT_OUTPUT_FOLDER,
):
    pass


def display_ablation_technique_i2i(
    models,
    seed=DEFAULT_SEED,
    ref_image_filepath=DEFAULT_REF_IMAGE_FILEPATH,
    prompt=None,
    inference_steps=DEFAULT_INFERENCE_STEP,
    height=DEFAULT_HEIGHT,
    width=DEFAULT_WIDTH,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    output_folder=DEFAULT_OUTPUT_FOLDER,
):
    # Constants
    dpi = 100  # Dots per inch

    # Dynamically extract model names and variants from the configuration
    model_names = list({model["name"] for model in models})
    model_variants = list({model["model_varient"] for model in models})

    n_rows = len(model_names)
    n_cols = (
        len(model_variants) + 2
    )  # Columns for model name, reference image, and each model variant result

    # Calculate figure size in inches to maintain original image size
    fig_width = n_cols * (width / dpi)
    fig_height = n_rows * (height / dpi)

    # Create a figure with specified size
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        dpi=dpi,
    )

    ref_image = PIL.Image.open(ref_image_filepath).convert("RGB")
    generator = torch.Generator(device="cuda").manual_seed(seed)

    for i, model_name in enumerate(model_names):
        # Display model name in the first column
        axes[i, 0].text(
            0.5,
            0.5,
            model_name,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=36,
        )
        axes[i, 0].axis("off")

        # Display reference image in the second column
        axes[i, 1].imshow(ref_image)
        axes[i, 1].axis("off")

        if i == 0:
            axes[i, 1].set_title("Reference Image")

        for j, model_variant in enumerate(model_variants):
            model = next(
                (
                    m
                    for m in models
                    if m["name"] == model_name and m["model_varient"] == model_variant
                ),
                None,
            )
            if model is not None:
                pipe = load_model(
                    model_name,
                    model["path"],
                    model_type="image-to-image",
                    model_varient=model_variant,
                )
                try:
                    image = pipe(
                        image=ref_image,
                        prompt=prompt,
                        num_inference_steps=inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
                    axes[i, j + 2].imshow(image)
                    axes[i, j + 2].axis("off")

                    if i == 0:
                        axes[i, j + 2].set_title(f"{model_variant} Result", fontsize=36)

                except Exception as e:
                    logging.error(
                        f"Error generating image with {model_name} and {model_variant} guidance scale: {e}"
                    )
                    axes[i, j + 2].set_visible(False)

    result_image_path = os.path.join(output_folder, "ablation_experiment_results.png")
    plt.savefig(result_image_path)
    logging.info(f"Combined image saved to {result_image_path}")
    plt.show()
