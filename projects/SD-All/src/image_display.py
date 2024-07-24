import os
import logging
import matplotlib.pyplot as plt
import torch
import PIL
from io import BytesIO
from model_loader import load_model


def display_ablation_steps_t2i(
    models, seed, prompt, inference_steps, height, width, guidance_scale, output_folder
):
    # Constants
    image_width, image_height = width, height  # Using provided dimensions
    dpi = 100  # Dots per inch
    n_rows = len(models)
    n_cols = len(inference_steps) + 1

    # Calculate figure size in inches to maintain original image size
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
            fontsize=18,
        )
        axes[i, 0].axis("off")

        if pipe is not None:
            for j, steps in enumerate(inference_steps):
                try:
                    if model_name == "sd3":
                        image = pipe(
                            prompt=prompt,
                            prompt_2=prompt,
                            num_inference_steps=steps,
                            height=height,
                            width=width,
                            guidance_scale=guidance_scale,
                            generator=generator,
                        ).images[0]
                    else:
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

                except Exception as e:
                    logging.error(
                        f"Error generating image with {model_name} and {steps} inference steps: {e}"
                    )
                    axes[i, j].set_visible(False)

            # Add titles below the bottom row images
            if i == n_rows - 1:
                for j, steps in enumerate(inference_steps):
                    fig.text(
                        (j + 1.5) / n_cols,
                        0.1,
                        f"{steps} steps",
                        ha="center",
                        fontsize=18,
                    )

    # plt.title("Comparison of Inference Steps", fontsize=36)
    # plt.title(
    #     f"Comparison of Inference Steps\nPrompt: {prompt}",
    # )
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title and caption

    result_image_path = os.path.join(output_folder, "ablation_experiment_results.png")
    plt.savefig(result_image_path)
    logging.info(f"Combined image saved to {result_image_path}")
    plt.show()


def display_ablation_steps_i2i(
    models,
    seed,
    ref_image_filepath,
    prompt,
    inference_steps,
    height,
    width,
    guidance_scale,
    output_folder,
):
    # Constants
    image_width, image_height = width, height  # Using provided dimensions
    dpi = 100  # Dots per inch
    n_rows = len(models)
    n_cols = len(inference_steps) + 2  # Additional column for model names

    # Calculate figure size in inches to maintain original image size
    fig_width = n_cols * (image_width / dpi)
    fig_height = n_rows * (image_height / dpi)

    # Create a figure with specified size
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        dpi=dpi,
        gridspec_kw={"width_ratios": [0.2, 1] + [1] * len(inference_steps)},
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
            fontsize=18,
        )
        axes[i, 0].axis("off")

        # Display ref_image in the second column
        axes[i, 1].imshow(ref_image)
        axes[i, 1].axis("off")

        if i == n_rows - 1:
            # Add title below the bottom row images
            fig.text(
                1.5 / n_cols,
                0.1,
                "Reference",
                ha="center",
                fontsize=18,
            )

        for j, steps in enumerate(inference_steps):
            try:
                if model_name == "sd3":
                    image = pipe(
                        image=ref_image,
                        prompt=prompt,
                        prompt_2=prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
                else:
                    image = pipe(
                        image=ref_image,
                        prompt=prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
                axes[i, j + 2].imshow(image)
                axes[i, j + 2].axis("off")

                if i == n_rows - 1:
                    # Add title below the bottom row images
                    fig.text(
                        (j + 1.5) / n_cols,
                        0.1,
                        f"{steps} steps",
                        ha="center",
                        fontsize=18,
                    )

            except Exception as e:
                logging.error(
                    f"Error generating image with {model_name} and {steps} inference steps: {e}"
                )
                axes[i, j + 2].set_visible(False)

    # plt.title("Comparison of Inference Steps", fontsize=36)
    # plt.title(
    #     f"Comparison of Inference Steps\nPrompt: {prompt}",
    # )
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title and caption

    result_image_path = os.path.join(output_folder, "ablation_experiment_results.png")
    plt.savefig(result_image_path)
    logging.info(f"Combined image saved to {result_image_path}")
    plt.show()
