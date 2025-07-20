import gradio as gr
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from utils import preprocess, prepare_mask_and_masked_image, recover_image

# Load pipeline (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

pipe_inpaint = pipe_inpaint.to(device)

# Set scheduler (optional if needed)
# from diffusers import EulerAncestralDiscreteScheduler
# pipe_inpaint.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_inpaint.scheduler.config)

def run_inpaint(prompt, image, mask):
    if image is None or mask is None:
        return None

    mask, masked_image = prepare_mask_and_masked_image(image, mask)
    result = pipe_inpaint(
        prompt=prompt,
        image=masked_image,
        mask_image=mask,
        guidance_scale=7.5,
    ).images[0]

    return [recover_image(result, image, mask)]


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# ✨ Inpainting with Stable Diffusion")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="e.g. A cat sitting on a beach")
            image_input = gr.Image(label="Upload Image", type="pil")
            mask_input = gr.Image(label="Upload Mask", type="pil")
            submit = gr.Button("Generate")

        with gr.Column():
            gallery = gr.Gallery(label="Output", show_label=False, columns=2, height="auto")


    submit.click(fn=run_inpaint, inputs=[prompt, image_input, mask_input], outputs=gallery)

demo.launch(share=True)
