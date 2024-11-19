import os
import gradio as gr
import cv2
import numpy as np
from copy import deepcopy

from PIL import Image
import torch
from gradio_image_prompter import ImagePrompter
from sam2.sam2_image_predictor import SAM2ImagePredictor

from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
from diffusers.models.controlnet_sd3 import SD3ControlNetModel


# -------------- initialization --------------

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# segmentation model
segmentor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny", cache_dir="ckpt", device=device)

# inpainting model
controlnet = SD3ControlNetModel.from_pretrained(
    "alimama-creative/SD3-Controlnet-Inpainting", use_safetensors=True, extra_conditioning_channels=1, cache_dir="ckpt"
)
inpaint_pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    cache_dir="ckpt",
)
inpaint_pipe.text_encoder.to(torch.float16)
inpaint_pipe.controlnet.to(torch.float16)
inpaint_pipe.to(device)

# -------------- general UI functionality --------------
def clear_all(length=480):
    return gr.Image.update(value=None, height=length, width=length, interactive=True), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        [], None, None

def clear_all_gen(length=480):
    return gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        [], None, None, None

def resize_image_and_points(img_array, points, max_resolution):
    # Get the dimensions of the original image
    original_height, original_width = img_array.shape[:2]

    # Determine the larger side
    max_side = max(original_width, original_height)

    # Check if resizing is needed
    if max_side > max_resolution:
        # Calculate the scaling factor
        scale_factor = max_resolution / max_side
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Resize the image
        img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Adjust the points
        scaled_points = []
        for point in points:
            x1, y1, l1, x2, y2, l2 = point
            x1 = int(x1 * scale_factor)
            y1 = int(y1 * scale_factor)
            x2 = int(x2 * scale_factor)
            y2 = int(y2 * scale_factor)
            scaled_points.append([x1, y1, l1, x2, y2, l2])

        return img_array, scaled_points

    # If no resizing is needed, return the original image and points
    return img_array, points

def get_subject_points(canvas):
    return canvas["image"], canvas["points"]

def segment(canvas, logits, max_resolution):
    if logits is not None:
        logits *=  32.0
    image, points = get_subject_points(canvas)
    image, points = resize_image_and_points(image, points, max_resolution)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        segmentor.set_image(image)
        input_points = []
        input_boxes = []
        for p in points:
            [x1, y1, _, x2, y2, _] = p
            if x2==0 and y2==0:
                input_points.append([x1, y1])
            else:
                input_boxes.append([x1, y1, x2, y2])
        if len(input_points) == 0:
            input_points = None
            input_labels = None
        else:
            input_points = np.array(input_points)
            input_labels = np.ones(len(input_points))
        if len(input_boxes) == 0:
            input_boxes = None
        else:
            input_boxes = np.array(input_boxes)
        masks, _, logits = segmentor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_boxes,
            multimask_output=False,
            return_logits=True,
            mask_input=logits,
        )
        mask = masks > 0
        masked_img = mask_image(image, mask[0], color=[252, 140, 90], alpha=0.9)
    return image, mask[0], masked_img, masked_img, logits / 32.0

def mask_image(image,
               mask,
               color=[255,0,0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose.
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    return out

def get_points(img,
               sel_pix,
               evt: gr.SelectData):
    # collect the selected point
    img = deepcopy(img)
    sel_pix.append(evt.index)
    # only draw the last two points
    if len(sel_pix) > 2:
        sel_pix = sel_pix[-2:]
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            # points = []
    return img if isinstance(img, np.ndarray) else np.array(img), sel_pix

# clear all handle/target points
def undo_points(original_image,
                mask):
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(original_image, mask, color=[252, 140, 90], alpha=0.9)
    else:
        masked_img = original_image.copy()
    return masked_img, []

def move_subject(image, mask, selected_points):
    start_point, end_point = selected_points[-2], selected_points[-1]
    # Step 1: Calculate the translation vector
    translation_vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])

    # Step 2: Ensure mask is in uint8 format and scaled to [0, 255] for OpenCV operations
    mask = (mask * 255).astype(np.uint8)

    # Step 3: Extract the subject area from the image using the mask
    subject_area = cv2.bitwise_and(image, image, mask=mask)

    # Step 4: Find the bounding box of the mask to crop the subject
    x, y, w, h = cv2.boundingRect(mask)
    cropped_subject = subject_area[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    # Step 5: Calculate new position for the subject
    new_x, new_y = x + translation_vector[0], y + translation_vector[1]

    # Step 6: Check boundaries to keep the subject within the image
    image_height, image_width = image.shape[:2]

    # Calculate the region where the subject will be placed
    target_x_start = max(new_x, 0)
    target_y_start = max(new_y, 0)
    target_x_end = min(new_x + w, image_width)
    target_y_end = min(new_y + h, image_height)

    # Calculate cropping needed if the subject moves outside the image
    crop_x_start = max(0, -new_x)
    crop_y_start = max(0, -new_y)
    crop_x_end = min(w, image_width - new_x)
    crop_y_end = min(h, image_height - new_y)

    cropped_subject = cropped_subject[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    cropped_mask = cropped_mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # Step 7: Remove the subject from the original location in the image by applying inverse mask
    inverse_mask = cv2.bitwise_not(mask)
    image_without_subject = cv2.bitwise_and(image, image, mask=inverse_mask)

    # Step 8: Overlay the cropped subject at the new location
    # Create a blank canvas for placing the moved subject and moved mask
    canvas = np.zeros_like(image)
    mask_canvas = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Place the subject at the adjusted position
    canvas[target_y_start:target_y_end, target_x_start:target_x_end] = cropped_subject
    mask_canvas[target_y_start:target_y_end, target_x_start:target_x_end] = cropped_mask

    # Step 9: Compute unfilled region mask
    # Start with the original mask of the subject
    unfilled_mask = mask.copy()
    # Subtract the moved mask area to leave only the empty areas
    unfilled_mask = cv2.bitwise_and(unfilled_mask, cv2.bitwise_not(mask_canvas))

    # Step 10: Combine the image without the original subject and the moved subject
    moved_image = cv2.bitwise_and(image_without_subject, image_without_subject, mask=cv2.bitwise_not(mask_canvas))
    moved_image += canvas

    return moved_image, unfilled_mask

def inpaint(image, mask, pos_prompt, neg_prompt, gen_seed, guidance_scale, n_inference_step, controlnet_conditioning_scale):
    height, width = image.shape[:2]
    generator = torch.Generator(device=device).manual_seed(gen_seed)
    image = Image.fromarray(image).convert("RGB")
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = Image.fromarray(mask).convert("L")
    res_image = inpaint_pipe(
        negative_prompt=neg_prompt,
        prompt=pos_prompt,
        height=height,
        width=width,
        control_image=image,
        control_mask=mask,
        num_inference_steps=n_inference_step,
        generator=generator,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=guidance_scale,
    ).images[0]
    # only change masked region
    res_image = np.where(np.array(mask)[..., None] > 0, np.array(res_image), np.array(image))
    return res_image

def inpaint_completion(image, pos_prompt, neg_prompt, gen_seed, guidance_scale, n_inference_step, controlnet_conditioning_scale):
    image, mask = image['background'], image['layers'][0]
    mask = np.mean(mask, axis=2)
    mask[mask > 0] = 255
    mask = mask.astype(np.uint8)
    return inpaint(image, mask, pos_prompt, neg_prompt, gen_seed, guidance_scale, n_inference_step, controlnet_conditioning_scale)

def inpaint_harmonization(image, pos_prompt, neg_prompt, gen_seed, guidance_scale, n_inference_step, controlnet_conditioning_scale):
    image, mask = image['background'], image['layers'][0]
    mask = np.mean(mask, axis=2)
    mask[mask > 0] = 255
    mask = mask.astype(np.uint8)
    return inpaint(image, mask, pos_prompt, neg_prompt, gen_seed, guidance_scale, n_inference_step, controlnet_conditioning_scale)
# ------------------------------------------------------

# -------------- UI definition --------------
with gr.Blocks() as demo:
    # layout definition
    with gr.Row():
        gr.Markdown("""# <center>Repositioning the Subject within Image </center>""")
    mask = gr.State(value=None) # store mask
    removal_mask = gr.State(value=None) # store removal mask
    selected_points = gr.State([]) # store points
    original_image = gr.State(value=None) # store original input image
    masked_original_image = gr.State(value=None) # store masked input image
    mask_logits = gr.State(value=None) # store mask logits
    with gr.Row():
        with gr.Column():
            gr.Markdown("""<p style="text-align: center; font-size: 20px">Input Image</p>""")
            canvas = ImagePrompter(type="numpy", label="Input Image", show_label=True) # for mask painting
            with gr.Row():
                select_button = gr.Button("Segment Subject")
                subject_points = gr.Dataframe(label="Subject Points", visible=False)
        with gr.Column():
            gr.Markdown("""<p style="text-align: center; font-size: 20px">Click Points</p>""")
            input_image = gr.Image(type="numpy", label="Moving Points", show_label=True, interactive=False) # for points clicking
            with gr.Row():
                undo_button = gr.Button("Undo point")
                move_button = gr.Button("Move Subject")
    with gr.Row():
        with gr.Column():
            gr.Markdown("""<p style="text-align: center; font-size: 20px">Moving Results</p>""")
            output_image = gr.Image(type="numpy", label="Subject Moved", show_label=True, interactive=False)
            with gr.Row():
                removal_button = gr.Button("Run Subject Removal")
        with gr.Column():
            gr.Markdown("""<p style="text-align: center; font-size: 20px">Subject Removal</p>""")
            removal_result = gr.ImageEditor(type="numpy", label="Removal Result", show_label=True, interactive=True, canvas_size=(1100, 1100),)
            with gr.Row():
                completion_button = gr.Button("Run Subject Completion")
    with gr.Row():
        with gr.Column():
            gr.Markdown("""<p style="text-align: center; font-size: 20px">Subject Completion</p>""")
            completion_result = gr.ImageEditor(type="numpy", label="Completion Result", show_label=True, interactive=True, canvas_size=(1100, 1100))
            with gr.Row():
                harmonization_button = gr.Button("Run Subject Harmonization")
        with gr.Column():
            gr.Markdown("""<p style="text-align: center; font-size: 20px">Subject Harmonization</p>""")
            harmonization_result = gr.Image(type="numpy", label="Harmonization Result", show_label=True, interactive=False)

    with gr.Tab("Prompts"):
        with gr.Row():
            subject_prompt = gr.Textbox(label="Subject Selection Prompt")
        with gr.Row():
            removal_pos_prompt = gr.Textbox(label="Subject Removal Positive Prompt", interactive=True)
            removal_neg_prompt = gr.Textbox(label="Subject Removal Negative Prompt", value="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW", interactive=True)
        with gr.Row():
            completion_pos_prompt = gr.Textbox(label="Subject Completion Positive Prompt", interactive=True)
            completion_neg_prompt = gr.Textbox(label="Subject Completion Negative Prompt", value="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW", interactive=True)
        with gr.Row():
            harmonization_pos_prompt = gr.Textbox(label="Subject Harmonization Positive Prompt", interactive=True)
            harmonization_neg_prompt = gr.Textbox(label="Subject Harmonization Negative Prompt", value="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW", interactive=True)

    # algorithm specific parameters
    with gr.Tab("Configs"):
        with gr.Row():
            gen_seed = gr.Number(value=42, label="Generation Seed", precision=0, interactive=True)
            max_resolution = gr.Number(value=1024, label="Max Resolution", precision=0, interactive=True)
            guidance_scale = gr.Number(value=7, label="CFG Scale", interactive=True)
            controlnet_conditioning_scale = gr.Number(value=0.95, label="Controlnet Conditioning Scale", interactive=True)
            n_inference_step = gr.Number(value=28, label="Total Sampling Steps", precision=0, interactive=True)

    with gr.Tab("Backbone Model Config"):
        with gr.Row():
            local_models_dir = 'ckpt'
            local_models_choice = \
                [os.path.join(local_models_dir,d) for d in os.listdir(local_models_dir) if os.path.isdir(os.path.join(local_models_dir,d))]
            model_path = gr.Dropdown(value="alimama-creative/SD3-Controlnet-Inpainting",
                label="Diffusion Model",
                choices=[
                    "alimama-creative/SD3-Controlnet-Inpainting",
                ] + local_models_choice
            )

    # event definition
    select_button.click(
        segment,
        [canvas, mask_logits, max_resolution],
        [original_image, mask, input_image, masked_original_image, mask_logits]
    )
    input_image.select(
        get_points,
        [masked_original_image, selected_points],
        [input_image, selected_points],
    )
    undo_button.click(
        undo_points,
        [original_image, mask],
        [input_image, selected_points]
    )
    move_button.click(
        move_subject,
        [original_image, mask, selected_points],
        [output_image, removal_mask]
    )
    removal_button.click(
        inpaint,
        [output_image, removal_mask, removal_pos_prompt, removal_neg_prompt, gen_seed, guidance_scale, n_inference_step, controlnet_conditioning_scale],
        [removal_result]
    )
    completion_button.click(
        inpaint_completion,
        [removal_result, completion_pos_prompt, completion_neg_prompt, gen_seed, guidance_scale, n_inference_step, controlnet_conditioning_scale],
        [completion_result]
    )
    harmonization_button.click(
        inpaint_harmonization,
        [completion_result, harmonization_pos_prompt, harmonization_neg_prompt, gen_seed, guidance_scale, n_inference_step, controlnet_conditioning_scale],
        [harmonization_result]
    )



demo.queue().launch(share=True)
