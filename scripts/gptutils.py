import base64
import io
from PIL import Image
import numpy as np
from openai import OpenAI


client = None

def set_openai_api(api_key):
    global client  
    client = OpenAI(api_key=api_key)


def encode_image(image):
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    

def call_gpt_model(prompt1=None, prompt2=None, prompt3=None, task=None, error_message=None, images=None):
    encoded_images = []

    if isinstance(images, (str, np.ndarray, Image.Image)):
        encoded_images.append(encode_image(images))
    elif isinstance(images, list):
        for image in images:
            encoded_images.append(encode_image(image))
    elif images is not None:
        raise ValueError(f"Unsupported input type for images: {type(images)}")

    task_prompt = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt1} if prompt1 else None,
            {"type": "text", "text": prompt2} if prompt2 else None,
            {"type": "text", "text": prompt3} if prompt3 else None,
            {"type": "text", "text": str(task)} if task else None,
            {"type": "text", "text": error_message} if error_message else None,
            # Add images to the content if any are encoded
            *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in encoded_images]
        ]
    }

    # Remove None values from the content
    task_prompt["content"] = [item for item in task_prompt["content"] if item is not None]

    messages = [
        {
            "role": "system",
            "content": "You are a single-arm gripper robot operation planning expert."
        },
        task_prompt
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500,
        temperature=0.2
    )

    return response.choices[0].message.content

