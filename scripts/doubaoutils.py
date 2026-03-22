import re
import base64

import cv2
import numpy as np
from PIL import Image
import supervision as sv
from openai import OpenAI
BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)

seed_vl_version = "doubao-1-5-thinking-vision-pro-250428"
client = None

def set_doubao_api(api_key):
    global client  
    client = OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


POINT_ANNOTATOR = sv.CircleAnnotator()


def draw_boxes_points_with_labels(
    image_path,
    boxes=None,
    points=None,
    classes=None,
    output_path=None,
):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if boxes is not None:
        detections = sv.Detections(
            xyxy=boxes,
            class_id=np.arange(len(boxes)),
            confidence=np.ones(len(boxes))
        )
        num_dets = len(boxes)
        annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(
            image, detections)
    if points is not None:
        points = np.concatenate([points, points], axis=1)
        detections = sv.Detections(
            xyxy=points,
            class_id=np.arange(len(points)),
            confidence=np.ones(len(points))
        )
        num_dets = len(points)
        annotated_image = POINT_ANNOTATOR.annotate(
            image, detections,
        )
    if classes is not None:
        annotated_image = LABEL_ANNOTATOR.annotate(
            annotated_image, detections, labels=classes
        )

    if output_path:
        cv2.imwrite(
            output_path,
            cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        )
    
    return annotated_image

def parse_transform_points_vanilla(image_path, message):
    pattern = r'<point>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</point>'
    matches = re.finditer(pattern, message)

    points = [
        [float(match.group(1)), float(match.group(2))]
        for match in matches
    ]
    image = Image.open(image_path)
    w, h = image.size
    points = np.array(points, dtype='float') / 1000
    points[:, 0] *= w
    points[:, 1] *= h
    return points


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode('utf-8')
    return image

def inference_image(prompt, image_path):
    base64_image = encode_image(image_path)
    image_format = image_path.split('.')[-1]
    assert image_format in ['jpg', 'jpeg', 'png', 'webp']
    
    response = client.chat.completions.create(
    model=seed_vl_version,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{base64_image}"
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ])
    return response.choices[0]


def get_2d_point(obs_image, task):
    result = inference_image(task, obs_image)

    points = parse_transform_points_vanilla(obs_image, result.message.content)
    image = draw_boxes_points_with_labels(obs_image, points=points)
    Image.fromarray(image).save("./images/output_with_points.jpg")
    return points
