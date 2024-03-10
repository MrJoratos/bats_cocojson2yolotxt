import json

def coco_to_yolo(coco_json_path, output_txt_path):
    with open(coco_json_path) as json_file:
        data = json.load(json_file)

    yolo_lines = []
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        keypoints = annotation['keypoints']
        width = annotation['width']
        height = annotation['height']

        for i in range(0, len(keypoints), 3):
            x = keypoints[i]
            y = keypoints[i + 1]
            visibility = keypoints[i + 2]

            if visibility > 0:  # Considering only visible keypoints
                # Normalizing coordinates to range [0, 1]
                x_normalized = x / width
                y_normalized = y / height

                yolo_line = f"{image_id} 0 {x_normalized} {y_normalized}\n"
                yolo_lines.append(yolo_line)

    with open(output_txt_path, 'w') as output_file:
        output_file.writelines(yolo_lines)

# Example usage
coco_json_path = 'annotations_trainval2017/annotations/person_keypoints_train2017.json'
output_txt_path = 'train_labels'
coco_to_yolo(coco_json_path, output_txt_path)