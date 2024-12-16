# hands23_detectorを評価するためのスクリプトを作成
import json
import numpy as np

# Load the JSON file
def convert_json_to_numpy(json_data):
    doh100_hand_boxes = []
    doh100_hand_scores = []
    pred_obj_boxes = []
    pred_obj_scores = []
    pred_sobj_boxes = []
    pred_sobj_scores = []
    pred_confidence_scores = []
    gt_hand_boxes = []
    gt_obj_boxes = []
    gt_sobj_boxes = []

    for image in json_data["images"]:
        hand_boxes = []
        hand_scores = []
        obj_boxes = []
        obj_scores = []
        sobj_boxes = []
        sobj_scores = []
        conf_scores = []

        for prediction in image["predictions"]:
            # Extract hand boxes and scores
            hand_box = list(map(float, prediction["hand_bbox"]))
            hand_boxes.append(hand_box)
            hand_scores.append(float(prediction["hand_pred_score"]))

            # Extract object boxes and scores
            if prediction["obj_bbox"]:
                obj_box = list(map(float, prediction["obj_bbox"]))
                obj_boxes.append(obj_box)
                obj_scores.append(float(prediction["obj_pred_score"]))
            else:
                obj_boxes.append([0.0, 0.0, 0.0, 0.0])
                obj_scores.append(0.0)

            # Extract secondary object boxes and scores
            if prediction["second_obj_bbox"]:
                sobj_box = list(map(float, prediction["second_obj_bbox"]))
                sobj_boxes.append(sobj_box)
                sobj_scores.append(float(prediction["sec_obj_pred_score"]))
            else:
                sobj_boxes.append([0.0, 0.0, 0.0, 0.0])
                sobj_scores.append(0.0)

            # Confidence scores (hand prediction scores)
            conf_scores.append(float(prediction["hand_pred_score"]))

        doh100_hand_boxes.append(hand_boxes)
        doh100_hand_scores.append(hand_scores)
        pred_obj_boxes.append(obj_boxes)
        pred_obj_scores.append(obj_scores)
        pred_sobj_boxes.append(sobj_boxes)
        pred_sobj_scores.append(sobj_scores)
        pred_confidence_scores.append(conf_scores)

    # Convert lists to NumPy arrays
    return {
        "doh100_hand_boxes": np.array(doh100_hand_boxes, dtype=object),
        "doh100_hand_scores": np.array(doh100_hand_scores, dtype=object),
        "pred_obj_boxes": np.array(pred_obj_boxes, dtype=object),
        "pred_obj_scores": np.array(pred_obj_scores, dtype=object),
        "pred_sobj_boxes": np.array(pred_sobj_boxes, dtype=object),
        "pred_sobj_scores": np.array(pred_sobj_scores, dtype=object),
        "pred_confidence_scores": np.array(pred_confidence_scores, dtype=object),
    }


# Example usage with a JSON file
if __name__ == "__main__":
    # Replace 'example.json' with the path to your JSON file
    with open("check/result.json", "r") as f:
        data = json.load(f)

    numpy_data = convert_json_to_numpy(data)

    # Print the converted NumPy arrays
    for key, value in numpy_data.items():
        print(f"{key}:\n{value}\n")
