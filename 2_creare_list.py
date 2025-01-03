import json
import os
import numpy as np  # NumPyをインポート
from hotr.metrics.utils import get_AP_HO, get_AP_HOS, get_AP_single

# JSONファイルのパスを指定（適切なパスに変更してください）
json_file_path = 'hands23_detector_result/GT_pred.json'  # 例: '/path/to/your/data.json'

# JSONファイルの存在を確認
if not os.path.exists(json_file_path):
    raise FileNotFoundError(f"指定されたJSONファイルが見つかりません: {json_file_path}")

# JSONデータの読み込み
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 各リストの初期化
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

# 画像名でソート（順序を固定するため）
for image_name in sorted(data.keys()):
    image_data = data[image_name]
    
    # 画像の幅と高さ（GT内で定義されているので、最初のGTエントリから取得）
    if image_data.get('GT'):
        width = image_data['GT'][0].get('width')   # デフォルト値を設定
        height = image_data['GT'][0].get('height')  # デフォルト値を設定
    else:
        raise ValueError(f"画像 '{image_name}' にはGTデータがありません。")

    # ---- Ground Truth (GT) ----
    gt_hand_boxes_image = []
    gt_obj_boxes_image = []
    gt_sobj_boxes_image = []
    
    for gt_entry in image_data.get('GT', []):
        # Hand Bounding Box (正規化 -> ピクセル)
        x1_norm = gt_entry.get('x1', 0)
        y1_norm = gt_entry.get('y1', 0)
        x2_norm = gt_entry.get('x2', 0)
        y2_norm = gt_entry.get('y2', 0)
        
        x1 = x1_norm * width
        y1 = y1_norm * height
        x2 = x2_norm * width
        y2 = y2_norm * height
        
        gt_hand_boxes_image.append([x1, y1, x2, y2])
        
        # Object Bounding Box
        obj_bbox = gt_entry.get('obj_bbox')
        if obj_bbox:
            obj_x1 = obj_bbox.get('x1', 0) * width
            obj_y1 = obj_bbox.get('y1', 0) * height
            obj_x2 = obj_bbox.get('x2', 0) * width
            obj_y2 = obj_bbox.get('y2', 0) * height
            gt_obj_boxes_image.append([obj_x1, obj_y1, obj_x2, obj_y2])
        else:
            # オブジェクトが存在しない場合は [0.0, 0.0, 0.0, 0.0] を追加
            gt_obj_boxes_image.append([0.0, 0.0, 0.0, 0.0])
        
        # Second Object Bounding Box
        second_obj_bbox = gt_entry.get('second_obj_bbox')
        if second_obj_bbox:
            sobj_x1 = second_obj_bbox.get('x1', 0) * width
            sobj_y1 = second_obj_bbox.get('y1', 0) * height
            sobj_x2 = second_obj_bbox.get('x2', 0) * width
            sobj_y2 = second_obj_bbox.get('y2', 0) * height
            gt_sobj_boxes_image.append([sobj_x1, sobj_y1, sobj_x2, sobj_y2])
        else:
            # セカンドオブジェクトが存在しない場合は [0.0, 0.0, 0.0, 0.0] を追加
            gt_sobj_boxes_image.append([0.0, 0.0, 0.0, 0.0])
    
    # 各画像のGTボックスをNumPy配列に変換
    gt_hand_boxes.append(np.array(gt_hand_boxes_image))
    gt_obj_boxes.append(np.array(gt_obj_boxes_image))
    gt_sobj_boxes.append(np.array(gt_sobj_boxes_image))
    
    # ---- Prediction ----
    doh100_hand_boxes_image = []
    doh100_hand_scores_image = []
    pred_obj_boxes_image = []
    pred_obj_scores_image = []
    pred_sobj_boxes_image = []
    pred_sobj_scores_image = []
    pred_confidence_scores_image = []
    
    for pred_entry in image_data.get('prediction', []):
        # Hand Bounding Box (文字列 -> 浮動小数点数)
        hand_bbox_str = pred_entry.get('hand_bbox', ["0.0", "0.0", "0.0", "0.0"])
        if hand_bbox_str is not None:
            try:
                hand_bbox = [float(coord) for coord in hand_bbox_str]
            except (ValueError, TypeError):
                hand_bbox = [0.0, 0.0, 0.0, 0.0]
        else:
            hand_bbox = [0.0, 0.0, 0.0, 0.0]
        doh100_hand_boxes_image.append(hand_bbox)
        
        #pred confidence score
        pred_confidence_score = 0.0

        # Hand Prediction Score
        hand_pred_score_str = pred_entry.get('hand_pred_score', "0.0")
        try:
            hand_pred_score = float(hand_pred_score_str)
            pred_confidence_score = hand_pred_score
        except (ValueError, TypeError):
            hand_pred_score = 0.0
        doh100_hand_scores_image.append(hand_pred_score)
        
        # Object Bounding Box
        obj_bbox_str = pred_entry.get('obj_bbox', ["0.0", "0.0", "0.0", "0.0"])
        if obj_bbox_str is not None:
            try:
                obj_bbox = [float(coord) for coord in obj_bbox_str]
            except (ValueError, TypeError):
                obj_bbox = [0.0, 0.0, 0.0, 0.0]
        else:
            obj_bbox = [0.0, 0.0, 0.0, 0.0]
        pred_obj_boxes_image.append(obj_bbox)
        
        
        # Object Prediction Score
        obj_pred_score_str = pred_entry.get('obj_pred_score', "0.0")
        try:
            obj_pred_score = float(obj_pred_score_str)
            pred_confidence_score = pred_confidence_score * obj_pred_score
        except (ValueError, TypeError):
            obj_pred_score = 0.0
        pred_obj_scores_image.append(obj_pred_score)
        

        
        # Second Object Bounding Box
        second_obj_bbox_str = pred_entry.get('second_obj_bbox', None)
        if second_obj_bbox_str:
            try:
                second_obj_bbox = [float(coord) for coord in second_obj_bbox_str]
            except (ValueError, TypeError):
                second_obj_bbox = [0.0, 0.0, 0.0, 0.0]
        else:
            # セカンドオブジェクトが存在しない場合は [0.0, 0.0, 0.0, 0.0] を追加
            second_obj_bbox = [0.0, 0.0, 0.0, 0.0]
        pred_sobj_boxes_image.append(second_obj_bbox)
        
        # Second Object Prediction Score
        sec_obj_pred_score_str = pred_entry.get('sec_obj_pred_score', "0.0")
        if sec_obj_pred_score_str and sec_obj_pred_score_str != "None":
            try:
                sec_obj_pred_score = float(sec_obj_pred_score_str)
                pred_confidence_score = pred_confidence_score * sec_obj_pred_score
            except (ValueError, TypeError):
                sec_obj_pred_score = 0.0
        else:
            sec_obj_pred_score = 0.0
        pred_sobj_scores_image.append(sec_obj_pred_score)
        pred_confidence_scores_image.append(pred_confidence_score)
        
    
    # 各画像のPredictionボックスをリストに追加
    doh100_hand_boxes.append(doh100_hand_boxes_image)  # リストのリスト
    doh100_hand_scores.append(doh100_hand_scores_image)
    pred_obj_boxes.append(pred_obj_boxes_image)
    pred_obj_scores.append(pred_obj_scores_image)
    pred_sobj_boxes.append(pred_sobj_boxes_image)
    pred_sobj_scores.append(pred_sobj_scores_image)
    pred_confidence_scores.append(pred_confidence_scores_image)

# 各リストを個別のNumPy配列のリストに変換
doh100_hand_boxes_np = [np.array(boxes) for boxes in doh100_hand_boxes]
doh100_hand_scores_np = [np.array(scores) for scores in doh100_hand_scores]
pred_obj_boxes_np = [np.array(boxes) for boxes in pred_obj_boxes]
pred_obj_scores_np = [np.array(scores) for scores in pred_obj_scores]
pred_sobj_boxes_np = [np.array(boxes) for boxes in pred_sobj_boxes]
pred_sobj_scores_np = [np.array(scores) for scores in pred_sobj_scores]
pred_confidence_scores_np = [np.array(scores) for scores in pred_confidence_scores]
gt_hand_boxes_np = [np.array(boxes) for boxes in gt_hand_boxes]
gt_obj_boxes_np = [np.array(boxes) for boxes in gt_obj_boxes]
gt_sobj_boxes_np = [np.array(boxes) for boxes in gt_sobj_boxes]


#各リストのサイズを表示
print("doh100_hand_boxes_np:", len(doh100_hand_boxes_np))
print("doh100_hand_scores_np:", len(doh100_hand_scores_np))
print("pred_obj_boxes_np:", len(pred_obj_boxes_np))
print("pred_obj_scores_np:", len(pred_obj_scores_np))
print("pred_sobj_boxes_np:", len(pred_sobj_boxes_np))
print("pred_sobj_scores_np:", len(pred_sobj_scores_np))
print("pred_confidence_scores_np:", len(pred_confidence_scores_np))
print("gt_hand_boxes_np:", len(gt_hand_boxes_np))
print("gt_obj_boxes_np:", len(gt_obj_boxes_np))


# 各リストの内容を確認（デバッグ用）
# print("doh100_hand_boxes_np:", doh100_hand_boxes_np)
# print("doh100_hand_scores_np:", doh100_hand_scores_np)
# print("pred_obj_boxes_np:", pred_obj_boxes_np)
# print("pred_obj_scores_np:", pred_obj_scores_np)
# print("pred_sobj_boxes_np:", pred_sobj_boxes_np)
# print("pred_sobj_scores_np:", pred_sobj_scores_np)
# print("pred_confidence_scores_np:", pred_confidence_scores_np)
# print("gt_hand_boxes_np:", gt_hand_boxes_np)
# print("gt_obj_boxes_np:", gt_obj_boxes_np)
# print("gt_sobj_boxes_np:", gt_sobj_boxes_np)

# ---- 修正ポイント ----
# JSONシリアライズ可能な形式に変換（NumPy配列をリストに変換）
output = {
    "doh100_hand_boxes": [boxes.tolist() for boxes in doh100_hand_boxes_np],
    "doh100_hand_scores": [scores.tolist() for scores in doh100_hand_scores_np],
    "pred_obj_boxes": [boxes.tolist() for boxes in pred_obj_boxes_np],
    "pred_obj_scores": [scores.tolist() for scores in pred_obj_scores_np],
    "pred_sobj_boxes": [boxes.tolist() for boxes in pred_sobj_boxes_np],
    "pred_sobj_scores": [scores.tolist() for scores in pred_sobj_scores_np],
    "pred_confidence_scores": [scores.tolist() for scores in pred_confidence_scores_np],
    "gt_hand_boxes": [boxes.tolist() for boxes in gt_hand_boxes_np],
    "gt_obj_boxes": [boxes.tolist() for boxes in gt_obj_boxes_np],
    "gt_sobj_boxes": [boxes.tolist() for boxes in gt_sobj_boxes_np]
}

# 出力をJSONファイルに保存
output_file_path = 'output_lists.json'  # 必要に応じて変更してください

with open(output_file_path, 'w', encoding='utf-8') as f_out:
    json.dump(output, f_out, ensure_ascii=False, indent=4)

print(f"リストの作成が完了し、'{output_file_path}' に保存されました。")

# AP計算のためにNumPy配列を使用（リストのリストをそのまま渡す）
ap_ho_results = {}
ap_hos_results = {}
ap_hand_results = {}
ap_active_obj_results = {}
ap_second_obj_results = {}

for iou_thres in [0.75, 0.5, 0.25]:
    prec, rec, ap_ho = get_AP_HO(
        doh100_hand_boxes_np, 
        doh100_hand_scores_np, 
        pred_obj_boxes_np, 
        pred_obj_scores_np, 
        pred_confidence_scores_np, 
        gt_hand_boxes_np, 
        gt_obj_boxes_np, 
        iou_thres=iou_thres
    )
    ap_ho_results[iou_thres] = ap_ho

    prec, rec, ap_hos = get_AP_HOS(
        doh100_hand_boxes_np, 
        doh100_hand_scores_np, 
        pred_obj_boxes_np, 
        pred_obj_scores_np, 
        pred_sobj_boxes_np, 
        pred_sobj_scores_np, 
        pred_confidence_scores_np, 
        gt_hand_boxes_np, 
        gt_obj_boxes_np, 
        gt_sobj_boxes_np, 
        iou_thres=iou_thres
    )
    ap_hos_results[iou_thres] = ap_hos

    prec, rec, ap_hand = get_AP_single(
        doh100_hand_boxes_np, 
        doh100_hand_scores_np,
        gt_hand_boxes_np, 
        iou_thres=iou_thres
    )
    ap_hand_results[iou_thres] = ap_hand

    prec, rec, ap_active_obj = get_AP_single(
        pred_obj_boxes_np, 
        pred_obj_scores_np,
        gt_obj_boxes_np, 
        iou_thres=iou_thres
    )
    ap_active_obj_results[iou_thres] = ap_active_obj

    prec, rec, ap_second_obj = get_AP_single(
        pred_sobj_boxes_np, 
        pred_sobj_scores_np,
        gt_sobj_boxes_np, 
        iou_thres=iou_thres
    )
    ap_second_obj_results[iou_thres] = ap_second_obj

print('----[Hand, Active-object] AP----')
print(f'| AP (0.75)\t\t: {ap_ho_results[0.75]:.5f}')
print(f'| AP (0.50)\t\t: {ap_ho_results[0.50]:.5f}')
print(f'| AP (0.25)\t\t: {ap_ho_results[0.25]:.5f}')
print('----[Hand, Active-object, Second-object] AP----')
print(f'| AP (0.75)\t\t: {ap_hos_results[0.75]:.5f}')
print(f'| AP (0.50)\t\t: {ap_hos_results[0.50]:.5f}')
print(f'| AP (0.25)\t\t: {ap_hos_results[0.25]:.5f}')
print('----Hand AP----')
print(f'| AP (0.75)\t\t: {ap_hand_results[0.75]:.5f}')
print(f'| AP (0.50)\t\t: {ap_hand_results[0.50]:.5f}')
print(f'| AP (0.25)\t\t: {ap_hand_results[0.25]:.5f}')
print('----Active-object AP----')
print(f'| AP (0.75)\t\t: {ap_active_obj_results[0.75]:.5f}')
print(f'| AP (0.50)\t\t: {ap_active_obj_results[0.50]:.5f}')
print(f'| AP (0.25)\t\t: {ap_active_obj_results[0.25]:.5f}')
print('----Second-object AP----')
print(f'| AP (0.75)\t\t: {ap_second_obj_results[0.75]:.5f}')
print(f'| AP (0.50)\t\t: {ap_second_obj_results[0.50]:.5f}')
print(f'| AP (0.25)\t\t: {ap_second_obj_results[0.25]:.5f}')


