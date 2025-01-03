import os
from PIL import Image
from tqdm import tqdm


# フォルダのパス
GT_folder = "checkpoints/hands23/SatoLab_HOTR/gpu_hands23_multi_hand_run_000010/result/best_second/full_second_only/GT_visualization"
# hands23_detector_folder = "hands23_detector_result/demo"
hands23_detector_folder = "checkpoints/hands23/SatoLab_HOTR/gpu_hands23_multi_hand_run_000004/result/best_second/full_second_only/unique_obj_pred_visualization"

pred_folder = "checkpoints/hands23/SatoLab_HOTR/gpu_hands23_multi_hand_run_000010/result/best_second/full_second_only/unique_obj_pred_visualization"
output_folder = "checkpoints/hands23/SatoLab_HOTR/gpu_hands23_multi_hand_run_000010/result/best_second/full_second_only/aligned"

# 出力フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)

# 出力フォルダ内の既存ファイルを削除
def clear_output_folder(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

clear_output_folder(output_folder)

def find_matching_file(base_name, folder):
    """
    指定されたフォルダ内で拡張子を無視して一致するファイルを探す。
    """
    for file in os.listdir(folder):
        if file.startswith(base_name) and (file.endswith(".png") or file.endswith(".jpg")):
            return os.path.join(folder, file)
    return None

# GTフォルダ内の画像をキーとして処理
for filename in tqdm(os.listdir(GT_folder), desc="Processing images"):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # ファイル名から接頭辞 [GT_] を削除
        base_name = filename.replace("GT_", "").rsplit('.', 1)[0]  # 拡張子を除外

        # 各フォルダのファイルパスを検索
        gt_image_path = os.path.join(GT_folder, filename)
        detector_image_path = find_matching_file(f"Pred_{base_name}", hands23_detector_folder)
        pred_image_path = find_matching_file(f"Pred_{base_name}", pred_folder)

        # 画像が存在する場合に結合
        if detector_image_path and pred_image_path:
            gt_image = Image.open(gt_image_path)
            detector_image = Image.open(detector_image_path)
            pred_image = Image.open(pred_image_path)

            # 高さを揃えて画像を横並びに結合
            total_width = gt_image.width + detector_image.width + pred_image.width
            max_height = max(gt_image.height, detector_image.height, pred_image.height)

            combined_image = Image.new("RGB", (total_width, max_height))

            # それぞれの画像を貼り付け
            combined_image.paste(gt_image, (0, 0))
            combined_image.paste(detector_image, (gt_image.width, 0))
            combined_image.paste(pred_image, (gt_image.width + detector_image.width, 0))

            # 保存
            output_path = os.path.join(output_folder, f"{base_name}.jpg")
            combined_image.save(output_path)

            # print(f"Processed and saved: {output_path}")
        else:
            # print(f"Missing images for {base_name}: {detector_image_path}, {pred_image_path}")
            pass
