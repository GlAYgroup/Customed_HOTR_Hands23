# Created By Maeda
# GT画像と可視化画像を横並びにする
from PIL import Image
import os
import re
from tqdm import tqdm

def combine_images(image_path_1, image_path_2, output_folder):
    # 画像を読み込む
    image1 = Image.open(image_path_1)
    image2 = Image.open(image_path_2)

    # 画像のサイズを取得
    width1, height1 = image1.size
    width2, height2 = image2.size

    # 新しい画像のサイズを計算
    total_width = width1 + width2
    max_height = max(height1, height2)

    # 新しい画像を作成
    new_image = Image.new('RGB', (total_width, max_height))

    # 画像を貼り付け
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))

    # 入力ファイル名を取得
    base_name_1 = os.path.basename(image_path_1)
    
    # 出力ファイル名を決定
    output_filename = f'aligned_{base_name_1}.jpg'
    
    # 新しい画像のパスを作成
    output_path = os.path.join(output_folder, output_filename)

    # 新しい画像を保存
    new_image.save(output_path)

def get_common_part(file_name):
    # ファイル名の共通部分を正規表現で抽出
    match = re.match(r'(GT_|Pred_)(.*)', file_name)
    return match.group(2) if match else None

def main(gt_folder, vis_folder):
    # 出力フォルダを決定
    output_folder = os.path.commonpath([gt_folder, vis_folder])
    output_folder = os.path.join(output_folder, 'aligned_images')

    # 出力フォルダが存在しない場合は作成する
    os.makedirs(output_folder, exist_ok=True)

    # GTフォルダ内の全ての画像ファイルを取得
    gt_images = sorted([f for f in os.listdir(gt_folder) if os.path.isfile(os.path.join(gt_folder, f))])
    vis_images = sorted([f for f in os.listdir(vis_folder) if os.path.isfile(os.path.join(vis_folder, f))])

    # tqdmを使って処理の進行状況を表示
    for gt_image in tqdm(gt_images, desc="Processing images"):
        gt_common_part = get_common_part(gt_image)
        for vis_image in vis_images:
            vis_common_part = get_common_part(vis_image)
            if gt_common_part and vis_common_part and gt_common_part == vis_common_part:
                # print(f"Processing {gt_common_part}")
                # print(f"Processing {vis_common_part}")
                gt_image_path = os.path.join(gt_folder, gt_image)
                vis_image_path = os.path.join(vis_folder, vis_image)
                combine_images(gt_image_path, vis_image_path, output_folder)
                break

# フォルダのパスを指定
gt_folder = 'checkpoints/doh/SatoLab_HOTR/doh_single_hand_run_000003/GT_visualization'
vis_folder = 'checkpoints/doh/SatoLab_HOTR/doh_single_hand_run_000003/unique_obj_pred_visualization'

# 関数を実行
main(gt_folder, vis_folder)
