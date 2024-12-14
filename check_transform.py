# check_transform.py

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import torch
from hotr.data.datasets.hands23 import build as build_hands23  # 修正済み
import torchvision
import os
import random
import shutil  # 既存ファイルの削除に使用
from torchvision import transforms as T
import torchvision.transforms.functional as F  # 必要に応じて追加

# 出力画像を保存するディレクトリ
SAVE_DIR = 'check'  # 絶対パスにする場合は '/check' に変更

def clean_save_dir(directory):
    """
    指定されたディレクトリ内のすべてのファイルを削除します。
    
    Parameters:
    - directory (str): ディレクトリのパス
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # ファイルを削除
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # サブディレクトリを削除
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(directory)

# ディレクトリのクリーンアップ
clean_save_dir(SAVE_DIR)
print(f"Cleaned up the directory: {SAVE_DIR}")

def visualize_sample(image, target, title=""):
    """
    画像とアノテーション（hand_bboxes、triplet_targets、hand_2d_key_points）を可視化し、保存する関数。

    Parameters:
    - image (PIL.Image.Image): 入力画像
    - target (dict): アノテーション情報
    - title (str): プロットのタイトル
    """
    # 画像をNumPy配列に変換 (RGB)
    image_np = np.array(image)

    # RGBからBGRに変換（OpenCVの仕様に合わせる）
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 画像のサイズを取得
    img_height, img_width = image_bgr.shape[:2]

    # トリプレットターゲットの描画
    if "triplet_targets" in target and "pair_boxes" in target:
        triplet_targets = target["triplet_targets"].numpy()  # [num_triplets]
        pair_boxes = target["pair_boxes"].numpy()  # [num_triplets, 12]  # 手、オブジェクト1、オブジェクト2の各4点

        for i, triplet in enumerate(pair_boxes):
            triplet_target = triplet_targets[i]
            # tripletは [cx, cy, w, h, cx, cy, w, h, cx, cy, w, h]
            hand_box = triplet[0:4]
            obj1_box = triplet[4:8]
            obj2_box = triplet[8:12]

            # 手のバウンディングボックスを青色で描画
            # if not np.all(hand_box == -1):
            #     # アンノーマライズ
            #     cx, cy, w_box, h_box = hand_box
            #     x_min = int((cx - w_box / 2) * img_width)
            #     y_min = int((cy - h_box / 2) * img_height)
            #     x_max = int((cx + w_box / 2) * img_width)
            #     y_max = int((cy + h_box / 2) * img_height)
            #     cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # 青色の矩形
            #     # ラベルを追加
            #     cv2.putText(image_bgr, f'Triplet {i} Hand: {triplet_target}', 
            #                 (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            #                 0.5, (255, 0, 0), 2)

            # オブジェクト1のバウンディングボックスを黄色で描画
            if not np.all(obj1_box == -1):
                # アンノーマライズ
                cx, cy, w_box, h_box = obj1_box
                x_min = int((cx - w_box / 2) * img_width)
                y_min = int((cy - h_box / 2) * img_height)
                x_max = int((cx + w_box / 2) * img_width)
                y_max = int((cy + h_box / 2) * img_height)
                cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)  # 黄色の矩形
                # ラベルを追加
                cv2.putText(image_bgr, f'Triplet {i} Target: {triplet_target}', 
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 0, 0), 2)

            # オブジェクト2のバウンディングボックスを緑色で描画
            if not np.all(obj2_box == -1):
                # アンノーマライズ
                cx, cy, w_box, h_box = obj2_box
                x_min = int((cx - w_box / 2) * img_width)
                y_min = int((cy - h_box / 2) * img_height)
                x_max = int((cx + w_box / 2) * img_width)
                y_max = int((cy + h_box / 2) * img_height)
                cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # 緑色の矩形
                # ラベルを追加
                cv2.putText(image_bgr, f'Triplet {i} Secondary Target: {triplet_target}', 
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 0, 255), 2)

    # 手のバウンディングボックスの描画
    if "hand_bboxes" in target:
        hand_boxes = target["hand_bboxes"].numpy()
        for i, hand_box in enumerate(hand_boxes):
            # hand_box は [cx, cy, w, h]
            if not np.all(hand_box == -1):
                # アンノーマライズ
                cx, cy, w_box, h_box = hand_box
                x_min = int((cx - w_box / 2) * img_width)
                y_min = int((cy - h_box / 2) * img_height)
                x_max = int((cx + w_box / 2) * img_width)
                y_max = int((cy + h_box / 2) * img_height)
                cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # 青色の矩形
                # ラベルを追加（必要に応じて）
                cv2.putText(image_bgr, f'Hand BBox {i}', 
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 0, 0), 2)

    # キーポイントの描画
    if "hand_2d_key_points" in target:
        keypoints = target["hand_2d_key_points"].numpy()

        # キーポイントの形状を確認
        print(f"{title} - keypoints shape after squeeze: {keypoints.shape}")

        # スケルトン定義（標準的な21キーポイントの接続）
        skeleton = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),       # Index Finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle Finger
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky Finger
        ]

        # 手の数に応じてキーポイントとスケルトンを描画
        if len(keypoints.shape) == 3:
            num_hands = keypoints.shape[0]
            for hand_idx in range(num_hands):
                # print(f"{title} - Hand {hand_idx} keypoints shape: {keypoints[hand_idx].shape}")
                # スケルトンの描画
                for connection in skeleton:
                    start_idx, end_idx = connection
                    start_kp = keypoints[hand_idx, start_idx]
                    end_kp = keypoints[hand_idx, end_idx]
                    
                    # キーポイントが無効な場合スキップ
                    if (start_kp < 0).any() or (end_kp < 0).any():
                        continue

                    # キーポイントが [0, 1] の範囲内にあるか確認
                    if not (0 <= start_kp[0] <= 1 and 0 <= start_kp[1] <= 1):
                        # print(f"{title} - Hand {hand_idx} Start Keypoint {start_idx} out of bounds: {start_kp}")
                        continue
                    if not (0 <= end_kp[0] <= 1 and 0 <= end_kp[1] <= 1):
                        # print(f"{title} - Hand {hand_idx} End Keypoint {end_idx} out of bounds: {end_kp}")
                        continue

                    # スケルトンラインの座標をピクセル単位に変換
                    start_x = int(start_kp[0] * img_width)
                    start_y = int(start_kp[1] * img_height)
                    end_x = int(end_kp[0] * img_width)
                    end_y = int(end_kp[1] * img_height)

                    # 手ごとに色を変更（例: 手1は赤、手2は青）
                    skeleton_color = (0, 0, 255) if hand_idx == 0 else (255, 0, 0)

                    # スケルトンラインを描画
                    cv2.line(image_bgr, (start_x, start_y), (end_x, end_y), skeleton_color, 2)

                # キーポイントの描画
                for kp_idx, kp in enumerate(keypoints[hand_idx]):
                    # 必要な値だけをアンパック（例: x, y）
                    if len(kp) >= 2:
                        x, y = kp[:2]
                    else:
                        x, y = kp  # もし (x, y) のみの場合

                    # 無効なキーポイント（例: x < 0 または y < 0）は描画しない
                    try:
                        x = float(x)
                        y = float(y)
                        # キーポイントが [0, 1] の範囲内にあるか確認
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            x_abs = int(x * img_width)
                            y_abs = int(y * img_height)
                            # 手ごとに色を変更（例: 手1は赤、手2は青）
                            keypoint_color = (0, 0, 255) if hand_idx == 0 else (255, 0, 0)
                            cv2.circle(image_bgr, (x_abs, y_abs), 3, keypoint_color, -1)  # 手ごとの色の円
                        else:
                            print(f"{title} - Hand {hand_idx} Keypoint {kp_idx} out of bounds: x={x}, y={y}")
                    except TypeError as e:
                        print(f"Error processing Hand {hand_idx} Keypoint {kp_idx} in {title}: {e}")
        elif len(keypoints.shape) == 2:
            # 単一の手
            for kp_idx, kp in enumerate(keypoints):
                # 必要な値だけをアンパック（例: x, y）
                if len(kp) >= 2:
                    x, y = kp[:2]
                else:
                    x, y = kp  # もし (x, y) のみの場合

                # 無効なキーポイント（例: x < 0 または y < 0）は描画しない
                try:
                    x = float(x)
                    y = float(y)
                    # キーポイントが [0, 1] の範囲内にあるか確認
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        x_abs = int(x * img_width)
                        y_abs = int(y * img_height)
                        cv2.circle(image_bgr, (x_abs, y_abs), 3, (0, 0, 255), -1)  # 赤色の円
                    else:
                        print(f"{title} - Keypoint {kp_idx} out of bounds: x={x}, y={y}")
                except TypeError as e:
                    print(f"Error processing Keypoint {kp_idx} in {title}: {e}")
        else:
            print(f"{title} - Unexpected keypoints shape: {keypoints.shape}")

    # 画像の保存
    save_path = os.path.join(SAVE_DIR, f"{title}.png")
    # BGR から RGB に変換（MatplotlibはRGBを使用）
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # 図を閉じてメモリを解放

    print(f"Saved visualization to {save_path}")

def inspect_keypoints(target, idx):
    """
    hand_2d_key_points の形状と内容を出力するデバッグ関数。

    Parameters:
    - target (dict): アノテーション情報
    - idx (int): サンプルのインデックス
    """
    if "hand_2d_key_points" in target:
        keypoints = target["hand_2d_key_points"]
        print(f"Sample {idx} - hand_2d_key_points:")
        print(f"Shape: {keypoints.shape}")
        # print(keypoints)
    else:
        print(f"Sample {idx} does not contain 'hand_2d_key_points'.")

def main_visualization():
    # データセットのビルド
    # args を適切に設定する必要があります。ここではダミーの args を作成します。
    class Args:
        data_path = 'hands23'  # データパスを適切に設定
        task = 'ASOD'  # タスクを設定（例: 'AOD' または 'ASOD'）
        hand_pose = 'add_in_d_0'  # 手のポーズフラグを設定
        dataset_file = 'hands23_dataset'  # 必要に応じて適切な値を設定
        check = True  # チェックモードのフラグ（必要に応じて設定）

    args = Args()

    # データセットの構築
    dataset = build_hands23(image_set='train', args=args)

    # データセットの長さを取得
    dataset_length = len(dataset)
    print(f"Dataset Length: {dataset_length}")

    # 可視化するサンプルの数
    num_samples_to_visualize = 20

    # ランダムなインデックスを選択
    if dataset_length < num_samples_to_visualize:
        random_indices = random.sample(range(dataset_length), dataset_length)
    else:
        random_indices = random.sample(range(dataset_length), num_samples_to_visualize)
    print(f"Randomly selected indices for visualization: {random_indices}")

    for idx in random_indices:
        print(f"\nVisualizing Index: {idx}")
        # サンプルの取得
        sample = dataset[idx]
        img_tensor, target = sample

        # ターゲットの内容を確認（デバッグ用）
        print(f"Target keys: {target.keys()}")
        inspect_keypoints(target, idx)  # デバッグ用にキーポイント情報を出力

        # 画像テンソルを PIL 画像に変換
        img_pil = torchvision.transforms.ToPILImage()(img_tensor)

        # タイトルを設定
        title = f"Sample_{idx}"

        # 可視化と保存
        visualize_sample(img_pil, target, title=title)

if __name__ == "__main__":
    main_visualization()
