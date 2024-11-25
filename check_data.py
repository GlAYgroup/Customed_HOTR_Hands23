#アノテーションの前処理関数(e.g. doh.py, hands23.py)の挙動が正しいか確認するコード
from hotr.data.datasets.doh  import main as build_doh
from hotr.data.datasets.hands23  import main as build_hands23
import tqdm

a = build_hands23(image_set='train', args=None)

# データセットの長さを取得
dataset_length = len(a)

# 全てのidxを確認して、可視化
for idx in range(dataset_length):
    print(f"Index: {idx}")
    print("img_info: ", a.img_infos[idx])
    # sample = a.__getitem__(idx)  # 各サンプルを取得
    # print(sample)  # サンプルの内容を表示
    
    # 可視化を行う
    a.vis_sample_from_pair_boxes(idx=idx)