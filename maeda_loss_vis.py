# Created By Maeda

import json
import matplotlib.pyplot as plt

def main(log_file_path):
    # ログデータを読み込む
    with open(log_file_path + '/log.txt', 'r') as file:
        log_data = [json.loads(line) for line in file]

    # train_lossの値を抽出
    train_losses = [entry['train_loss'] for entry in log_data]

    # エポック数を生成
    epochs = list(range(len(train_losses)))

    # train_lossの経過を描画
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', linestyle='-', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.grid(True)

    # 画像として保存
    plt.savefig(log_file_path + '/log_vis.png')
    print(log_file_path + '/log_vis.png')

# Execute
# log_file_path = 'checkpoints/check/doh/SatoLab_HOTR/doh_single_hand_run_000004'
# log_file_path = 'checkpoints/check/hands23/SatoLab_HOTR/hands23_single_hand_run_000004'
# main(log_file_path)