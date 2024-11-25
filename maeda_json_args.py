import torch

checkpoint_path = 'checkpoints/doh/SatoLab_HOTR/doh_single_hand_run_000001/checkpoint.pth'

# チェックポイントファイルからデータを読み込む
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# チェックポイントに含まれるデータを表示する
args = checkpoint['args']
print("argsの内容:")
print(args)
