import json

# JSONファイルを読み込む
with open('maeda/doh_eval_results.json', 'r') as file:
    data = json.load(file)
print("--Loaded--")
# キーが0から10までのデータを抽出
extracted_data = {}
for key in range(11):
    if str(key) in data:
        print("ture")
        extracted_data[str(key)] = data[str(key)]

# 抽出したデータを使う
# 例えば、抽出したデータを表示
print(extracted_data)
# 必要であれば、抽出したデータを新しいJSONファイルに保存
with open('check_doh_eval_results.json', 'w') as file:
    json.dump(extracted_data, file, indent=4)
