# Training-of-PYTHIA-410M-Model 進行隱私保護與醫學數據標準化


## 執行環境
- OS"Windows11
- 顯示卡
	- 獨顯:NVIDIA GeForce GTX 3090Ti

## 環境建立
- python==3.11.5
```python
pip install tensorflow==2.14.0
pip install torch==2.1.0
pip install pandas==2.1.2
pip install datasets==2.14.6
pip install tqdm==4.66.1
pip install peft==0.6.0

```


## 文件下載
訓練資料:https://codalab.lisn.upsaclay.fr/competitions/15425?secret_key=db7687a5-8fc7-4323-a94f-2cca2ac04d39

從Pythia 上載預訓練模型進行後續的訓練
Pythia 400M model:https://github.com/EleutherAI/pythia
| Params | n_layers | d_model | n_heads | d_head | Batch Size | Learning Rate | Hugging Face Checkpoints                                                |
| ------ | -------- | ------- | ------- | ------ | ---------- | ------------- | ---------------------------------------------------------- |
| 70M    | 6        | 512     | 8       | 64     | 2M         | 1.0e-3          | [Standard](https://huggingface.co/EleutherAI/pythia-70m), [Deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped)  |
| 160M   | 12       | 768     | 12      | 64     | 2M         | 6.0e-4          | [Standard](https://huggingface.co/EleutherAI/pythia-160m), [Deduped](https://huggingface.co/EleutherAI/pythia-160m-deduped)|
| 410M   | 24       | 1024    | 16      | 64     | 2M         | 3.0e-4          | [Standard](https://huggingface.co/EleutherAI/pythia-410m), [Deduped](https://huggingface.co/EleutherAI/pythia-410m-deduped)|

## 訓練方式
- 將資料轉換成下列型態，由左至右分別為病例檔名、句子起始位置、句子、對應答案
```python
10 52 Lab No: 09F01654 IDNUM:09F01654    
```
- 使用collate_batch_with_prompt_template將每段訓練資料處理成這個形式再丟進模型做訓練
```python
"<|endoftext|> 文章句子\n\n####\n\n句子對應的答案<|END|>" 
```
-Model載入模型方法
```python
model = AutoModelForCausalLM.from_pretrained(plm, revision = "step3000", config = config)
```
-  儲存權重檔
```python
torch.save(model.state_dict(), "D:/NN/models_160M.pt") 
```
## 資料後處理
將預測模型所產生的輸出進行篩選比較，並找出規律抓出關鍵字按照自訂格式填入且去除重複輸出的問題下列是關於AGE程式碼:
```python
age_patterns = [
        r"\b(\d+)\s?(?:yo|yr|yrs|years? old|years?-old)\b",  # 有 "yo", "yr", "yrs" 的年龄数字
        r"\b(?:at\s)?age\s(\d+)\b",  # 匹配 "age" 後面跟著数字的模式
        r"\bin\s(\d+)s\b",  # 匹配 "in 65s" 這種格式
        r"M\s/\s(\d+)\sSlides"  # 匹配 "M / 76 Slides" 這種格式
    ]
duration_patterns = [
        r"(\d+\s?(?:years?|yrs?)\s?)(?:\sago)",  # 匹配 '10 years ago', '13 yrs ago', '20years ago', " 1 year ago", " 1 years ago"
        r"(\d+\s?months\s?)",  # 匹配 '18 months ago', '3 months'
        r"(\d+\s?weeks?|wks?\s?)",  # 匹配 '2 weeks ago', '4 wks ago', '36 week'
        r"(\d+-\d+\s?months?\s?)"
    ]
    

```
## 結果討論
- 針對不同模型所計算出的結果差異
![image](https://github.com/alexcheng0316/Training-of-PYTHIA-400M-Model-Privacy-Protection-and-Standardization-of-Medical-Data/assets/152590195/15c202fe-c7d8-4d73-a1b6-dcc31c4e8474)

