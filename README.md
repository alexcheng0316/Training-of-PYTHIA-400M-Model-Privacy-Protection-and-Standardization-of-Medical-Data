# Training-of-PYTHIA-400M-Model 進行隱私保護與醫學數據標準化


## 執行環境
- OS"Windows11
- 顯示卡
	- 獨顯:NVIDIA GeForce GTX 3090Ti

## 環境建立
- python==3.11.5
- tensorflow==2.14.0
- torch==2.1.0
- pandas==2.1.2
- datasets==2.14.6
- tqdm==4.66.1
- peft==0.6.0

## 文件下載
從Pythia 上載預訓練模型進行後續的訓練
Pythia 400M model:https://github.com/EleutherAI/pythia

## 資料後處理
將預測模型所產生的輸出進行篩選比較，並找出規律抓出關鍵字按照自訂格式填入且去除重複輸出的問題下列是關於AGE程式碼:
```python
    age_patterns = [
        r"\b(\d+)\s?(?:yo|yr|yrs|years? old|years?-old)\b",  # 有 "yo", "yr", "yrs" 的年龄数字
        r"\b(?:at\s)?age\s(\d+)\b",  # 匹配 "age" 後面跟著数字的模式
        r"\bin\s(\d+)s\b",  # 匹配 "in 65s" 這種格式
        r"M\s/\s(\d+)\sSlides"  # 匹配 "M / 76 Slides" 這種格式
    ]

```
