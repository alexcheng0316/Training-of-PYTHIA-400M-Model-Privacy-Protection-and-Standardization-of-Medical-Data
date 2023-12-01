from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset, Features, Value
from islab.aicup import collate_batch_with_prompt_template, OpenDeidBatchSampler, aicup_predict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm, trange
from peft import LoraConfig, TaskType, get_peft_model
from collections import Counter
import torch
import random
import io
import datetime
import re
import pandas as pd

#紀錄時間
start_time = datetime.datetime.now()

model_load_path = "D:/NN/models_1b_epoch30_merged_dataset.pt"
plm = "EleutherAI/pythia-1b-deduped"
# plm = "EleutherAI/pythia-70m-deduped"

bos = "<|endoftext|>"
eos = "<|END|>"
pad = "<|pad|>"
sep = "\n\n####\n\n"
special_tokens_dict = {"eos_token":eos, "bos_token" : bos, "pad_token" : pad, "sep_token" : sep}

# peft_config = LoraConfig(task_type = TaskType.CAUSAL_LM, lora_alpha = 8, lora_dropout = 0.1)
peft_config = LoraConfig(task_type = TaskType.CAUSAL_LM, inference_mode = False, r = 8, lora_alpha = 32, lora_dropout = 0.1)
tokenizer = AutoTokenizer.from_pretrained(plm, revision = "step3000")
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
config = AutoConfig.from_pretrained(plm,
                                    bos_token_id = tokenizer.bos_token_id,
                                    eos_token_id = tokenizer.eos_token_id,
                                    pad_token_id = tokenizer.pad_token_id,
                                    sep_token_id = tokenizer.sep_token_id,
                                    output_hidden_states = False)

PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
IGNORED_PAD_IDX = -100
tokenizer.padding_side = "left"

#套用模型
model = AutoModelForCausalLM.from_pretrained(plm, revision = "step3000", config = config)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = AdamW(model.parameters(), lr = 5e-5)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.load_state_dict(torch.load(model_load_path))
# model_weight = torch.load('D:/NN/models_160m.pt')

#weight_tensor = model.gpt_neox.embed_in.weight



#model_weight['gpt_neox.embed_in.weight'] = model_weight['gpt_neox.embed_in.weight'][:50304, :]
#model_weight['embed_out.weight'] = model_weight['embed_out.weight'][:50304, :]


# model.load_state_dict(model_weight)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = AdamW(model.parameters(), lr = 5e-5)


model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.eval()
model.resize_token_embeddings(len(tokenizer))

valid_data =load_dataset("csv",data_files = "D:/NN/AICup/First_Phase_Release(Correction)/Tutorial/opendid_valid.tsv", delimiter ='\t',
                         features = Features({'fid':Value('string'),'idx':Value('int64'),'content':Value('string'),'label':Value('string')}),
                         column_names=['fid','idx','content','label'])

valid_list =list(valid_data['train'])

for sample in valid_list:
    if sample["content"] is None:
        sample["content"] = "None"
def convert_to_iso8601(text):
    # 定义正则表达式模式以匹配不同的时间表示方式
    duration_patterns = [
        (r"(\d+)\s?(?:years?|yrs?)\s?", "P{}Y"),  # 匹配 '10years ago', '13 yrs ago' 这样的表示方式
        (r"(\d+)\s?(?:years?)\s?", "P{}Y"),  # 匹配 '2 years' 这样的表示方式
        (r"(\d+)\s?(?:months?)\s?", "P{}M"),  # 匹配 '18 months ago', '3 months' 这样的表示方式
        (r"(\d+)\s?(?:weeks?|wks?)\s?", "P{}W")  # 匹配 '2 weeks ago', '4 wks ago' 这样的表示方式
        # (r"(\d+)-(\d+)\s?(?:month|months)\s?", "P{}.{}M")  # 匹配 '4-5 month' 这样的表示方式
    ]

    iso8601_times = []

    for pattern, iso_format in duration_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            iso_duration = iso_format.format(match)
    return iso_duration

def aicup_predict2(model, tokenizer, input, template = "<|endoftext|> __CONTENT__\n\n####\n\n"):
    seeds = [template.replace("__CONTENT__", data['content']) for data in input]
    sep = tokenizer.sep_token
    eos = tokenizer.eos_token
    pad = tokenizer.pad_token
    pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    """Generate text from a trained model."""
    model.eval()
    device = model.device
    texts = tokenizer(seeds, return_tensors = 'pt', padding=True).to(device)
    outputs = []
    #種類
    categories = [
        "PATIENT", "DOCTOR", "USERNAME", "PROFESSION", "ROOM", "DEPARTMENT", "HOSPITAL", 
        "ORGANIZATION", "STREET", "CITY", "STATE", "COUNTRY", "ZIP", "LOCATION-OTHER", 
        "AGE", "DATE", "TIME", "DURATION", "SET", "PHONE", "FAX", "EMAIL", "URL", 
        "IPADDR", "SSN", "MEDICALRECORD", "HEALTHPLAN", "ACCOUNT", "LICENSE", "VECHICLE", 
        "DEVICE", "BIOID", "IDNUM", "OTHER"
    ]
    age_patterns = [
        r"\b(\d+)\s?(?:yo|yr|yrs|years? old|years?-old)\b",  # 匹配带有 "yo", "yr", "yrs" 的年龄数字
        r"\b(?:at\s)?age\s(\d+)\b",  # 匹配 "age" 后面跟着数字的模式
        r"\bin\s(\d+)s\b",  # 匹配 "in 65s" 这种格式
        r"M\s/\s(\d+)\sSlides"  # 匹配 "M / 76 Slides" 这种格式
    ]
    duration_patterns = [
        r"(\d+\s?(?:years?|yrs?)\s?)(?:\sago)",  # 匹配 '10 years ago', '13 yrs ago', '20years ago', " 1 year ago", " 1 years ago"
        r"(\d+\s?months\s?)",  # 匹配 '18 months ago', '3 months'
        r"(\d+\s?weeks?|wks?\s?)",  # 匹配 '2 weeks ago', '4 wks ago', '36 week'
        r"(\d+-\d+\s?months?\s?)"
    ]
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**texts, max_new_tokens=400, pad_token_id = pad_idx,
                                        eos_token_id=tokenizer.convert_tokens_to_ids(eos))
        preds = tokenizer.batch_decode(output_tokens)
        # print(preds)
        for idx , pred in enumerate(preds):
            agetext= 0
            # aticle是將原句取出來
            article = pred[pred.index("<|endoftext|>") + len("<|endoftext|>"):pred.index(sep)]
            pred = pred[pred.index(sep)+len(sep):].replace(pad, "").replace(eos, "").strip()
            # print(article)
            # 匹配句子中的duration訊息
            matches_age = []
            matches_duration = []
            for pattern in duration_patterns:
                matches_duration.extend(re.findall(pattern, article))
            for durationtext in matches_duration:
                if durationtext != None:
                    durationtext = durationtext.strip()
                    iso8601 = convert_to_iso8601(article)
                    # print(article)
                    # print( durationtext,"###",iso8601)
                    lidxs = {}
                    lidx = 0
                    if agetext in lidxs:
                        lidx = lidxs[durationtext]
                    lidx = input[idx]['content'].find(durationtext, lidx)
                    eidx = lidx+len(durationtext)
                    lidxs[durationtext] = eidx
                    sidx=int(input[idx]["idx"])
                    # print(f'{input[idx]["fid"]}\tDURATION\t{lidx+sidx}\t{eidx+sidx}\t{durationtext}\t{iso8601}')
                    outputs.append(f'{input[idx]["fid"]}\tDURATION\t{lidx+sidx}\t{eidx+sidx}\t{durationtext}\t{iso8601}')
            # 匹配句子中的年齡訊息
            for pattern in age_patterns:
                matches_age.extend(re.findall(pattern, article))
            for agetext in matches_age:
                if int(agetext) > 0:
                    # print(article)
                    # print("###", agetext)
                    lidxs = {}
                    lidx = 0
                    if agetext in lidxs:
                        lidx = lidxs[agetext]
                    lidx = input[idx]['content'].find(agetext, lidx)
                    eidx = lidx+len(agetext)
                    lidxs[agetext] = eidx
                    sidx=int(input[idx]["idx"])
                    outputs.append(f'{input[idx]["fid"]}\tAGE\t{lidx+sidx}\t{lidx+sidx+2}\t{agetext}')
                    # print(f'{input[idx]["fid"]}\tAGE:\t{lidx+sidx}\t{lidx+sidx+2}\t{agetext}')
            #     agetext= 0
            if pred == "PHI: NULL":
                continue
            # print("pred:",pred)
            phis = pred.split("\\n")
            # print("phis:", phis)
            lidxs = {}
            for p in phis:
                tid = p.find(':')
                if tid > 0:
                    text = p[tid+1:].strip()
                    nv = text.find('=>')
                    normalizedV = None
                    if nv > 0:
                        normalizedV = text[nv+2:]
                        text = text[:nv]
                    lidx = 0
                    if text in lidxs:
                        lidx = lidxs[text]
                    lidx = input[idx]['content'].find(text, lidx)
                    eidx = lidx+len(text)
                    lidxs[text] = eidx
                    sidx=int(input[idx]["idx"])
                    if (len(text)==0) and (normalizedV is None):
                        print(f"{lidx+sidx}\t{eidx+sidx}\t{text}")
                    elif (len(text)==0) and (normalizedV is not None):
                        print(f"{lidx+sidx}\t{eidx+sidx}\t{text}\t{normalizedV}")
                    elif normalizedV is None and len(text)<50:
                        # 檢查 {p[:tid]} 是否在種類清單中
                        if p[:tid] in categories:
                            outputs.append(f'{input[idx]["fid"]}\t{p[:tid]}\t{lidx+sidx}\t{eidx+sidx}\t{text}')
                    elif normalizedV is not None and len(normalizedV)<100 and len(text)<50:
                        if p[:tid] in categories:
                            outputs.append(f'{input[idx]["fid"]}\t{p[:tid]}\t{lidx+sidx}\t{eidx+sidx}\t{text}\t{normalizedV}')
    return outputs

BATCH_SIZE = 128

# for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
#     with torch.no_grad():
#         seeds = valid_list[i:i+BATCH_SIZE]
#         outputs = aicup_predict2(model, tokenizer, input = seeds)


with io.open("D:/NN/answer_1B_epoch30_merged_dataset.txt", "w", encoding = "utf8") as f:
    for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
        with torch.no_grad():
            seeds = valid_list[i:i+BATCH_SIZE]
            outputs = aicup_predict2(model, tokenizer, input = seeds)
            for o in outputs:
                f.write(o)
                f.write("\n")

#結束時間
end_time = datetime.datetime.now()
#計算經過的時間
execution_time = end_time - start_time
#格式化時間
formatted_time = execution_time - datetime.timedelta(microseconds=execution_time.microseconds)
print(f"執行時間為: {formatted_time}")

 # 刪除重複的資料
# def remove_duplicates_from_file(file_path):
#     seen = set()
#     unique_data = []

#     with open(file_path, 'r', encoding="utf-8") as file:
#         for line in file:
#             parts = line.strip().split("\t")

#             if len(parts) < 4:
#                 continue

#             key = tuple(parts[:4])

#             if key not in seen:
#                 seen.add(key)
#                 unique_data.append(line)

#     # 這裡重新打開文件，並使用 'w' 模式寫入去重後的數據
#     with open(file_path, 'w', encoding="utf-8") as file:
#         file.writelines(unique_data)

# # 調用函數
# file_path = "D:/NN/answer.txt"
# remove_duplicates_from_file(file_path)
# print("ok")
