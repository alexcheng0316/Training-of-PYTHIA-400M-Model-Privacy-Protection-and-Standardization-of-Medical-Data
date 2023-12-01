from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset, Features, Value
from islab.aicup import collate_batch_with_prompt_template, OpenDeidBatchSampler, aicup_predict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm, trange
from tqdm.notebook import tqdm
import torch
import random
import io

# plm = "EleutherAI/pythia-70m-deduped"
plm = "EleutherAI/pythia-160m-deduped"
bos = "<|endoftext|>"
eos = "<|END|>"
pad = "<|pad|>"
sep = "\n\n####\n\n"
special_tokens_dict = {"eos_token":eos, "bos_token" : bos, "pad_token" : pad, "sep_token" : sep}

tokenizer = AutoTokenizer.from_pretrained(plm, revision = "step3000")
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
config = AutoConfig.from_pretrained(plm,
                                    bos_token_id = tokenizer.bos_token_id,
                                    eos_token_id = tokenizer.eos_token_id,
                                    pad_token_id = tokenizer.pad_token_id,
                                    sep_token_id = tokenizer.sep_token_id,
                                    output_hidden_states = False)

model = AutoModelForCausalLM.from_pretrained(plm, revision = "step3000", config = config)

dataset = load_dataset("csv", data_files = "D:/NN/AICup/First_Phase_Release(Correction)/Tutorial/opendid_set1.tsv", delimiter = "\t", 
                       column_names = ["fid", "idx", "content", "label"], 
                       features = Features({"fid":Value("string"), "idx":Value("int64"), "content":Value("string"), "label":Value("string")}))
for i, sample in enumerate(dataset["train"]):
    if sample["content"] is None:
        dataset["train"][i]["content"] = "None"
    else:
        dataset["train"][i]["content"] = str(sample["content"])

train_data = list(dataset["train"])


for sample in train_data:
    if sample["content"] is None:
        sample["content"] = "None"

# for sample in train_data:
#     content = sample["content"]
#     if isinstance(content, str):
#         if content == "None":
#             print(f"'content' is a string and its value is 'None': {content}")
#     else:
#         print(f"'content' is not a string: {content}")        
PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
IGNORED_PAD_IDX = -100

tokenizer.padding_side = "left"


train_dataloader = DataLoader(train_data, batch_size = 3, shuffle = False, collate_fn = lambda batch: collate_batch_with_prompt_template(batch, tokenizer, template = "<|endoftext|> __CONTENT__\n\n####\n\n__LABEL__ <|END|>", IGNORED_PAD_IDX = -100))
titer = iter(train_dataloader)
tks, labels, masks = next(titer)

BATCH_SIZE = 16

class BatchSampler():
    def __init__(self, data, batch_size):
        self.pooled_indices = []
        self.data = data
        self.batch_size = batch_size
        self.len = len(list(data))
    def __iter__(self):
        self.pooled_indices = []
        indices = [(index, len(data["content"])) for index, data in enumerate(self.data)]
        random.shuffle(indices)
        
        for i in range(0, len(indices), BATCH_SIZE * 100):
            self.pooled_indices.extend(sorted(indices[i:i + BATCH_SIZE *100], key = lambda x:x[1], reverse = True))
        self.pooled_indices = [x[0] for x in self.pooled_indices]
        
        for i in range(0, len(self.pooled_indices), BATCH_SIZE):
            yield self.pooled_indices[i:i + BATCH_SIZE]
    def __len__(self):
        return (self.len + self.batch_size - 1) // self.batch_size

bucket_train_dataloader = DataLoader(train_data, batch_sampler = BatchSampler(train_data, BATCH_SIZE), 
                                     collate_fn = lambda batch : collate_batch_with_prompt_template(batch, tokenizer, template = "<|endoftext|> __CONTENT__\n\n####\n\n__LABEL__ <|END|>", IGNORED_PAD_IDX = -100),
                                     pin_memory = True)
for idx, batch in enumerate(bucket_train_dataloader):
    print(batch)
    print(batch[0].shape)
    print(batch[1].shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr = 5e-5)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

model.train()

for _ in trange(EPOCHS, desc = "Epoch"):
  model.train()
  total_loss = 0
  predictions, true_labels = [], []
  for step ,(seqs, labels, masks) in enumerate(bucket_train_dataloader):
    seqs = seqs.to(device)
    labels = labels.to(device)
    masks = masks.to(device)
    model.zero_grad()
    outputs = model(seqs, labels = labels)
    logits = outputs.logits
    loss = outputs.loss
    loss = loss.mean()

    total_loss += loss.item()
    loss.backward()
    optimizer.step()
  avg_train_loss = total_loss / len(bucket_train_dataloader)
  print("Average train loss: {}".format(avg_train_loss))
  
torch.save(model.state_dict(), "D:/NN/models_160M.pt")

valid_data =load_dataset("csv",data_files = "D:/NN/AICup/First_Phase_Release(Correction)/Tutorial/opendid_valid.tsv", delimiter ='\t',
                         features = Features({'fid':Value('string'),'idx':Value('int64'),'content':Value('string'),'label':Value('string')}),
                         column_names=['fid','idx','content','label'])

valid_list =list(valid_data['train'])

for sample in valid_list:
    if sample["content"] is None:
        sample["content"] = "None"
        
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
    #return
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**texts, max_new_tokens=400, pad_token_id = pad_idx,
                                        eos_token_id=tokenizer.convert_tokens_to_ids(eos))
        preds = tokenizer.batch_decode(output_tokens)
        for idx , pred in enumerate(preds):
            
            pred = pred[pred.index(sep)+len(sep):].replace(pad, "").replace(eos, "").strip()
            # print(pred)
            if pred == "PHI: NULL":
                continue
            phis = pred.split('\n')
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
                    if len(text) > 150:
                        break
                    if normalizedV is None:
                        outputs.append(f'{input[idx]["fid"]}\t{p[:tid]}\t{lidx+sidx}\t{eidx+sidx}\t{text}')
                    else:
                        outputs.append(f'{input[idx]["fid"]}\t{p[:tid]}\t{lidx+sidx}\t{eidx+sidx}\t{text}\t{normalizedV}')
    return outputs

BATCH_SIZE = 64
with io.open("D:/NN/answer.txt", "w", encoding = "utf8") as f:
  for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
    with torch.no_grad():
      seeds = valid_list[i:i+BATCH_SIZE]
      outputs = aicup_predict2(model, tokenizer, input = seeds)
      for o in outputs:
        f.write(o)
        f.write("\n")
