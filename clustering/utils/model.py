import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import time
import numpy as np
from utils.dataset import Collate,TrainDataset
from torch.utils.data import  DataLoader
from Config import config
import gc



class CustomModel1(nn.Module):


    def __init__(self, CF, config_path=None, pretrained=True):
        super().__init__()
        self.CF = CF
        if config_path is None:
            self.config = AutoConfig.from_pretrained(CF.model,output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(CF.model,config=self.config)
            self.model.resize_token_embeddings(len(self.CF.tokenizer))
        else:
            self.model = AutoModel.from_config(self.config)

        self.ln1 = nn.LayerNorm(self.config.hidden_size)
        self.linear1 = nn.Sequential(nn.Linear(self.config.hidden_size, 128),
                                     nn.LayerNorm(128), nn.ReLU(),
                                     nn.Dropout(0.2))

        self.linear2 = nn.Sequential(nn.Linear(len(self.CF.numeric_cols), 64), nn.LayerNorm(64),
                                     nn.ReLU(), nn.Dropout(0.2))

        self.linear3 = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )


    def forward(self, inputs,numeric_x):
        # pooler
        out = self.model(**inputs)['last_hidden_state'][:,0,:]
        out =  self.ln1(out)
        out = self.linear1(out)
        out2 = self.linear2(numeric_x)
        out = torch.cat([out,out2],axis=-1)
        out = self.linear3(out)
        return out

    def predict(self, inputs, numeric_x):
        out = self.model(**inputs)['last_hidden_state'][:,0,:]
        out =  self.ln1(out)
        out = self.linear1(out)
        out2 = self.linear2(numeric_x)
        out = torch.cat([out,out2],axis=-1)
        out = self.linear3(out)
        return out


def inference_fn(test_loader, model, device):
    preds = []
    model.to(device)
    model.eval()
    start = end = time.time()
    target = []

    if config.is_inference == False:
        with torch.no_grad():
            for step, (inputs, numeric_x, labels) in tqdm(enumerate(test_loader),total = len(test_loader)):
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                numeric_x = numeric_x.to(device)
                
                with torch.cuda.amp.autocast(enabled=True):
                    y_preds = model.predict(inputs, numeric_x)
                preds.append(y_preds.sigmoid().detach().cpu().numpy())
                target.append(labels.cpu().detach().numpy())

        predictions = np.concatenate(preds)
        target = np.concatenate(target)
        print(f"Total time taken = {round((time.time() - start)/60,2)} minutes")
        return predictions,target
    else:
        with torch.no_grad():
            for step, (inputs, numeric_x) in tqdm(enumerate(test_loader),total = len(test_loader)):
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                numeric_x = numeric_x.to(device)
                
                with torch.cuda.amp.autocast(enabled=True):
                    y_preds = model.predict(inputs, numeric_x)
                preds.append(y_preds.sigmoid().detach().cpu().numpy())

        predictions = np.concatenate(preds)
        print(f"Total time taken = {round((time.time() - start)/60,2)} minutes")
        return predictions


    

def prediction(CF,df_pair_test,df,model_ckpt,model_dir):


    df['text_len'] = df[CF.text_cols[0]].fillna('').str.cat(df[CF.text_cols[1:]].fillna('').astype(str)).str.len()
    df_pair_test['text_len'] = df['text_len'].values[df_pair_test['ltable_id']] + df['text_len'].values[df_pair_test['rtable_id']]
    df_pair_test = df_pair_test.sort_values('text_len', ascending=False).reset_index(drop=True)
    
    CF.tokenizer = AutoTokenizer.from_pretrained(CF.out_dir + 'tokenizer/')
    CF.SEP_TOKEN = CF.tokenizer.sep_token
    CF.COL_TOKEN = '[COL]'
    CF.NAN_TOKEN = '[NAN]'
    collate_fn = Collate(CF.tokenizer)

    test_dataset = TrainDataset(CF, df_pair_test, df,test=config.is_inference)
    test_loader = DataLoader(test_dataset,
                             batch_size=CF.batch_size,
                             shuffle=False,
                             num_workers=4, 
                             pin_memory=True, 
                             drop_last=False, 
                             collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CustomModel1(CF, config_path=CF.out_dir + 'config.pth', pretrained=False)
        
    state = torch.load(CF.out_dir+ model_ckpt,map_location=torch.device('cpu'))

    
    model.load_state_dict(state)

    if config.is_inference ==False:
        prediction,target = inference_fn(test_loader, model, device)
        df_pair_test['duplicate_flag'] =target.tolist()
    else:
        prediction = inference_fn(test_loader, model, device)



    df_pair_test[f'pred_{model_dir}'] =prediction.tolist()
    df_pair_test[f'pred_{model_dir}'] = df_pair_test[f'pred_{model_dir}'].apply(lambda x : x[0])
        

    del model, state, prediction; 
    gc.collect()
    torch.cuda.empty_cache()
    
    return df_pair_test