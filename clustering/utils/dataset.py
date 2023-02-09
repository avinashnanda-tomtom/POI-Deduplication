import torch
from torch.utils.data import  DataLoader,Dataset


def prepare_input(CF, text):
    inputs = CF.tokenizer(text,
                           add_special_tokens=True,
                           max_length=CF.max_len,
                           padding=False,
                           truncation=True,
                           return_offsets_mapping=False)
    return inputs

class Collate:

    def __init__(self, tokenizer,test=False):
        self.tokenizer = tokenizer
        self.test = test

    def __call__(self, batch):
        inputs = dict()
        inputs['input_ids'] = [sample[0]['input_ids'] for sample in batch]
        inputs['attention_mask'] = [sample[0]['attention_mask'] for sample in batch]
        numeric_x = [sample[1] for sample in batch]
        numeric_x = torch.stack(numeric_x)
        if self.test == False:
            labels = [sample[2] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(token_id) for token_id in inputs["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            inputs["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in inputs["input_ids"]]
            inputs["attention_mask"] = [s + (batch_max - len(s)) * [0]for s in inputs["attention_mask"]]

        else:
            inputs["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in inputs["input_ids"]]
            inputs["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in inputs["attention_mask"]]

        # convert to tensors
        inputs["input_ids"] = torch.tensor(inputs["input_ids"],dtype=torch.long)
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"],dtype=torch.long)
        if self.test == False:
            labels = torch.tensor(labels, dtype=torch.float)
            return inputs, numeric_x, labels
        else:
            return inputs, numeric_x


class TrainDataset(Dataset):

    def __init__(self, CF, df_pair, df,test = False, swap=False):
        self.CF = CF
        self.texts = df[self.CF.text_cols].fillna(self.CF.NAN_TOKEN).astype(str).to_numpy()
        self.n_col = self.texts.shape[1]
        self.numeric_x = df_pair[CF.numeric_cols].to_numpy()
        self.links = df_pair[['ltable_id', 'rtable_id']].to_numpy()
        self.swap = swap
        self.test = test
        if self.test == False:
            self.labels = df_pair['duplicate_flag'].to_numpy()

    def __len__(self):
        return len(self.links)

    def __getitem__(self, item):
        ltable_id, rtable_id = self.links[item, :]
        if self.swap:
            ltable_id, rtable_id = rtable_id, ltable_id
        sentence = self.connect_text(ltable_id, rtable_id)
        inputs = prepare_input(self.CF, sentence)
        numeric_x = torch.tensor(self.numeric_x[item], dtype=torch.float)
        if self.test == False:
            labels = self.labels[item]
            return inputs, numeric_x, labels
        else:
            return inputs, numeric_x

    def connect_text(self, ltable_id, rtable_id):
        """
        example:
          ["name_i", "categories_i", "name_j", "categories_j"] -> "name_i[SEP]name_j[COL]categories_i[SEP]categories_j"
          ["name_j", "[NAN]", "name_j", "categories_j"] -> "name_i[SEP]name_j[COL][NAN][SEP]categories_j"
        """
        for col_dim in range(self.n_col):
            text_i, text_j = self.texts[[ltable_id, rtable_id], col_dim]

            if col_dim == 0:
                sentence = text_i + self.CF.SEP_TOKEN + text_j
            else:
                sentence = sentence + self.CF.COL_TOKEN + text_i + self.CF.SEP_TOKEN + text_j

        return sentence