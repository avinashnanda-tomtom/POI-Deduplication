import lightgbm as lgbm
import numpy as np
import torch
from tqdm.auto import tqdm
import gc

def ForestInfer_lgb(ml_model, X, model_file, batch_size=128):
    """_summary_

    Args:
        ml_model (model): ForestInference model
        X (array): array of dataframe to predict on.
        model_file (text): ligtgbm pretrained model filename.
        batch_size (int, optional): batch size of prediction. Defaults to 128.

    Returns:
        array: prediction probabilities.
    """
    if torch.cuda.is_available():
        step = X.shape[0] // batch_size
        if batch_size * step < X.shape[0]:
            step += 1
        ret = []
        start = 0
        for i in tqdm(range(step)):
            end = start + batch_size

            pred = ml_model.predict(X[start:end, :])
            ret.append(pred)
            start += batch_size
        pred = np.concatenate(ret)

    else:
        model = lgbm.Booster(model_file=model_file)
        pred = model.pred(X)
    return pred

def pred_multi(model_file,df):
    from cuml import ForestInference
    fi = ForestInference(output_type='numpy')
    ml_model = fi.load(filename=model_file, model_type='lightgbm')
    prediction = []
    pred = ForestInfer_lgb(ml_model,df.values,model_file,batch_size = 512)
    prediction = prediction + list(pred)

    del ml_model
    gc.collect()
    torch.cuda.empty_cache()
    return prediction