import sys
from os.path import abspath, dirname

sys.path.append(dirname(dirname(abspath(__file__))))
import numpy as np
import pandas as pd
from Config import config
from pathlib import Path
from utils.explode import explode_df
from utils.explode_multilingual import explode_df_multilingual
from utils.logger_helper import log_helper
from utils.text_clean import clean_name, clean_streets, clean_text
from utils.trace import Trace
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=4,verbose=1)

def explode_clean(log):
    df = pd.read_csv(config.raw_dir + f"Fuse_{config.country}_clustered_v2.csv",engine='c')
    Path(config.input_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"length of overall dataframe {len(df)}")
    df = explode_df(df)
        
    log.info(f"length of after exploding {len(df)}")
    print("Cleaning the exploded data")
    # cols = [
    #     "sourceNames","brands","houseNumber","phoneNumbers",
    #     "category","streets","cities","postalCode","subCategory"
    # ]
    # for c in cols:
    #     if c == "cities":
    #         df[c] = df[c].apply(clean_text, remove_number=True)
    #     else:
    #         df[c] = df[c].apply(clean_text)

    # df["cities"] = df["cities"].apply(clean_streets)
    # df["streets"] = df["streets"].apply(clean_streets)

    # df["sourceNames"] = df["sourceNames"].apply(clean_name)

    # df["sourceNames"] = df["sourceNames"].replace("", np.nan)

    # df = df[~(df["sourceNames"].isnull())]

    # df = imputing_values(df, dict_update_null=config.dict_update_null)

    # log.info(f"length of after cleaning {len(df)}")
    df = df.sample(frac=1).reset_index(drop=True)
    df["Id"] = df.index

    for col in df.columns:
        if col not in config.columns_keep:
            df.drop(col, axis=1, inplace=True)

    df.drop_duplicates(inplace=True)

    log.info(f"length of after cleaning {len(df)}")

    df.to_csv(
        config.input_dir + f"Fuse_exploded_{config.country}_cleaned.csv", index=False
    )
    print("completed exploding and cleaning")
    log.info("completed exploding and cleaning")


if __name__ == "__main__":
    trace = Trace()

    log = log_helper(f"explode_and_clean_{config.country}", config.country)

    with trace.timer("explode_and_clean", log):
        explode_clean(log)
