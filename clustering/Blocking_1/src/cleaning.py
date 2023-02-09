import sys
from os.path import abspath, dirname

sys.path.append(dirname(dirname(abspath(__file__))))
import numpy as np
import pandas as pd
from Config import config
from pathlib import Path
from utils.logger_helper import log_helper
from utils.cleaning_utils import canonical_url, clean_email, process_phone, clean_text, unique_list, clean_name, clean_streets, rem_words
from tqdm.auto import tqdm
from utils.trace import Trace
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=4, verbose=1)


def clean(log):
    df = pd.read_csv(config.raw_dir + f"Fuse_{config.country}.csv", engine='c')
    Path(config.input_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"length of overall dataframe {len(df)}")

    df["latitude"] = df["latitude"].astype(np.float32)
    df["longitude"] = df["longitude"].astype(np.float32)
    df = df[~(df["longitude"].isnull())]
    df["officialName"] = df["officialName"].str.strip()
    df["officialName"] = df["officialName"].replace("", np.nan)
    df = df[~(df["officialName"].isnull())]
    df.drop_duplicates(inplace=True)

    df["internet"] = df["internet"].apply(canonical_url)
    df["email"] = df["email"].apply(canonical_url)
    df["internet"] = df["internet"].apply(clean_email)
    df["email"] = df["email"].apply(clean_email)

    df["phoneNumbers"] = df["phoneNumbers"].apply(process_phone)

    print("Cleaning the data")
    text_columns = [
        "officialName", "brands", "houseNumber", "category", "streets",
        "cities", "postalCode", "subCategory", "email", "internet"
    ]

    for col in tqdm(text_columns):
        df[col] = df[col].astype("str").astype(str).replace('nan', np.nan)
        df[col] = df[col].apply(clean_text)

    # remove duplicate words
    df["brands"] = df["brands"].apply(unique_list)
    df["subCategory"] = df["subCategory"].apply(unique_list)
    df["officialName"] = df["officialName"].apply(unique_list)
    df["streets"] = df["streets"].apply(unique_list)
    df["cities"] = df["cities"].apply(unique_list)

    # remove unspecified sub category

    df["subCategory"].replace('unspecified', np.nan, regex=True, inplace=True)

    df["cities"] = df["cities"].apply(clean_streets)
    df["streets"] = df["streets"].apply(clean_streets)
    df["officialName"] = df["officialName"].apply(clean_name)

    df["officialName"] = df["officialName"].apply(rem_words)
    df["cities"] = df["cities"].apply(rem_words)
    df["streets"] = df["streets"].apply(rem_words)

    df = df.replace(r'^\s*$', np.nan, regex=True)

    df["officialName"] = df["officialName"].replace("", np.nan)
    df = df[~(df["officialName"].isnull())]

    log.info(f"length of after cleaning {len(df)}")
    df = df.sample(frac=1).reset_index(drop=True)
    df["Id"] = df.index

    for col in df.columns:
        if col not in config.columns_keep:
            df.drop(col, axis=1, inplace=True)

    df["category"].replace('caf pub', 'cafe pub', regex=True,inplace=True)

    log.info(f"length of after cleaning {len(df)}")

    df.to_csv(config.input_dir + f"Fuse_{config.country}_cleaned.csv",
              index=False)
    print("completed cleaning")
    log.info("completed cleaning")


if __name__ == "__main__":
    trace = Trace()

    log = log_helper(f"cean_{config.country}", config.country)

    with trace.timer("clean", log):
        clean(log)
