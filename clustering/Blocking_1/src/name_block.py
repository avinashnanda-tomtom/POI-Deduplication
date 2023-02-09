import pandas as pd
from Config import config
from utils import blocking_utils
from utils.logger_helper import log_helper
from utils.trace import Trace
from tqdm.auto import tqdm
import numpy as np
import rapidfuzz
from itertools import compress

def name_block(log):
    df = pd.read_csv(config.input_dir + f"Fuse_{config.country}_cleaned.csv",engine='c',dtype={"postalCode": "str", "houseNumber": "str"})

    log.info("Finding nearest neigbhours using officialName")

    p3 = df[["Id", "placeId", "officialName", "latitude", "longitude"]].copy()

    # rounded coordinates
    # rounding: 1=10Km, 2=1Km
    p3["long2"] = p3["longitude"].astype("float32")
    p3["latitude"] = np.round(p3["latitude"], 1).astype("float32")
    p3["longitude"] = np.round(p3["longitude"], 2).astype("float32")

    p3 = p3.sort_values(by=["latitude", "longitude", "officialName"]).reset_index(
        drop=True
    )
    p3["officialName"] = p3["officialName"].map(str)
    names = p3["officialName"].to_numpy()
    lon2 = p3["longitude"].to_numpy()
    idx1 = []
    idx2 = []
    for i in tqdm(range(p3.shape[0])):
        li = lon2[i]
        selected_li = [j for j in range(i+1,min(i+500, p3.shape[0] - 1)) if lon2[j]== li]
        if len(selected_li)>=1:
            values = rapidfuzz.process.cdist(list([names[i]]), list(names[selected_li]), scorer=rapidfuzz.distance.Jaro.similarity, score_cutoff=0.7)
            sel_idx = list(compress(selected_li, values[0] > 0.7))
            for j in sel_idx:
                idx1.append(i)
                idx2.append(j)

    p1 = p3[["Id"]].loc[idx1].reset_index(drop=True)
    p2 = p3[["Id"]].loc[idx2].reset_index(drop=True)
    pairs_officialName = pd.DataFrame(zip(list(p1["Id"]), list(p2["Id"])))
    pairs_officialName.columns = ["Id1", "Id2"]
    pairs_officialName.drop_duplicates(inplace=True)
    pairs_officialName = pairs_officialName.reset_index(drop=True)
    # flip - only keep one of the flipped pairs, the other one is truly redundant
    idx = pairs_officialName["Id1"] > pairs_officialName["Id2"]
    pairs_officialName["t"] = pairs_officialName["Id1"]
    pairs_officialName["Id1"].loc[idx] = pairs_officialName["Id2"].loc[idx]
    pairs_officialName["Id2"].loc[idx] = pairs_officialName["t"].loc[idx]
    pairs_officialName = pairs_officialName[["Id1", "Id2"]]
    pairs_officialName = pairs_officialName.drop_duplicates(subset=["Id1", "Id2"])

    pairs_officialName = pairs_officialName[["Id1", "Id2"]]
    pairs_officialName.columns = ["ltable_id", "rtable_id"]

    candidate_set_df = blocking_utils.clean_placeid(pairs_officialName, df)

    candidate_set_df.reset_index(drop=True, inplace=True)

    log.info(f"Total number of candidates = {len(candidate_set_df)}")

    candidate_set_df.to_parquet(
        config.root_dir
        + f"Blocking_1/outputs/candidates_name_{config.country}.parquet",
        compression="zstd",
        index=None,
    )


if __name__ == "__main__":

    trace = Trace()
    log = log_helper(
        f"Sourcename_neighours_{config.country}",
        config.country,
    )

    with trace.timer("generate_officialName_candidates", log):
        name_block(log)
