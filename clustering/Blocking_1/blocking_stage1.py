from Blocking_1.src import name_block
from utils.trace import Trace
from clustering.Blocking_1.src import (
    cleaning,
    cosine_blocking,
    gen_embeddings,
    combine_blocks,
    housenumber_block,
    phonenumber_block,
    name_block
)
import time
import warnings
from Config import config
from utils.logger_helper import log_helper

warnings.filterwarnings("ignore")


def block1(log):
    log.info(
        "************************** Blocking stage 1 *************************************"
    )

    start_time = time.time()
    trace1 = Trace()


    with trace1.timer("cleaning", log):
        cleaning.clean(log)

    with trace1.timer("Generating_sbert_embeddings", log):
        gen_embeddings.gen_embeddings(log)

    with trace1.timer("generate_cosine_candidates", log):
        cosine_blocking.gen_cos_candidates(log)

    # with trace1.timer("generate_house_candidates", log):
    #     housenumber_block.house_block(log)

    # with trace1.timer("generate_phone_candidates", log):
    #     phonenumber_block.phone_block(log)

    # with trace1.timer("generate_name_candidates", log):
    #     name_block.name_block(log)

    with trace1.timer("combine_blocks", log):
        combine_blocks.combine(log)

    log.info(
        f"Total time taken by blocking1 = {round((time.time() - start_time)/60,2)} minutes"
    )
    print(
        f"Total time taken by blocking1 = {round((time.time() - start_time)/60,2)} minutes"
    )


if __name__ == "__main__":
    start_time = time.time()
    trace = Trace()
    log = log_helper(f"Blocking_stage_1_{config.country}", config.country)
    with trace.timer("Blocking_stage_1", log):
        block1(log)
