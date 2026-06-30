#include "strategic_tft.h"

#define OBS_SIZE STRATEGIC_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {STRATEGIC_NUM_ACTIONS}
#define OBS_TENSOR_T FloatTensor
#define MY_ACTION_MASK STRATEGIC_NUM_ACTIONS

#define Env StrategicTFT
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->base_seed = (int)env->rng;
    DictItem* seed_item = dict_get_unsafe(kwargs, "seed");
    if (seed_item != NULL) {
        env->base_seed += (int)seed_item->value;
    }
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "placement", log->placement);
    dict_set(out, "final_round", log->final_round);
    dict_set(out, "hp", log->hp);
}
