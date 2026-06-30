#include <stdio.h>
#include <time.h>

#include "strategic_tft.h"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

int main(int argc, char** argv) {
    int envs_count = argc > 1 ? atoi(argv[1]) : 4096;
    int total_steps = argc > 2 ? atoi(argv[2]) : 10000000;
    if (envs_count <= 0 || total_steps <= 0) {
        fprintf(stderr, "usage: strategic_tft [envs] [steps]\n");
        return 2;
    }

    StrategicTFT* envs = (StrategicTFT*)calloc((size_t)envs_count, sizeof(StrategicTFT));
    for (int i = 0; i < envs_count; i++) {
        StrategicTFT* env = &envs[i];
        env->num_agents = 1;
        env->base_seed = i;
        env->observations = (float*)calloc(STRATEGIC_OBS_SIZE, sizeof(float));
        env->actions = (float*)calloc(1, sizeof(float));
        env->rewards = (float*)calloc(1, sizeof(float));
        env->terminals = (float*)calloc(1, sizeof(float));
        env->action_mask = (unsigned char*)calloc(STRATEGIC_NUM_ACTIONS, sizeof(unsigned char));
        c_reset(env);
    }

    double started = now_seconds();
    long long steps = 0;
    while (steps < total_steps) {
        for (int i = 0; i < envs_count && steps < total_steps; i++) {
            StrategicTFT* env = &envs[i];
            env->actions[0] = (float)strategic_heuristic_action(env);
            c_step(env);
            steps++;
        }
    }
    double elapsed = now_seconds() - started;
    double sps = (double)steps / elapsed;

    float episodes = 0.0f;
    float placement = 0.0f;
    float score = 0.0f;
    for (int i = 0; i < envs_count; i++) {
        episodes += envs[i].log.n;
        placement += envs[i].log.placement;
        score += envs[i].log.score;
    }
    if (episodes > 0.0f) {
        placement /= episodes;
        score /= episodes;
    }
    printf("{\"envs\":%d,\"steps\":%lld,\"elapsed_sec\":%.6f,\"steps_per_sec\":%.2f,"
           "\"episodes\":%.0f,\"mean_placement\":%.6f,\"mean_score\":%.6f}\n",
           envs_count, steps, elapsed, sps, episodes, placement, score);

    for (int i = 0; i < envs_count; i++) {
        free(envs[i].observations);
        free(envs[i].actions);
        free(envs[i].rewards);
        free(envs[i].terminals);
        free(envs[i].action_mask);
    }
    free(envs);
    return 0;
}
