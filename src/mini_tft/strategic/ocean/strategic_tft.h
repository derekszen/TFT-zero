#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define STRATEGIC_OBS_SIZE 38
#define STRATEGIC_NUM_ACTIONS 11
#define STRATEGIC_SHOP_SIZE 5
#define STRATEGIC_UNIT_COUNT 9
#define STRATEGIC_OWNED_SIZE 10
#define STRATEGIC_MAX_LEVEL 9
#define STRATEGIC_ROLE_COUNT 3
#define STRATEGIC_TRAIT_COUNT 6

enum {
    STRATEGIC_HOLD = 0,
    STRATEGIC_LEVEL = 1,
    STRATEGIC_ROLL = 2,
    STRATEGIC_BUY_BEST_UPGRADE = 3,
    STRATEGIC_BUY_BEST_SYNERGY = 4,
    STRATEGIC_BUY_HIGHEST_COST = 5,
    STRATEGIC_FIELD_STRONGEST = 6,
    STRATEGIC_GREED_ECON = 7,
    STRATEGIC_SLAM_CARRY_ITEM = 8,
    STRATEGIC_SLAM_TANK_ITEM = 9,
    STRATEGIC_SLAM_SUPPORT_ITEM = 10,
};

enum {
    STRATEGIC_CARRY = 0,
    STRATEGIC_TANK = 1,
    STRATEGIC_SUPPORT = 2,
};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float placement;
    float final_round;
    float hp;
    float n;
} Log;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int num_agents;
    unsigned int rng;
    int base_seed;
    int seed;
    uint64_t rng_key;
    int round;
    int hp;
    int gold;
    int level;
    int xp;
    int shop[STRATEGIC_SHOP_SIZE];
    int owned[STRATEGIC_OWNED_SIZE];
    int fielded[STRATEGIC_MAX_LEVEL];
    int role_items[STRATEGIC_ROLE_COUNT];
    int role_item_slots[STRATEGIC_ROLE_COUNT];
    int done;
    int final_reason;
    int action_count;
    float last_board_strength;
    float last_enemy_strength;
    int last_damage;
    int last_win;
    int total_rolls;
    int total_xp_buys;
    int total_units_bought;
    int total_item_slams;
    int total_illegal_actions;
    int episode_steps;
    float episode_return;
} StrategicTFT;

static const int UNIT_COST[STRATEGIC_OWNED_SIZE] = {0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
static const float UNIT_POWER[STRATEGIC_OWNED_SIZE] = {
    0.0f, 10.0f, 11.0f, 8.5f, 16.0f, 17.5f, 14.0f, 24.0f, 27.0f, 21.0f};
static const int UNIT_ROLE[STRATEGIC_OWNED_SIZE] = {
    -1, STRATEGIC_TANK, STRATEGIC_CARRY, STRATEGIC_SUPPORT, STRATEGIC_TANK,
    STRATEGIC_CARRY, STRATEGIC_SUPPORT, STRATEGIC_TANK, STRATEGIC_CARRY,
    STRATEGIC_SUPPORT};
static const int UNIT_TRAIT[STRATEGIC_OWNED_SIZE] = {-1, 0, 1, 2, 0, 1, 3, 4, 5, 3};
static const float STAR_MULTIPLIER[4] = {0.0f, 1.0f, 1.9f, 3.5f};
static const float ROLE_ITEM_POWER[STRATEGIC_ROLE_COUNT] = {9.0f, 8.0f, 6.5f};
static const int UNITS_BY_COST[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

static inline uint64_t strategic_next_u64(StrategicTFT* env) {
    env->rng_key = env->rng_key * 6364136223846793005ULL + 1442695040888963407ULL;
    return env->rng_key;
}

static inline float strategic_random_float(StrategicTFT* env) {
    return (float)((strategic_next_u64(env) >> 11U) * (1.0 / 9007199254740992.0));
}

static inline int strategic_random_int(StrategicTFT* env, int limit) {
    return (int)(strategic_next_u64(env) % (uint64_t)limit);
}

static inline int strategic_star_level(int copies) {
    if (copies >= 9) return 3;
    if (copies >= 3) return 2;
    if (copies >= 1) return 1;
    return 0;
}

static inline int strategic_xp_to_next_level(int level) {
    return 2 + (level > 1 ? level - 1 : 0) * 2;
}

static inline float strategic_enemy_strength_for_round(int round_num) {
    float r = (float)(round_num > 1 ? round_num : 1);
    return (13.0f + r * 3.2f + powf(r, 1.18f) * 1.35f) * 5.0f;
}

static inline int strategic_placement_proxy(const StrategicTFT* env, int final_max_round) {
    if (final_max_round || (env->round >= 36 && env->hp > 0)) return 1;
    if (env->round >= 36) return 2;
    if (env->round >= 32) return 3;
    if (env->round >= 29) return 4;
    if (env->round >= 25) return 5;
    if (env->round >= 18) return 6;
    if (env->round >= 11) return 7;
    return 8;
}

static inline float strategic_board_strength(const StrategicTFT* env) {
    float strength = 0.0f;
    int trait_counts[STRATEGIC_TRAIT_COUNT] = {0};
    int role_counts[STRATEGIC_ROLE_COUNT] = {0};

    for (int slot = 0; slot < STRATEGIC_MAX_LEVEL; slot++) {
        int unit_id = env->fielded[slot];
        if (unit_id == 0) continue;
        int stars = strategic_star_level(env->owned[unit_id]);
        strength += UNIT_POWER[unit_id] * STAR_MULTIPLIER[stars];
        trait_counts[UNIT_TRAIT[unit_id]]++;
        role_counts[UNIT_ROLE[unit_id]]++;
    }

    for (int i = 0; i < STRATEGIC_TRAIT_COUNT; i++) {
        if (trait_counts[i] >= 3) strength += 8.0f;
        else if (trait_counts[i] >= 2) strength += 3.0f;
    }
    for (int i = 0; i < STRATEGIC_ROLE_COUNT; i++) {
        if (role_counts[i] > 0) strength += env->role_item_slots[i] * ROLE_ITEM_POWER[i];
    }
    if (role_counts[STRATEGIC_TANK] > 0 && role_counts[STRATEGIC_CARRY] > 0) {
        strength += 6.0f;
    }
    if (role_counts[0] + role_counts[1] + role_counts[2] == 0) {
        strength -= 12.0f;
    }
    return strength > 0.0f ? strength : 0.0f;
}

static inline float strategic_scenario_score(const StrategicTFT* env) {
    float round_frac = fminf(1.0f, fmaxf(0.0f, (float)env->round / 36.0f));
    float hp_frac = fminf(1.0f, fmaxf(0.0f, (float)env->hp / 100.0f));
    float enemy = fmaxf(1.0f, strategic_enemy_strength_for_round(env->round));
    float strength_ratio = fminf(1.4f, env->last_board_strength / enemy) / 1.4f;
    return fminf(1.0f, fmaxf(0.0f, 0.45f * round_frac + 0.25f * hp_frac + 0.30f * strength_ratio));
}

static inline void strategic_fielded_role_presence(const StrategicTFT* env, int present[3]) {
    present[0] = present[1] = present[2] = 0;
    for (int slot = 0; slot < STRATEGIC_MAX_LEVEL; slot++) {
        int unit_id = env->fielded[slot];
        if (unit_id > 0) present[UNIT_ROLE[unit_id]] = 1;
    }
}

static inline void strategic_strongest_field(const StrategicTFT* env, int out[STRATEGIC_MAX_LEVEL]) {
    int candidates[STRATEGIC_UNIT_COUNT];
    int count = 0;
    for (int unit_id = 1; unit_id <= STRATEGIC_UNIT_COUNT; unit_id++) {
        if (env->owned[unit_id] > 0) candidates[count++] = unit_id;
    }
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            int left = candidates[i];
            int right = candidates[j];
            float lp = UNIT_POWER[left] * STAR_MULTIPLIER[strategic_star_level(env->owned[left])];
            float rp = UNIT_POWER[right] * STAR_MULTIPLIER[strategic_star_level(env->owned[right])];
            int swap = 0;
            if (rp > lp) swap = 1;
            else if (rp == lp && UNIT_COST[right] > UNIT_COST[left]) swap = 1;
            else if (rp == lp && UNIT_COST[right] == UNIT_COST[left] && right < left) swap = 1;
            if (swap) {
                candidates[i] = right;
                candidates[j] = left;
            }
        }
    }
    for (int i = 0; i < STRATEGIC_MAX_LEVEL; i++) out[i] = 0;
    int limit = env->level < STRATEGIC_MAX_LEVEL ? env->level : STRATEGIC_MAX_LEVEL;
    for (int i = 0; i < count && i < limit; i++) out[i] = candidates[i];
}

static inline int strategic_best_buy(const StrategicTFT* env, int mode, int* out_shop, int* out_unit) {
    int total_owned = 0;
    for (int i = 0; i < STRATEGIC_OWNED_SIZE; i++) total_owned += env->owned[i];
    if (total_owned >= 27) return 0;

    int owned_traits[STRATEGIC_TRAIT_COUNT] = {0};
    int present_roles[STRATEGIC_ROLE_COUNT] = {0};
    strategic_fielded_role_presence(env, present_roles);
    for (int unit_id = 1; unit_id <= STRATEGIC_UNIT_COUNT; unit_id++) {
        if (env->owned[unit_id] > 0) owned_traits[UNIT_TRAIT[unit_id]] = 1;
    }

    int found = 0;
    float best_score = -1.0e30f;
    int best_shop = -1;
    int best_unit = 0;
    for (int shop_index = 0; shop_index < STRATEGIC_SHOP_SIZE; shop_index++) {
        int unit_id = env->shop[shop_index];
        if (unit_id == 0 || env->gold < UNIT_COST[unit_id]) continue;
        int copies = env->owned[unit_id];
        float score = 0.0f;
        if (mode == 0) {
            if (copies <= 0) continue;
            int next = copies + 1;
            int immediate = (next == 3 || next == 9);
            int d3 = (3 - (next % 3)) % 3;
            int d9 = (9 - (next % 9)) % 9;
            int distance = d3 < d9 ? d3 : d9;
            score = (immediate ? 1000.0f : 0.0f) + (20.0f - (float)distance) + UNIT_POWER[unit_id];
        } else if (mode == 1) {
            score = (owned_traits[UNIT_TRAIT[unit_id]] ? 100.0f : 0.0f)
                + (!present_roles[UNIT_ROLE[unit_id]] ? 20.0f : 0.0f)
                + UNIT_POWER[unit_id] + (float)UNIT_COST[unit_id];
        } else {
            score = (float)UNIT_COST[unit_id] * 100.0f + UNIT_POWER[unit_id];
        }
        if (!found || score > best_score
                || (score == best_score && shop_index > best_shop)
                || (score == best_score && shop_index == best_shop && unit_id > best_unit)) {
            found = 1;
            best_score = score;
            best_shop = shop_index;
            best_unit = unit_id;
        }
    }
    *out_shop = best_shop;
    *out_unit = best_unit;
    return found;
}

static inline void strategic_write_obs_and_mask(StrategicTFT* env) {
    float* obs = env->observations;
    int cursor = 0;
    obs[cursor++] = (float)env->round / 36.0f;
    obs[cursor++] = (float)env->hp / 100.0f;
    obs[cursor++] = (float)env->gold / 100.0f;
    obs[cursor++] = (float)env->level / 9.0f;
    obs[cursor++] = (float)env->xp / 18.0f;
    obs[cursor++] = (float)env->action_count / 3.0f;
    obs[cursor++] = env->last_board_strength / 200.0f;
    obs[cursor++] = env->last_enemy_strength / 200.0f;
    for (int i = 0; i < STRATEGIC_SHOP_SIZE; i++) obs[cursor++] = (float)env->shop[i] / 9.0f;
    for (int i = 0; i < STRATEGIC_OWNED_SIZE; i++) obs[cursor++] = (float)env->owned[i] / 9.0f;
    for (int i = 0; i < STRATEGIC_MAX_LEVEL; i++) obs[cursor++] = (float)env->fielded[i] / 9.0f;
    for (int i = 0; i < STRATEGIC_ROLE_COUNT; i++) obs[cursor++] = (float)env->role_items[i] / 5.0f;
    for (int i = 0; i < STRATEGIC_ROLE_COUNT; i++) obs[cursor++] = (float)env->role_item_slots[i] / 3.0f;

    if (env->action_mask == NULL) return;
    memset(env->action_mask, 0, STRATEGIC_NUM_ACTIONS * sizeof(unsigned char));
    if (env->done) return;
    env->action_mask[STRATEGIC_HOLD] = 1;
    env->action_mask[STRATEGIC_GREED_ECON] = 1;
    env->action_mask[STRATEGIC_LEVEL] = (env->gold >= 4 && env->level < 9);
    env->action_mask[STRATEGIC_ROLL] = (env->gold >= 2);
    int shop_index = -1;
    int unit_id = 0;
    env->action_mask[STRATEGIC_BUY_BEST_UPGRADE] = strategic_best_buy(env, 0, &shop_index, &unit_id);
    env->action_mask[STRATEGIC_BUY_BEST_SYNERGY] = strategic_best_buy(env, 1, &shop_index, &unit_id);
    env->action_mask[STRATEGIC_BUY_HIGHEST_COST] = strategic_best_buy(env, 2, &shop_index, &unit_id);
    int strongest[STRATEGIC_MAX_LEVEL];
    strategic_strongest_field(env, strongest);
    env->action_mask[STRATEGIC_FIELD_STRONGEST] =
        memcmp(strongest, env->fielded, sizeof(strongest)) != 0;
    int roles[STRATEGIC_ROLE_COUNT];
    strategic_fielded_role_presence(env, roles);
    env->action_mask[STRATEGIC_SLAM_CARRY_ITEM] =
        env->role_items[STRATEGIC_CARRY] > 0 && env->role_item_slots[STRATEGIC_CARRY] < 3
        && roles[STRATEGIC_CARRY];
    env->action_mask[STRATEGIC_SLAM_TANK_ITEM] =
        env->role_items[STRATEGIC_TANK] > 0 && env->role_item_slots[STRATEGIC_TANK] < 3
        && roles[STRATEGIC_TANK];
    env->action_mask[STRATEGIC_SLAM_SUPPORT_ITEM] =
        env->role_items[STRATEGIC_SUPPORT] > 0 && env->role_item_slots[STRATEGIC_SUPPORT] < 3
        && roles[STRATEGIC_SUPPORT];
}

static inline int strategic_sample_unit_id(StrategicTFT* env) {
    float roll = strategic_random_float(env);
    float tier_one = env->level <= 3 ? 0.72f : env->level <= 5 ? 0.48f : env->level <= 7 ? 0.25f : 0.12f;
    float tier_two = env->level <= 3 ? 0.26f : env->level <= 5 ? 0.42f : env->level <= 7 ? 0.50f : 0.43f;
    int cost = roll < tier_one ? 1 : roll < tier_one + tier_two ? 2 : 3;
    return UNITS_BY_COST[cost - 1][strategic_random_int(env, 3)];
}

static inline void strategic_refresh_shop(StrategicTFT* env) {
    for (int i = 0; i < STRATEGIC_SHOP_SIZE; i++) env->shop[i] = strategic_sample_unit_id(env);
}

static inline void strategic_add_log(StrategicTFT* env, int final_max_round) {
    int placement = strategic_placement_proxy(env, final_max_round);
    float score = strategic_scenario_score(env);
    env->log.perf += (8.0f - (float)placement) / 7.0f;
    env->log.score += score;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->episode_steps;
    env->log.placement += (float)placement;
    env->log.final_round += (float)env->round;
    env->log.hp += (float)env->hp;
    env->log.n += 1.0f;
}

static inline void c_reset(StrategicTFT* env) {
    memset(env->shop, 0, sizeof(env->shop));
    memset(env->owned, 0, sizeof(env->owned));
    memset(env->fielded, 0, sizeof(env->fielded));
    memset(env->role_items, 0, sizeof(env->role_items));
    memset(env->role_item_slots, 0, sizeof(env->role_item_slots));
    env->seed = env->base_seed;
    env->base_seed += 1000003;
    env->rng_key = ((uint64_t)(uint32_t)env->seed) ^ 0x9E3779B97F4A7C15ULL;
    env->round = 1;
    env->hp = 100;
    env->gold = 3;
    env->level = 3;
    env->xp = 0;
    env->done = 0;
    env->final_reason = 0;
    env->action_count = 0;
    env->last_board_strength = 0.0f;
    env->last_enemy_strength = 0.0f;
    env->last_damage = 0;
    env->last_win = 0;
    env->total_rolls = 0;
    env->total_xp_buys = 0;
    env->total_units_bought = 0;
    env->total_item_slams = 0;
    env->total_illegal_actions = 0;
    env->episode_steps = 0;
    env->episode_return = 0.0f;
    strategic_refresh_shop(env);
    strategic_write_obs_and_mask(env);
}

static inline float strategic_normal_noise(StrategicTFT* env) {
    return ((strategic_random_float(env) + strategic_random_float(env) + strategic_random_float(env)) - 1.5f)
        * (2.0f * 1.6f);
}

static inline int strategic_damage_from_margin(int round_num, float enemy, float board) {
    int base = round_num < 8 ? 2 : round_num < 16 ? 4 : round_num < 24 ? 6 : 8;
    return (int)((float)base + fmaxf(0.0f, enemy - board) / 22.0f);
}

static inline float strategic_end_round(StrategicTFT* env, int greed, int* done, int* final_max_round) {
    float previous_strength = env->last_board_strength;
    float current_strength = strategic_board_strength(env);
    float enemy_strength = strategic_enemy_strength_for_round(env->round) + strategic_normal_noise(env);
    float p_win = 1.0f / (1.0f + expf(-((current_strength - enemy_strength) / 14.0f)));
    int won = strategic_random_float(env) < p_win;
    int damage = won ? 0 : strategic_damage_from_margin(env->round, enemy_strength, current_strength);
    int previous_hp = env->hp;
    env->hp = env->hp - damage > 0 ? env->hp - damage : 0;
    env->last_board_strength = current_strength;
    env->last_enemy_strength = enemy_strength;
    env->last_damage = damage;
    env->last_win = won;
    env->gold += 5 + (env->gold / 10 < 5 ? env->gold / 10 : 5);
    if (won) env->gold += 1;
    if (greed && env->gold >= 10) env->gold += 1;
    if (env->round % 4 == 0) env->role_items[strategic_random_int(env, 3)] += 1;
    strategic_refresh_shop(env);
    env->action_count = 0;

    if (env->hp <= 0) {
        *done = 1;
        *final_max_round = 0;
        env->done = 1;
        env->final_reason = 1;
    } else if (env->round >= 36) {
        *done = 1;
        *final_max_round = 1;
        env->done = 1;
        env->final_reason = 2;
    } else {
        env->round += 1;
    }

    float reward = (float)(env->hp - previous_hp) * 0.04f;
    reward += (current_strength - previous_strength) * 0.015f;
    reward += won ? 0.25f : -0.10f;
    reward += strategic_scenario_score(env) * 0.10f;
    if (greed) reward += env->hp >= 45 ? 0.04f : -0.08f;
    if (*done && *final_max_round) reward += 1.0f;
    if (*done && !*final_max_round) reward -= 0.8f;
    return reward;
}

static inline float strategic_apply_action(StrategicTFT* env, int action) {
    if (action == STRATEGIC_LEVEL) {
        env->gold -= 4;
        env->xp += 4;
        int leveled = 0;
        while (env->level < 9 && env->xp >= strategic_xp_to_next_level(env->level)) {
            env->xp -= strategic_xp_to_next_level(env->level);
            env->level += 1;
            leveled = 1;
        }
        env->total_xp_buys += 1;
        return leveled ? 0.08f : 0.01f;
    }
    if (action == STRATEGIC_ROLL) {
        env->gold -= 2;
        strategic_refresh_shop(env);
        env->total_rolls += 1;
        return 0.0f;
    }
    if (action == STRATEGIC_BUY_BEST_UPGRADE || action == STRATEGIC_BUY_BEST_SYNERGY
            || action == STRATEGIC_BUY_HIGHEST_COST) {
        int mode = action == STRATEGIC_BUY_BEST_UPGRADE ? 0
            : action == STRATEGIC_BUY_BEST_SYNERGY ? 1 : 2;
        int shop_index = -1;
        int unit_id = 0;
        if (!strategic_best_buy(env, mode, &shop_index, &unit_id)) return 0.0f;
        env->gold -= UNIT_COST[unit_id];
        env->owned[unit_id] += 1;
        env->shop[shop_index] = 0;
        env->total_units_bought += 1;
        return env->owned[unit_id] == 3 || env->owned[unit_id] == 9 ? 0.10f : 0.04f;
    }
    if (action == STRATEGIC_FIELD_STRONGEST) {
        float before = strategic_board_strength(env);
        int strongest[STRATEGIC_MAX_LEVEL];
        strategic_strongest_field(env, strongest);
        int changed = memcmp(strongest, env->fielded, sizeof(strongest)) != 0;
        memcpy(env->fielded, strongest, sizeof(strongest));
        float after = strategic_board_strength(env);
        return changed && after > before ? (after - before) * 0.01f : 0.0f;
    }
    int role = -1;
    if (action == STRATEGIC_SLAM_CARRY_ITEM) role = STRATEGIC_CARRY;
    else if (action == STRATEGIC_SLAM_TANK_ITEM) role = STRATEGIC_TANK;
    else if (action == STRATEGIC_SLAM_SUPPORT_ITEM) role = STRATEGIC_SUPPORT;
    if (role >= 0) {
        env->role_items[role] -= 1;
        env->role_item_slots[role] += 1;
        env->total_item_slams += 1;
        return ROLE_ITEM_POWER[role] * 0.02f;
    }
    return 0.0f;
}

static inline void strategic_step_impl(StrategicTFT* env, int auto_reset) {
    int action = (int)env->actions[0];
    int legal = action >= 0 && action < STRATEGIC_NUM_ACTIONS
        && (env->action_mask == NULL || env->action_mask[action]);
    int done = 0;
    int final_max_round = 0;
    float reward = 0.0f;
    env->terminals[0] = 0.0f;
    env->rewards[0] = 0.0f;

    if (!legal) {
        env->total_illegal_actions += 1;
        reward -= 1.0f;
    } else if (action == STRATEGIC_HOLD) {
        reward += strategic_end_round(env, 0, &done, &final_max_round);
    } else if (action == STRATEGIC_GREED_ECON) {
        reward += strategic_end_round(env, 1, &done, &final_max_round);
    } else {
        reward += strategic_apply_action(env, action);
        env->action_count += 1;
        if (env->action_count >= 3) {
            reward -= 0.05f;
            reward += strategic_end_round(env, 0, &done, &final_max_round);
        }
    }

    env->episode_steps += 1;
    env->episode_return += reward;
    env->rewards[0] = reward;
    if (done) {
        env->terminals[0] = 1.0f;
        strategic_add_log(env, final_max_round);
        if (auto_reset) {
            c_reset(env);
        } else {
            strategic_write_obs_and_mask(env);
        }
        return;
    }
    strategic_write_obs_and_mask(env);
}

static inline void strategic_step_no_reset(StrategicTFT* env) {
    strategic_step_impl(env, 0);
}

static inline void c_step(StrategicTFT* env) {
    strategic_step_impl(env, 1);
}

static inline int strategic_heuristic_action(StrategicTFT* env) {
    strategic_write_obs_and_mask(env);
    unsigned char* mask = env->action_mask;
    float pressure = strategic_enemy_strength_for_round(env->round);
    float strength = strategic_board_strength(env);
    if (mask[STRATEGIC_FIELD_STRONGEST] && env->action_count >= 1) return STRATEGIC_FIELD_STRONGEST;
    if (mask[STRATEGIC_SLAM_CARRY_ITEM] && strength < pressure * 0.95f) return STRATEGIC_SLAM_CARRY_ITEM;
    if (mask[STRATEGIC_SLAM_TANK_ITEM] && strength < pressure * 0.95f) return STRATEGIC_SLAM_TANK_ITEM;
    if (mask[STRATEGIC_SLAM_SUPPORT_ITEM] && strength < pressure * 0.95f) return STRATEGIC_SLAM_SUPPORT_ITEM;
    if (mask[STRATEGIC_BUY_BEST_UPGRADE]) return STRATEGIC_BUY_BEST_UPGRADE;
    if (mask[STRATEGIC_BUY_HIGHEST_COST] && (env->gold < 20 || strength < pressure * 1.05f)) return STRATEGIC_BUY_HIGHEST_COST;
    if (mask[STRATEGIC_BUY_BEST_SYNERGY]) return STRATEGIC_BUY_BEST_SYNERGY;
    if (mask[STRATEGIC_LEVEL] && ((env->level < 4 && env->round >= 3)
            || (env->level < 6 && env->round >= 9 && env->gold >= 8)
            || (env->level < 8 && env->round >= 18 && env->gold >= 20))) return STRATEGIC_LEVEL;
    if (mask[STRATEGIC_ROLL] && env->gold >= 16 && strength < pressure * 0.85f) return STRATEGIC_ROLL;
    if (mask[STRATEGIC_FIELD_STRONGEST]) return STRATEGIC_FIELD_STRONGEST;
    if (mask[STRATEGIC_GREED_ECON] && env->gold >= 20 && env->hp >= 60 && strength >= pressure) return STRATEGIC_GREED_ECON;
    return STRATEGIC_HOLD;
}

static inline void c_render(StrategicTFT* env) {
    (void)env;
}

static inline void c_close(StrategicTFT* env) {
    (void)env;
}
