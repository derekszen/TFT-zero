#pragma once

#include <array>
#include <cstdint>

namespace mini_tft::strategic::native {

constexpr int kNumActions = 11;
constexpr int kShopSize = 5;
constexpr int kUnitCount = 9;
constexpr int kOwnedSize = kUnitCount + 1;
constexpr int kMaxLevel = 9;
constexpr int kRoleCount = 3;
constexpr int kTraitCount = 6;

enum Action {
    kHold = 0,
    kLevel = 1,
    kRoll = 2,
    kBuyBestUpgrade = 3,
    kBuyBestSynergy = 4,
    kBuyHighestCost = 5,
    kFieldStrongest = 6,
    kGreedEcon = 7,
    kSlamCarryItem = 8,
    kSlamTankItem = 9,
    kSlamSupportItem = 10,
};

enum Role {
    kCarry = 0,
    kTank = 1,
    kSupport = 2,
};

enum FinalReason {
    kFinalNone = 0,
    kFinalHpZero = 1,
    kFinalMaxRound = 2,
};

struct StrategicConfig {
    int max_round = 36;
    int max_actions_per_round = 3;
    int starting_hp = 100;
    int starting_gold = 3;
    int starting_level = 3;
    int starting_xp = 0;
    int max_level = kMaxLevel;
    int shop_size = kShopSize;
    int roll_cost = 2;
    int xp_buy_cost = 4;
    int xp_per_buy = 4;
    int base_income = 5;
    int max_interest = 5;
    int win_gold = 1;
    int max_role_item_slots = 3;
    int item_drop_interval = 4;
    double combat_sigmoid_scale = 14.0;
    double enemy_strength_multiplier = 5.0;
    double combat_noise = 2.0;
    int max_owned_copies = 27;
};

struct StrategicState {
    std::int64_t seed = 0;
    std::uint64_t rng_key = 0;
    int round = 1;
    int hp = 100;
    int gold = 3;
    int level = 3;
    int xp = 0;
    std::array<int, kShopSize> shop{};
    std::array<int, kOwnedSize> owned{};
    std::array<int, kMaxLevel> fielded{};
    std::array<int, kRoleCount> role_items{};
    std::array<int, kRoleCount> role_item_slots{};
    bool done = false;
    int final_reason = kFinalNone;
    int action_count = 0;
    double last_board_strength = 0.0;
    double last_enemy_strength = 0.0;
    int last_damage = 0;
    bool last_win = false;
    int total_rolls = 0;
    int total_xp_buys = 0;
    int total_units_bought = 0;
    int total_item_slams = 0;
    int total_illegal_actions = 0;
};

struct StepResult {
    double reward = 0.0;
    bool terminated = false;
    bool truncated = false;
    bool legal = false;
    bool ended_round = false;
};

}  // namespace mini_tft::strategic::native
