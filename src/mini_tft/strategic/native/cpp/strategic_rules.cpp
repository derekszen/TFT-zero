#include "strategic_rules.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace mini_tft::strategic::native {
namespace {

constexpr std::uint64_t kLcgMultiplier = 6364136223846793005ULL;
constexpr std::uint64_t kLcgIncrement = 1442695040888963407ULL;
constexpr std::uint64_t kSeedXor = 0x9E3779B97F4A7C15ULL;
constexpr double kInvDouble53 = 1.0 / 9007199254740992.0;

constexpr std::array<int, kOwnedSize> kUnitCost = {0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
constexpr std::array<double, kOwnedSize> kUnitPower = {
    0.0, 10.0, 11.0, 8.5, 16.0, 17.5, 14.0, 24.0, 27.0, 21.0};
constexpr std::array<int, kOwnedSize> kUnitRole = {
    -1, kTank, kCarry, kSupport, kTank, kCarry, kSupport, kTank, kCarry, kSupport};
constexpr std::array<int, kOwnedSize> kUnitTrait = { -1, 0, 1, 2, 0, 1, 3, 4, 5, 3 };
constexpr std::array<double, 4> kStarMultiplier = {0.0, 1.0, 1.9, 3.5};
constexpr std::array<double, kRoleCount> kRoleItemPower = {9.0, 8.0, 6.5};
constexpr std::array<std::array<int, 3>, 3> kUnitsByCost = {{
    {{1, 2, 3}},
    {{4, 5, 6}},
    {{7, 8, 9}},
}};

int star_level(int copies) {
    if (copies >= 9) {
        return 3;
    }
    if (copies >= 3) {
        return 2;
    }
    if (copies >= 1) {
        return 1;
    }
    return 0;
}

int xp_to_next_level(int level) {
    return 2 + std::max(0, level - 1) * 2;
}

std::array<bool, kRoleCount> fielded_role_presence(const StrategicState& state) {
    std::array<bool, kRoleCount> present{};
    for (int unit_id : state.fielded) {
        if (unit_id > 0) {
            present[static_cast<std::size_t>(kUnitRole[static_cast<std::size_t>(unit_id)])] = true;
        }
    }
    return present;
}

std::pair<int, int> best_buy(
    const StrategicState& state,
    const StrategicConfig& config,
    const std::string& mode) {
    int total_owned = 0;
    for (int copies : state.owned) {
        total_owned += copies;
    }
    if (total_owned >= config.max_owned_copies) {
        return {-1, 0};
    }

    std::array<bool, kTraitCount> owned_traits{};
    for (int unit_id = 1; unit_id <= kUnitCount; ++unit_id) {
        if (state.owned[static_cast<std::size_t>(unit_id)] > 0) {
            owned_traits[static_cast<std::size_t>(kUnitTrait[static_cast<std::size_t>(unit_id)])] =
                true;
        }
    }
    const auto present_roles = fielded_role_presence(state);

    bool found = false;
    std::tuple<double, int, int> best{-std::numeric_limits<double>::infinity(), -1, 0};
    for (int shop_index = 0; shop_index < config.shop_size; ++shop_index) {
        const int unit_id = state.shop[static_cast<std::size_t>(shop_index)];
        if (unit_id == 0) {
            continue;
        }
        const int cost = kUnitCost[static_cast<std::size_t>(unit_id)];
        if (state.gold < cost) {
            continue;
        }
        const int copies = state.owned[static_cast<std::size_t>(unit_id)];
        double score = 0.0;
        if (mode == "upgrade") {
            if (copies <= 0) {
                continue;
            }
            const int next_copies = copies + 1;
            const double immediate = (next_copies == 3 || next_copies == 9) ? 1.0 : 0.0;
            const int distance = std::min(
                (3 - (next_copies % 3)) % 3,
                (9 - (next_copies % 9)) % 9);
            score = immediate * 1000.0 + (20.0 - static_cast<double>(distance))
                + kUnitPower[static_cast<std::size_t>(unit_id)];
        } else if (mode == "synergy") {
            const bool trait_match =
                owned_traits[static_cast<std::size_t>(kUnitTrait[static_cast<std::size_t>(unit_id)])];
            const bool role_need =
                !present_roles[static_cast<std::size_t>(kUnitRole[static_cast<std::size_t>(unit_id)])];
            score = (trait_match ? 100.0 : 0.0) + (role_need ? 20.0 : 0.0)
                + kUnitPower[static_cast<std::size_t>(unit_id)] + static_cast<double>(cost);
        } else if (mode == "highest_cost") {
            score = static_cast<double>(cost) * 100.0
                + kUnitPower[static_cast<std::size_t>(unit_id)];
        } else {
            throw std::invalid_argument("unknown buy mode");
        }

        const std::tuple<double, int, int> candidate{score, shop_index, unit_id};
        if (!found || candidate > best) {
            best = candidate;
            found = true;
        }
    }
    if (!found) {
        return {-1, 0};
    }
    return {std::get<1>(best), std::get<2>(best)};
}

double unit_field_power(int unit_id, int copies) {
    return kUnitPower[static_cast<std::size_t>(unit_id)]
        * kStarMultiplier[static_cast<std::size_t>(star_level(copies))];
}

std::array<int, kMaxLevel> strongest_field_signature(
    const StrategicState& state,
    const StrategicConfig& config) {
    std::vector<int> candidates;
    candidates.reserve(kUnitCount);
    for (int unit_id = 1; unit_id <= kUnitCount; ++unit_id) {
        if (state.owned[static_cast<std::size_t>(unit_id)] > 0) {
            candidates.push_back(unit_id);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [&state](int left, int right) {
        const double left_power = unit_field_power(left, state.owned[static_cast<std::size_t>(left)]);
        const double right_power =
            unit_field_power(right, state.owned[static_cast<std::size_t>(right)]);
        if (left_power != right_power) {
            return left_power > right_power;
        }
        if (kUnitCost[static_cast<std::size_t>(left)] != kUnitCost[static_cast<std::size_t>(right)]) {
            return kUnitCost[static_cast<std::size_t>(left)]
                > kUnitCost[static_cast<std::size_t>(right)];
        }
        return left < right;
    });

    std::array<int, kMaxLevel> fielded{};
    const int limit = std::min(config.max_level, state.level);
    for (int index = 0; index < limit && index < static_cast<int>(candidates.size()); ++index) {
        fielded[static_cast<std::size_t>(index)] = candidates[static_cast<std::size_t>(index)];
    }
    return fielded;
}

bool should_level(const StrategicState& state) {
    if (state.level < 4 && state.round >= 3) {
        return true;
    }
    if (state.level < 6 && state.round >= 9 && state.gold >= 8) {
        return true;
    }
    if (state.level < 8 && state.round >= 18 && state.gold >= 20) {
        return true;
    }
    return false;
}

double normal_noise(StrategicState& state, const StrategicConfig& config) {
    if (config.combat_noise <= 0.0) {
        return 0.0;
    }
    return ((random_float(state) + random_float(state) + random_float(state)) - 1.5)
        * (config.combat_noise * 1.6);
}

int damage_from_margin(int round_num, double enemy_strength, double board) {
    const int base = round_num < 8 ? 2 : round_num < 16 ? 4 : round_num < 24 ? 6 : 8;
    return static_cast<int>(static_cast<double>(base) + std::max(0.0, enemy_strength - board) / 22.0);
}

void maybe_drop_role_item(StrategicState& state, const StrategicConfig& config) {
    if (state.round % config.item_drop_interval != 0) {
        return;
    }
    const int role_index = random_int(state, kRoleCount);
    state.role_items[static_cast<std::size_t>(role_index)] += 1;
}

int sample_unit_id(StrategicState& state, const StrategicConfig& config) {
    const double roll = random_float(state);
    std::array<double, 3> tiers{};
    if (state.level <= 3) {
        tiers = {0.72, 0.26, 0.02};
    } else if (state.level <= 5) {
        tiers = {0.48, 0.42, 0.10};
    } else if (state.level <= 7) {
        tiers = {0.25, 0.50, 0.25};
    } else {
        tiers = {0.12, 0.43, 0.45};
    }
    const int cost = roll < tiers[0] ? 1 : roll < tiers[0] + tiers[1] ? 2 : 3;
    const int index = random_int(state, 3);
    return kUnitsByCost[static_cast<std::size_t>(cost - 1)][static_cast<std::size_t>(index)];
}

double end_round(StrategicState& state, const StrategicConfig& config, bool greed) {
    const double previous_strength = state.last_board_strength;
    const double current_strength = board_strength(state, config);
    const double enemy_strength = enemy_strength_for_round(state.round, config) + normal_noise(state, config);
    const double p_win = 1.0 / (1.0 + std::exp(-((current_strength - enemy_strength) / config.combat_sigmoid_scale)));
    const bool won = random_float(state) < p_win;
    const int damage = won ? 0 : damage_from_margin(state.round, enemy_strength, current_strength);
    const int previous_hp = state.hp;
    state.hp = std::max(0, state.hp - damage);

    state.last_board_strength = current_strength;
    state.last_enemy_strength = enemy_strength;
    state.last_damage = damage;
    state.last_win = won;

    state.gold += config.base_income + std::min(config.max_interest, state.gold / 10);
    if (won) {
        state.gold += config.win_gold;
    }
    if (greed && state.gold >= 10) {
        state.gold += 1;
    }

    maybe_drop_role_item(state, config);
    refresh_shop(state, config);
    state.action_count = 0;

    if (state.hp <= 0) {
        state.done = true;
        state.final_reason = kFinalHpZero;
    } else if (state.round >= config.max_round) {
        state.done = true;
        state.final_reason = kFinalMaxRound;
    } else {
        state.round += 1;
    }

    double reward = static_cast<double>(state.hp - previous_hp) * 0.04;
    reward += (current_strength - previous_strength) * 0.015;
    reward += won ? 0.25 : -0.10;
    reward += scenario_score(state, config) * 0.10;
    if (greed) {
        reward += state.hp >= config.starting_hp * 0.45 ? 0.04 : -0.08;
    }
    if (state.done && state.final_reason == kFinalMaxRound) {
        reward += 1.0;
    }
    if (state.done && state.final_reason == kFinalHpZero) {
        reward -= 0.8;
    }
    return reward;
}

double apply_non_terminal_action(
    StrategicState& state,
    int action,
    const StrategicConfig& config) {
    if (action == kLevel) {
        state.gold -= config.xp_buy_cost;
        state.xp += config.xp_per_buy;
        bool leveled = false;
        while (state.level < config.max_level && state.xp >= xp_to_next_level(state.level)) {
            state.xp -= xp_to_next_level(state.level);
            state.level += 1;
            leveled = true;
        }
        state.total_xp_buys += 1;
        return leveled ? 0.08 : 0.01;
    }
    if (action == kRoll) {
        state.gold -= config.roll_cost;
        state.total_rolls += 1;
        refresh_shop(state, config);
        return 0.0;
    }
    if (action == kBuyBestUpgrade || action == kBuyBestSynergy || action == kBuyHighestCost) {
        const std::string mode = action == kBuyBestUpgrade
            ? "upgrade"
            : action == kBuyBestSynergy ? "synergy" : "highest_cost";
        const auto [shop_index, unit_id] = best_buy(state, config, mode);
        if (shop_index < 0) {
            return 0.0;
        }
        state.gold -= kUnitCost[static_cast<std::size_t>(unit_id)];
        state.owned[static_cast<std::size_t>(unit_id)] += 1;
        state.shop[static_cast<std::size_t>(shop_index)] = 0;
        state.total_units_bought += 1;
        const int copies = state.owned[static_cast<std::size_t>(unit_id)];
        return (copies == 3 || copies == 9) ? 0.10 : 0.04;
    }
    if (action == kFieldStrongest) {
        const double before = board_strength(state, config);
        const auto next_field = strongest_field_signature(state, config);
        const bool changed = next_field != state.fielded;
        state.fielded = next_field;
        const double after = board_strength(state, config);
        return changed ? std::max(0.0, (after - before) * 0.01) : 0.0;
    }
    int role_index = -1;
    if (action == kSlamCarryItem) {
        role_index = kCarry;
    } else if (action == kSlamTankItem) {
        role_index = kTank;
    } else if (action == kSlamSupportItem) {
        role_index = kSupport;
    }
    if (role_index >= 0) {
        state.role_items[static_cast<std::size_t>(role_index)] -= 1;
        state.role_item_slots[static_cast<std::size_t>(role_index)] += 1;
        state.total_item_slams += 1;
        return kRoleItemPower[static_cast<std::size_t>(role_index)] * 0.02;
    }
    return 0.0;
}

}  // namespace

StrategicState reset(std::int64_t seed, const StrategicConfig& config) {
    StrategicState state;
    state.seed = seed;
    state.rng_key = static_cast<std::uint64_t>(seed) ^ kSeedXor;
    state.round = 1;
    state.hp = config.starting_hp;
    state.gold = config.starting_gold;
    state.level = config.starting_level;
    state.xp = config.starting_xp;
    refresh_shop(state, config);
    return state;
}

std::array<bool, kNumActions> legal_action_mask(
    const StrategicState& state,
    const StrategicConfig& config) {
    std::array<bool, kNumActions> mask{};
    if (state.done) {
        return mask;
    }

    mask[static_cast<std::size_t>(kHold)] = true;
    mask[static_cast<std::size_t>(kGreedEcon)] = true;
    mask[static_cast<std::size_t>(kLevel)] =
        state.gold >= config.xp_buy_cost && state.level < config.max_level;
    mask[static_cast<std::size_t>(kRoll)] = state.gold >= config.roll_cost;
    mask[static_cast<std::size_t>(kBuyBestUpgrade)] =
        best_buy(state, config, "upgrade").first >= 0;
    mask[static_cast<std::size_t>(kBuyBestSynergy)] =
        best_buy(state, config, "synergy").first >= 0;
    mask[static_cast<std::size_t>(kBuyHighestCost)] =
        best_buy(state, config, "highest_cost").first >= 0;
    mask[static_cast<std::size_t>(kFieldStrongest)] =
        strongest_field_signature(state, config) != state.fielded;

    const auto present_roles = fielded_role_presence(state);
    mask[static_cast<std::size_t>(kSlamCarryItem)] =
        state.role_items[kCarry] > 0
        && state.role_item_slots[kCarry] < config.max_role_item_slots
        && present_roles[kCarry];
    mask[static_cast<std::size_t>(kSlamTankItem)] =
        state.role_items[kTank] > 0
        && state.role_item_slots[kTank] < config.max_role_item_slots
        && present_roles[kTank];
    mask[static_cast<std::size_t>(kSlamSupportItem)] =
        state.role_items[kSupport] > 0
        && state.role_item_slots[kSupport] < config.max_role_item_slots
        && present_roles[kSupport];
    return mask;
}

StepResult step(StrategicState& state, int action, const StrategicConfig& config) {
    if (state.done) {
        throw std::runtime_error("Strategic episode is done. Call reset() before stepping.");
    }

    const auto mask = legal_action_mask(state, config);
    const bool legal = action >= 0 && action < kNumActions && mask[static_cast<std::size_t>(action)];
    StepResult result;
    result.legal = legal;

    if (!legal) {
        state.total_illegal_actions += 1;
        result.reward -= 1.0;
    } else if (action == kHold) {
        result.reward += end_round(state, config, false);
        result.ended_round = true;
    } else if (action == kGreedEcon) {
        result.reward += end_round(state, config, true);
        result.ended_round = true;
    } else {
        result.reward += apply_non_terminal_action(state, action, config);
        state.action_count += 1;
        if (state.action_count >= config.max_actions_per_round && !state.done) {
            result.reward -= 0.05;
            result.reward += end_round(state, config, false);
            result.ended_round = true;
        }
    }

    result.terminated =
        state.done && (state.final_reason == kFinalHpZero || state.final_reason == kFinalMaxRound);
    result.truncated = false;
    return result;
}

void refresh_shop(StrategicState& state, const StrategicConfig& config) {
    for (int index = 0; index < config.shop_size; ++index) {
        state.shop[static_cast<std::size_t>(index)] = sample_unit_id(state, config);
    }
}

double board_strength(const StrategicState& state, const StrategicConfig& config) {
    (void)config;
    double strength = 0.0;
    std::array<int, kTraitCount> trait_counts{};
    std::array<int, kRoleCount> role_counts{};
    for (int unit_id : state.fielded) {
        if (unit_id == 0) {
            continue;
        }
        const int copies = state.owned[static_cast<std::size_t>(unit_id)];
        strength += kUnitPower[static_cast<std::size_t>(unit_id)]
            * kStarMultiplier[static_cast<std::size_t>(star_level(copies))];
        trait_counts[static_cast<std::size_t>(kUnitTrait[static_cast<std::size_t>(unit_id)])] += 1;
        role_counts[static_cast<std::size_t>(kUnitRole[static_cast<std::size_t>(unit_id)])] += 1;
    }

    for (int count : trait_counts) {
        if (count >= 3) {
            strength += 8.0;
        } else if (count >= 2) {
            strength += 3.0;
        }
    }
    for (int role_index = 0; role_index < kRoleCount; ++role_index) {
        if (role_counts[static_cast<std::size_t>(role_index)] > 0) {
            strength += static_cast<double>(state.role_item_slots[static_cast<std::size_t>(role_index)])
                * kRoleItemPower[static_cast<std::size_t>(role_index)];
        }
    }
    if (role_counts[kTank] > 0 && role_counts[kCarry] > 0) {
        strength += 6.0;
    }
    const int role_total = role_counts[0] + role_counts[1] + role_counts[2];
    if (role_total == 0) {
        strength -= 12.0;
    }
    return std::max(0.0, strength);
}

double enemy_strength_for_round(int round_num, const StrategicConfig& config) {
    const double round_float = static_cast<double>(std::max(1, round_num));
    const double base = 13.0 + round_float * 3.2 + std::pow(round_float, 1.18) * 1.35;
    return base * config.enemy_strength_multiplier;
}

int placement_proxy(const StrategicState& state, const StrategicConfig& config) {
    if (state.final_reason == kFinalMaxRound || (state.round >= config.max_round && state.hp > 0)) {
        return 1;
    }
    if (state.round >= 36) {
        return 2;
    }
    if (state.round >= 32) {
        return 3;
    }
    if (state.round >= 29) {
        return 4;
    }
    if (state.round >= 25) {
        return 5;
    }
    if (state.round >= 18) {
        return 6;
    }
    if (state.round >= 11) {
        return 7;
    }
    return 8;
}

double scenario_score(const StrategicState& state, const StrategicConfig& config) {
    const double round_frac = std::min(1.0, std::max(0.0, static_cast<double>(state.round) / config.max_round));
    const double hp_frac = std::min(1.0, std::max(0.0, static_cast<double>(state.hp) / config.starting_hp));
    const double enemy = std::max(1.0, enemy_strength_for_round(std::max(1, state.round), config));
    const double strength_ratio = std::min(1.4, state.last_board_strength / enemy) / 1.4;
    return std::min(1.0, std::max(0.0, 0.45 * round_frac + 0.25 * hp_frac + 0.30 * strength_ratio));
}

int heuristic_action(
    const StrategicState& state,
    const std::array<bool, kNumActions>& mask,
    const StrategicConfig& config) {
    const double pressure = enemy_strength_for_round(state.round, config);
    const double strength = board_strength(state, config);

    if (mask[kFieldStrongest] && state.action_count >= 1) {
        return kFieldStrongest;
    }
    for (int action : {kSlamCarryItem, kSlamTankItem, kSlamSupportItem}) {
        if (mask[static_cast<std::size_t>(action)] && strength < pressure * 0.95) {
            return action;
        }
    }
    if (mask[kBuyBestUpgrade]) {
        return kBuyBestUpgrade;
    }
    if (mask[kBuyHighestCost] && (state.gold < 20 || strength < pressure * 1.05)) {
        return kBuyHighestCost;
    }
    if (mask[kBuyBestSynergy]) {
        return kBuyBestSynergy;
    }
    if (mask[kLevel] && should_level(state)) {
        return kLevel;
    }
    if (mask[kRoll] && state.gold >= 16 && strength < pressure * 0.85) {
        return kRoll;
    }
    if (mask[kFieldStrongest]) {
        return kFieldStrongest;
    }
    if (mask[kGreedEcon] && state.gold >= 20 && state.hp >= 60 && strength >= pressure) {
        return kGreedEcon;
    }
    return kHold;
}

int random_action(const StrategicState& state, const std::array<bool, kNumActions>& mask) {
    const auto actions = legal_actions(mask);
    if (actions.empty()) {
        return kHold;
    }
#if defined(__SIZEOF_INT128__)
    const auto index = static_cast<std::size_t>(
        (static_cast<unsigned __int128>(state.rng_key) + static_cast<unsigned>(state.round)
         + static_cast<unsigned>(state.action_count))
        % actions.size());
#else
    const auto index = static_cast<std::size_t>(
        (state.rng_key + static_cast<std::uint64_t>(state.round + state.action_count))
        % actions.size());
#endif
    return actions[index];
}

std::vector<int> legal_actions(const std::array<bool, kNumActions>& mask) {
    std::vector<int> actions;
    actions.reserve(kNumActions);
    for (int action = 0; action < kNumActions; ++action) {
        if (mask[static_cast<std::size_t>(action)]) {
            actions.push_back(action);
        }
    }
    return actions;
}

const char* action_name(int action) {
    switch (action) {
        case kHold:
            return "hold";
        case kLevel:
            return "level";
        case kRoll:
            return "roll";
        case kBuyBestUpgrade:
            return "buy_best_upgrade";
        case kBuyBestSynergy:
            return "buy_best_synergy";
        case kBuyHighestCost:
            return "buy_highest_cost";
        case kFieldStrongest:
            return "field_strongest";
        case kGreedEcon:
            return "greed_econ";
        case kSlamCarryItem:
            return "slam_carry_item";
        case kSlamTankItem:
            return "slam_tank_item";
        case kSlamSupportItem:
            return "slam_support_item";
        default:
            return "unknown";
    }
}

const char* final_reason_name(int final_reason) {
    if (final_reason == kFinalHpZero) {
        return "hp_zero";
    }
    if (final_reason == kFinalMaxRound) {
        return "max_round";
    }
    return "";
}

std::uint64_t next_u64(StrategicState& state) {
    state.rng_key = state.rng_key * kLcgMultiplier + kLcgIncrement;
    return state.rng_key;
}

double random_float(StrategicState& state) {
    return static_cast<double>(next_u64(state) >> 11U) * kInvDouble53;
}

int random_int(StrategicState& state, int limit) {
    if (limit <= 0) {
        throw std::invalid_argument("limit must be positive");
    }
    return static_cast<int>(next_u64(state) % static_cast<std::uint64_t>(limit));
}

}  // namespace mini_tft::strategic::native
