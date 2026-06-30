#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "strategic_state.hpp"

namespace mini_tft::strategic::native {

StrategicState reset(std::int64_t seed, const StrategicConfig& config = StrategicConfig());
std::array<bool, kNumActions> legal_action_mask(
    const StrategicState& state,
    const StrategicConfig& config = StrategicConfig());
StepResult step(StrategicState& state, int action, const StrategicConfig& config = StrategicConfig());

void refresh_shop(StrategicState& state, const StrategicConfig& config = StrategicConfig());
double board_strength(const StrategicState& state, const StrategicConfig& config = StrategicConfig());
double enemy_strength_for_round(int round_num, const StrategicConfig& config = StrategicConfig());
int placement_proxy(const StrategicState& state, const StrategicConfig& config = StrategicConfig());
double scenario_score(const StrategicState& state, const StrategicConfig& config = StrategicConfig());
int heuristic_action(const StrategicState& state, const std::array<bool, kNumActions>& mask,
                     const StrategicConfig& config = StrategicConfig());
int random_action(const StrategicState& state, const std::array<bool, kNumActions>& mask);
std::vector<int> legal_actions(const std::array<bool, kNumActions>& mask);
const char* action_name(int action);
const char* final_reason_name(int final_reason);

std::uint64_t next_u64(StrategicState& state);
double random_float(StrategicState& state);
int random_int(StrategicState& state, int limit);

}  // namespace mini_tft::strategic::native
