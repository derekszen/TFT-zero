#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "strategic_rules.hpp"
#include "strategic_state.hpp"

namespace mini_tft::strategic::native {

struct MCTSConfig {
    int simulations = 16;
    int max_depth = 12;
    int rollout_steps = 8;
    double exploration = 1.4;
    double gamma = 0.97;
    std::string prior_mode = "uniform";
};

struct NativeDecision {
    int selected_action = kHold;
    std::array<double, kNumActions> visit_policy{};
    std::array<double, kNumActions> action_values{};
    std::array<int, kNumActions> action_visits{};
    std::array<bool, kNumActions> action_present{};
    int simulations_run = 0;
    double elapsed_ms = 0.0;
    int max_depth = 0;
};

struct EpisodeRow {
    std::string policy;
    int episode = 0;
    std::int64_t seed = 0;
    int placement = 8;
    int final_round = 1;
    int hp = 0;
    int final_reason = kFinalNone;
    double scenario_score = 0.0;
    int illegal_actions = 0;
    double total_reward = 0.0;
    int steps = 0;
    int decisions = 0;
    int simulations = 0;
    double elapsed_sec = 0.0;
};

struct DecisionRow {
    std::string policy;
    int episode = 0;
    std::int64_t seed = 0;
    int step = 0;
    int round = 1;
    int action_id = kHold;
    bool legal = false;
    double reward = 0.0;
    bool ended_round = false;
    int hp = 0;
    int gold = 0;
    int level = 0;
    int placement_proxy = 8;
    double scenario_score = 0.0;
    int simulations = 0;
    double mcts_elapsed_ms = 0.0;
    int mcts_max_depth = 0;
    std::array<double, kNumActions> visit_policy{};
    std::array<double, kNumActions> action_values{};
    std::array<int, kNumActions> action_visits{};
    std::array<bool, kNumActions> action_present{};
};

struct SmokeResult {
    std::vector<EpisodeRow> episode_rows;
    std::vector<DecisionRow> decision_rows;
    double elapsed_sec = 0.0;
};

struct BatchPlanResult {
    std::vector<int> selected_actions;
    std::vector<std::array<double, kNumActions>> visit_policies;
    std::vector<double> values;
    double elapsed_sec = 0.0;
    double simulations_per_sec = 0.0;
};

class StrategicMCTSPlanner {
public:
    StrategicMCTSPlanner(MCTSConfig config, StrategicConfig simulator_config);

    NativeDecision plan(const StrategicState& state);

private:
    struct SearchNode {
        double prior = 1.0;
        int visits = 0;
        double value_sum = 0.0;
        std::array<SearchNode*, kNumActions> children{};
        std::array<bool, kNumActions> child_present{};
        bool terminal = false;

        SearchNode();
        ~SearchNode();
        SearchNode(const SearchNode&) = delete;
        SearchNode& operator=(const SearchNode&) = delete;

        double mean_value() const;
    };

    MCTSConfig config_;
    StrategicConfig simulator_config_;
    int max_depth_seen_ = 0;

    double simulate(SearchNode& node, StrategicState& state, int depth);
    void expand(SearchNode& node, const StrategicState& state) const;
    int select_child(const SearchNode& node) const;
    int select_final_action(const SearchNode& root, const StrategicState& state) const;
    double rollout_value(const StrategicState& state) const;
    double value(const StrategicState& state) const;
    std::vector<double> priors(const StrategicState& state, const std::vector<int>& actions) const;
};

SmokeResult run_native_mcts_smoke(
    int episodes,
    std::int64_t seed,
    const std::vector<int>& simulations,
    int max_depth,
    int rollout_steps,
    const std::string& prior_mode);

BatchPlanResult plan_batch_from_seeds(
    const std::vector<std::int64_t>& seeds,
    int simulations,
    int max_depth,
    int rollout_steps,
    const std::string& prior_mode);

}  // namespace mini_tft::strategic::native
