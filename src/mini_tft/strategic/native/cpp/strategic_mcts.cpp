#include "strategic_mcts.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mini_tft::strategic::native {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point started) {
    return std::chrono::duration<double>(Clock::now() - started).count();
}

double elapsed_milliseconds(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

EpisodeRow make_episode_row(
    const std::string& policy,
    int episode,
    std::int64_t seed,
    double total_reward,
    int steps,
    const StrategicState& state,
    const StrategicConfig& config,
    int decisions,
    double elapsed_sec,
    int simulations) {
    EpisodeRow row;
    row.policy = policy;
    row.episode = episode;
    row.seed = seed;
    row.placement = placement_proxy(state, config);
    row.final_round = state.round;
    row.hp = state.hp;
    row.final_reason = state.final_reason;
    row.scenario_score = scenario_score(state, config);
    row.illegal_actions = state.total_illegal_actions;
    row.total_reward = total_reward;
    row.steps = steps;
    row.decisions = decisions;
    row.simulations = simulations;
    row.elapsed_sec = elapsed_sec;
    return row;
}

EpisodeRow run_policy_episode(
    const std::string& policy,
    int episode,
    std::int64_t seed,
    const StrategicConfig& config) {
    StrategicState state = reset(seed, config);
    double total_reward = 0.0;
    int steps = 0;
    const int max_steps = config.max_round * (config.max_actions_per_round + 1);
    while (!state.done && steps < max_steps) {
        const auto mask = legal_action_mask(state, config);
        const int action = policy == "random" ? random_action(state, mask) : heuristic_action(state, mask, config);
        const StepResult result = step(state, action, config);
        total_reward += result.reward;
        steps += 1;
    }
    return make_episode_row(policy, episode, seed, total_reward, steps, state, config, 0, 0.0, 0);
}

std::pair<EpisodeRow, std::vector<DecisionRow>> run_mcts_episode(
    const std::string& policy_name,
    int episode,
    std::int64_t seed,
    int simulations,
    const MCTSConfig& mcts_config,
    const StrategicConfig& simulator_config) {
    StrategicState state = reset(seed, simulator_config);
    StrategicMCTSPlanner planner(mcts_config, simulator_config);
    std::vector<DecisionRow> rows;
    rows.reserve(64);
    double total_reward = 0.0;
    int steps = 0;
    const int max_steps = simulator_config.max_round * (simulator_config.max_actions_per_round + 1);
    const auto started = Clock::now();

    while (!state.done && steps < max_steps) {
        const auto mask = legal_action_mask(state, simulator_config);
        const NativeDecision decision = planner.plan(state);
        const int action = decision.selected_action;
        const bool legal =
            action >= 0 && action < kNumActions && mask[static_cast<std::size_t>(action)];
        const StepResult result = step(state, action, simulator_config);
        total_reward += result.reward;

        DecisionRow row;
        row.policy = policy_name;
        row.episode = episode;
        row.seed = seed;
        row.step = steps;
        row.round = state.round;
        row.action_id = action;
        row.legal = legal;
        row.reward = result.reward;
        row.ended_round = result.ended_round;
        row.hp = state.hp;
        row.gold = state.gold;
        row.level = state.level;
        row.placement_proxy = placement_proxy(state, simulator_config);
        row.scenario_score = scenario_score(state, simulator_config);
        row.simulations = simulations;
        row.mcts_elapsed_ms = decision.elapsed_ms;
        row.mcts_max_depth = decision.max_depth;
        row.visit_policy = decision.visit_policy;
        row.action_values = decision.action_values;
        row.action_visits = decision.action_visits;
        row.action_present = decision.action_present;
        rows.push_back(row);
        steps += 1;
    }

    const double elapsed_sec = elapsed_seconds(started);
    return {
        make_episode_row(
            policy_name,
            episode,
            seed,
            total_reward,
            steps,
            state,
            simulator_config,
            static_cast<int>(rows.size()),
            elapsed_sec,
            simulations),
        rows,
    };
}

}  // namespace

StrategicMCTSPlanner::SearchNode::SearchNode() {
    children.fill(nullptr);
    child_present.fill(false);
}

StrategicMCTSPlanner::SearchNode::~SearchNode() {
    for (SearchNode* child : children) {
        delete child;
    }
}

double StrategicMCTSPlanner::SearchNode::mean_value() const {
    if (visits <= 0) {
        return 0.0;
    }
    return value_sum / static_cast<double>(visits);
}

StrategicMCTSPlanner::StrategicMCTSPlanner(
    MCTSConfig config,
    StrategicConfig simulator_config)
    : config_(std::move(config)), simulator_config_(simulator_config) {
    if (config_.simulations <= 0) {
        throw std::invalid_argument("simulations must be positive");
    }
    if (config_.max_depth <= 0) {
        throw std::invalid_argument("max_depth must be positive");
    }
    if (config_.rollout_steps < 0) {
        throw std::invalid_argument("rollout_steps must be non-negative");
    }
    if (config_.gamma <= 0.0) {
        throw std::invalid_argument("gamma must be positive");
    }
}

NativeDecision StrategicMCTSPlanner::plan(const StrategicState& state) {
    const auto started = Clock::now();
    SearchNode root;
    max_depth_seen_ = 0;
    expand(root, state);

    int simulations_run = 0;
    for (int index = 0; index < config_.simulations; ++index) {
        StrategicState sim_state = state;
        simulate(root, sim_state, 0);
        simulations_run += 1;
    }

    NativeDecision decision;
    decision.selected_action = select_final_action(root, state);
    decision.simulations_run = simulations_run;
    decision.elapsed_ms = elapsed_milliseconds(started);
    decision.max_depth = max_depth_seen_;

    double visit_total = 0.0;
    for (int action = 0; action < kNumActions; ++action) {
        const SearchNode* child = root.children[static_cast<std::size_t>(action)];
        if (child == nullptr) {
            continue;
        }
        decision.action_present[static_cast<std::size_t>(action)] = true;
        decision.action_visits[static_cast<std::size_t>(action)] = child->visits;
        decision.action_values[static_cast<std::size_t>(action)] = child->mean_value();
        decision.visit_policy[static_cast<std::size_t>(action)] = child->visits;
        visit_total += static_cast<double>(child->visits);
    }
    if (visit_total > 0.0) {
        for (double& value : decision.visit_policy) {
            value /= visit_total;
        }
    }
    return decision;
}

double StrategicMCTSPlanner::simulate(SearchNode& node, StrategicState& state, int depth) {
    max_depth_seen_ = std::max(max_depth_seen_, depth);
    if (state.done) {
        const double terminal_value = value(state);
        node.visits += 1;
        node.value_sum += terminal_value;
        node.terminal = true;
        return terminal_value;
    }
    if (depth >= config_.max_depth) {
        const double leaf_value = rollout_value(state);
        node.visits += 1;
        node.value_sum += leaf_value;
        return leaf_value;
    }

    bool has_children = false;
    for (bool present : node.child_present) {
        has_children = has_children || present;
    }
    if (!has_children) {
        expand(node, state);
        const double leaf_value = rollout_value(state);
        node.visits += 1;
        node.value_sum += leaf_value;
        return leaf_value;
    }

    const int action = select_child(node);
    SearchNode* child = node.children[static_cast<std::size_t>(action)];
    StepResult result = step(state, action, simulator_config_);
    double backed_value = result.reward;
    if (!(result.terminated || result.truncated)) {
        backed_value += config_.gamma * simulate(*child, state, depth + 1);
    } else {
        backed_value += config_.gamma * value(state);
        child->visits += 1;
        child->value_sum += backed_value;
        child->terminal = true;
    }
    node.visits += 1;
    node.value_sum += backed_value;
    return backed_value;
}

void StrategicMCTSPlanner::expand(SearchNode& node, const StrategicState& state) const {
    const auto actions = legal_actions(legal_action_mask(state, simulator_config_));
    const std::vector<double> priors_for_actions = priors(state, actions);
    for (std::size_t index = 0; index < actions.size(); ++index) {
        const int action = actions[index];
        auto* child = new SearchNode();
        child->prior = priors_for_actions[index];
        node.children[static_cast<std::size_t>(action)] = child;
        node.child_present[static_cast<std::size_t>(action)] = true;
    }
}

int StrategicMCTSPlanner::select_child(const SearchNode& node) const {
    const int parent_visits = std::max(1, node.visits);
    const double log_parent = std::log(static_cast<double>(parent_visits) + 1.0);
    double best_score = -std::numeric_limits<double>::infinity();
    int best_action = kHold;
    for (int action = 0; action < kNumActions; ++action) {
        if (!node.child_present[static_cast<std::size_t>(action)]) {
            continue;
        }
        const SearchNode* child = node.children[static_cast<std::size_t>(action)];
        const double exploit = child->mean_value();
        const double explore = config_.exploration * child->prior
            * std::sqrt(log_parent / static_cast<double>(1 + child->visits));
        const double score = exploit + explore;
        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }
    return best_action;
}

int StrategicMCTSPlanner::select_final_action(
    const SearchNode& root,
    const StrategicState& state) const {
    bool has_children = false;
    int best_action = kHold;
    int best_visits = -1;
    double best_value = -std::numeric_limits<double>::infinity();
    for (int action = 0; action < kNumActions; ++action) {
        if (!root.child_present[static_cast<std::size_t>(action)]) {
            continue;
        }
        has_children = true;
        const SearchNode* child = root.children[static_cast<std::size_t>(action)];
        const int visits = child->visits;
        const double mean = child->mean_value();
        if (visits > best_visits || (visits == best_visits && mean > best_value)) {
            best_action = action;
            best_visits = visits;
            best_value = mean;
        }
    }
    if (has_children) {
        return best_action;
    }
    const auto actions = legal_actions(legal_action_mask(state, simulator_config_));
    return actions.empty() ? kHold : actions.front();
}

double StrategicMCTSPlanner::rollout_value(const StrategicState& state) const {
    StrategicState rollout_state = state;
    double rollout = 0.0;
    double discount = 1.0;
    for (int index = 0; index < config_.rollout_steps; ++index) {
        if (rollout_state.done) {
            break;
        }
        const auto mask = legal_action_mask(rollout_state, simulator_config_);
        const int action = heuristic_action(rollout_state, mask, simulator_config_);
        const StepResult result = step(rollout_state, action, simulator_config_);
        rollout += discount * result.reward;
        discount *= config_.gamma;
        if (result.terminated || result.truncated) {
            break;
        }
    }
    return rollout + discount * value(rollout_state);
}

double StrategicMCTSPlanner::value(const StrategicState& state) const {
    const double score = scenario_score(state, simulator_config_);
    const int placement = placement_proxy(state, simulator_config_);
    const double placement_score = (8.0 - static_cast<double>(placement)) / 7.0;
    return 0.65 * score + 0.35 * placement_score;
}

std::vector<double> StrategicMCTSPlanner::priors(
    const StrategicState& state,
    const std::vector<int>& actions) const {
    std::vector<double> output;
    output.reserve(actions.size());
    if (actions.empty()) {
        return output;
    }
    if (config_.prior_mode == "heuristic") {
        const auto mask = legal_action_mask(state, simulator_config_);
        const int policy_action = heuristic_action(state, mask, simulator_config_);
        const double base = 0.15 / static_cast<double>(actions.size());
        double total = 0.0;
        for (int action : actions) {
            double prior = base;
            if (action == policy_action) {
                prior += 0.85;
            }
            output.push_back(prior);
            total += prior;
        }
        if (total > 0.0) {
            for (double& prior : output) {
                prior /= total;
            }
        }
        return output;
    }
    const double uniform = 1.0 / static_cast<double>(actions.size());
    output.assign(actions.size(), uniform);
    return output;
}

SmokeResult run_native_mcts_smoke(
    int episodes,
    std::int64_t seed,
    const std::vector<int>& simulations,
    int max_depth,
    int rollout_steps,
    const std::string& prior_mode) {
    if (episodes <= 0) {
        throw std::invalid_argument("episodes must be positive");
    }
    if (simulations.empty()) {
        throw std::invalid_argument("at least one simulation count is required");
    }
    const auto started = Clock::now();
    const StrategicConfig simulator_config;
    SmokeResult result;
    result.episode_rows.reserve(static_cast<std::size_t>(episodes * (2 + simulations.size())));

    for (const std::string& policy : {"random", "heuristic"}) {
        for (int episode = 0; episode < episodes; ++episode) {
            const std::int64_t episode_seed = seed + episode;
            result.episode_rows.push_back(
                run_policy_episode(policy, episode, episode_seed, simulator_config));
        }
    }

    for (int simulation_count : simulations) {
        MCTSConfig mcts_config;
        mcts_config.simulations = simulation_count;
        mcts_config.max_depth = max_depth;
        mcts_config.rollout_steps = rollout_steps;
        mcts_config.prior_mode = prior_mode;
        const std::string policy_name = "mcts_" + std::to_string(simulation_count);
        for (int episode = 0; episode < episodes; ++episode) {
            const std::int64_t episode_seed = seed + episode;
            auto [episode_row, decision_rows] = run_mcts_episode(
                policy_name,
                episode,
                episode_seed,
                simulation_count,
                mcts_config,
                simulator_config);
            result.episode_rows.push_back(episode_row);
            result.decision_rows.insert(
                result.decision_rows.end(),
                decision_rows.begin(),
                decision_rows.end());
        }
    }
    result.elapsed_sec = elapsed_seconds(started);
    return result;
}

BatchPlanResult plan_batch_from_seeds(
    const std::vector<std::int64_t>& seeds,
    int simulations,
    int max_depth,
    int rollout_steps,
    const std::string& prior_mode) {
    const auto started = Clock::now();
    const StrategicConfig simulator_config;
    MCTSConfig mcts_config;
    mcts_config.simulations = simulations;
    mcts_config.max_depth = max_depth;
    mcts_config.rollout_steps = rollout_steps;
    mcts_config.prior_mode = prior_mode;
    StrategicMCTSPlanner planner(mcts_config, simulator_config);

    BatchPlanResult result;
    result.selected_actions.reserve(seeds.size());
    result.visit_policies.reserve(seeds.size());
    result.values.reserve(seeds.size());
    for (std::int64_t seed : seeds) {
        const StrategicState state = reset(seed, simulator_config);
        const NativeDecision decision = planner.plan(state);
        result.selected_actions.push_back(decision.selected_action);
        result.visit_policies.push_back(decision.visit_policy);
        result.values.push_back(decision.action_values[static_cast<std::size_t>(decision.selected_action)]);
    }
    result.elapsed_sec = elapsed_seconds(started);
    const double total_simulations = static_cast<double>(seeds.size()) * static_cast<double>(simulations);
    result.simulations_per_sec = result.elapsed_sec > 0.0 ? total_simulations / result.elapsed_sec : 0.0;
    return result;
}

}  // namespace mini_tft::strategic::native
