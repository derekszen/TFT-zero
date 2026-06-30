#include <Python.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "strategic_mcts.hpp"
#include "strategic_rules.hpp"
#include "strategic_state.hpp"

namespace {

using mini_tft::strategic::native::BatchPlanResult;
using mini_tft::strategic::native::DecisionRow;
using mini_tft::strategic::native::EpisodeRow;
using mini_tft::strategic::native::FinalReason;
using mini_tft::strategic::native::MCTSConfig;
using mini_tft::strategic::native::SmokeResult;
using mini_tft::strategic::native::StrategicConfig;
using mini_tft::strategic::native::StrategicState;
using mini_tft::strategic::native::action_name;
using mini_tft::strategic::native::final_reason_name;
using mini_tft::strategic::native::kFinalNone;
using mini_tft::strategic::native::kNumActions;
using mini_tft::strategic::native::plan_batch_from_seeds;
using mini_tft::strategic::native::reset;
using mini_tft::strategic::native::run_native_mcts_smoke;
using mini_tft::strategic::native::step;

bool set_new(PyObject* dict, const char* key, PyObject* value) {
    if (value == nullptr) {
        return false;
    }
    const int result = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return result == 0;
}

PyObject* py_final_reason(int final_reason) {
    if (final_reason == kFinalNone) {
        Py_RETURN_NONE;
    }
    return PyUnicode_FromString(final_reason_name(final_reason));
}

template <std::size_t N>
PyObject* int_tuple(const std::array<int, N>& values) {
    PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(N));
    if (tuple == nullptr) {
        return nullptr;
    }
    for (std::size_t index = 0; index < N; ++index) {
        PyTuple_SET_ITEM(tuple, static_cast<Py_ssize_t>(index), PyLong_FromLong(values[index]));
    }
    return tuple;
}

template <std::size_t N>
PyObject* double_list(const std::array<double, N>& values) {
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(N));
    if (list == nullptr) {
        return nullptr;
    }
    for (std::size_t index = 0; index < N; ++index) {
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(index), PyFloat_FromDouble(values[index]));
    }
    return list;
}

template <std::size_t N>
PyObject* action_int_dict(const std::array<int, N>& values, const std::array<bool, N>& present) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }
    for (std::size_t action = 0; action < N; ++action) {
        if (!present[action]) {
            continue;
        }
        PyObject* value = PyLong_FromLong(values[action]);
        if (value == nullptr || PyDict_SetItemString(dict, action_name(static_cast<int>(action)), value) != 0) {
            Py_XDECREF(value);
            Py_DECREF(dict);
            return nullptr;
        }
        Py_DECREF(value);
    }
    return dict;
}

template <std::size_t N>
PyObject* action_double_dict(const std::array<double, N>& values, const std::array<bool, N>& present) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }
    for (std::size_t action = 0; action < N; ++action) {
        if (!present[action]) {
            continue;
        }
        PyObject* value = PyFloat_FromDouble(values[action]);
        if (value == nullptr || PyDict_SetItemString(dict, action_name(static_cast<int>(action)), value) != 0) {
            Py_XDECREF(value);
            Py_DECREF(dict);
            return nullptr;
        }
        Py_DECREF(value);
    }
    return dict;
}

double round6(double value) {
    return std::round(value * 1000000.0) / 1000000.0;
}

PyObject* state_signature(const StrategicState& state) {
    PyObject* tuple = PyTuple_New(24);
    if (tuple == nullptr) {
        return nullptr;
    }
    PyTuple_SET_ITEM(tuple, 0, PyLong_FromLongLong(state.seed));
    PyTuple_SET_ITEM(tuple, 1, PyLong_FromUnsignedLongLong(state.rng_key));
    PyTuple_SET_ITEM(tuple, 2, PyLong_FromLong(state.round));
    PyTuple_SET_ITEM(tuple, 3, PyLong_FromLong(state.hp));
    PyTuple_SET_ITEM(tuple, 4, PyLong_FromLong(state.gold));
    PyTuple_SET_ITEM(tuple, 5, PyLong_FromLong(state.level));
    PyTuple_SET_ITEM(tuple, 6, PyLong_FromLong(state.xp));
    PyTuple_SET_ITEM(tuple, 7, int_tuple(state.shop));
    PyTuple_SET_ITEM(tuple, 8, int_tuple(state.owned));
    PyTuple_SET_ITEM(tuple, 9, int_tuple(state.fielded));
    PyTuple_SET_ITEM(tuple, 10, int_tuple(state.role_items));
    PyTuple_SET_ITEM(tuple, 11, int_tuple(state.role_item_slots));
    PyTuple_SET_ITEM(tuple, 12, PyBool_FromLong(state.done ? 1 : 0));
    PyTuple_SET_ITEM(tuple, 13, py_final_reason(state.final_reason));
    PyTuple_SET_ITEM(tuple, 14, PyLong_FromLong(state.action_count));
    PyTuple_SET_ITEM(tuple, 15, PyFloat_FromDouble(round6(state.last_board_strength)));
    PyTuple_SET_ITEM(tuple, 16, PyFloat_FromDouble(round6(state.last_enemy_strength)));
    PyTuple_SET_ITEM(tuple, 17, PyLong_FromLong(state.last_damage));
    PyTuple_SET_ITEM(tuple, 18, PyBool_FromLong(state.last_win ? 1 : 0));
    PyTuple_SET_ITEM(tuple, 19, PyLong_FromLong(state.total_rolls));
    PyTuple_SET_ITEM(tuple, 20, PyLong_FromLong(state.total_xp_buys));
    PyTuple_SET_ITEM(tuple, 21, PyLong_FromLong(state.total_units_bought));
    PyTuple_SET_ITEM(tuple, 22, PyLong_FromLong(state.total_item_slams));
    PyTuple_SET_ITEM(tuple, 23, PyLong_FromLong(state.total_illegal_actions));
    return tuple;
}

std::vector<int> parse_int_sequence(PyObject* value, const char* name) {
    PyObject* sequence = PySequence_Fast(value, name);
    if (sequence == nullptr) {
        throw std::invalid_argument(name);
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    std::vector<int> output;
    output.reserve(static_cast<std::size_t>(size));
    PyObject** items = PySequence_Fast_ITEMS(sequence);
    for (Py_ssize_t index = 0; index < size; ++index) {
        const long parsed = PyLong_AsLong(items[index]);
        if (PyErr_Occurred()) {
            Py_DECREF(sequence);
            throw std::invalid_argument(name);
        }
        output.push_back(static_cast<int>(parsed));
    }
    Py_DECREF(sequence);
    return output;
}

std::vector<std::int64_t> parse_seed_sequence(PyObject* value, const char* name) {
    PyObject* sequence = PySequence_Fast(value, name);
    if (sequence == nullptr) {
        throw std::invalid_argument(name);
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    std::vector<std::int64_t> output;
    output.reserve(static_cast<std::size_t>(size));
    PyObject** items = PySequence_Fast_ITEMS(sequence);
    for (Py_ssize_t index = 0; index < size; ++index) {
        const long long parsed = PyLong_AsLongLong(items[index]);
        if (PyErr_Occurred()) {
            Py_DECREF(sequence);
            throw std::invalid_argument(name);
        }
        output.push_back(static_cast<std::int64_t>(parsed));
    }
    Py_DECREF(sequence);
    return output;
}

PyObject* py_episode_row(const EpisodeRow& row) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }
    if (!set_new(dict, "policy", PyUnicode_FromString(row.policy.c_str()))
        || !set_new(dict, "episode", PyLong_FromLong(row.episode))
        || !set_new(dict, "seed", PyLong_FromLongLong(row.seed))
        || !set_new(dict, "placement", PyLong_FromLong(row.placement))
        || !set_new(dict, "final_round", PyLong_FromLong(row.final_round))
        || !set_new(dict, "hp", PyLong_FromLong(row.hp))
        || !set_new(dict, "final_reason", py_final_reason(row.final_reason))
        || !set_new(dict, "scenario_score", PyFloat_FromDouble(row.scenario_score))
        || !set_new(dict, "illegal_actions", PyLong_FromLong(row.illegal_actions))
        || !set_new(dict, "total_reward", PyFloat_FromDouble(row.total_reward))
        || !set_new(dict, "steps", PyLong_FromLong(row.steps))
        || !set_new(dict, "decisions", PyLong_FromLong(row.decisions))
        || !set_new(dict, "simulations", PyLong_FromLong(row.simulations))
        || !set_new(dict, "elapsed_sec", PyFloat_FromDouble(row.elapsed_sec))) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

PyObject* py_decision_row(const DecisionRow& row) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }
    if (!set_new(dict, "policy", PyUnicode_FromString(row.policy.c_str()))
        || !set_new(dict, "episode", PyLong_FromLong(row.episode))
        || !set_new(dict, "seed", PyLong_FromLongLong(row.seed))
        || !set_new(dict, "step", PyLong_FromLong(row.step))
        || !set_new(dict, "round", PyLong_FromLong(row.round))
        || !set_new(dict, "action_id", PyLong_FromLong(row.action_id))
        || !set_new(dict, "action", PyUnicode_FromString(action_name(row.action_id)))
        || !set_new(dict, "legal", PyBool_FromLong(row.legal ? 1 : 0))
        || !set_new(dict, "reward", PyFloat_FromDouble(row.reward))
        || !set_new(dict, "ended_round", PyBool_FromLong(row.ended_round ? 1 : 0))
        || !set_new(dict, "hp", PyLong_FromLong(row.hp))
        || !set_new(dict, "gold", PyLong_FromLong(row.gold))
        || !set_new(dict, "level", PyLong_FromLong(row.level))
        || !set_new(dict, "placement_proxy", PyLong_FromLong(row.placement_proxy))
        || !set_new(dict, "scenario_score", PyFloat_FromDouble(row.scenario_score))
        || !set_new(dict, "simulations", PyLong_FromLong(row.simulations))
        || !set_new(dict, "mcts_elapsed_ms", PyFloat_FromDouble(row.mcts_elapsed_ms))
        || !set_new(dict, "mcts_max_depth", PyLong_FromLong(row.mcts_max_depth))
        || !set_new(dict, "visit_policy", double_list(row.visit_policy))
        || !set_new(dict, "action_visits", action_int_dict(row.action_visits, row.action_present))
        || !set_new(dict, "action_values", action_double_dict(row.action_values, row.action_present))) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

PyObject* py_trace_script(PyObject*, PyObject* args, PyObject* kwargs) {
    long long seed = 0;
    PyObject* actions_obj = nullptr;
    int max_round = 36;
    int max_actions_per_round = 3;
    static const char* keywords[] = {
        "seed",
        "actions",
        "max_round",
        "max_actions_per_round",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "LO|ii",
            const_cast<char**>(keywords),
            &seed,
            &actions_obj,
            &max_round,
            &max_actions_per_round)) {
        return nullptr;
    }

    try {
        const std::vector<int> actions = parse_int_sequence(actions_obj, "actions must be a sequence");
        StrategicConfig config;
        config.max_round = max_round;
        config.max_actions_per_round = max_actions_per_round;
        StrategicState state = reset(seed, config);
        PyObject* rows = PyList_New(0);
        if (rows == nullptr) {
            return nullptr;
        }
        PyObject* signature = state_signature(state);
        if (signature == nullptr || PyList_Append(rows, signature) != 0) {
            Py_XDECREF(signature);
            Py_DECREF(rows);
            return nullptr;
        }
        Py_DECREF(signature);
        for (int action : actions) {
            if (state.done) {
                break;
            }
            step(state, action, config);
            signature = state_signature(state);
            if (signature == nullptr || PyList_Append(rows, signature) != 0) {
                Py_XDECREF(signature);
                Py_DECREF(rows);
                return nullptr;
            }
            Py_DECREF(signature);
        }
        return rows;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyObject* py_run_mcts_smoke(PyObject*, PyObject* args, PyObject* kwargs) {
    int episodes = 0;
    long long seed = 0;
    PyObject* simulations_obj = nullptr;
    int max_depth = 0;
    int rollout_steps = 0;
    const char* prior_mode = "uniform";
    static const char* keywords[] = {
        "episodes",
        "seed",
        "simulations",
        "max_depth",
        "rollout_steps",
        "prior_mode",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "iLOii|s",
            const_cast<char**>(keywords),
            &episodes,
            &seed,
            &simulations_obj,
            &max_depth,
            &rollout_steps,
            &prior_mode)) {
        return nullptr;
    }

    try {
        const std::vector<int> simulations =
            parse_int_sequence(simulations_obj, "simulations must be a sequence");
        const SmokeResult result = run_native_mcts_smoke(
            episodes,
            seed,
            simulations,
            max_depth,
            rollout_steps,
            prior_mode);

        PyObject* dict = PyDict_New();
        PyObject* episode_rows = PyList_New(static_cast<Py_ssize_t>(result.episode_rows.size()));
        PyObject* decision_rows = PyList_New(static_cast<Py_ssize_t>(result.decision_rows.size()));
        if (dict == nullptr || episode_rows == nullptr || decision_rows == nullptr) {
            Py_XDECREF(dict);
            Py_XDECREF(episode_rows);
            Py_XDECREF(decision_rows);
            return nullptr;
        }
        for (std::size_t index = 0; index < result.episode_rows.size(); ++index) {
            PyObject* row = py_episode_row(result.episode_rows[index]);
            if (row == nullptr) {
                Py_DECREF(dict);
                Py_DECREF(episode_rows);
                Py_DECREF(decision_rows);
                return nullptr;
            }
            PyList_SET_ITEM(episode_rows, static_cast<Py_ssize_t>(index), row);
        }
        for (std::size_t index = 0; index < result.decision_rows.size(); ++index) {
            PyObject* row = py_decision_row(result.decision_rows[index]);
            if (row == nullptr) {
                Py_DECREF(dict);
                Py_DECREF(episode_rows);
                Py_DECREF(decision_rows);
                return nullptr;
            }
            PyList_SET_ITEM(decision_rows, static_cast<Py_ssize_t>(index), row);
        }

        if (!set_new(dict, "episode_rows", episode_rows)
            || !set_new(dict, "decision_rows", decision_rows)
            || !set_new(dict, "elapsed_sec", PyFloat_FromDouble(result.elapsed_sec))) {
            Py_DECREF(dict);
            return nullptr;
        }
        return dict;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyObject* py_plan_batch_from_seeds(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* seeds_obj = nullptr;
    int simulations = 0;
    int max_depth = 0;
    int rollout_steps = 0;
    const char* prior_mode = "uniform";
    static const char* keywords[] = {
        "seeds",
        "simulations",
        "max_depth",
        "rollout_steps",
        "prior_mode",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "Oiii|s",
            const_cast<char**>(keywords),
            &seeds_obj,
            &simulations,
            &max_depth,
            &rollout_steps,
            &prior_mode)) {
        return nullptr;
    }

    try {
        const std::vector<std::int64_t> seeds =
            parse_seed_sequence(seeds_obj, "seeds must be a sequence");
        const BatchPlanResult result =
            plan_batch_from_seeds(seeds, simulations, max_depth, rollout_steps, prior_mode);

        PyObject* dict = PyDict_New();
        PyObject* selected = PyList_New(static_cast<Py_ssize_t>(result.selected_actions.size()));
        PyObject* policies = PyList_New(static_cast<Py_ssize_t>(result.visit_policies.size()));
        PyObject* values = PyList_New(static_cast<Py_ssize_t>(result.values.size()));
        if (dict == nullptr || selected == nullptr || policies == nullptr || values == nullptr) {
            Py_XDECREF(dict);
            Py_XDECREF(selected);
            Py_XDECREF(policies);
            Py_XDECREF(values);
            return nullptr;
        }
        for (std::size_t index = 0; index < result.selected_actions.size(); ++index) {
            PyList_SET_ITEM(
                selected,
                static_cast<Py_ssize_t>(index),
                PyLong_FromLong(result.selected_actions[index]));
            PyList_SET_ITEM(
                policies,
                static_cast<Py_ssize_t>(index),
                double_list(result.visit_policies[index]));
            PyList_SET_ITEM(
                values,
                static_cast<Py_ssize_t>(index),
                PyFloat_FromDouble(result.values[index]));
        }
        if (!set_new(dict, "selected_actions", selected)
            || !set_new(dict, "visit_policies", policies)
            || !set_new(dict, "values", values)
            || !set_new(dict, "elapsed_sec", PyFloat_FromDouble(result.elapsed_sec))
            || !set_new(dict, "simulations_per_sec", PyFloat_FromDouble(result.simulations_per_sec))) {
            Py_DECREF(dict);
            return nullptr;
        }
        return dict;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {
        "trace_script",
        reinterpret_cast<PyCFunction>(py_trace_script),
        METH_VARARGS | METH_KEYWORDS,
        "Return Python-compatible state signatures after reset and scripted actions.",
    },
    {
        "run_mcts_smoke",
        reinterpret_cast<PyCFunction>(py_run_mcts_smoke),
        METH_VARARGS | METH_KEYWORDS,
        "Run the compiled simulator-backed strategic MCTS smoke.",
    },
    {
        "plan_batch_from_seeds",
        reinterpret_cast<PyCFunction>(py_plan_batch_from_seeds),
        METH_VARARGS | METH_KEYWORDS,
        "Plan one native MCTS decision for each reset seed.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Compiled strategic-lane simulator and MCTS backend.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__native() {
    return PyModule_Create(&module);
}
