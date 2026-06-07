const ACTION = {
  END_TURN: 0,
  ROLL: 1,
  BUY_XP: 2,
  BUY_SHOP_0: 3,
  SELL_BENCH_0: 8,
  FIELD_BEST_BOARD: 17,
  SLAM_BEST_ITEM: 18,
};

const COST_NAMES = ["", "one", "two", "three", "four", "five"];
let state = null;
let autoplay = null;
let stepMode = true;

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status}`);
  }
  return response.json();
}

async function loadState() {
  state = await api("/api/state");
  render();
}

async function postAction(action) {
  stopAutoplay();
  state = await api("/api/action", {
    method: "POST",
    body: JSON.stringify({ action }),
  });
  render();
}

async function botStep() {
  state = await api("/api/bot-step", { method: "POST", body: "{}" });
  render();
  if (state.status.done) {
    stopAutoplay();
  }
}

async function moveUnit(fromZone, fromIndex, toZone, toIndex) {
  stopAutoplay();
  state = await api("/api/move-unit", {
    method: "POST",
    body: JSON.stringify({
      from_zone: fromZone,
      from_index: fromIndex,
      to_zone: toZone,
      to_index: toIndex,
    }),
  });
  render();
}

async function reset() {
  stopAutoplay();
  const seed = state?.seed ?? 0;
  state = await api("/api/reset", {
    method: "POST",
    body: JSON.stringify({ seed }),
  });
  render();
}

function actionLegal(action) {
  return Boolean(state?.actions?.find((item) => item.id === action)?.legal);
}

function render() {
  if (!state) return;

  document.getElementById("stage").textContent = state.status.stage_label;
  document.getElementById("phase").textContent = state.status.done ? "Done" : "Planning";
  document.getElementById("timer").textContent = state.status.done ? "--" : "30";
  document.getElementById("capacity").textContent = `${boardCount()} / ${state.status.level}`;
  document.getElementById("hpText").textContent = `${state.status.hp} HP`;
  document.getElementById("level").textContent = `Lv. ${state.status.level}`;
  document.getElementById("xp").textContent = `${state.status.xp}/${state.status.xp_needed}`;
  document.getElementById("gold").textContent = state.status.gold;
  document.getElementById("strength").textContent = `Power ${state.status.strength}`;
  document.getElementById("enemyNext").textContent = `Enemy ${state.status.enemy_next}`;

  renderTraits();
  renderEnemy();
  renderBoard();
  renderBench();
  renderShop();
  renderItems();
  renderLog();
  renderScoreboard();
  bindActionButtons();
}

function boardCount() {
  return state.board.filter(Boolean).length;
}

function renderEnemy() {
  const root = document.getElementById("enemySquad");
  const gate = document.getElementById("enemyGate");
  root.innerHTML = "";
  const level = state.enemy.display_level ? ` · Lv. ${state.enemy.display_level}` : "";
  gate.textContent = `${state.enemy.label} · ${state.enemy.unit_count} units${level} · ${state.enemy.strength}`;
  for (const slot of state.enemy.slots) {
    const token = document.createElement("div");
    token.className = `enemy-token tier-${slot.tier}`;
    token.innerHTML = `
      <div class="enemy-orb">${slot.tier}</div>
      <div class="enemy-name">${slot.name}</div>
    `;
    root.append(token);
  }
}

function renderTraits() {
  const root = document.getElementById("traits");
  root.innerHTML = "";
  if (!state.traits.length) {
    root.append(emptyState("No traits"));
    return;
  }
  for (const trait of state.traits) {
    const row = document.createElement("div");
    row.className = `trait-row ${trait.active ? "active" : ""}`;
    const progress = trait.active_breakpoint ?? trait.next_breakpoint ?? trait.breakpoints[0];
    row.innerHTML = `
      <div class="trait-icon">${trait.label.slice(0, 1)}</div>
      <div class="trait-copy">
        <strong>${trait.label}</strong>
        <span>${trait.count} / ${progress}</span>
      </div>
    `;
    root.append(row);
  }
}

function renderBoard() {
  const root = document.getElementById("board");
  root.innerHTML = "";
  for (let index = 0; index < 9; index += 1) {
    const cell = document.createElement("div");
    cell.className = "hex-cell drop-slot";
    cell.dataset.zone = "board";
    cell.dataset.index = String(index);
    const unit = state.board[index];
    if (unit) {
      cell.append(unitToken(unit, "board"));
      makeDraggable(cell, "board", index);
      cell.classList.add("occupied");
    } else {
      cell.classList.add("empty");
    }
    makeDropTarget(cell, "board", index);
    root.append(cell);
  }
}

function renderBench() {
  const root = document.getElementById("bench");
  root.innerHTML = "";
  for (let index = 0; index < state.bench.length; index += 1) {
    const slot = document.createElement("button");
    slot.type = "button";
    slot.className = "bench-slot drop-slot";
    slot.dataset.zone = "bench";
    slot.dataset.index = String(index);
    const unit = state.bench[index];
    if (unit) {
      slot.append(unitToken(unit, "bench"));
      slot.title = `Drag ${unit.name} to the board. Shift-click or right-click to sell.`;
      slot.addEventListener("click", (event) => {
        if (event.shiftKey && actionLegal(ACTION.SELL_BENCH_0 + index)) {
          postAction(ACTION.SELL_BENCH_0 + index);
        }
      });
      slot.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        if (actionLegal(ACTION.SELL_BENCH_0 + index)) {
          postAction(ACTION.SELL_BENCH_0 + index);
        }
      });
      makeDraggable(slot, "bench", index);
      slot.classList.add("occupied");
    } else {
      slot.innerHTML = "<span>Empty</span>";
      slot.classList.add("empty");
    }
    makeDropTarget(slot, "bench", index);
    root.append(slot);
  }
}

function renderShop() {
  const root = document.getElementById("shop");
  root.innerHTML = "";
  for (let index = 0; index < state.shop.length; index += 1) {
    const unit = state.shop[index];
    const card = document.createElement("button");
    card.type = "button";
    card.className = "shop-card";
    if (!unit) {
      card.classList.add("empty");
      card.disabled = true;
      card.innerHTML = "<span>Sold</span>";
      root.append(card);
      continue;
    }
    card.classList.add(`cost-${COST_NAMES[unit.cost]}`);
    card.disabled = !actionLegal(ACTION.BUY_SHOP_0 + index);
    card.addEventListener("click", () => postAction(ACTION.BUY_SHOP_0 + index));
    card.innerHTML = `
      <div class="shop-art">${unit.image ? `<img src="${unit.image}" alt="" />` : initials(unit.name)}</div>
      <div class="shop-meta">
        <div class="unit-name">${unit.name}</div>
        <div class="unit-traits">${unit.traits.join(" / ")}</div>
        <div class="cost">${unit.cost}g</div>
      </div>
    `;
    root.append(card);
  }
}

function renderItems() {
  const root = document.getElementById("items");
  root.innerHTML = "";
  const itemAction = state.item_action || {};
  const label = document.createElement("span");
  label.className = "items-label";
  label.textContent = "Items";
  label.title = itemAction.detail || "";
  root.append(label);
  if (!state.items.length) {
    root.append(emptyState("None"));
    return;
  }
  for (const item of state.items) {
    const chip = document.createElement("span");
    chip.className = `item-chip ${item.kind}`;
    chip.textContent = item.name;
    chip.title = itemChipTitle(item);
    if (itemAction.legal) {
      chip.classList.add("actionable");
      chip.setAttribute("role", "button");
      chip.tabIndex = 0;
      chip.addEventListener("click", () => postAction(ACTION.SLAM_BEST_ITEM));
      chip.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          postAction(ACTION.SLAM_BEST_ITEM);
        }
      });
    }
    root.append(chip);
  }
}

function itemChipTitle(item) {
  const tags = item.tags?.length ? ` · ${item.tags.join("/")}` : "";
  const effects = Object.entries(item.effects || {})
    .map(([key, value]) => `${key.replaceAll("_", " ")} ${value}`)
    .join(", ");
  if (item.kind === "component") {
    return `Component${tags}`;
  }
  return effects ? `Completed${tags} · ${effects}` : `Completed${tags}`;
}

function renderLog() {
  const root = document.getElementById("log");
  root.innerHTML = "";
  const latest = state.status.done && state.summary
    ? [`Final HP ${state.summary.final_hp}; reason ${state.summary.final_reason}`, ...state.log]
    : state.log;
  for (const line of latest.slice(0, 6)) {
    const item = document.createElement("div");
    item.className = "log-line";
    item.textContent = line;
    root.append(item);
  }
}

function renderScoreboard() {
  const root = document.getElementById("scoreboard");
  root.innerHTML = "";
  const rows = [
    ["Curve", state.status.enemy_next],
    ["Power", state.status.strength],
    ["Penalty", state.status.enemy_power_penalty],
    ["Step", state.status.step_count],
    ["Reward", state.last.reward],
  ];
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    root.append(row);
  }
}

function bindActionButtons() {
  setButton("buyXp", ACTION.BUY_XP);
  setButton("roll", ACTION.ROLL);
  setButton("fieldBest", ACTION.FIELD_BEST_BOARD);
  setButton(
    "slamItem",
    ACTION.SLAM_BEST_ITEM,
    state.item_action?.label || "Slam Item",
    state.item_action?.detail || "",
  );
  setButton("endTurn", ACTION.END_TURN);
  document.getElementById("botStep").disabled = state.status.done;
  renderPlayControls();
}

function setButton(id, action, label = null, title = "") {
  const button = document.getElementById(id);
  if (label !== null) {
    button.textContent = label;
  }
  button.title = title;
  button.disabled = !actionLegal(action);
  button.onclick = () => postAction(action);
}

function makeDraggable(element, zone, index) {
  element.draggable = true;
  element.addEventListener("dragstart", (event) => {
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("application/json", JSON.stringify({ zone, index }));
    element.classList.add("dragging");
  });
  element.addEventListener("dragend", () => {
    element.classList.remove("dragging");
  });
}

function makeDropTarget(element, zone, index) {
  element.addEventListener("dragover", (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  });
  element.addEventListener("dragenter", () => element.classList.add("drop-hover"));
  element.addEventListener("dragleave", () => element.classList.remove("drop-hover"));
  element.addEventListener("drop", (event) => {
    event.preventDefault();
    element.classList.remove("drop-hover");
    const raw = event.dataTransfer.getData("application/json");
    if (!raw) return;
    const source = JSON.parse(raw);
    if (source.zone === zone && source.index === index) return;
    moveUnit(source.zone, source.index, zone, index).catch((error) => console.error(error));
  });
}

function unitToken(unit, size) {
  const wrapper = document.createElement("div");
  wrapper.className = `unit-token ${size} cost-${COST_NAMES[unit.cost]}`;
  const stars = "*".repeat(unit.stars ?? 1);
  wrapper.innerHTML = `
    <div class="portrait">${unit.image ? `<img src="${unit.image}" alt="" />` : initials(unit.name)}</div>
    <div class="healthbar"><span></span></div>
    <div class="token-name">${unit.name}</div>
    <div class="stars">${stars}</div>
  `;
  return wrapper;
}

function initials(name) {
  return `<span>${name.split(" ").map((part) => part[0]).join("").slice(0, 2)}</span>`;
}

function emptyState(text) {
  const item = document.createElement("div");
  item.className = "empty-state";
  item.textContent = text;
  return item;
}

function stopAutoplay() {
  if (autoplay) {
    clearInterval(autoplay);
    autoplay = null;
  }
  renderPlayControls();
}

function toggleAutoplay() {
  if (autoplay) {
    stopAutoplay();
    return;
  }
  if (stepMode) {
    botStep().catch((error) => console.error(error));
    return;
  }
  document.getElementById("autoPlay").textContent = "Pause";
  autoplay = setInterval(() => {
    botStep().catch((error) => {
      stopAutoplay();
      console.error(error);
    });
  }, 700);
}

function toggleStepMode() {
  stepMode = !stepMode;
  if (stepMode) {
    stopAutoplay();
  }
  renderPlayControls();
}

function renderPlayControls() {
  const playButton = document.getElementById("autoPlay");
  const modeButton = document.getElementById("stepMode");
  if (!playButton || !modeButton || !state) return;

  modeButton.textContent = stepMode ? "Step: On" : "Step: Off";
  modeButton.classList.toggle("active-toggle", stepMode);
  modeButton.setAttribute("aria-pressed", String(stepMode));
  playButton.disabled = state.status.done;
  if (autoplay) {
    playButton.textContent = "Pause";
  } else {
    playButton.textContent = stepMode ? "Next Bot" : "Auto Play";
  }
}

document.getElementById("botStep").addEventListener("click", () => {
  stopAutoplay();
  botStep();
});
document.getElementById("stepMode").addEventListener("click", toggleStepMode);
document.getElementById("autoPlay").addEventListener("click", toggleAutoplay);
document.getElementById("reset").addEventListener("click", () => {
  stopAutoplay();
  reset();
});

loadState().catch((error) => {
  console.error(error);
  document.body.innerHTML = `<pre>${error.message}</pre>`;
});
