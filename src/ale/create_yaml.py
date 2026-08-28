
import yaml

component_variants = {
    "full":        {"dueling": True,  "noisy": True,  "nstep": 3},
    "no_dueling":  {"dueling": False, "noisy": True,  "nstep": 3},
    "no_noisy":    {"dueling": True,  "noisy": False, "nstep": 3},
    "nstep1":      {"dueling": True,  "noisy": True,  "nstep": 1},
}

combos = []

# ── Baselines: no delay, sparse vs full reward, full Rainbow, hard target, no predict ──
for reward_name, full_rewards in [("sparse", False), ("full", True)]:
    combos.append({
        "run_group": f"baseline_{reward_name}",
        "full_rewards": full_rewards,
        "action_delay": 0,
        "predict": False,
        "target_update": "hard",
        **component_variants["full"],
    })

# ── Full cross: delay x component_variant x target_update x predict ──
for delay in [5]:
    for variant_name, variant in component_variants.items():
        for target_update in ["hard", "soft"]:
            for predict in [False, True]:
                combos.append({
                    "run_group": f"d{delay}_{variant_name}_{target_update}_{'pred' if predict else 'noPred'}",
                    "full_rewards": False,
                    "action_delay": delay,
                    "predict": predict,
                    "target_update": target_update,
                    **variant,
                })

sweep_config = {
    "program": "sweep_train.py",
    "method": "grid",
    "metric": {"name": "episode/total_reward", "goal": "maximize"},
    "parameters": {
        "combo": {"values": combos},
        "environment": {"value": 4},
        "episode": {"value": 5000},
    },
}

with open("sweep_config.yaml", "w") as f:
    yaml.dump(sweep_config, f, sort_keys=False, default_flow_style=False)

print(f"Generated {len(combos)} combos -> sweep_config.yaml")