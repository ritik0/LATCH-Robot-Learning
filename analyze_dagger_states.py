import numpy as np 

old_obs = np.load("observations.npy")
all_obs = np.load("dagger_observations.npy")
new_obs = all_obs[len(old_obs):]

names = [
    "cos(q1)",
    "cos(q2)",
    "sin(q1)",
    "sin(q2)",
    "target_x",
    "target_y",
    "qd1",
    "qd2",
    "dx",
    "dy"
]

for i, name in enumerate(names):
    old_values = old_obs[:, i]
    new_values = new_obs[:, i]

    print(f"\n{name}")

    print(
        "OLD:",
        "min =", round(old_values.min(), 3),
        "max =", round(old_values.max(), 3),
        "mean =", round(old_values.mean(), 3),
        "std =", round(old_values.std(), 3)
    )

    print(
        "NEW:",
        "min =", round(new_values.min(), 3),
        "max =", round(new_values.max(), 3),
        "mean =", round(new_values.mean(), 3),
        "std =", round(new_values.std(), 3)
    )