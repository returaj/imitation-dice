from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import wrappers
import safedice
import utils
import time
import pickle


def evaluate(env, actor, train_env_id, num_episodes=10):
    """Evaluates the policy.
    Args:
        actor: A policy to evaluate
        env: Environment to evaluate the policy on
        train_env_id: train_env_id to compute normalized score
        num_episodes: A number of episodes to average the policy on
    Returns:
        Averaged reward and a total number of steps.
    """
    total_timesteps = 0
    total_returns = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            if "ant" in train_env_id.lower():
                state = np.concatenate((state[:27], [0.0]), -1)
            action = actor.step(state)[0].numpy()
            # print(f'!!!!!step: {env.step(action)}')
            next_state, reward, done, _ = env.step(action)

            total_returns += reward
            total_timesteps += 1
            state = next_state

    mean_score = total_returns / num_episodes
    mean_timesteps = total_timesteps / num_episodes
    return mean_score, mean_timesteps


def run(config):
    seed = config["seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env_id = config["env_id"]

    # unsafe data info
    unsafe_dataset_name = config["unsafe_dataset_name"]
    unsafe_num_traj = config["unsafe_num_traj"]
    # union data info
    union_dataset_names = config["union_dataset_names"]
    union_num_trajs = config["union_num_trajs"]
    if len(union_dataset_names) == 0:
        union_dataset_names, union_num_trajs = config["union_dataset_default_info"]
    assert len(union_dataset_names) == len(union_num_trajs)

    dataset_dir = config["dataset_dir"]

    (
        unsafe_initial_states,
        unsafe_states,
        unsafe_actions,
        unsafe_next_states,
        unsafe_dones,
    ) = utils.load_safety_gym_data(
        dataset_dir, env_id, unsafe_dataset_name, unsafe_num_traj, start_idx=0
    )

    # load union dataset
    (
        union_init_states,
        union_states,
        union_actions,
        union_next_states,
        union_dones,
    ) = ([], [], [], [], [])
    if len(union_dataset_names) > 0:
        for union_datatype_idx, (
            union_dataset_name,
            union_num_traj,
        ) in enumerate(zip(union_dataset_names, union_num_trajs)):
            start_idx = (
                unsafe_num_traj
                if (unsafe_dataset_name == union_dataset_name)
                else 0
            )

            (initial_states, states, actions, next_states, dones) = (
                utils.load_safety_gym_data(
                    dataset_dir,
                    env_id,
                    union_dataset_name,
                    union_num_traj,
                    start_idx=start_idx,
                )
            )

            union_init_states.append(initial_states)
            union_states.append(states)
            union_actions.append(actions)
            union_next_states.append(next_states)
            union_dones.append(dones)

    union_init_states = np.concatenate(union_init_states).astype(np.float32)
    union_states = np.concatenate(union_states).astype(np.float32)
    union_actions = np.concatenate(union_actions).astype(np.float32)
    union_next_states = np.concatenate(union_next_states).astype(np.float32)
    union_dones = np.concatenate(union_dones).astype(np.float32)

    union_init_states = np.concatenate(
        [union_init_states, unsafe_initial_states]
    ).astype(np.float32)
    union_states = np.concatenate([union_states, unsafe_states]).astype(np.float32)
    union_actions = np.concatenate([union_actions, unsafe_actions]).astype(
        np.float32
    )
    union_next_states = np.concatenate(
        [union_next_states, unsafe_next_states]
    ).astype(np.float32)
    union_dones = np.concatenate([union_dones, unsafe_dones]).astype(np.float32)

    print("# of expert demonstraions: {}".format(unsafe_states.shape[0]))
    print("# of imperfect demonstraions: {}".format(union_states.shape[0]))

    # normalize
    shift = -np.mean(union_states, 0)
    scale = 1.0 / (np.std(union_states, 0) + 1e-3)
    union_init_states = (union_init_states + shift) * scale
    unsafe_states = (unsafe_states + shift) * scale
    unsafe_next_states = (unsafe_next_states + shift) * scale
    union_states = (union_states + shift) * scale
    union_next_states = (union_next_states + shift) * scale

    # environment setting
    if "ant" in env_id.lower():
        shift_env = np.concatenate((shift, np.zeros(84)))
        scale_env = np.concatenate((scale, np.ones(84)))
    else:
        shift_env = shift
        scale_env = scale
    env = wrappers.create_il_env(
        env_id, seed, shift_env, scale_env, normalized_box_actions=False
    )
    eval_env = wrappers.create_il_env(
        env_id, seed + 1, shift_env, scale_env, normalized_box_actions=False
    )

    if config["using_absorbing"]:
        # using absorbing state
        union_init_states = np.c_[
            union_init_states, np.zeros(len(union_init_states), dtype=np.float32)
        ]
        (expert_states, expert_actions, expert_next_states, expert_dones) = (
            utils.add_absorbing_states(
                expert_states, expert_actions, expert_next_states, expert_dones, env
            )
        )
        (union_states, union_actions, union_next_states, union_dones) = (
            utils.add_absorbing_states(
                union_states, union_actions, union_next_states, union_dones, env
            )
        )
    else:
        # ignore absorbing state
        union_init_states = np.c_[
            union_init_states, np.zeros(len(union_init_states), dtype=np.float32)
        ]
        expert_states = np.c_[
            expert_states, np.zeros(len(expert_states), dtype=np.float32)
        ]
        expert_next_states = np.c_[
            expert_next_states, np.zeros(len(expert_next_states), dtype=np.float32)
        ]
        union_states = np.c_[
            union_states, np.zeros(len(union_states), dtype=np.float32)
        ]
        union_next_states = np.c_[
            union_next_states, np.zeros(len(union_next_states), dtype=np.float32)
        ]

    algorithm = config["algorithm"]
    if "ant" in env_id.lower():
        observation_dim = 28
    else:
        observation_dim = env.observation_space.shape[0]

    # Create imitator
    is_discrete_action = env.action_space.dtype == int
    action_dim = env.action_space.n if is_discrete_action else env.action_space.shape[0]
    if algorithm == "demodice":
        imitator = demodice.DemoDICE(
            observation_dim, action_dim, is_discrete_action, config=config
        )
    else:
        raise ValueError(f"{algorithm} is not supported algorithm name")

    print("Save interval :", config["save_interval"])
    # checkpoint dir
    checkpoint_dir = (
        f"checkpoint_imitator/{config['algorithm']}/{config['env_id']}/"
        f"{expert_dataset_name}_{expert_num_traj}_"
        f"{imperfect_dataset_names}_{imperfect_num_trajs}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filepath = f"{checkpoint_dir}/{config['seed']}.pickle"
    if config["resume"] and os.path.exists(checkpoint_filepath):
        # Load checkpoint.
        imitator.init_dummy(observation_dim, action_dim)
        checkpoint_data = imitator.load(checkpoint_filepath)
        training_info = checkpoint_data["training_info"]
        training_info["iteration"] += 1
        print(f"Checkpoint '{checkpoint_filepath}' is resumed")
    else:
        print(f"No checkpoint is found: {checkpoint_filepath}")
        training_info = {
            "iteration": 0,
            "logs": [],
        }
    print(config["save_interval"])
    config["total_iterations"] = config["total_iterations"] + 1

    # Start training
    start_time = time.time()
    with tqdm(
        total=config["total_iterations"],
        initial=training_info["iteration"],
        desc="",
        disable=os.environ.get("DISABLE_TQDM", False),
        ncols=70,
    ) as pbar:
        while training_info["iteration"] < config["total_iterations"]:
            if algorithm in ["demodice"]:
                union_init_indices = np.random.randint(
                    0, len(union_init_states), size=config["batch_size"]
                )
                expert_indices = np.random.randint(
                    0, len(expert_states), size=config["batch_size"]
                )
                union_indices = np.random.randint(
                    0, len(union_states), size=config["batch_size"]
                )

                info_dict = imitator.update(
                    union_init_states[union_init_indices],
                    expert_states[expert_indices],
                    expert_actions[expert_indices],
                    expert_next_states[expert_indices],
                    union_states[union_indices],
                    union_actions[union_indices],
                    union_next_states[union_indices],
                )
            else:
                raise ValueError(f"Undefined algorithm {algorithm}")

            if training_info["iteration"] % config["log_interval"] == 0:
                average_returns, evaluation_timesteps = evaluate_d4rl(
                    eval_env, imitator, env_id
                )

                info_dict.update({"eval": average_returns})
                print(
                    f"Eval: ave returns=d: {average_returns}"
                    f" ave episode length={evaluation_timesteps}"
                    f' / elapsed_time={time.time() - start_time} ({training_info["iteration"] / (time.time() - start_time)} it/sec)'
                )
                print("=========================")
                for key, val in info_dict.items():
                    print(f"{key:25}: {val:8.3f}")
                print("=========================")

                training_info["logs"].append(
                    {"step": training_info["iteration"], "log": info_dict}
                )
                print(f'timestep {training_info["iteration"]} - log update...')
                print("Done!", flush=True)

            # Save checkpoint
            if (
                training_info["iteration"] % config["save_interval"] == 0
                and training_info["iteration"] > 0
            ):
                imitator.save(checkpoint_filepath, training_info)

            training_info["iteration"] += 1
            pbar.update(1)


if __name__ == "__main__":
    from config.lfd_default_config import get_parser

    # configurations
    args = get_parser().parse_args()
    config = vars(args)

    print("Start running")
    run(config)
