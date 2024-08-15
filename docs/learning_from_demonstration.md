# Learning from Demonstration (LfD)

## Prerequisites:

1. Follow [instructions](../README.md#advanced-installation) for advanced installation of stretch_ai with Python 3.10

1. Install Hello-Robot's fork of HuggingFace LeRobot

   ```bash
   # Install in same conda environment as stretch_ai
   conda activate stretch_ai

   git clone git@github.com:hello-yiche/lerobot.git
   cd lerobot

   # Support for Stretch is currently implemented on this branch
   git switch stretch-act

   # Editable install makes editing configs useful, though configs can also be specified via cli
   pip install -e .
   ```

## Overview of LfD process

1. [Collect demonstration dataset with dex teleop](data_collection.md) (`50 episodes`)
1. [Format dataset and push to HuggingFace Hub](#format-data-and-push-to-huggingface-repo)
1. [Train policy](#train-a-policy) (`1000 epochs`)
1. [Load and evaluate policy](#evaluating-a-policy)
1. [Integrate as skill in a long horizon task](#integrating-a-skill-in-a-long-horizon-task-example)

## Format data and push to huggingface repo

### [Authenticate with huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

### Process and push demonstration folder to HuggingFace repo

```bash
# --raw-dir:  where the episodes for this task are stored
# --repo-id: Name of huggingface dataset, consists of your ID and the dataset name separated by a `/`
# --local-dir: where a local copy of the final huggingface dataset will be stored, last two layers of local_dir should be in same format as the repo-id
# --video: If true, dataset will only contain videos and no images
# --fps: FPS of demonstrations

python ./lerobot/scripts/push_dataset_to_hub.py \
--raw-dir ./path-to-raw-dir \
--repo-id <huggingface-id>/<your-dataset-name> \
--raw-format dobbe \
--local-dir ./path-to-local-dataset-folder/<huggingface-id>/<your-dataset-name> \
--video 0 \
--fps 6
```

### Sample command assuming following folder structure:

```
my_workspace
├── stretch_ai
├── lerobot
└── data/
   ├── my-huggingface-id/
   │   └── my-dataset
   └── default_task/
       └── default_user/
           └── default_env/
               ├── episode_1_folder
               ├── episode_2_folder
               └── episode_3_folder
```

```bash
# Assuming cwd is my_workspace/lerobot
python ./lerobot/scripts/push_dataset_to_hub.py \
--raw-dir ../data/default_task/default_user/default_env \
--repo-id my-huggingface-id/my-dataset \
--raw-format dobbe \
--local-dir ../data/my-huggingface-id/my-dataset \
--video 0
--fps 6
```

### Visualizing dataset with Rerun.io (Optional)

![](images/rerun_dataset.png)

```bash
# Sample command
python ./lerobot/scripts/visualize_dataset.py --repo-id my-huggingface-id/my-dataset --episode-index 0 --root ../data/default_task/default_user
```

```bash
# Specify root if you wish to use local copy of the dataset, else dataset will be pulled from web
# --repo-id: Huggingface dataset repo
# --episode-index: Which episode to visualize
# --root: Where the local copy of the huggingface dataset is stored (e.g. local-dir in the previous step, but without specific folder )
python ./lerobot/scripts/visualize_dataset.py \
--repo-id hellorobotinc/<your-dataset-name> \
--episode-index <episode-idx> \
--root ../data/default_task/default_user
```

## Train a policy

Policy config files are located in `./lerobot/configs/policy`. Env config files are located in `./lerobot/configs/env`

Available policy configs for Stretch:

- `stretch_diffusion` - default training configs for [Diffusion Policy](https://arxiv.org/abs/2303.04137v4)
- `stretch_diffusion_depth` - default training configs for [Diffusion Policy](https://arxiv.org/abs/2303.04137v4) with depth included as input
- `stretch_act_real` - default training configs for [ACT](https://arxiv.org/abs/2304.13705)
- `stretch_vqbet` - default training configs for [VQ-BeT](https://arxiv.org/abs/2403.03181)

Available env configs for Stretch:

- `stretch_real` - Standard 9 dim state space and (9+1) dim action space for Stretch
  - state: (x, y, theta, lift, arm, roll, pitch, yaw, gripper)
  - action: (x, y, theta, lift, arm, roll, pitch, yaw, gripper, progress)

Training configs defined in the policy yaml file can be overridden in CLI.
If the config looks like below:

```yaml
training:
  learning_rate: 0.001
```

At runtime we can override this by adding the snippet below. For more details see [Hydra docs](https://hydra.cc/docs/intro/) and [LeRobot](https://github.com/huggingface/lerobot?tab=readme-ov-file#train-your-own-policy).

```bash
training.learning_rate=0.00001
```

Like training any other model with a GPU, try to adjust batch size to maximally utilize GPU vRAM.

Sample training command:

```bash
python3 lerobot/scripts/train.py \
policy=stretch_diffusion \
env=stretch_real \
wandb.enable=true \
training.batch_size=64 \
training.num_workers=16
```

### Sample loss curve for "two different tasks" trained with the same parameters

This shows that even for different tasks, loss curves should look similar

- `num_episodes` = 50
- policy=stretch_diffusion
- env=stretch_real
- Y-axis is `loss` in log scale
- X-axis is `num_epochs`, from experience 1000 epochs is enough for good performance
- Training took ~19 hours for 2000 epochs on RTX 4090
- [Full run available here](https://wandb.ai/jensenhuang2/diffusion-kitchen-diagonal?nw=nwuserjensenhuang2)

![](images/sample_loss.png)

## Evaluating a policy

### On Robot:

```bash
ros2 launch stretch_ros2_bridge server.launch.py
```

### On PC:

Specify the policy name of the weights provided:

- Available policies: `diffusion`,`diffusion_depth`,`act`,`vqbet`

Specify the teleop mode according to the teleop mode used to train the policy

- Available teleop modes: `base_x`,`stationary_base`,`old_stationary_base`

```bash
python3 -m stretch.app.lfd.ros2_lfd_leader \
--robot_ip $ROBOT_IP \
--policy_name <name-of-policy> \
--policy_path <path-to-weights-folder> \
--teleop-mode <teleop-mode> \
--record-success
```

Sample command:

```bash
python3 -m stretch.app.lfd.ros2_lfd_leader \
--robot_ip $ROBOT_IP \
--policy_path lerobot/outputs/train/2024-07-28/17-34-36_stretch_real_diffusion_default/checkpoints/100000/pretrained_model \
--policy_name diffusion \
--teleop-mode base_x
--record-success
```

## Integrating a skill in a long horizon task (example)

`stretch.app.goto_and_execute_skill` demonstrates an implementation of multiple skills integrated into a long horizon task. The current implementation is environment specific (e.g. it will not work in your home), but provides an example for how LfD skills can be integrated.