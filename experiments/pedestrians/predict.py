import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
from utils import prediction_output_to_trajectories

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to use", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output directory", type=str)
parser.add_argument("--output_tag", help="name tag for output files", type=str)
parser.add_argument("--node_type", help="node type to predict", type=str)
parser.add_argument("--num_samples", help="number of trajectory samples", type=int, default=20)
args = parser.parse_args()


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')
    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    prediction_rows = []
    history_rows = []

    with torch.no_grad():
        for scene_idx, scene in enumerate(scenes):
            print(f"-- Predicting Scene {scene_idx + 1}/{len(scenes)}")
            timesteps = np.arange(scene.timesteps)

            predictions_dict = eval_stg.predict(scene,
                                                timesteps,
                                                ph,
                                                num_samples=args.num_samples,
                                                min_history_timesteps=7,
                                                min_future_timesteps=12,
                                                z_mode=False,
                                                gmm_mode=False,
                                                full_dist=False)

            if not predictions_dict:
                continue

            (output_dict,
             histories_dict,
             futures_dict) = prediction_output_to_trajectories(predictions_dict,
                                                               scene.dt,
                                                               max_hl,
                                                               ph,
                                                               prune_ph_to_future=True)

            for t in output_dict.keys():
                for node in output_dict[t].keys():
                    if node.type.name != args.node_type:
                        continue

                    pred = output_dict[t][node]    # (1, num_samples, future_len, 2)
                    gt = futures_dict[t][node]      # (future_len, 2)
                    hist = histories_dict[t][node]  # (hist_len, 2)

                    future_len = gt.shape[0]
                    num_samples = pred.shape[1]

                    # Prediction rows
                    for s in range(num_samples):
                        for ft in range(future_len):
                            prediction_rows.append((
                                scene_idx,
                                t,
                                str(node),
                                s,
                                ft + 1,
                                pred[0, s, ft, 0],
                                pred[0, s, ft, 1],
                                gt[ft, 0],
                                gt[ft, 1]
                            ))

                    # History rows (once per pedestrian per timestep, no samples)
                    hist_len = hist.shape[0]
                    for ht in range(hist_len):
                        offset = ht - (hist_len - 1)  # e.g. -7, -6, ..., 0
                        history_rows.append((
                            scene_idx,
                            t,
                            str(node),
                            offset,
                            hist[ht, 0],
                            hist[ht, 1]
                        ))

    print(f"-- Saving {len(prediction_rows)} prediction rows, {len(history_rows)} history rows")

    os.makedirs(args.output_path, exist_ok=True)

    pred_df = pd.DataFrame(prediction_rows,
                           columns=['scene_id', 'timestep', 'node_id', 'sample_id',
                                    'future_t', 'pred_x', 'pred_y', 'gt_x', 'gt_y'])
    pred_df.to_csv(os.path.join(args.output_path, args.output_tag + '_predictions.csv'), index=False)

    hist_df = pd.DataFrame(history_rows,
                           columns=['scene_id', 'timestep', 'node_id', 'history_t',
                                    'obs_x', 'obs_y'])
    hist_df.to_csv(os.path.join(args.output_path, args.output_tag + '_histories.csv'), index=False)

    print("-- Done")
