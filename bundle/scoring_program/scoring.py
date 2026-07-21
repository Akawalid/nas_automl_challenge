import json
import numpy as np
import pickle as pkl
import os
import sys

from sklearn.metrics import accuracy_score

# argv: [1] = input dir (contains ref/ and res/), [2] = output dir (for scores.json)
INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

LABELS_DIR = os.path.join(INPUT_DIR, 'ref')
PREDICTIONS_DIR = os.path.join(INPUT_DIR, 'res')

if __name__ == '__main__':
    try:
        print("=" * 75)
        print("="*13 + "    Your Unseen Data 2026 Submission is scoring     " + "="*13)
        print("=" * 75)

        total_score = 0
        overall_stats = {}
        for dataset in os.listdir(LABELS_DIR):
            print("== Scoring {} ==".format(dataset))

            data_path = os.path.join(LABELS_DIR, dataset)

            with open(os.path.join(data_path, 'metadata'), "r") as f:
                metadata = json.load(f)
            with open(os.path.join(PREDICTIONS_DIR, "{}_stats.pkl".format(metadata['codename'])), "rb") as f:
                run_stats = pkl.load(f)

            if run_stats['Failed']:
                raw_score = -1
                adj_score = -10
            else:
                labels = np.load(os.path.join(data_path, 'test_y.npy'))
                prediction_file = [p for p in os.listdir(PREDICTIONS_DIR) if metadata['codename'] == p.replace(".npy", "")][0]
                predictions = np.load(os.path.join(PREDICTIONS_DIR, prediction_file))
                labels = labels[:len(predictions)]

                raw_score = 100 * accuracy_score(labels, predictions)
                benchmark = metadata['benchmark']

                scaling_factor = 10 / (100 - benchmark)
                adj_score = (raw_score - benchmark) * scaling_factor
                adj_score = max(-10, adj_score)

            total_score += adj_score

            print("Raw Score:    {:.3f}".format(raw_score))
            print("Adj Score:    {:.3f}".format(adj_score))
            print(f'Model Params: {"N/A" if run_stats["Params"] is None else run_stats["Params"]}')
            print("Runtime:      {:,.1f}s".format(run_stats['Runtime']))
            run_stats['Raw_Score'] = float(np.round(raw_score, 3))
            run_stats['Adj_Score'] = float(np.round(adj_score, 3))

            overall_stats.update({"{}_{}".format(metadata['codename'], k): v for k, v in run_stats.items()})

        print("===========================")
        print("Final Score: {:.3f}".format(total_score))
        overall_stats['Final_Score'] = np.round(total_score, 3)
        with open(os.path.join(OUTPUT_DIR, "scores.json"), "w") as f:
            json.dump(overall_stats, f)
    except Exception as e:
        print(e)