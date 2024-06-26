import os
import torch
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm

from scripts.utils import calculate_metrics,plot_sample
from timeit import default_timer as timer

def evaluate_test_data(model, torch_dataset, torch_device, RESULT_DIR, THRESHOLD):
    cnt = 0
    results_dict = {
        "sample_id": [],
        "jaccard": [],
        "f1": [],
        "recall": [],
        "precision": [],
        "accuracy": [],
    }
    with tqdm(total=len(torch_dataset)) as pbar:
        with torch.no_grad():
            for x, y in torch_dataset:
                #         print(x.shape,y.shape)
                cnt += 1
                out = model(x.to(torch_device).unsqueeze(0)).squeeze(0)
                out = F.sigmoid(out)
                plot_sample(
                    x, y, out, cnt, RESULT_DIR, THRESHOLD
                )
                (jaccard, f1, recall_score, precision_score, accuracy_score) = (
                    calculate_metrics(y_true=y, y_pred=out, threshold=THRESHOLD)
                )
                results_dict["jaccard"].append(jaccard)
                results_dict["f1"].append(f1)
                results_dict["recall"].append(recall_score)
                results_dict["precision"].append(precision_score)
                results_dict["accuracy"].append(accuracy_score)
                results_dict["sample_id"].append(cnt)
                # Update the progress bar and set postfix with metrics
                pbar.set_postfix({
                    'jaccard': jaccard,
                    'f1': f1,
                    'recall': recall_score,
                    'precision': precision_score,
                    'accuracy': accuracy_score
                })
                pbar.update(1)
                # break
    df = pd.DataFrame(results_dict)
    df.to_csv(os.path.join(RESULT_DIR, "results.csv"), index=False)
    return df


if __name__ == "__main__":
    # %matplotlib inline
    pass
