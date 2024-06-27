import os
import torch
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm

from scripts.utils import calculate_metrics, plot_sample
from timeit import default_timer as timer


def evaluate_test_data(model, torch_dataset,
                    torch_device, RESULT_DIR,
                    THRESHOLD,save_plots = False,
                    save_csv_file = False,
                    show_progress = False):
    """ Evaluate the model on the test data

    Args:
        model (nn.Module): _description_
        torch_dataset (_type_): _description_
        torch_device (_type_): _description_
        RESULT_DIR (str): _description_
        THRESHOLD (int): _description_
        save_plots (bool, optional): True if wants to visualize results. Defaults to False.
        save_csv_file (bool, optional): True if wants to save evaluation scores for test dataset. Defaults to False.
        show_progress (bool, optional): True if want to see the score in CLI. Defaults to False.

    Returns:
        pd.DataFrame: Evaluation Summary
    """
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
                if save_plots:
                    plot_sample(x, y, out, cnt, RESULT_DIR, THRESHOLD)
                
                (jaccard, f1, recall_score, precision_score, accuracy_score) = (
                    calculate_metrics(y_true=y, y_pred=out, threshold=THRESHOLD)
                )
                results_dict["jaccard"].append(jaccard)
                results_dict["f1"].append(f1)
                results_dict["recall"].append(recall_score)
                results_dict["precision"].append(precision_score)
                results_dict["accuracy"].append(accuracy_score)
                results_dict["sample_id"].append(cnt)
                if show_progress:
                    # Update the progress bar and set postfix with metrics
                    pbar.set_postfix(
                        {
                            "jaccard": jaccard,
                            "f1": f1,
                            "recall": recall_score,
                            "precision": precision_score,
                            "accuracy": accuracy_score,
                        }
                    )
                pbar.update(1)
                # break
    df = pd.DataFrame(results_dict)
    if save_csv_file:
        df.to_csv(os.path.join(RESULT_DIR, "results.csv"), index=False)
    return df


if __name__ == "__main__":
    # %matplotlib inline
    pass
