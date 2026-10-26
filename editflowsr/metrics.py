from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict
import numpy as np
import scipy

def compute_metrics(infos, metrics="r2"):
    results = defaultdict(list)
    if metrics == "":
        return {}

    if "true" in infos:
        true, predicted = infos["true"], infos["predicted"]
        assert len(true) == len(predicted), "issue with len, true: {}, predicted: {}".format(len(true), len(predicted))
        for i in range(len(true)):
            if predicted[i] is None: continue
            if len(true[i].shape)==2:
                true[i]=true[i][:,0]
            if len(predicted[i].shape)==2:
                predicted[i]=predicted[i][:,0]
            assert true[i].shape == predicted[i].shape, "Problem with shapes: {}, {}".format(true[i].shape, predicted[i].shape)

    for metric in metrics.split(","):
        if metric == "r2":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or not np.all(np.isfinite(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    results[metric].append(r2_score(true[i], predicted[i]))
        if metric == "r2_zero":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or not np.all(np.isfinite(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    results[metric].append(max(0, r2_score(true[i], predicted[i])))

        elif metric.startswith("accuracy_l1"):
            if metric == "accuracy_l1":
                atol, rtol = 0.0, 0.1
                tolerance_point = 0.95
            elif metric == "accuracy_l1_biggio":

                atol, rtol = 1e-3, 0.05
                tolerance_point = 0.95
            else:
                atol = 0
                rtol = float(metric.split("_")[-1])
                tolerance_point = 0.95

            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or not np.all(np.isfinite(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    is_close = np.isclose(predicted[i], true[i], atol=atol, rtol=rtol)
                    results[metric].append(float(is_close.mean()>=tolerance_point))

        elif metric == "_mse":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or not np.all(np.isfinite(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    results[metric].append(mean_squared_error(true[i], predicted[i]))
        elif metric == "_nmse":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or not np.all(np.isfinite(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    mean_y = np.mean(true[i])
                    NMSE = (np.mean(np.square(true[i]- predicted[i])))/mean_y
                    results[metric].append(NMSE)
        elif metric == "_rmse":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or not np.all(np.isfinite(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    results[metric].append(float(np.sqrt(mean_squared_error(true[i], predicted[i]))))
        elif metric == "_complexity":
            if "predicted_tree" not in infos:
                results[metric].extend([np.nan for _ in range(len(infos["true"]))])
                continue
            predicted_tree = infos["predicted_tree"]
            for i in range(len(predicted_tree)):
                if predicted_tree[i] is None:
                    results[metric].append(np.nan)
                else:
                    results[metric].append(len(predicted_tree[i].prefix().split(",")))

        elif metric == "_relative_complexity":
            if "tree" not in infos or "predicted_tree" not in infos:
                results[metric].extend([np.nan for _ in range(len(infos["true"]))])
                continue
            tree = infos["tree"]
            predicted_tree = infos["predicted_tree"]
            for i in range(len(predicted_tree)):
                if predicted_tree[i] is None:
                    results[metric].append(np.nan)
                else:
                    results[metric].append(len(predicted_tree[i].prefix().split(",")) - len(tree[i].prefix().split(",")))

        elif metric == "is_symbolic_solution":

            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or not np.all(np.isfinite(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    diff = true[i] - predicted[i]
                    div = true[i] / (predicted[i] + 1e-100)
                    std_diff = scipy.linalg.norm(
                        np.abs(diff - diff.mean(0))
                    )
                    std_div = scipy.linalg.norm(
                        np.abs(div - div.mean(0))
                    )
                    if std_diff<1e-10:
                        results[metric].append(float(std_div < 1e-10))
                    else:
                        results[metric].append(0.0)

        elif metric == "_l1_error":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or not np.all(np.isfinite(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    l1_error = np.mean(np.abs((true[i] - predicted[i])))
                    if np.isnan(l1_error):
                        results[metric].append(np.inf)
                    else:
                        results[metric].append(l1_error)
    return results
