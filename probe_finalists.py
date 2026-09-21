"""决赛圈对比：在最佳特征配置上比较 LR / SVM / 集成。"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
from sklearn.ensemble import VotingClassifier

from run_experiments import resolve_feature
from src.config import RANDOM_SEED, VAL_SIZE
from src.data_utils import load_data, split_train_val
from src.experiment import evaluate_pipeline, reset_seed
from src.features import build_pipeline
from src.models import make_classifier

X, y, Xt = load_data()
X_tr, X_val, y_tr, y_val = split_train_val(X, y, VAL_SIZE, RANDOM_SEED)


def run(label, clf, fk):
    reset_seed(RANDOM_SEED)
    pipe = build_pipeline(resolve_feature(fk), clf)
    r = evaluate_pipeline(pipe, X_tr, y_tr, X_val, y_val)
    print(
        f"{label:<26} acc={r['val_accuracy']:.4f} "
        f"macroF1={r['val_f1_macro']:.4f} fit={r['fit_time_s']:.0f}s"
    )
    return r


run("A  LR C=10 | F1", make_classifier("lr", C=10.0), "F1")
run("G  LR C=1  | F1", make_classifier("lr", C=1.0), "F1")
run("B  LR C=10 | F9", make_classifier("lr", C=10.0), "F9")
run("H  LR C=1  | F9", make_classifier("lr", C=1.0), "F9")
run("D  SVM C=10 | F1", make_classifier("svm", kernel="linear", C=10.0), "F1")
ens_nb_lr = VotingClassifier(
    estimators=[
        ("nb", make_classifier("nb", alpha=0.1)),
        ("lr", make_classifier("lr", C=10.0)),
    ],
    voting="soft",
)
run("E1 NB+LR 软投票 | F1", ens_nb_lr, "F1")
ens3 = VotingClassifier(
    estimators=[
        ("nb", make_classifier("nb", alpha=0.1)),
        ("lr", make_classifier("lr", C=10.0)),
        ("svm", make_classifier("svm", kernel="linear", C=10.0, probability=True)),
    ],
    voting="soft",
)
run("E3 NB+LR+SVM 软投票 | F1", ens3, "F1")
print("done")
