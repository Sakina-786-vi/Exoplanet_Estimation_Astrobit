from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from detection.bls import detect_star
from models.calibration import calibrate, fit_oof_calibrator
from models.train import (
    fit_model,
    fit_vetting_model,
    predict_probability,
    predict_vetting_probability,
    save_model,
)
from preprocessing import preprocess_star
from submission.generate_submission import write_submission


def _load_split(directory, labels_path=None):
    files = sorted(Path(directory).glob("*.parquet"))
    labels = pd.read_csv(labels_path).set_index("kepid") if labels_path else None
    records = []
    for path in files:
        result = preprocess_star(pd.read_parquet(path), star_id=path.stem)
        if not result["ok"]:
            print(f"{path.stem}: preprocessing failed; excluded from inference")
            continue
        gap = np.diff(result["time"], prepend=result["time"][0]) > 3 * 29.4 / (60 * 24)
        candidates = detect_star(path.stem, result["time"], result["detrended_flux"],
                                 result["quarter"], gap, CONFIG)
        best = max(candidates, key=lambda candidate: candidate.score, default=None)
        label = None
        if labels is not None:
            label = int(labels.loc[int(path.stem.split("_")[-1]), "label"])
        records.append((result, candidates, best, label))
    return records


def _rows(records, model, calibrator, threshold, score_source="model", vetter=None):
    rows = []
    for result, candidates, best, _ in records:
        if best is None:
            rows.append({"star_id": result["star_id"], "prediction": 0, "confidence": 0.0,
                         "period": "", "depth_ppm": "", "duration_hours": ""})
            continue
        if score_source == "candidate_vetting":
            vetting_score = float(predict_vetting_probability(vetter, [best])[0])
            score = vetting_score
        else:
            model_scores = predict_probability(model, [best])
            score = (best.confidence if score_source == "statistical" else
                     (float(model_scores[0]) if np.isfinite(model_scores[0]) else best.confidence))
        confidence = float(calibrate(calibrator, [score])[0])
        decision_score = best.confidence if score_source == "statistical" else score
        positive = decision_score >= threshold
        rows.append({"star_id": result["star_id"], "prediction": int(positive),
                     "confidence": round(confidence, 8),
                     "period": round(best.period_days, 8) if positive else "",
                     "depth_ppm": round(best.depth_ppm, 5) if positive else "",
                     "duration_hours": round(best.duration_hours, 5) if positive else ""})
    return rows


def run(args):
    global CONFIG
    CONFIG = json.loads(Path(args.config).read_text())
    train = _load_split(args.train_dir, args.labels)
    dev = _load_split(args.dev_dir, args.dev_labels) if args.dev_dir else []
    private = _load_split(args.input_dir)
    private_files = sorted(Path(args.input_dir).glob("*.parquet"))
    if len(private) != len(private_files):
        raise RuntimeError(
            f"PRIVATE/input split has {len(private_files)} files but only "
            f"{len(private)} successful inference records"
        )
    private_ids = [path.stem for path in private_files]
    if all(star_id.startswith("STAR_") for star_id in private_ids):
        expected_private_ids = [f"STAR_{index:04d}" for index in range(len(private_ids))]
        if private_ids != expected_private_ids:
            raise RuntimeError("PRIVATE IDs must be the contiguous STAR_0000... sequence")
    train_candidates = [best for _, _, best, label in train if best is not None]
    train_labels = [label for _, _, best, label in train if best is not None]
    model = fit_model(train_candidates, train_labels, CONFIG["random_seed"]) if CONFIG.get("use_ml") else None
    vetter = None
    vetting_config = CONFIG.get("candidate_vetting", {})
    if vetting_config.get("enabled", False) and train_candidates:
        vetter = fit_vetting_model(train_candidates, train_labels, CONFIG["random_seed"] + 17)
    # Calibrate only on held-out fold scores, never on the scores used to fit
    # the final model.
    fold_count = min(int(CONFIG.get("calibration_folds", 5)), len(train_candidates))
    oof_scores = np.full(len(train_candidates), np.nan)
    if fold_count >= 2:
        for fold in range(fold_count):
            held_out = np.arange(len(train_candidates)) % fold_count == fold
            fold_model = fit_model(
                [candidate for index, candidate in enumerate(train_candidates) if not held_out[index]],
                [label for index, label in enumerate(train_labels) if not held_out[index]],
                CONFIG["random_seed"] + fold,
            )
            oof_scores[held_out] = predict_probability(
                fold_model, [candidate for index, candidate in enumerate(train_candidates) if held_out[index]]
            )
    raw_scores = oof_scores
    fallback_scores = np.asarray([candidate.confidence for candidate in train_candidates])
    raw_scores = np.where(np.isfinite(raw_scores), raw_scores, fallback_scores)
    # Select the score family and threshold on TRAIN OOF predictions. A fixed
    # probability threshold is unsafe here because rare positives and isotonic
    # calibration can legitimately compress probabilities.
    labels_array = np.asarray(train_labels)
    model_source = raw_scores
    statistical_source = fallback_scores
    vetting_source = np.asarray([predict_vetting_probability(vetter, [candidate])[0] for candidate in train_candidates]) if vetter is not None else model_source

    def best_threshold(scores):
        thresholds = np.unique(np.r_[scores, np.linspace(0, 1, 101)])
        best = (-1.0, float(thresholds[-1]))
        for threshold in thresholds:
            pred = scores >= threshold
            tp = np.sum(pred & (labels_array == 1))
            fp = np.sum(pred & (labels_array == 0))
            fn = np.sum(~pred & (labels_array == 1))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            if f1 > best[0]:
                best = (float(f1), float(threshold))
        return best

    model_quality = best_threshold(model_source)
    statistical_quality = best_threshold(statistical_source)
    vetting_quality = best_threshold(vetting_source) if vetter is not None else (-1.0, 0.0)
    score_candidates = [("statistical", statistical_source, statistical_quality), ("model", model_source, model_quality)]
    if vetter is not None:
        score_candidates.append(("candidate_vetting", vetting_source, vetting_quality))
    score_source, selected_scores, selected_quality = max(score_candidates, key=lambda item: item[2][0])
    calibrator = fit_oof_calibrator(selected_scores, train_labels)
    threshold = selected_quality[1]
    frozen = CONFIG.get("frozen", {})
    if frozen.get("status") == "TRAIN_SELECTED_DEV_FROZEN":
        score_source = frozen["selected_score_source"]
        threshold = float(frozen["selected_threshold"])
        selected_scores = statistical_source if score_source == "statistical" else model_source
        calibrator = fit_oof_calibrator(selected_scores, train_labels)
    CONFIG["selected_score_source"] = score_source
    CONFIG["selected_threshold"] = threshold
    CONFIG["train_oof_f1"] = selected_quality[0]
    save_model(model, Path(args.output_dir) / "model.json")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_used.json").write_text(json.dumps(CONFIG, indent=2))
    for name, records in (("train", train), ("dev", dev)):
        if records:
            pd.DataFrame(_rows(records, model, calibrator, threshold, score_source, vetter)).to_csv(
                output / f"{name}_predictions.csv", index=False)
    rows = _rows(private, model, calibrator, threshold, score_source, vetter)
    write_submission(rows, output / "submission.csv", private_ids)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--train-dir", default="train_pack/train")
    parser.add_argument("--labels", default="train_pack/train_labels.csv")
    parser.add_argument("--dev-dir", default="dev_pack/dev")
    parser.add_argument("--dev-labels", default="dev_pack/dev_labels.csv")
    parser.add_argument("--config", default="configs/final_config.json")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    rows = run(args)
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise RuntimeError(f"expected {args.expected_count} rows, produced {len(rows)}")
    print(f"processed {len(rows)} stars; output={args.output_dir}")


if __name__ == "__main__":
    main()
