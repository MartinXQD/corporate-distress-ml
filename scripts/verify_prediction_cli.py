"""Check the public inference CLI against frozen scores without fitting."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from distress_ml.common import ROOT, ART, PUBLIC, dump
from distress_ml.finalize import reproduce


def main():
    freeze = json.loads((ROOT / 'configs/final_freeze.json').read_text())
    item = next(m for m in freeze['models'] if 'ML' in m['representative'] and m['seed'] == 42)
    reproduce(item['label'])
    raw = pd.read_csv(Path(os.environ.get('DISTRESS_DATA', ROOT.parent / 'american_bankruptcy.csv')))
    subset = raw[raw.year >= 2015].head(128).copy()
    folder = Path(tempfile.mkdtemp(prefix='prediction_cli_check_', dir=ART / 'audit'))
    outputs = []
    for name, frame in [('original_labels', subset), ('changed_labels', subset.assign(status_label='ignored'))]:
        input_path = folder / (name + '.csv')
        output_path = folder / (name + '_scores.csv')
        frame.to_csv(input_path, index=False)
        subprocess.run([
            str(ROOT / 'scripts/run.sh'), '-m', 'distress_ml.cli', 'predict',
            '--model', item['path'], '--input', str(input_path), '--output', str(output_path)
        ], cwd=ROOT, check=True)
        outputs.append(pd.read_csv(output_path, float_precision='round_trip'))
    # Threaded forest reductions may differ at the final floating-point bit.
    for field in ['raw_score', 'calibrated_score']:
        np.testing.assert_allclose(outputs[0][field], outputs[1][field], atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(outputs[0].predicted_label, outputs[1].predicted_label)
    saved = pd.read_csv(ART / f'predictions/{item["label"]}_historical.csv.gz', float_precision='round_trip')
    aligned = outputs[0].merge(saved, on=['company_id', 'year'], suffixes=('_cli', '_frozen'), validate='one_to_one')
    assert len(aligned) == 128
    for field in ['raw_score', 'calibrated_score']:
        np.testing.assert_allclose(aligned[field + '_cli'], aligned[field + '_frozen'], atol=1e-6, rtol=1e-6)
    result = dict(passed=True, rows=128, inference_label_independent=True,
                  frozen_prediction_match=True, ml_full_historical_checkpoint_reproduced=True,
                  model=item['label'], tolerance=1e-6, fitting_performed=False)
    result['label_change_max_score_difference'] = float(np.max(np.abs(outputs[0].calibrated_score - outputs[1].calibrated_score)))
    dump(PUBLIC / 'prediction_cli_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
