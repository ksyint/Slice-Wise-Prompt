"""Patient-disjoint train/validation/test partitioning."""
import numpy as np


def patient_split(records, seed=42, fractions=(.7, .1, .2)):
    if not np.isclose(sum(fractions), 1) or any(value < 0 for value in fractions):
        raise ValueError('Split fractions must be nonnegative and sum to one.')
    patients = sorted({str(record['patient_id']) for record in records})
    np.random.default_rng(seed).shuffle(patients)
    cut1 = int(round(len(patients) * fractions[0], 10))
    cut2 = int(round(len(patients) * sum(fractions[:2]), 10))
    allocation = {patient: ('train' if index < cut1 else 'val' if index < cut2 else 'test') for index, patient in enumerate(patients)}
    result = {key: [] for key in ('train', 'val', 'test')}
    for record in records:
        result[allocation[str(record['patient_id'])]].append(record)
    return result
