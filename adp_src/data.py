from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_TOKEN = {aa: index + 1 for index, aa in enumerate(AMINO_ACIDS)}


@dataclass(frozen=True)
class PeptideRecord:
    identifier: str
    sequence: str
    label: int


def read_labeled_fasta(path: str | Path) -> list[PeptideRecord]:
    """Read AntiDMPpred FASTA headers of the form ``>id|label|split``."""
    records: list[PeptideRecord] = []
    header: str | None = None
    pieces: list[str] = []

    def finish() -> None:
        nonlocal header, pieces
        if header is None:
            return
        fields = header.split("|")
        if len(fields) < 2 or fields[1] not in {"0", "1"}:
            raise ValueError(f"Missing binary label in FASTA header: >{header}")
        sequence = "".join(pieces).upper()
        invalid = sorted(set(sequence) - set(AMINO_ACIDS))
        if invalid:
            raise ValueError(f"Non-canonical residues in >{header}: {invalid}")
        if not 5 <= len(sequence) <= 50:
            raise ValueError(f"Sequence length outside 5-50 in >{header}")
        # Preserve the complete first FASTA token because the numeric prefix is
        # reused across the positive and negative source collections.
        records.append(PeptideRecord(header.split()[0], sequence, int(fields[1])))

    with Path(path).open(encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                header, pieces = line[1:], []
            elif header is None:
                raise ValueError("Sequence line encountered before a FASTA header")
            else:
                pieces.append(line)
    finish()
    if not records:
        raise ValueError(f"No FASTA records in {path}")
    return records


def records_to_arrays(records: list[PeptideRecord]) -> tuple[list[str], np.ndarray, list[str]]:
    return (
        [record.sequence for record in records],
        np.asarray([record.label for record in records], dtype=np.int64),
        [record.identifier for record in records],
    )


def tokenize(sequences: list[str], max_length: int = 50) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((len(sequences), max_length), dtype=np.int64)
    mask = np.zeros((len(sequences), max_length), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        length = min(len(sequence), max_length)
        tokens[row, :length] = [AA_TO_TOKEN[aa] for aa in sequence[:length]]
        mask[row, :length] = 1.0
    return tokens, mask
