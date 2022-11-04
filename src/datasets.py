"""Contains code for reading in CoNLL datasets for parsing."""
from collections.abc import Iterable, Iterator, Mapping, Sequence
import enum
import os
from typing import NamedTuple, Optional

from absl import logging
from torchtext import data

_CONLL_SEPARATOR = "\t"
FieldMapping = tuple[str, data.Field] | tuple[None, None]


class ConllField(enum.IntEnum):
    """Matches CoNLL field names to their indices.

    Can be used anywhere an int can be used, due to subclassing enum.IntEnum.
    """
    ID = 0
    FORM = 1
    LEMMA = 2
    UPOS = 3
    XPOS = 4
    FEATS = 5
    HEAD = 6
    DEPREL = 7
    DEPS = 8
    MISC = 9


_CONLL_POS_MAPPING = {
    ConllField.FORM: 0,
    ConllField.UPOS: 1,
}


class ConllRow(NamedTuple):
    """Contains the entries in a CoNLL file's row."""
    id: str
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: str
    deprel: str
    deps: str
    misc: str

    @classmethod
    def from_string(cls, string):
        """Creates instance from a string containing 10 tab-separated fields."""
        return cls(*string.split(_CONLL_SEPARATOR))


def _make_fields() -> Sequence[FieldMapping]:
    word = data.Field(init_token="<bos>", pad_token=None, eos_token="<eos>")
    pos = data.Field(
        init_token="<bos>", include_lengths=True, pad_token=None, eos_token="<eos>"
    )
    fields = [("word", word), ("pos", pos), (None, None)]
    return fields


def _iterate_conll_sentences(file: Iterable[str]) -> Iterator[list[ConllRow]]:
    """Groups lines of a CoNLL-X file into sentences.

    Each sentence in a CoNLL-X file has one token per line, with a blank
    line between sentences. This groups the file into lists of tokens,
    each of which represents a sentence.
    """
    rows: list[ConllRow] = []
    for line in file:
        line = line.strip()
        if line:
            rows.append(ConllRow.from_string(line))
        else:
            yield rows
            rows = []
    if rows:  # If the file lacks final blank line, still yield last entry.
        yield rows


class ConllXDatasetPos(data.Dataset):
    """Stores a subset of the 10-column CoNLL-X file."""

    def __init__(
        self,
        path: os.PathLike,
        fields,
        encoding: str = "utf-8",
        column_mapping: Optional[Mapping[ConllField, int]] = None,
        **kwargs,
    ):
        if column_mapping is None:
            column_mapping = _CONLL_POS_MAPPING

        examples: list[data.Example] = []

        with open(path, mode="r", encoding=encoding) as input_file:
            for sentence in _iterate_conll_sentences(input_file):
                examples.append(self._make_example(sentence, fields, column_mapping))
        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def _make_example(
        sentence: list[ConllRow],
        fields: Sequence[FieldMapping],
        column_mapping: Mapping[ConllField, int],
    ):
        """Constructs a data.Example instance from the tab-separated rows."""
        # Initialize empty lists.
        relevant_columns = [[] for _ in column_mapping]
        for row in sentence:
            for i, column_val in enumerate(row):
                if i in column_mapping:
                    relevant_columns[column_mapping[i]].append(column_val)
        return data.Example.fromlist(relevant_columns, fields)

    @staticmethod
    def _validate_column_map(column_map: Mapping[int, int]) -> None:
        """Raises exception if the column map is ill-formed."""
        # Check that all indices are valid.
        if not all(0 <= x < 10 for x in column_map.keys()):
            raise ValueError(
                "CoNLL-X has 10 fields 0...9; found out-of-range indices"
                f" {[x for x in column_map.keys() if not 0 <= x < 10]}"
            )
        # Check that destinations are unique.
        values = list(column_map.values())
        if sorted(values) != sorted(set(values)):
            raise ValueError(
                f"Destinations in column mapping must be unique; found {values}"
            )
        # Check that destinations are contiguous.
        if sorted(values) != list(range(len(values))):
            raise ValueError(
                "Destinations in column mapping must be contiguous;"
                f" found {sorted(values)}"
            )


def build_train_test(
    train_path: os.PathLike,
    test_path: os.PathLike,
    *,
    batch_size: int = 20,
    device: str = "cpu",
    shuffle_train: bool = False,
) -> tuple[data.BucketIterator, data.BucketIterator]:
    """Loads a train/test pair in from disk."""
    fields = _make_fields()
    train = ConllXDatasetPos(train_path, fields, filter_pred=lambda x: len(x.word) < 50)
    test = ConllXDatasetPos(test_path, fields)

    logging.info(f"Total train sentences: {len(train)}")
    logging.info(f"Total test sentences: {len(test)}")

    train_iter = data.BucketIterator(
        train, batch_size=batch_size, device=device, shuffle=shuffle_train
    )
    test_iter = data.BucketIterator(
        test, batch_size=batch_size, device=device, shuffle=False
    )
    return train_iter, test_iter
