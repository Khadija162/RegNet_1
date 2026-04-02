from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class PreparedData:
    dataframe: pd.DataFrame
    text_values: np.ndarray
    categorical_indices: np.ndarray
    numerical_values: np.ndarray
    ground_truth: np.ndarray
    categorical_encoders: Dict[str, LabelEncoder]
    num_classes_per_cat_attr: List[int]
    modalities: List[str]
    all_attribute_names: List[str]


class DataPreprocessor:
    def __init__(self, spec):
        self.spec = spec

    def validate(self, df: pd.DataFrame) -> None:
        required = self.spec.textual_columns + self.spec.categorical_columns + self.spec.numerical_columns
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset '{self.spec.name}': {missing}")
        overlap = (
            set(self.spec.textual_columns) & set(self.spec.categorical_columns)
        ) | (
            set(self.spec.textual_columns) & set(self.spec.numerical_columns)
        ) | (
            set(self.spec.categorical_columns) & set(self.spec.numerical_columns)
        )
        if overlap:
            raise ValueError(f"Columns assigned to multiple modalities: {sorted(overlap)}")

    def prepare(self, df: pd.DataFrame) -> PreparedData:
        self.validate(df)

        text_values = df[self.spec.textual_columns].fillna("").astype(str).to_numpy() if self.spec.textual_columns else np.empty((len(df), 0), dtype=object)

        cat_arrays = []
        encoders: Dict[str, LabelEncoder] = {}
        num_classes = []
        for col in self.spec.categorical_columns:
            encoder = LabelEncoder()
            vals = df[col].fillna("MISSING").astype(str)
            transformed = encoder.fit_transform(vals)
            cat_arrays.append(transformed)
            encoders[col] = encoder
            num_classes.append(len(encoder.classes_))
        categorical_indices = np.stack(cat_arrays, axis=1) if cat_arrays else np.empty((len(df), 0), dtype=np.int64)

        numerical_values = (
            df[self.spec.numerical_columns].apply(pd.to_numeric, errors="coerce").fillna(-1).to_numpy(dtype=np.float32)
            if self.spec.numerical_columns else np.empty((len(df), 0), dtype=np.float32)
        )

        ground_truth_parts = []
        if text_values.shape[1] > 0:
            ground_truth_parts.append(text_values)
        if categorical_indices.shape[1] > 0:
            ground_truth_parts.append(categorical_indices.astype(object))
        if numerical_values.shape[1] > 0:
            ground_truth_parts.append(numerical_values.astype(object))
        ground_truth = np.concatenate(ground_truth_parts, axis=1) if ground_truth_parts else np.empty((len(df), 0), dtype=object)

        modalities = (["txt"] * len(self.spec.textual_columns) + ["cat"] * len(self.spec.categorical_columns) + ["num"] * len(self.spec.numerical_columns))
        all_attribute_names = self.spec.textual_columns + self.spec.categorical_columns + self.spec.numerical_columns

        return PreparedData(
            dataframe=df,
            text_values=text_values,
            categorical_indices=categorical_indices,
            numerical_values=numerical_values,
            ground_truth=ground_truth,
            categorical_encoders=encoders,
            num_classes_per_cat_attr=num_classes,
            modalities=modalities,
            all_attribute_names=all_attribute_names,
        )
