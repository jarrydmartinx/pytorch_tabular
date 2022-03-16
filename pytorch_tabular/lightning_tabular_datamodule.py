# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Data Module"""
import logging
import re
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import category_encoders as ce
from sklearn.base import TransformerMixin, copy
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)

from categorical_encoders import OrdinalEncoder

logger = logging.getLogger(__name__)

class TabDatamodule(LightningDataModule):

    CONTINUOUS_TRANSFORMS = {
        "quantile_uniform": {
            "callable": QuantileTransformer,
            "params": dict(output_distribution="uniform", random_state=42),
        },
        "quantile_normal": {
            "callable": QuantileTransformer,
            "params": dict(output_distribution="normal", random_state=42),
        },
        "box-cox": {
            "callable": PowerTransformer,
            "params": dict(method="box-cox", standardize=False),
        },
        "yeo-johnson": {
            "callable": PowerTransformer,
            "params": dict(method="yeo-johnson", standardize=False),
        },
    }

    def __init__(
        self,
        train_df: pd.DataFrame,       
        val_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        task: str = "metric_learning",
        target_cols: List[str] = [],
        continuous_cols: List[str] = [],
        categorical_cols: List = [],
        embedding_dims: Optional[List[int]] = None,
        date_cols: List[Tuple[str, str]] = [],
        encode_date_columns: bool = False,
        validation_split: Optional[float] = 0.25,
        batch_size: int = 16,
        continuous_feature_transform: Optional[str] = None,
        normalize_continuous_features: bool = False, 
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = False,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None):
        
        """The Pytorch Lightning Datamodule for Tabular Data
        """
        super().__init__()
        self.train = train_df.copy()
        self.validation = val_df if val_df is None else val_df.copy()
        self.test = test_df if test_df is None else test_df.copy()
        self._fitted = False
        self.save_hyperparameters()
        if target_transform is not None:
            if isinstance(target_transform, Iterable):
                target_transform = FunctionTransformer(
                    func=target_transform[0], inverse_func=target_transform[1]
                )
            self.hparams["do_target_transform"] = True
        else:
            self.hparams["do_target_transform"] = False

    def set_output_dim(self) -> None:
        if self.hparams.task == "regression":
            self.hparams.output_dim = len(self.hparams.target)
        elif self.hparams.task == "classification":
            self.hparams.output_dim = len(self.train[self.hparams.target[0]].unique())
        # elif self.hparams.task == "ssl":
        #     self.output_dim = len(self.categorical_cols) + len(
        #         self.hparams.continuous_cols)
        elif self.hparams.task == "metric_learning":
            self.output_dim = len(self.hparams.categorical_cols) + len(
                self.hparams.continuous_cols)

        self.categorical_cardinality = [
            int(self.train[col].fillna("NA").nunique()) + 1
            for col in self.hparams.categorical_cols
        ]
        if (
            hasattr(self.hparams, "embedding_dims")
            and self.embedding_dims is None
        ):
            self.embedding_dims = [
                (x, min(50, (x + 1) // 2))
                for x in self.hparams.categorical_cardinality
            ]

    def preprocess_data(
        self, data: pd.DataFrame, stage: str = "inference"
    ) -> Tuple[pd.DataFrame, list]:
        """The preprocessing, like Categorical Encoding, Normalization, etc. which any dataframe should undergo before feeding into the dataloder

        Args:
            data (pd.DataFrame): A dataframe with the features and target
            stage (str, optional): Internal parameter. Used to distinguisj between fit and inference. Defaults to "inference".

        Returns:
            tuple[pd.DataFrame, list]: Returns the processed dataframe and the added features(list) as a tuple
        """
        logger.info(f"Preprocessing data: Stage: {stage}...")
        added_features = None
        if self.hparams.date_cols:
            for field_name, freq in self.hparams.date_cols:
                data = self.make_date(data, field_name)
                data, added_features = self.add_datepart(
                    data, field_name, frequency=freq, prefix=None, drop=True
                )
        # The only features that are added are the date features extracted
        # from the date which are categorical in nature
        if (added_features is not None) and (stage == "fit"):
            logger.debug(
                f"Added {added_features} features after encoding the date_columns"
            )
            self.hparams.categorical_cols += added_features
            self.hparams.categorical_dim = (
                len(self.hparams.categorical_cols)
                if self.hparams.categorical_cols is not None
                else 0
            )
        # Encoding Categorical Columns
        if len(self.hparams.categorical_cols) > 0:
            if stage == "fit":
                if self.do_leave_one_out_encoder():
                    logger.debug("Encoding Categorical Columns using LeavOneOutEncoder")
                    self.categorical_encoder = ce.LeaveOneOutEncoder(
                        cols=self.hparams.categorical_cols, random_state=42
                    )
                    # Multi-Target Regression uses the first target to encode the categorical columns
                    if len(self.hparams.target) > 1:
                        logger.warning(
                            f"Multi-Target Regression: using the first target({self.hparams.target[0]}) to encode the categorical columns"
                        )
                    data = self.categorical_encoder.fit_transform(
                        data, data[self.hparams.target[0]]
                    )
                else:
                    logger.debug("Encoding Categorical Columns using OrdinalEncoder")
                    self.categorical_encoder = OrdinalEncoder(
                        cols=self.hparams.categorical_cols
                    )
                    data = self.categorical_encoder.fit_transform(data)
            else:
                data = self.categorical_encoder.transform(data)

        # Transforming Continuous Columns
        if (self.hparams.continuous_feature_transform is not None) and (
            len(self.hparams.continuous_cols) > 0
        ):
            if stage == "fit":
                transform = self.CONTINUOUS_TRANSFORMS[
                    self.hparams.continuous_feature_transform
                ]
                self.continuous_transform = transform["callable"](**transform["params"])
                # TODO implement quantile noise
                data.loc[
                    :, self.hparams.continuous_cols
                ] = self.continuous_transform.fit_transform(
                    data.loc[:, self.hparams.continuous_cols]
                )
            else:
                data.loc[
                    :, self.hparams.continuous_cols
                ] = self.continuous_transform.transform(
                    data.loc[:, self.hparams.continuous_cols]
                )

        # Normalizing Continuous Columns
        if (self.hparams.normalize_continuous_features) and (
            len(self.hparams.continuous_cols) > 0
        ):
            if stage == "fit":
                self.scaler = StandardScaler()
                data.loc[:, self.hparams.continuous_cols] = self.scaler.fit_transform(
                    data.loc[:, self.hparams.continuous_cols]
                )
            else:
                data.loc[:, self.hparams.continuous_cols] = self.scaler.transform(
                    data.loc[:, self.hparams.continuous_cols]
                )

        # Converting target labels to a 0 indexed label
        if self.hparams.task == "classification":
            if stage == "fit":
                self.label_encoder = LabelEncoder()
                data[self.hparams.target[0]] = self.label_encoder.fit_transform(
                    data[self.hparams.target[0]]
                )
            else:
                if self.hparams.target[0] in data.columns:
                    data[self.hparams.target[0]] = self.label_encoder.transform(
                        data[self.hparams.target[0]]
                    )
        # Target Transforms
        if all([col in data.columns for col in self.hparams.target]):
            if self.do_target_transform:
                if stage == "fit":
                    target_transforms = []
                    for col in self.hparams.target:
                        _target_transform = copy.deepcopy(
                            self.target_transform
                        )
                        data[col] = _target_transform.fit_transform(
                            data[col].values.reshape(-1, 1)
                        )
                        target_transforms.append(_target_transform)
                    self.target_transforms = target_transforms
                else:
                    for col, _target_transform in zip(
                        self.hparams.target, self.target_transforms
                    ):
                        data[col] = _target_transform.transform(
                            data[col].values.reshape(-1, 1)
                        )
        return data, added_features

    def setup(self, stage: Optional[str] = None) -> None:
        """Data Operations you want to perform on all GPUs, like train-test split, transformations, etc.
        This is called before accessing the dataloaders

        Args:
            stage (Optional[str], optional): Internal parameter to distinguish between fit and inference. Defaults to None.
        """
        if stage == "fit" or stage is None:
            if self.validation is None:
                logger.debug(
                    f"No validation data provided. Using {self.hparams.validation_split*100}% of train data as validation"
                )
                val_idx = self.train.sample(
                    int(self.hparams.validation_split * len(self.train)), random_state=42
                ).index
                self.validation = self.train[self.train.index.isin(val_idx)]
                self.train = self.train[~self.train.index.isin(val_idx)]
            else:
                self.validation = self.validation.copy()
            # Preprocessing Train, Validation
            self.train, _ = self.preprocess_data(self.train, stage="fit")
            self.validation, _ = self.preprocess_data(
                self.validation, stage="inference"
            )
            if self.test is not None:
                self.test, _ = self.preprocess_data(self.test, stage="inference")
            # Calculating the categorical dims and embedding dims etc and updating the hparams
            self.update_hparams()
            self._fitted = True

    # adapted from gluonts
    @classmethod
    def time_features_from_frequency_str(cls, freq_str: str) -> List[str]:
        """
        Returns a list of time features that will be appropriate for the given frequency string.

        Parameters
        ----------

        freq_str
            Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

        """

        features_by_offsets = {
            offsets.YearBegin: [],
            offsets.YearEnd: [],
            offsets.MonthBegin: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
            ],
            offsets.MonthEnd: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
            ],
            offsets.Week: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "Week",
            ],
            offsets.Day: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "Week" "Day",
                "Dayofweek",
                "Dayofyear",
            ],
            offsets.BusinessDay: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "Week" "Day",
                "Dayofweek",
                "Dayofyear",
            ],
            offsets.Hour: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "Week" "Day",
                "Dayofweek",
                "Dayofyear",
                "Hour",
            ],
            offsets.Minute: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "Week" "Day",
                "Dayofweek",
                "Dayofyear",
                "Hour",
                "Minute",
            ],
        }

        offset = to_offset(freq_str)

        for offset_type, feature in features_by_offsets.items():
            if isinstance(offset, offset_type):
                return feature

        supported_freq_msg = f"""
        Unsupported frequency {freq_str}

        The following frequencies are supported:

            Y, YS   - yearly
                alias: A
            M, MS   - monthly
            W   - weekly
            D   - daily
            B   - business days
            H   - hourly
            T   - minutely
                alias: min
        """
        raise RuntimeError(supported_freq_msg)

    # adapted from fastai
    @classmethod
    def make_date(cls, df: pd.DataFrame, date_field: str):
        "Make sure `df[date_field]` is of the right date type."
        field_dtype = df[date_field].dtype
        if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            field_dtype = np.datetime64
        if not np.issubdtype(field_dtype, np.datetime64):
            df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)
        return df

    # adapted from fastai
    @classmethod
    def add_datepart(
        cls,
        df: pd.DataFrame,
        field_name: str,
        frequency: str,
        prefix: str = None,
        drop: bool = True,
    ):
        "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
        field = df[field_name]
        prefix = (
            re.sub("[Dd]ate$", "", field_name) if prefix is None else prefix
        ) + "_"
        attr = cls.time_features_from_frequency_str(frequency)
        added_features = []
        for n in attr:
            if n == "Week":
                continue
            df[prefix + n] = getattr(field.dt, n.lower())
            added_features.append(prefix + n)
        # Pandas removed `dt.week` in v1.1.10
        if "Week" in attr:
            week = (
                field.dt.isocalendar().week
                if hasattr(field.dt, "isocalendar")
                else field.dt.week
            )
            df.insert(3, prefix + "Week", week)
            added_features.append(prefix + "Week")
        # TODO Not adding Elapsed by default. Need to route it through hparams
        # mask = ~field.isna()
        # df[prefix + "Elapsed"] = np.where(
        #     mask, field.values.astype(np.int64) // 10 ** 9, None
        # )
        # added_features.append(prefix + "Elapsed")
        if drop:
            df.drop(field_name, axis=1, inplace=True)

        # Removing features woth zero variations
        # for col in added_features:
        #     if len(df[col].unique()) == 1:
        #         df.drop(columns=col, inplace=True)
        #         added_features.remove(col)
        return df, added_features

    def train_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Function that loads the train set."""
        dataset = TabularDataset(
            task=self.hparams.task,
            data=self.train,
            categorical_cols=self.hparams.categorical_cols,
            continuous_cols=self.hparams.continuous_cols,
            embed_categorical=(not self.do_leave_one_out_encoder()),
            target=self.target,
        )
        return DataLoader(
            dataset,
            batch_size if batch_size is not None else self.batch_size,
            shuffle=True if self.train_sampler is None else False,
            num_workers=self.hparams.num_workers,
            sampler=self.train_sampler,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Function that loads the validation set."""
        dataset = TabularDataset(
            task=self.hparams.task,
            data=self.validation,
            categorical_cols=self.hparams.categorical_cols,
            continuous_cols=self.hparams.continuous_cols,
            embed_categorical=(not self.do_leave_one_out_encoder()),
            target=self.target,
        )
        return DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Function that loads the validation set."""
        if self.test is not None:
            dataset = TabularDataset(
                task=self.hparams.task,
                data=self.test,
                categorical_cols=self.hparams.categorical_cols,
                continuous_cols=self.hparams.continuous_cols,
                embed_categorical=(not self.do_leave_one_out_encoder()),
                target=self.target,
            )
            return DataLoader(
                dataset,
                self.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )

    def prepare_inference_dataloader(self, df: pd.DataFrame) -> DataLoader:
        """Function that prepares and loads the new data.

        Args:
            df (pd.DataFrame): Dataframe with the features and target

        Returns:
            DataLoader: The dataloader for the passed in dataframe
        """
        df = df.copy()
        if len(set(self.target) - set(df.columns)) > 0:
            if self.hparams.task == "classification":
                df.loc[:, self.target] = np.array(
                    [self.label_encoder.classes_[0]] * len(df)
                )
            else:
                df.loc[:, self.target] = np.zeros((len(df), len(self.target)))
        df, _ = self.preprocess_data(df, stage="inference")

        dataset = TabularDataset(
            task=self.hparams.task,
            data=df,
            categorical_cols=self.hparams.categorical_cols,
            continuous_cols=self.hparams.continuous_cols,
            embed_categorical=(not self.do_leave_one_out_encoder()),
            target=self.target
            if all([col in df.columns for col in self.target])
            else None,
        )
        return DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


class TabularDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        task: str,
        continuous_cols: List[str] = None,
        categorical_cols: List[str] = None,
        embed_categorical: bool = True,
        target: List[str] = None,
    ):
        """Dataset to Load Tabular Data

        Args:
            data (pd.DataFrame): Pandas DataFrame to load during training
            task (str): Whether it is a classification or regression task. If classification, it returns a LongTensor as target
            continuous_cols (List[str], optional): A list of names of continuous columns. Defaults to None.
            categorical_cols (List[str], optional): A list of names of categorical columns.
            These columns must be ordinal encoded beforehand. Defaults to None.
            embed_categorical (bool): Flag to tell the dataset whether to convert categorical columns to LongTensor or retain as float.
            If we are going to embed categorical cols with an embedding layer, we need to convert the columns to LongTensor
            target (List[str], optional): A list of strings with target column name(s). Defaults to None.
        """

        self.task = task
        self.n = data.shape[0]

        if target:
            self.y = data[target].astype(np.float32).values
            if isinstance(target, str):
                self.y = self.y.reshape(-1, 1)  # .astype(np.int64)
        else:
            self.y = np.zeros((self.n, 1))  # .astype(np.int64)

        if task == "classification":
            self.y = self.y.astype(np.int64)
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.continuous_cols = continuous_cols if continuous_cols else []

        if self.continuous_cols:
            self.continuous_X = data[self.continuous_cols].astype(np.float32).values

        if self.categorical_cols:
            self.categorical_X = data[categorical_cols]
            if embed_categorical:
                self.categorical_X = self.categorical_X.astype(np.int64).values
            else:
                self.categorical_X = self.categorical_X.astype(np.float32).values

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return {
            "target": self.y[idx],
            "continuous": self.continuous_X[idx]
            if self.continuous_cols
            else Tensor(),
            "categorical": self.categorical_X[idx]
            if self.categorical_cols
            else Tensor(),
        }
