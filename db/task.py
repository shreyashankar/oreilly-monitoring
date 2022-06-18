import logging
import typing

from collections import defaultdict, namedtuple
from datetime import datetime, timedelta

import duckdb
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture

SubgroupKey = namedtuple(
    "SubgroupKey", ["subgroup_id", "window_start", "window_size"]
)
WindowKey = namedtuple("WindowKey", ["window_start", "window_size"])


class Task(object):
    def __init__(self, name: str, **kwargs) -> None:
        """Task constructor.
        Args:
            name (str): Name of the task.
        """
        self.name = name
        self.metrics = {}
        self.metric_fns = {}

        # Initialize IW parameters
        self.train_metric_vals = {}
        self.train_df_sample = None
        self.n_components = kwargs.get("n_components", 10)
        self.max_iter = kwargs.get("max_iter", 100)
        self.random_state = kwargs.get("random_state", 42)

        # Setup database connection and tables
        self.con = duckdb.connect(database=":memory")
        self._setup_tables()

    def _setup_tables(self) -> None:
        """Creates tables for predictions, feedbacks, and metrics."""
        self.con.execute(
            """CREATE TABLE predictions (id text PRIMARY KEY, t_pred timestamp, prediction real, subgroup_id integer, features MAP(text, real))"""
        )
        self.con.execute(
            """CREATE TABLE feedbacks (id text PRIMARY KEY, t_feedback timestamp, feedback real)"""
        )
        self.con.execute(
            """CREATE TABLE metrics (ts timestamp, name text, window_start timestamp, window_size integer, value real, num_labeled integer, num_unlabeled integer, labeled_value real, unlabeled_value real)"""
        )
        self.con.execute(
            """CREATE VIEW preds_with_feedbacks AS SELECT t_pred, t_feedback, prediction, feedback from predictions INNER JOIN feedbacks ON predictions.id = feedbacks.id"""
        )
        self.con.execute(
            """CREATE UNIQUE INDEX idx_metrics on metrics (name, window_start, window_size)"""
        )

        # Create indexes on ts, t_pred, t_feedback
        self.con.execute(
            """CREATE INDEX idx_preds_ts ON predictions (t_pred)"""
        )
        self.con.execute(
            """CREATE INDEX idx_feedbacks_ts ON feedbacks (t_feedback)"""
        )

        self.window_subgroup_counter = defaultdict(int)
        self.tainted_windows = set()

    def clear(self) -> None:
        """Method to clear all predictions and feedbacks for the task."""
        self.con.execute("DELETE FROM predictions")
        self.con.execute("DELETE FROM feedbacks")
        self.con.execute("DELETE FROM metrics")

        # Reset window subgroup counter
        self.window_subgroup_counter = defaultdict(int)
        self.tainted_windows = set()

    def register_training_set(
        self,
        train_df: pd.DataFrame,
        feature_cols: typing.List[str],
        label_col: str,
        prediction_col: str,
        n_rows: int = 100000,
    ) -> None:
        """Registers training set and creates binning function.
        Args:
            train_df (pd.DataFrame): Training dataframe
            feature_cols (typing.List[str]): List of feature columns
            loss_col (str): Column that represents model loss value
            n_rows (int, optional): Number of rows to use for subsample.
                Defaults to 100000.
            weighted_subsample (bool, optional): Whether to use weighted
                subsample. Defaults to True.
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.prediction_col = prediction_col

        # Subsample training dataframe based on losses if specified
        self.train_df_sample = train_df.sample(
            n=n_rows if n_rows < len(train_df) else len(train_df),
            random_state=self.random_state,
        )
        # Fit binning function to subsampled dataframe
        ## TODO: Fill out binning function
        self.binning_fn = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
        ).fit(
            self.train_df_sample[feature_cols + [self.prediction_col]].values
        )
        self.train_df_sample["iw_bin"] = self.binning_fn.predict(
            self.train_df_sample[feature_cols + [self.prediction_col]].values
        )

        # For each metric fn, compute training metrics
        self.compute_training_metrics()

    def compute_training_metrics(self, metric_fns: typing.Dict = None) -> None:
        """Computes training metrics for the task."""
        if metric_fns is None:
            metric_fns = self.metric_fns

        for metric_name, metric_fn in metric_fns.items():
            group_metric_vals = self.train_df_sample.groupby("iw_bin")[
                [self.label_col, self.prediction_col]
            ].apply(
                lambda x: metric_fn(
                    x[self.label_col].values, x[self.prediction_col].values
                )
            )
            self.train_metric_vals[metric_name] = group_metric_vals.to_dict()

        logging.info(self.train_metric_vals)

    def register_metric(
        self,
        name: str,
        metric_fn: typing.Callable,
        window_sizes: typing.List[int],
    ) -> None:
        """Method to register a metric function for the task.
        Args:
            name (str): Name of the metric, like accuracy.
            metric_fn (typing.Callable): Metric function that accepts 2 args --
                feedbacks and predictions -- and returns a float. Each arg is
                a list of floats.
            window_sizes (typing.List[int]): List of window sizes (in seconds).
                Does not support cumulative windows!
        """

        for ws in window_sizes:
            if ws not in self.metrics:
                self.metrics[ws] = []
            if name not in self.metrics[ws]:
                self.metrics[ws].append(name)
            self.metric_fns[name] = metric_fn

        # For each bin, compute the metric
        if self.train_df_sample is not None:
            self.compute_training_metrics(metric_fns={name: metric_fn})

    def get_subgroup_id(
        self, features: typing.Dict[str, float], prediction: float
    ) -> int:
        """Method to get the subgroup id for a given feature vector.
        Args:
            features (typing.Dict[str, float]): Feature vector.
        Returns:
            int: Subgroup id.
        """
        # TODO: Change this method if changing binning function
        return int(
            self.binning_fn.predict(
                np.array(
                    [features[f] for f in self.feature_cols] + [prediction]
                ).reshape(1, -1)
            )[0]
        )

    def log_prediction(
        self,
        ts: datetime,
        id: str,
        prediction: float,
        features: typing.Dict[str, float],
    ) -> None:
        """Method to log a prediction. Computes the subgroup id,
        increments its counter for relevant windows, and refreshes
        metrics.
        Args:
            ts (datetime): Time the prediction was made.
            id (str): Unique identifier for the prediction.
            prediction (float): Value of the prediction.
            features (typing.Dict[str, float]): Feature vector.
        """
        if isinstance(ts, np.datetime64):
            ts = datetime.fromtimestamp(ts.astype("O") / 1e9)
        # Compute subgroup id
        subgroup_id = self.get_subgroup_id(features, prediction)

        execute_stmt = (
            f"LIST_VALUE({','.join(['?' for _ in range(len(features))])})"
        )

        self.con.execute(
            f"INSERT INTO predictions VALUES (?, ?, ?, ?, MAP({execute_stmt}, {execute_stmt}))",
            (
                id,
                str(ts),
                prediction,
                subgroup_id,
                *features.keys(),
                *features.values(),
            ),
        )

        # Increment window counter for subgroup
        self.update_counters_only((subgroup_id, 1), ts)

    def log_feedback(
        self,
        ts: datetime,
        id: str,
        feedback: float,
    ) -> None:
        """Method to log a feedback / label. Gets the prediction for
        the given id, decrements its counter, and refreshes metrics.
        Args:
            ts (datetime): Time the feedback was given.
            id (str): Unique identifier for the feedback to join with
                predictions.
            feedback (float): Value of the feedback.
        """
        if isinstance(ts, np.datetime64):
            ts = datetime.fromtimestamp(ts.astype("O") / 1e9)
        self.con.execute(
            "INSERT INTO feedbacks VALUES (?, ?, ?)", (id, str(ts), feedback)
        )

        # Get prediction time and refresh that window's metrics
        rows = self.con.execute(
            "SELECT t_pred, subgroup_id FROM predictions WHERE id = ?",
            (id,),
        ).fetchall()
        prediction_ts, subgroup_id = rows[0]
        assert prediction_ts is not None
        self.update_counters_only((int(subgroup_id), -1), prediction_ts)

    def update_counters_only(
        self,
        offset_subgroup: typing.Tuple[int, int],
        current_ts: datetime = None,
    ):
        """Updates counters corresponding to labeled or unlabeled
        predictions.
        Args:
            current_ts (datetime, optional): Time of prediction or feedback.
                Defaults to datetime.now().
            offset_subgroup (typing.Tuple[int, int], optional): (Subgroup id,
                +1 or -1) to offset the window counter. Defaults to ().
        """
        if current_ts is None:
            current_ts = datetime.now()

        # Get all windows that need to be refreshed
        for ws, _ in self.metrics.items():
            lower_bound = current_ts - timedelta(seconds=ws)
            res = (
                self.con.execute(
                    """
                SELECT DISTINCT
                    t_pred
                FROM
                    predictions
                WHERE
                    t_pred > ?
                    AND t_pred <= ?
                """,
                    (str(lower_bound), str(current_ts)),
                )
                .fetchdf()["t_pred"]
                .values.astype("datetime64[s]")
                .tolist()
            )

            for t_pred in res:
                sk = SubgroupKey(offset_subgroup[0], t_pred, ws)
                self.window_subgroup_counter[sk] += offset_subgroup[1]
                self.tainted_windows.add(WindowKey(t_pred, ws))

    def compute_metrics(self, current_ts: datetime = None) -> None:
        """Gets all windows and refreshes the corresponding metrics.
        Args:
            current_ts (datetime, optional): Time of prediction or feedback.
                Defaults to datetime.now().
        """
        if current_ts is None:
            current_ts = datetime.now()
        metric_results = {
            "ts": [],
            "name": [],
            "window_start": [],
            "window_size": [],
            "value": [],
            "num_labeled": [],
            "num_unlabeled": [],
            "labeled_value": [],
            "unlabeled_value": [],
        }
        # Do the join between predictions and feedbacks once
        self.con.execute(
            "CREATE TABLE preds_with_feedbacks_materialized AS SELECT * FROM preds_with_feedbacks;"
        )

        for window in self.tainted_windows:
            # Get predictions & feedbacks for this window
            preds_and_feedbacks = self.con.execute(
                "SELECT feedback, prediction FROM preds_with_feedbacks_materialized WHERE t_pred >= ? AND t_pred < ?",
                (
                    str(window.window_start),
                    str(
                        window.window_start
                        + timedelta(seconds=window.window_size)
                    ),
                ),
            ).fetchdf()
            feedbacks = preds_and_feedbacks["feedback"].values
            predictions = preds_and_feedbacks["prediction"].values

            for metric_name, metric_fn in self.metric_fns.items():
                # Compute labeled metric value
                labeled_metric_value = 0.0
                num_labeled_points = 0
                unlabeled_metric_value = 0.0
                num_unlabeled_points = 0

                if len(feedbacks) > 0:
                    labeled_metric_value = metric_fn(
                        feedbacks.astype(float),
                        predictions.astype(float),
                    )
                    num_labeled_points = len(feedbacks)

                # Compute unlabeled metric value
                if self.train_df_sample is not None:
                    num_unlabeled_per_subgroup = {
                        subgroup_id: self.window_subgroup_counter[
                            SubgroupKey(
                                subgroup_id,
                                window.window_start,
                                window.window_size,
                            )
                        ]
                        for subgroup_id in self.train_metric_vals[
                            metric_name
                        ].keys()
                    }
                    num_unlabeled_points = sum(
                        num_unlabeled_per_subgroup.values()
                    )
                    if num_unlabeled_points > 0:
                        subgroup_density = {
                            subgroup_id: float(
                                num_unlabeled_per_subgroup[subgroup_id]
                                / num_unlabeled_points
                            )
                            for subgroup_id in num_unlabeled_per_subgroup.keys()
                        }
                        unlabeled_metric_value = sum(
                            [
                                subgroup_density[subgroup_id] * v
                                for subgroup_id, v in self.train_metric_vals[
                                    metric_name
                                ].items()
                            ]
                        )

                # Compute weights
                weighted_metric_value = None
                if num_labeled_points + num_unlabeled_points > 0:
                    labeled_weight = float(
                        num_labeled_points
                        / (num_labeled_points + num_unlabeled_points)
                    )
                    unlabeled_weight = 1.0 - labeled_weight
                    weighted_metric_value = (
                        labeled_weight * labeled_metric_value
                    ) + (unlabeled_weight * unlabeled_metric_value)

                metric_results["ts"].append(current_ts)
                metric_results["name"].append(metric_name)
                metric_results["window_start"].append(window.window_start)
                metric_results["window_size"].append(window.window_size)
                metric_results["value"].append(weighted_metric_value)
                metric_results["num_labeled"].append(num_labeled_points)
                metric_results["num_unlabeled"].append(num_unlabeled_points)
                metric_results["labeled_value"].append(labeled_metric_value)
                metric_results["unlabeled_value"].append(
                    unlabeled_metric_value
                )

        # Commit new metric values to the DB
        if len(metric_results["ts"]) > 0:
            temp_metric_df = pd.DataFrame(metric_results)

            # Create new metric df as view
            self.con.register("temp_metric_df", temp_metric_df)

            self.con.execute(
                "DELETE FROM metrics USING temp_metric_df WHERE metrics.name = temp_metric_df.name AND metrics.window_start = temp_metric_df.window_start AND metrics.window_size = temp_metric_df.window_size"
            )

            self.con.execute(
                "INSERT INTO metrics SELECT * FROM temp_metric_df"
            )

            self.con.unregister("temp_metric_df")

            del temp_metric_df

        self.tainted_windows = set()
        self.con.execute("DROP TABLE preds_with_feedbacks_materialized")

    def _get_max_ts(self) -> datetime:
        """Returns maximum prediction or feedback time.
        Returns:
            datetime: max ts.
        """
        max_t_pred = self.con.execute(
            "SELECT MAX(t_pred) as ts FROM predictions"
        ).fetchall()[0][0]
        max_t_feedback = self.con.execute(
            "SELECT MAX(t_feedback) as ts FROM feedbacks"
        ).fetchall()[0][0]

        tses = [max_t_pred, max_t_feedback]
        tses = [ts for ts in tses if ts is not None]
        return max(tses)

    def get_metrics(
        self, name: str = None, update: bool = True
    ) -> pd.DataFrame:
        """Method to get all metrics for the task.
        Args:
            name (str, optional): Name of the metric to get. Defaults to None.
            update (bool, optional): Whether to update the metric values.
        Returns:
            pd.DataFrame: DataFrame containing all metrics for the task.
        """
        if update:
            max_ts = self._get_max_ts()
            # Check if max_ts in metrics table
            existence = self.con.execute(
                "SELECT COUNT(*) FROM metrics WHERE ts = ?", (max_ts,)
            ).fetchall()[0][0]
            if existence == 0:
                self.compute_metrics(max_ts)

        all_metrics = (
            self.con.execute(
                "SELECT * FROM metrics ORDER BY window_start DESC, window_size ASC, name ASC"
            ).fetchdf()
            if not name
            else self.con.execute(
                f"SELECT * FROM metrics WHERE name='{name}' ORDER BY window_start DESC, window_size ASC"
            ).fetchdf()
        )
        return all_metrics

    def get_predictions(self) -> pd.DataFrame:
        """Method to get all predictions for the task.
        Returns:
            pd.DataFrame: DataFrame containing all predictions for the task.
        """
        return self.con.execute(
            "SELECT * FROM predictions ORDER BY t_pred ASC"
        ).fetchdf()

    def get_feedbacks(self) -> pd.DataFrame:
        """Method to get all feedbacks for the task.
        Returns:
            pd.DataFrame: DataFrame containing all feedbacks for the task.
        """
        return self.con.execute(
            "SELECT * FROM feedbacks ORDER BY t_feedback ASC"
        ).fetchdf()
