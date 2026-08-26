"""CMR - Comparing Mean Responses Algorithm"""

# pylint: disable = line-too-long
import pandas as pd

from otava.analysis import TTestStats
from otava.series import  ChangePoint
from orion.logger import SingletonLogger
from orion.algorithms.algorithm import Algorithm


class CMR(Algorithm):
    """Implementation of the CMR algorithm
    Will Combine metrics into 2 lines and compare with a tolerancy to set pass fail

    Args:
        Algorithm (Algorithm): Inherits
    """


    def _analyze(self):
        """Analyze the dataframe with meaning any previous data and generate percent change with a current uuid

        Returns:
            series: data series that contains attributes and full dataframe
            change_points_by_metric: list of ChangePoints
        """
        logger = SingletonLogger.get_logger("Orion")
        logger.info("Starting analysis using CMR")
        if not (pd.api.types.is_numeric_dtype(self.dataframe["timestamp"]) and self.dataframe["timestamp"].astype(int).min() > 1e9):
            self.dataframe["timestamp"] = pd.to_datetime(self.dataframe["timestamp"])
            self.dataframe["timestamp"] = self.dataframe["timestamp"].astype(int) // 10**9

        if len(self.dataframe.index) == 1:
            series= self.setup_series()
            series.data = self.dataframe
            return series, {}
        # if larger than 2 rows, need to get the mean of 0 through -2
        self._original_dataframe = self.dataframe.copy()
        self.dataframe = self.combine_and_average_runs(self.dataframe)

        series= self.setup_series()

        df, change_points_by_metric = self.run_cmr(self.dataframe)
        series.data= df

        for metric, cps in change_points_by_metric.items():
            direction = self.metrics_config[metric]["direction"]
            if direction != 0:
                filtered = []
                for cp in cps:
                    delta = cp.stats.mean_2 - cp.stats.mean_1
                    if abs(cp.stats.mean_1) < 1e-12:
                        if abs(delta) > 1e-12 and delta * direction > 0:
                            filtered.append(cp)
                    elif (delta / cp.stats.mean_1 * 100) * direction > 0:
                        filtered.append(cp)
                change_points_by_metric[metric] = filtered

        self.regression_flag = any(change_points_by_metric.values())

        return series, change_points_by_metric


    def run_cmr(self, dataframe_list: pd.DataFrame):
        """
        Generate the percent difference in a 2 row dataframe

        Args:
            dataframe_list (pd.DataFrame): data frame of all data to compare on

        Returns:
            pd.Dataframe, dict[metric_name, ChangePoint]: Returned data frame and change points
        """
        metric_columns = self.metrics_config.keys()
        change_points_by_metric={ k:[] for k in metric_columns }

        for column in metric_columns:
            try:
                m1 = float(dataframe_list[column][0])
                m2 = float(dataframe_list[column][1])
            except (ValueError, TypeError):
                continue

            change_point = ChangePoint(metric=column,
                                index=1,
                                qhat=0.0,
                                time=0,
                                stats=TTestStats(
                                        mean_1=m1,
                                        mean_2=m2,
                                        std_1=0.0,
                                        std_2=0.0,
                                        pvalue=1.0
                                    ))
            change_points_by_metric[column].append(change_point)

        # based on change point generate pass/fail
        return dataframe_list, change_points_by_metric

    def combine_and_average_runs(self, dataFrame: pd.DataFrame):
        """
        If more than 1 previous run, mean data together into 1 single row
        Combine with current run into 1 data frame (current run being -1 index)

        Args:
            dataFrame (pd.DataFrame): data to combine into 2 rows

        Returns:
            pd.Dataframe: data frame of most recent run and averaged previous runs
        """
        last_row = dataFrame.tail(1)
        baseline_runs = dataFrame[:-1]

        # Preserve metadata from the most recent baseline run.
        baseline_data = {
            column: [baseline_runs[column].iloc[-1]]
            for column in dataFrame.columns
        }

        # Average only configured metrics across the baseline runs.
        for column in self.metrics_config:
            if column not in baseline_runs.columns:
                continue

            numeric_col = pd.to_numeric(
                baseline_runs[column], errors='coerce'
            )
            baseline_data[column] = [
                numeric_col.mean()
                if numeric_col.notna().any()
                else float('nan')
            ]

        baseline_row = pd.DataFrame(baseline_data)
        result = pd.concat([baseline_row, last_row], ignore_index=True)
        return result
