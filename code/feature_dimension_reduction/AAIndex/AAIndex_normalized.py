import numpy as np
import pandas as pd
from typing import Union, List
import os

class DataNormalizer:

    def __init__(self):
        self.stats = {}

    def z_score_normalize(self, data: Union[np.ndarray, pd.Series, pd.DataFrame],
                         columns: List[str] = None,
                         inplace: bool = False) -> Union[np.ndarray, pd.Series, pd.DataFrame]:

        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()

            if not inplace:
                data = data.copy()

            for col in columns:
                mean = data[col].mean()
                std = data[col].std()
                if std == 0:
                    std = 1e-10
                data[col] = (data[col] - mean) / std
                self.stats[f'z_score_{col}'] = {'mean': mean, 'std': std}

            return data if not inplace else None

        elif isinstance(data, pd.Series):
            mean = data.mean()
            std = data.std()
            if std == 0:
                std = 1e-10
            result = (data - mean) / std
            self.stats[f'z_score_{data.name}'] = {'mean': mean, 'std': std}
            return result if not inplace else None

        else:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1e-10
            result = (data - mean) / std
            self.stats['z_score'] = {'mean': mean, 'std': std}
            return result

    def min_max_normalize(self, data: Union[np.ndarray, pd.Series, pd.DataFrame],
                         columns: List[str] = None,
                         feature_range: tuple = (0, 1),
                         inplace: bool = False) -> Union[np.ndarray, pd.Series, pd.DataFrame]:

        min_val, max_val = feature_range

        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()

            if not inplace:
                data = data.copy()

            for col in columns:
                col_min = data[col].min()
                col_max = data[col].max()
                if col_max == col_min:
                    col_max = col_min + 1e-10
                data[col] = min_val + (data[col] - col_min) * (max_val - min_val) / (col_max - col_min)
                self.stats[f'min_max_{col}'] = {'min': col_min, 'max': col_max}

            return data if not inplace else None

        elif isinstance(data, pd.Series):
            d_min = data.min()
            d_max = data.max()
            if d_max == d_min:
                d_max = d_min + 1e-10
            result = min_val + (data - d_min) * (max_val - min_val) / (d_max - d_min)
            self.stats[f'min_max_{data.name}'] = {'min': d_min, 'max': d_max}
            return result if not inplace else None

        else:
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            data_max[data_max == data_min] = data_min[data_max == data_min] + 1e-10
            result = min_val + (data - data_min) * (max_val - min_val) / (data_max - data_min)
            self.stats['min_max'] = {'min': data_min, 'max': data_max}
            return result

    def log_transform(self, data: Union[np.ndarray, pd.Series, pd.DataFrame],
                     columns: List[str] = None,
                     inplace: bool = False,
                     add_constant: bool = True) -> Union[np.ndarray, pd.Series, pd.DataFrame]:

        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()

            if not inplace:
                data = data.copy()

            for col in columns:
                if add_constant:
                    constant = 1 - data[col].min() if data[col].min() <= 0 else 0
                    data[col] = np.log(data[col] + constant)
                    self.stats[f'log_{col}'] = {'constant': constant}
                else:
                    if any(data[col] <= 0):
                        raise ValueError(f"Column {col} contains non-positive values, cannot apply log transform without adding constant")
                    data[col] = np.log(data[col])
                    self.stats[f'log_{col}'] = {'constant': 0}

            return data if not inplace else None

        elif isinstance(data, pd.Series):
            if add_constant:
                constant = 1 - data.min() if data.min() <= 0 else 0
                result = np.log(data + constant)
                self.stats[f'log_{data.name}'] = {'constant': constant}
            else:
                if any(data <= 0):
                    raise ValueError("Series contains non-positive values, cannot apply log transform without adding constant")
                result = np.log(data)
                self.stats[f'log_{data.name}'] = {'constant': 0}

            return result if not inplace else None

        else:
            if add_constant:
                constant = 1 - np.min(data) if np.min(data) <= 0 else 0
                result = np.log(data + constant)
                self.stats['log'] = {'constant': constant}
            else:
                if np.any(data <= 0):
                    raise ValueError("Array contains non-positive values, cannot apply log transform without adding constant")
                result = np.log(data)
                self.stats['log'] = {'constant': 0}

            return result

    def get_stats(self):
        return self.stats

if __name__ == "__main__":
    file_path = "/Optimized-Ensemble-Learning-AIP/code/feature_dimension_reduction/AAIndex_RF.xlsx"

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist. Please check the path and file name.")
    else:
        try:
            df = pd.read_excel(file_path, header=None)
            df.info()
            rows, columns = df.shape

            normalizer = DataNormalizer()
            normalized_df = normalizer.min_max_normalize(df.copy())

            #output_file = "AAIndex_normalized.xlsx"
            #normalized_df.to_excel(output_file, index=False)
            #print(f"\nThe normalized data has been saved to {output_file}")

        except Exception as e:
            print(f"An unknown error occurred: {e}")

