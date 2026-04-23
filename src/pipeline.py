"""
Финальный пайплайн прогнозирования почасового трафика.

Загружает очищенный временной ряд, обучает выбранную модель на всей истории
и сохраняет прогноз на заданный горизонт в CSV. Дополнительно замеряет
время обучения и инференса.

Доступные модели:
    - snaive   : SeasonalNaive с периодом 168 (недельный бейзлайн, очень быстрый)
    - lgbm     : LightGBM через MLForecast с лагами 1/24/168 и скользящими средними
    - autoarima: AutoARIMA через StatsForecast (медленнее, но полностью параметрическая)

Запуск из корня проекта:
    python src/pipeline.py --model lgbm --horizon 24

Результат — файл results/forecast.csv со столбцами unique_id, ds, y_hat.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / 'data' / 'traffic_clean.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'results' / 'forecast.csv'

FREQ = 'h'
SEASON_WEEK = 168


def load_series(path: Path) -> pd.DataFrame:
    """Читает очищенный ряд в формате Nixtla (unique_id, ds, y)."""
    if not path.exists():
        raise FileNotFoundError(
            f'Не найден файл {path}. Сначала прогоните ноутбук, чтобы он '
            f'создал data/traffic_clean.csv.'
        )
    df = pd.read_csv(path, parse_dates=['ds'])
    required = {'unique_id', 'ds', 'y'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'В файле отсутствуют столбцы: {missing}')
    return df.sort_values(['unique_id', 'ds']).reset_index(drop=True)


def fit_predict_snaive(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, float, float]:
    """SeasonalNaive с периодом 168."""
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    sf = StatsForecast(
        models=[SeasonalNaive(season_length=SEASON_WEEK, alias='SNaive_168')],
        freq=FREQ,
        n_jobs=1,
    )
    t0 = time.perf_counter()
    sf.fit(df=df)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = sf.predict(h=horizon)
    pred_time = time.perf_counter() - t0

    pred = pred.rename(columns={'SNaive_168': 'y_hat'})[['unique_id', 'ds', 'y_hat']]
    return pred, fit_time, pred_time


def fit_predict_lgbm(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, float, float]:
    """LightGBM поверх mlforecast с лагами 1/24/168 и rolling mean."""
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean

    mlf = MLForecast(
        models={
            'LGBM': lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=64,
                random_state=42, verbose=-1,
            ),
        },
        freq=FREQ,
        lags=[1, 24, 168],
        lag_transforms={1: [RollingMean(window_size=24), RollingMean(window_size=168)]},
        date_features=['hour', 'dayofweek'],
    )
    t0 = time.perf_counter()
    mlf.fit(df=df)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = mlf.predict(h=horizon)
    pred_time = time.perf_counter() - t0

    pred = pred.rename(columns={'LGBM': 'y_hat'})[['unique_id', 'ds', 'y_hat']]
    return pred, fit_time, pred_time


def fit_predict_autoarima(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, float, float]:
    """AutoARIMA с суточной сезонностью. Обучение долгое, поэтому берём
    последние 60 суток ряда."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    tail_rows = 60 * 24
    df_tail = df.groupby('unique_id', group_keys=False).tail(tail_rows).reset_index(drop=True)

    sf = StatsForecast(
        models=[AutoARIMA(season_length=24, alias='AutoARIMA')],
        freq=FREQ,
        n_jobs=1,
    )
    t0 = time.perf_counter()
    sf.fit(df=df_tail)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = sf.predict(h=horizon)
    pred_time = time.perf_counter() - t0

    pred = pred.rename(columns={'AutoARIMA': 'y_hat'})[['unique_id', 'ds', 'y_hat']]
    return pred, fit_time, pred_time


MODEL_REGISTRY = {
    'snaive':    fit_predict_snaive,
    'lgbm':      fit_predict_lgbm,
    'autoarima': fit_predict_autoarima,
}


def run(input_path: Path, output_path: Path, model: str, horizon: int) -> pd.DataFrame:
    if model not in MODEL_REGISTRY:
        raise ValueError(
            f'Неизвестная модель: {model}. Доступные: {list(MODEL_REGISTRY)}'
        )

    print(f'[1/4] Загрузка данных из {input_path}')
    df = load_series(input_path)
    print(f'      Строк: {len(df):,}, период: {df["ds"].min()} — {df["ds"].max()}')

    print(f'[2/4] Обучение модели: {model}')
    fit_fn = MODEL_REGISTRY[model]
    pred, fit_time, pred_time = fit_fn(df, horizon)

    print(f'[3/4] Прогноз на {horizon} часов')
    print(f'      Время обучения: {fit_time:.2f} c')
    print(f'      Время инференса: {pred_time:.3f} c')
    print(f'      Среднее прогнозное значение: {pred["y_hat"].mean():.2f}')

    print(f'[4/4] Сохранение прогноза в {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output_path, index=False)
    print('      Готово.')

    return pred


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Traffic forecasting pipeline')
    parser.add_argument(
        '--model', type=str, default='lgbm',
        choices=sorted(MODEL_REGISTRY.keys()),
        help='Модель для прогноза (по умолчанию: lgbm)',
    )
    parser.add_argument(
        '--horizon', type=int, default=24,
        help='Горизонт прогноза в часах (по умолчанию: 24)',
    )
    parser.add_argument(
        '--input', type=Path, default=DEFAULT_INPUT,
        help=f'Путь к входному CSV (по умолчанию: {DEFAULT_INPUT})',
    )
    parser.add_argument(
        '--output', type=Path, default=DEFAULT_OUTPUT,
        help=f'Путь к выходному CSV (по умолчанию: {DEFAULT_OUTPUT})',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.input, args.output, args.model, args.horizon)
    return 0


if __name__ == '__main__':
    sys.exit(main())
