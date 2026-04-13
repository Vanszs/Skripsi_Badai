"""
Central configuration for the final parameter contract.
"""

FINAL_TARGET_COLS = [
    'precipitation',
    'wind_speed_10m',
    'relative_humidity_2m',
]

FINAL_FEATURE_COLS = [
    'temperature_2m',
    'relative_humidity_2m',
    'dewpoint_2m',
    'surface_pressure',
    'wind_speed_10m',
    'wind_direction_10m',
    'cloud_cover',
    'precipitation_lag1',
    'elevation',
]

OPTIONAL_LEGACY_RENAMES = {
    'cloudcover': 'cloud_cover',
    'dew_point_2m': 'dewpoint_2m',
}


def harmonize_weather_columns(df):
    """
    Normalize legacy column names to the final contract used by the pipeline.
    """
    rename_map = {old: new for old, new in OPTIONAL_LEGACY_RENAMES.items() if old in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def validate_feature_schema(df, feature_cols=None, target_cols=None):
    """
    Raise a clear error when the dataset does not match the required contract.
    """
    feature_cols = feature_cols or FINAL_FEATURE_COLS
    target_cols = target_cols or FINAL_TARGET_COLS
    missing = [col for col in [*feature_cols, *target_cols] if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset schema mismatch. Missing required columns: {missing}"
        )


def validate_feature_values(df, feature_cols=None):
    """
    Raise an error if a required feature exists in schema but contains no usable values.
    """
    feature_cols = feature_cols or FINAL_FEATURE_COLS
    invalid = [col for col in feature_cols if col in df.columns and df[col].notna().sum() == 0]
    if invalid:
        raise ValueError(
            f"Dataset contains required features with all-NaN values: {invalid}"
        )
