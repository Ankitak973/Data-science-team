"""
Helper functions for data cleaning operations
Following industry best practices for data quality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_excluded_columns(df: pd.DataFrame, target_column: Optional[str] = None) -> set:
    """
    Identify columns that should be excluded from encoding/scaling:
      - The target column
      - ID-like columns (ending in "id" or named "id")
      - High-cardinality columns that look like IDs
    """
    exclude_cols = set()

    # Always exclude target
    if target_column and target_column in df.columns:
        exclude_cols.add(target_column)

    # Auto-detect ID-like columns
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower.endswith("id") or col_lower == "id":
            exclude_cols.add(col)

        # High-cardinality IDs (heuristic: > 95% unique)
        # Check only if not float (floats rarely IDs unless integer-like)
        if not pd.api.types.is_float_dtype(df[col]):
            try:
                if df[col].nunique() / len(df) > 0.95:
                    exclude_cols.add(col)
            except Exception:
                pass

    return exclude_cols


class DataCleaner:
    """Industry-standard data cleaning utilities"""

    def __init__(self, schema: Optional[Dict[str, Any]] = None, business_rules: Optional[List[Dict]] = None):
        self.cleaning_log = []
        self.stats = {
            'duplicates_removed': 0,
            'missing_values_handled': 0,
            'outliers_detected': 0,
            'columns_cleaned': 0
        }
        # Optional schema and rules can be provided
        self.schema = schema or {}
        self.business_rules = business_rules or []

    def separate_target(self, df: pd.DataFrame, target_column: Optional[str]):
        if target_column and target_column in df.columns:
            y = df[target_column].copy()

            # Convert common categorical targets to numeric
            if y.dtype == "object":
                y = y.str.lower().map({"yes": 1, "no": 0})

            X = df.drop(columns=[target_column])
            self.log_action("Target Handling", f"Separated target column '{target_column}'")
            return X, y

        return df, None

    def log_action(self, action: str, details: str):
        """Track all cleaning actions for transparency"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {action}: {details}"
        self.cleaning_log.append(log_entry)
        logger.info(log_entry)

    @staticmethod
    def clean_string(s: str) -> str:
        """Normalized string: lowercase, strip, snake_case, alpha-numeric only"""
        if not isinstance(s, str):
            return str(s)
        import re
        # lower & strip
        s = s.strip().lower()
        # replace spaces with underscore
        s = re.sub(r'\s+', '_', s)
        # remove non-alphanumeric (except underscore)
        s = re.sub(r'[^\w_]', '', s)
        return s

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert column names to snake_case using clean_string"""
        df_clean = df.copy()
        df_clean.columns = [self.clean_string(c) for c in df_clean.columns]
        self.log_action("Normalize Columns", "Converted column names to snake_case")
        return df_clean

    def detect_data_types(self, df: pd.DataFrame) -> Dict:
        """Intelligently detect and suggest correct data types"""
        type_suggestions = {}

        for col in df.columns:
            current_type = df[col].dtype

            # Try to detect dates
            if current_type == 'object':
                try:
                    coerced = pd.to_datetime(df[col], errors='coerce')
                    non_null = coerced.notnull().sum()
                    if non_null > len(df) * 0.5:
                        type_suggestions[col] = 'datetime'
                        continue
                except Exception:
                    pass

            # Detect numeric columns stored as strings
            if current_type == 'object':
                try:
                    numeric_coerced = pd.to_numeric(df[col], errors='coerce')
                    numeric_non_null = numeric_coerced.notnull().sum()
                    if numeric_non_null > len(df) * 0.8:
                        type_suggestions[col] = 'numeric'
                    else:
                        type_suggestions[col] = 'text'
                except Exception:
                    type_suggestions[col] = 'text'
            else:
                type_suggestions[col] = str(current_type)

        return type_suggestions

    # --- new helpers for 13-step pipeline ---

    def apply_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply schema corrections (if a schema is provided)"""
        df_clean = df.copy()
        if not self.schema:
            self.log_action("Schema", "No schema provided - skipping schema enforcement")
            return df_clean

        for col, expected in self.schema.items():
            if col not in df_clean.columns:
                self.log_action("Schema", f"Column '{col}' missing from data")
                continue
            # expected example: {'dtype': 'numeric'/'datetime'/'text'}
            dtype = expected.get('dtype')
            if dtype == 'numeric':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                self.log_action("Schema", f"Coerced '{col}' to numeric")
            elif dtype == 'datetime':
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                self.log_action("Schema", f"Coerced '{col}' to datetime")
            elif dtype == 'text':
                df_clean[col] = df_clean[col].astype(str)
                self.log_action("Schema", f"Coerced '{col}' to text")

        return df_clean

    def correct_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt intelligent dtype corrections for common cases"""
        df_clean = df.copy()

        suggestions = self.detect_data_types(df_clean)
        for col, typ in suggestions.items():
            try:
                if typ == 'numeric':
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                elif typ == 'datetime':
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                # text -> keep as object / string
            except Exception as e:
                self.log_action("Dtype Correction", f"Failed to coerce '{col}': {e}")

        self.log_action("Dtype Correction", f"Suggested dtype corrections applied for {len(suggestions)} columns")
        return df_clean

    def remove_duplicates(self, df: pd.DataFrame, subset=None) -> pd.DataFrame:
        """Remove duplicate rows with logging"""
        initial_rows = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep='first')
        duplicates_removed = initial_rows - len(df_clean)

        self.stats['duplicates_removed'] = duplicates_removed
        self.log_action(
            "Remove Duplicates",
            f"Removed {duplicates_removed} duplicate rows ({duplicates_removed/initial_rows*100:.2f}%)"
        )

        return df_clean

    def handle_missing_values(self, df: pd.DataFrame, strategy='smart') -> pd.DataFrame:
        """Handle missing values using intelligent strategies"""
        df_clean = df.copy()
        missing_summary = df.isnull().sum()

        for col in df.columns:
            missing_count = int(missing_summary[col])

            if missing_count == 0:
                continue

            missing_pct = (missing_count / len(df)) * 100

            # Drop column if >50% missing
            if missing_pct > 50:
                df_clean = df_clean.drop(columns=[col])
                self.log_action(
                    "Drop Column",
                    f"Dropped '{col}' - {missing_pct:.1f}% missing values"
                )
                continue

            # Handle based on data type
            if pd.api.types.is_numeric_dtype(df[col]):
                # Use median for numeric (more robust than mean)
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                self.log_action(
                    "Fill Missing",
                    f"Filled {missing_count} missing values in '{col}' with median ({median_val})"
                )
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Fill with mode or forward fill
                if df_clean[col].mode().empty:
                    df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                self.log_action(
                    "Fill Missing",
                    f"Filled {missing_count} missing datetime values in '{col}'"
                )
            else:
                # Use mode (most common) for categorical/text
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_value)
                self.log_action(
                    "Fill Missing",
                    f"Filled {missing_count} missing values in '{col}' with mode: '{mode_value}'"
                )

            self.stats['missing_values_handled'] += missing_count

        return df_clean

    def detect_outliers(self, df: pd.DataFrame, method='iqr') -> Dict:
        """Detect outliers using IQR method"""
        outlier_info = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            non_null = df[col].dropna()
            if non_null.empty:
                continue

            Q1 = non_null.quantile(0.25)
            Q3 = non_null.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            if len(outliers) > 0:
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                self.log_action(
                    "Detect Outliers",
                    f"Found {len(outliers)} outliers in '{col}' ({outlier_info[col]['percentage']:.2f}%)"
                )

        self.stats['outliers_detected'] = sum(info['count'] for info in outlier_info.values())
        return outlier_info

    def standardize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text columns - trim, lowercase, etc."""
        df_clean = df.copy()
        text_cols = df.select_dtypes(include=['object', 'string']).columns

        for col in text_cols:
            # Strip whitespace
            df_clean[col] = df_clean[col].astype(str).str.strip()

            # Remove extra spaces
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)

            # Optionally lowercase
            df_clean[col] = df_clean[col].str.lower()

            self.log_action(
                "Standardize Text",
                f"Cleaned text in column '{col}'"
            )
            self.stats['columns_cleaned'] += 1

        return df_clean

    def drop_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop ID-like columns (ending in 'id')"""
        df_clean = df.copy()
        id_cols = [c for c in df_clean.columns if str(c).lower().endswith("id")]
        
        if id_cols:
            df_clean = df_clean.drop(columns=id_cols)
            self.log_action("Drop IDs", f"Dropped ID columns: {id_cols}")
        
        return df_clean

    def encode_categoricals(self, df: pd.DataFrame, max_onehot: int = 30, target_column: Optional[str] = None) -> pd.DataFrame:
        """Encode categorical variables, excluding target"""
        # -----------------------------
        # Exclusion logic (Target only, IDs should be dropped beforehand)
        # -----------------------------
        exclude_cols = set()
        if target_column and target_column in df.columns:
            exclude_cols.add(target_column)

        # -----------------------------
        # Categorical Encoding
        # -----------------------------
        df_encoded = df.copy()
        
        cat_cols = df_encoded.select_dtypes(include=["object", "category"]).columns
        cat_cols = [c for c in cat_cols if c not in exclude_cols]

        # 🔴 SAFETY CHECK
        if len(cat_cols) > 0:
            df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)
            self.log_action("Encode", f"One-hot encoded columns: {cat_cols}")
        
        return df_encoded

    def scale_numeric(self, df: pd.DataFrame, method: str = 'standard', target_column: Optional[str] = None) -> pd.DataFrame:
        """Scale numeric columns, excluding target"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        df_clean = df.copy()
        
        # Identify columns to exclude (Target only)
        exclude_cols = set()
        if target_column and target_column in df.columns:
            exclude_cols.add(target_column)

        num_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns 
                    if c not in exclude_cols]

        if len(num_cols) == 0:
            self.log_action("Scale", "No numeric columns to scale")
            return df_clean

        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
        self.log_action("Scale", f"Scaled {len(num_cols)} numeric columns using {method} scaler")
        return df_clean

    def detect_leakage(self, df: pd.DataFrame, target_col: Optional[str] = None) -> List[str]:
        """
        Basic leakage detection:
         - identical columns to target
         - extremely high correlation with target (if numeric)
         - features with timestamps after target timestamp (if datetime)
        """
        issues = []
        if target_col and target_col in df.columns:
            # identical values
            for col in df.columns:
                if col == target_col:
                    continue
                try:
                    if df[col].equals(df[target_col]):
                        issues.append(f"Column '{col}' is identical to target '{target_col}'")
                except Exception:
                    pass

            # numeric correlation
            if pd.api.types.is_numeric_dtype(df[target_col]):
                target_series = pd.to_numeric(df[target_col], errors='coerce')
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col == target_col:
                        continue
                    corr = target_series.corr(pd.to_numeric(df[col], errors='coerce'))
                    if corr is not None and abs(corr) >= 0.95:
                        issues.append(f"High correlation ({corr:.2f}) between '{col}' and target '{target_col}'")

        # datetime leakage heuristic: if any datetime column has max > target max or > now
        datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns
        for col in datetime_cols:
            try:
                max_val = df[col].max()
                if isinstance(max_val, pd.Timestamp) and max_val > pd.Timestamp.now():
                    issues.append(f"Datetime column '{col}' contains future timestamps (max={max_val})")
            except Exception:
                pass

        if issues:
            self.log_action("Leakage Detection", f"Leakage issues found: {issues}")
        else:
            self.log_action("Leakage Detection", "No obvious leakage detected")

        return issues

    def apply_business_rules(self, df: pd.DataFrame) -> List[str]:
        """
        Apply simple business rules provided as list of dicts.
        Rule example: {'column': 'age', 'op': '>=', 'value': 0, 'action': 'clip'/'drop'/'flag'}
        Returns list of violation descriptions (if any).
        """
        violations = []
        for rule in self.business_rules:
            col = rule.get('column')
            op = rule.get('op')
            val = rule.get('value')
            action = rule.get('action', 'flag')

            if col not in df.columns:
                violations.append(f"Rule column missing: {col}")
                continue

            try:
                if op == '>=':
                    mask = df[col] < val
                elif op == '<=':
                    mask = df[col] > val
                elif op == '>':
                    mask = df[col] <= val
                elif op == '<':
                    mask = df[col] >= val
                elif op == '==':
                    mask = df[col] != val
                else:
                    mask = pd.Series([False] * len(df))

                count = mask.sum()
                if count > 0:
                    if action == 'drop':
                        df.drop(df[mask].index, inplace=True)
                        violations.append(f"Dropped {count} rows violating {col} {op} {val}")
                    elif action == 'clip' and pd.api.types.is_numeric_dtype(df[col]):
                        if op in ('>=', '>'):
                            df.loc[mask, col] = val
                        else:
                            df.loc[mask, col] = val
                        violations.append(f"Clipped {count} values in {col} to satisfy {op} {val}")
                    else:
                        violations.append(f"{count} rows violate rule {col} {op} {val} (action={action})")
            except Exception as e:
                violations.append(f"Error evaluating rule for {col}: {e}")

        if violations:
            self.log_action("Business Rules", f"Violations: {violations}")
        else:
            self.log_action("Business Rules", "No violations detected")

        return violations

    def generate_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic metadata about the cleaned dataset"""
        metadata = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'dtypes': {col: str(dt) for col, dt in df.dtypes.items()},
            'missing_per_column': df.isnull().sum().to_dict(),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'generated_at': datetime.now().isoformat()
        }
        self.log_action("Metadata", f"Generated metadata: rows={metadata['rows']}, cols={metadata['columns']}")
        return metadata

    def save_cleaned(self, df: pd.DataFrame, path: str) -> str:
        """Save cleaned DataFrame to disk and log it"""
        try:
            df.to_csv(path, index=False)
            self.log_action("Save Cleaned", f"Saved cleaned dataset to {path}")
            return path
        except Exception as e:
            self.log_action("Save Error", f"Failed to save cleaned dataset: {e}")
            raise

    def get_cleaning_report(self) -> str:
        """Generate a comprehensive cleaning report"""
        report = "="*50 + "\n"
        report += "DATA CLEANING REPORT\n"
        report += "="*50 + "\n\n"

        report += "STATISTICS:\n"
        for key, value in self.stats.items():
            report += f"  • {key.replace('_', ' ').title()}: {value}\n"

        report += "\nCLEANING ACTIONS:\n"
        for log in self.cleaning_log:
            report += f"  {log}\n"

        report += "\n" + "="*50
        return report


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate dataframe meets basic quality standards"""
    issues = []

    # Check if empty
    if df.empty:
        issues.append("DataFrame is empty")

    # Check for columns
    if len(df.columns) == 0:
        issues.append("DataFrame has no columns")

    # Check for excessive missing data
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 and len(df.columns) > 0 else 0
    if missing_pct > 30:
        issues.append(f"High missing data: {missing_pct:.1f}%")

    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        issues.append("Duplicate column names detected")

    is_valid = len(issues) == 0
    return is_valid, issues
