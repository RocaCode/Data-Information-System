# Handles the importing of files from different sources

import os
import pandas as pd
import magic
import logging
import hashlib
from datetime import datetime
from functools import lru_cache

#Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#custom exception handling
class SchemaError(Exception):
    """Raised when data does not match expected schema."""
    pass

#util functions
def get_file_hash(file_path):
    """Generate a hash for the file to ensure data integrity."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def detect_file_type(file_path):
    """detect file type by examining file content."""
    # First try file extension as a fallback
    extension = file_path.lower().split('.')[-1]
    
    # Use magic to detect mime type
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)

    # CSV files are often detected as text/plain, so check extension too
    if file_type == 'text/csv' or (file_type == 'text/plain' and extension == 'csv'):
        return 'csv'
    elif file_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
        return 'excel'
    elif file_type == 'application/json':
        return 'json'
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
#Validation functions
def validate_file(file_path):
    """Validate that a file exist and is readable."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"The file is not readable: {file_path}")
    
def validate_schema(df, schema):
    """Validate DataFrame against a given schema."""
    for column, expected_type in schema.items():
        if column not in df.columns:
            raise SchemaError(f"Missing expected column: {column}")
        if str(df[column].dtype) != expected_type:
            raise SchemaError(f"Column {column} expected type {expected_type}, found {df[column].dtype}")

#load CSV, Excel, JSON files w/ error handling and return as DataFrame
def load_csv(file_path):
    """Load a CSV file into DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error Loading CSV file {file_path}: {str(e)}")

def load_excel(file_path):
    """Load an Excel file into DataFrame."""
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        raise Exception(f"Error Loading Excel file {file_path}: {str(e)}")

def load_json(file_path):
    """Load a JSON file into DataFrame."""
    try:
        return pd.read_json(file_path)
    except Exception as e:
        raise Exception(f"Error Loading JSON file {file_path}: {str(e)}")
    
#advanced loading for larger files with chunking support
def load_csv_in_chunks(file_path, chunk_size=10000):
    """Load csv files in chunks to handle large files."""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk

@lru_cache(maxsize=32)
def load_file_cached(file_path, file_hash):
    """Load file with caching based on content hash."""
    # Return a single DataFrame for the given file path (not the batch loader)
    return load_file(file_path)

#Main interface functions
def load_file(file_path):
    """smart loader to detect file type and load accordingly."""
    validate_file(file_path)
    file_type = detect_file_type(file_path)

    if file_type == 'csv':
        return load_csv(file_path)
    elif file_type == 'excel':
        return load_excel(file_path)
    elif file_type == 'json':
        return load_json(file_path)
    
def smart_load(file_path, use_cache=True):
    """Main entry point for loading files with above features."""
    logger.info(f"Loading file: {file_path}")
    start_time = datetime.now()

    try:
        validate_file(file_path)

        if use_cache:
            file_hash = get_file_hash(file_path)
            df = load_file_cached(file_path, file_hash)
        else:
            df = load_file(file_path)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Successfully Loaded {file_path} in {duration:.2f} seconds")
        return df
    
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {str(e)}")
        raise

#Preview function
def preview_file(file_path, rows=5, sample_size=None):
    """Preview the file content without fully loading it."""
    df = smart_load(file_path, use_cache=True)

    preview = {
        'head': df.head(rows).to_dict(),
        'total_rows': len(df),
        'columns': list(df.columns),
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }

    if sample_size:
        preview['Sample'] = df.sample(n=min(sample_size, len(df))).to_dict()

    return preview

#Batch file proccessing
def load_files(file_paths, schema=None):
    """Load multiple files into DataFrames."""
    loaded_files = {}
    for file_path in file_paths:
        df = smart_load(file_path)
        if schema:
            validate_schema(df, schema)
        loaded_files[file_path] = df
    return loaded_files