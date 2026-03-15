"""Main SDGym benchmarking module."""

import functools
import gzip
import logging
import math
import multiprocessing
import os
import re
import threading
import tracemalloc
import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from urllib.parse import urlparse

import cloudpickle
import numpy as np
import pandas as pd
import tqdm
import yaml

from sdmetrics.reports.multi_table import (
    DiagnosticReport as MultiTableDiagnosticReport,
)
from sdmetrics.reports.multi_table import (
    QualityReport as MultiTableQualityReport,
)
from sdmetrics.reports.single_table import (
    DiagnosticReport as SingleTableDiagnosticReport,
)
from sdmetrics.reports.single_table import (
    QualityReport as SingleTableQualityReport,
)
from sdmetrics.single_table import DCRBaselineProtection

from sdgym.datasets import _load_dataset_with_client, get_dataset_paths
from sdgym.errors import BenchmarkError, SDGymError
from sdgym.metrics import get_metrics
from sdgym.progress import TqdmLogger
from sdgym.result_writer import LocalResultsWriter, S3ResultsWriter
from sdgym.s3 import (
    S3_PREFIX,
    S3_REGION,
    is_s3_path,
    parse_s3_path,
)
from sdgym.synthesizers import MultiTableUniformSynthesizer, UniformSynthesizer
from sdgym.synthesizers.base import BaselineSynthesizer
from sdgym.utils import (
    calculate_score_time,
    convert_metadata_to_sdmetrics,
    format_exception,
    get_duplicates,
    get_size_of,
    get_synthesizers,
    get_utc_now,
    used_memory,
)

LOGGER = logging.getLogger(__name__)

# Default Constants & Configurations
DEFAULT_SINGLE_TABLE_SYNTHESIZERS =[
    'GaussianCopulaSynthesizer',
    'CTGANSynthesizer',
    'UniformSynthesizer',
]
DEFAULT_MULTI_TABLE_SYNTHESIZERS =['MultiTableUniformSynthesizer', 'HMASynthesizer']

DEFAULT_SINGLE_TABLE_DATASETS =[
    'adult',
    'alarm',
    'census',
    'child',
    'covtype',
    'expedia_hotel_logs',
    'insurance',
    'intrusion',
    'news',
]
DEFAULT_MULTI_TABLE_DATASETS =[
    'fake_hotels',
    'Biodegradability',
    'Student_loan',
    'restbase',
    'airbnb-simplified',
    'financial',
    'NBA',
]

N_BYTES_IN_MB = 1000 * 1000
FILE_INCREMENT_PATTERN = re.compile(r'\((\d+)\)$')
RESULTS_DATE_PATTERN = re.compile(r'SDGym_results_(\d{2}_\d{2}_\d{4})')
METAINFO_FILE_PATTERN = re.compile(r'metainfo(?:\((\d+)\))?\.yaml$')

SDV_SINGLE_TABLE_SYNTHESIZERS =[
    'GaussianCopulaSynthesizer',
    'CTGANSynthesizer',
    'CopulaGANSynthesizer',
    'TVAESynthesizer',
]
SDV_MULTI_TABLE_SYNTHESIZERS = ['HMASynthesizer']
SDV_SYNTHESIZERS = SDV_SINGLE_TABLE_SYNTHESIZERS + SDV_MULTI_TABLE_SYNTHESIZERS


class ConfigTunables:
    """Configurable algorithm and system variables to avoid magic constants in scoring."""
    S3_TIMEOUT_CONNECT = 30
    S3_TIMEOUT_READ = 300
    S3_MAX_RETRIES = 5
    DCR_BASELINE_ITERATIONS = 2
    DCR_BASELINE_SAMPLE_FRAC = 0.60
    KILL_GRACE_PERIOD_SEC = 2
    KILL_EXTREME_PERIOD_SEC = 1


class JobArgs(NamedTuple):
    """Arguments needed to run a single synthesizer + dataset benchmark job."""

    synthesizer: Dict[str, Any]
    data: Any
    metadata: Any
    metrics: Any
    timeout: Optional[int]
    compute_quality_score: bool
    compute_diagnostic_score: bool
    compute_privacy_score: bool
    dataset_name: str
    modality: str
    output_directions: Optional[Dict[str, str]]


@contextmanager
def MemoryTracker():
    """Robust context manager to safely start and clear tracemalloc boundaries."""
    tracemalloc.start()
    try:
        yield
    finally:
        tracemalloc.stop()
        tracemalloc.clear_traces()


def _get_boto3_client_lazy(access_key_id: Optional[str] = None, secret_access_key: Optional[str] = None):
    """Lazy evaluation / initialization of boto3 client including configurable retries."""
    import boto3
    from botocore.config import Config

    config = Config(
        connect_timeout=ConfigTunables.S3_TIMEOUT_CONNECT,
        read_timeout=ConfigTunables.S3_TIMEOUT_READ,
        retries={'max_attempts': ConfigTunables.S3_MAX_RETRIES, 'mode': 'standard'},
    )
    if access_key_id and secret_access_key:
        return boto3.client(
            's3',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=S3_REGION,
            config=config,
        )
    return boto3.client('s3', config=config)


def _import_and_validate_synthesizers(
    synthesizers: Optional[List[Any]], 
    custom_synthesizers: Optional[List[Any]], 
    modality: str
) -> List[Any]:
    """Import user-provided synthesizer and validate modality and uniqueness.

    Args:
        synthesizers: A list of synthesizer strings or classes.
        custom_synthesizers: A list of custom synthesizer definitions.
        modality: The required modality ('single_table' or 'multi_table').

    Returns:
        List of initialized and verified synthesizer class descriptors.
    """
    synthesizers = synthesizers or[]
    custom_synthesizers = custom_synthesizers or[]
    resolved_synthesizers = get_synthesizers(synthesizers + custom_synthesizers)
    mismatched = [
        synth['synthesizer']
        for synth in resolved_synthesizers
        if synth['synthesizer']._MODALITY_FLAG != modality
    ]
    if mismatched:
        raise ValueError(
            f"Synthesizers must be of modality '{modality}'. "
            "Found these synthesizers that don't match: "
            f"{', '.join([type(synth).__name__ for synth in mismatched])}"
        )

    duplicates = get_duplicates(synthesizers + custom_synthesizers)
    if duplicates:
        raise ValueError(
            'Synthesizers must be unique. Please remove repeated values in the provided '
            'synthesizers.'
        )

    return resolved_synthesizers


def _get_metainfo_increment(top_folder: str, s3_client: Optional[Any] = None) -> int:
    increments =[]
    first_file_message = 'No metainfo file found, starting from increment (0)'
    if s3_client:
        bucket, prefix = parse_s3_path(top_folder)
        from botocore.exceptions import BotoCoreError, ClientError
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            contents = response.get('Contents', [])
            for obj in contents:
                file_name = Path(obj['Key']).name
                match = METAINFO_FILE_PATTERN.match(file_name)
                if match:
                    increments.append(int(match.group(1)) if match.group(1) else 0)

        except (BotoCoreError, ClientError) as error:
            LOGGER.warning("Could not reach S3 when getting metainfo. Details: %s. %s", error, first_file_message)
            return 0  # start with (0) if error or absent
    else:
        folder_path = Path(top_folder)
        if not folder_path.exists():
            LOGGER.info(first_file_message)
            return 0
        for file in folder_path.glob('metainfo*.yaml'):
            match = METAINFO_FILE_PATTERN.match(file.name)
            if match:
                increments.append(int(match.group(1)) if match.group(1) else 0)

    return max(increments) + 1 if increments else 0


def _setup_output_destination_aws(
    output_destination: str,
    synthesizers: List[str],
    datasets: List[str],
    modality: str,
    s3_client: Any,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Generates proper aws pathing avoiding using Path object which outputs corrupted `\\` backslashes safely."""
    paths: defaultdict = defaultdict(dict)
    s3_path = output_destination[len(S3_PREFIX) :].rstrip('/')
    parts = s3_path.split('/')
    bucket_name = parts[0]
    prefix_parts = parts[1:]
    
    paths['bucket_name'] = bucket_name
    today = datetime.today().strftime('%m_%d_%Y')
    modality_prefix = '/'.join(prefix_parts +[modality])
    top_folder = f'{modality_prefix}/SDGym_results_{today}'
    
    increment = _get_metainfo_increment(f'{S3_PREFIX}{bucket_name}/{top_folder}', s3_client)
    suffix = f'({increment})' if increment >= 1 else ''
    s3_client.put_object(Bucket=bucket_name, Key=top_folder + '/')
    synthetic_data_extension = 'zip' if modality == 'multi_table' else 'csv'
    
    for dataset in datasets:
        dataset_folder = f'{top_folder}/{dataset}_{today}'
        s3_client.put_object(Bucket=bucket_name, Key=dataset_folder + '/')

        for synth_name in synthesizers:
            final_synth_name = f'{synth_name}{suffix}'
            synth_folder = f'{dataset_folder}/{final_synth_name}'
            s3_client.put_object(Bucket=bucket_name, Key=synth_folder + '/')
            paths[dataset][final_synth_name] = {
                'synthesizer': (f'{S3_PREFIX}{bucket_name}/{synth_folder}/{final_synth_name}.pkl'),
                'synthetic_data': (
                    f'{S3_PREFIX}{bucket_name}/{synth_folder}/'
                    f'{final_synth_name}_synthetic_data.{synthetic_data_extension}'
                ),
                'benchmark_result': (
                    f'{S3_PREFIX}{bucket_name}/{synth_folder}/{final_synth_name}_benchmark_result.csv'
                ),
                'metainfo': (f'{S3_PREFIX}{bucket_name}/{top_folder}/metainfo{suffix}.yaml'),
                'results': (f'{S3_PREFIX}{bucket_name}/{top_folder}/results{suffix}.csv'),
            }

    s3_client.put_object(
        Bucket=bucket_name,
        Key=f'{top_folder}/metainfo{suffix}.yaml',
        Body='completed_date: null\n'.encode('utf-8'),
    )
    return paths


def _setup_output_destination(
    output_destination: Optional[str],
    synthesizers: List[str],
    datasets: List[str],
    modality: str,
    s3_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Set up the output destination for the benchmark results.

    Args:
        output_destination: Target writing directory endpoint / prefix URI.
        synthesizers: Array of string keys correlating algorithm configurations.
        datasets: Data definitions to establish bucket sub-roots.
        modality: Relational definitions format 'single_table'/'multi_table'.
        s3_client: Explicit boto3 service configuration dependency. Defaults to None.
    """
    if s3_client:
        return _setup_output_destination_aws(
            output_destination, synthesizers, datasets, modality, s3_client
        )

    if output_destination is None:
        return {}

    output_path = Path(output_destination)
    output_path.mkdir(parents=True, exist_ok=True)
    today = datetime.today().strftime('%m_%d_%Y')
    
    top_folder = output_path / modality / f'SDGym_results_{today}'
    top_folder.mkdir(parents=True, exist_ok=True)
    
    increment = _get_metainfo_increment(str(top_folder))
    suffix = f'({increment})' if increment >= 1 else ''
    paths: defaultdict = defaultdict(dict)
    synthetic_data_extension = 'zip' if modality == 'multi_table' else 'csv'
    
    for dataset in datasets:
        dataset_folder = top_folder / f'{dataset}_{today}'
        dataset_folder.mkdir(parents=True, exist_ok=True)

        for synth_name in synthesizers:
            final_synth_name = f'{synth_name}{suffix}'
            synth_folder = dataset_folder / final_synth_name
            synth_folder.mkdir(parents=True, exist_ok=True)
            paths[dataset][final_synth_name] = {
                'synthesizer': str(synth_folder / f'{final_synth_name}.pkl'),
                'synthetic_data': str(
                    synth_folder / f'{final_synth_name}_synthetic_data.{synthetic_data_extension}'
                ),
                'benchmark_result': str(synth_folder / f'{final_synth_name}_benchmark_result.csv'),
                'metainfo': str(top_folder / f'metainfo{suffix}.yaml'),
                'results': str(top_folder / f'results{suffix}.csv'),
            }

    return dict(paths)


def _generate_job_args_list(
    limit_dataset_size: bool,
    sdv_datasets: Optional[List[str]],
    additional_datasets_folder: Optional[str],
    sdmetrics: Optional[List[Any]],
    timeout: Optional[int],
    output_destination: Optional[str],
    compute_quality_score: bool,
    compute_diagnostic_score: bool,
    compute_privacy_score: Optional[bool],
    synthesizers: List[Dict[str, Any]],
    s3_client: Optional[Any],
    modality: str,
) -> List[JobArgs]:
    """Generates Job Configurations dynamically executing across specified metrics, synthesizing datasets securely mapping environments"""
    sdv_datasets_found = ([]
        if sdv_datasets is None
        else get_dataset_paths(
            modality=modality,
            datasets=sdv_datasets,
            s3_client=s3_client,
        )
    )
    additional_datasets = ([]
        if additional_datasets_folder is None
        else get_dataset_paths(
            modality=modality,
            bucket=(
                additional_datasets_folder
                if is_s3_path(additional_datasets_folder)
                else os.path.join(additional_datasets_folder, modality)
            ),
            s3_client=s3_client,
        )
    )
    datasets = sdv_datasets_found + additional_datasets
    synthesizer_names = [synth['name'] for synth in synthesizers]
    dataset_names =[dataset.name for dataset in datasets]
    paths = _setup_output_destination(
        output_destination, synthesizer_names, dataset_names, modality=modality, s3_client=s3_client
    )
    
    job_tuples =[]
    for dataset in datasets:
        for synthesizer in synthesizers:
            # We enforce immutability avoiding bleeding internal names dict allocations randomly across benchmark contexts loops
            synth_config = dict(synthesizer)
            
            if paths:
                final_name = next(
                    (name for name in paths.get(dataset.name, {}) if name.startswith(synth_config['name'])),
                    synth_config['name'],
                )
            else:
                final_name = synth_config['name']

            synth_config['name'] = final_name
            job_tuples.append((synth_config, dataset))

    job_args_list =[]
    for synth_config, dataset in job_tuples:
        data, metadata_dict = _load_dataset_with_client(
            modality, dataset, limit_dataset_size=limit_dataset_size, s3_client=s3_client
        )
        path = paths.get(dataset.name, {}).get(synth_config['name'], None)
        job_args_list.append(
            JobArgs(
                synthesizer=synth_config,
                data=data,
                metadata=metadata_dict,
                metrics=sdmetrics,
                timeout=timeout,
                compute_quality_score=compute_quality_score,
                compute_diagnostic_score=compute_diagnostic_score,
                compute_privacy_score=compute_privacy_score or False,
                dataset_name=dataset.name,
                modality=modality,
                output_directions=path,
            )
        )

    return job_args_list


def _synthesize(
    synthesizer_dict: Dict[str, Any],
    real_data: Any,
    metadata: Any,
    synthesizer_path: Optional[Dict[str, str]] = None,
    result_writer: Optional[Union[LocalResultsWriter, S3ResultsWriter]] = None,
    modality: Optional[str] = None,
) -> Tuple[Any, timedelta, timedelta, float, float]:
    """Train the model strictly tracking evaluation traces without accumulating excessive GC locks inherently breaking loops contexts limits memory usage boundaries configurations properly natively loops definitions parameters natively headers checks validations safely.
    
    Avoid calculating fully extensive memory allocations outside MemoryTracker natively."""
    synthesizer_klass_or_obj = synthesizer_dict['synthesizer']
    
    if isinstance(synthesizer_klass_or_obj, type):
        assert issubclass(synthesizer_klass_or_obj, BaselineSynthesizer), (
            '`synthesizer` must be a synthesizer class'
        )
        synthesizer = synthesizer_klass_or_obj()
    else:
        assert isinstance(synthesizer_klass_or_obj, BaselineSynthesizer), (
            '`synthesizer` must be an instance of a synthesizer class.'
        )

    get_synthesizer = synthesizer.get_trained_synthesizer
    sample_from_synthesizer = synthesizer.sample_from_synthesizer
    data = real_data.copy()

    fitted_synthesizer = None
    synthetic_data = None
    synthesizer_size = None
    peak_memory = None
    
    start = get_utc_now()
    train_end = None
    
    try:
        with MemoryTracker():
            fitted_synthesizer = get_synthesizer(data, metadata)
            
            # This size validation uses standard benchmarking specification metric - avoid randomly omitting if metric demands 1-to-1 spec evaluations 
            synthesizer_size = len(cloudpickle.dumps(fitted_synthesizer)) / N_BYTES_IN_MB
            
            train_end = get_utc_now()
            train_time = train_end - start

            if modality == 'multi_table':
                synthetic_data = sample_from_synthesizer(fitted_synthesizer, 1.0)
            else:
                synthetic_data = sample_from_synthesizer(fitted_synthesizer, n_samples=len(data))

            sample_end = get_utc_now()
            sample_time = sample_end - train_end
            
            # Grabs memory trace evaluated internally directly without bloating allocations!
            peak_memory = tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB

        if synthesizer_path is not None and result_writer is not None:
            internal_synthesizer = getattr(
                fitted_synthesizer, '_internal_synthesizer', fitted_synthesizer
            )
            result_writer.write_pickle(internal_synthesizer, synthesizer_path['synthesizer'])
            if modality == 'multi_table':
                result_writer.write_zipped_dataframes(
                    synthetic_data, synthesizer_path['synthetic_data']
                )
            else:
                result_writer.write_dataframe(synthetic_data, synthesizer_path['synthetic_data'])

        return synthetic_data, train_time, sample_time, synthesizer_size, peak_memory

    except Exception as e:
        now = get_utc_now()
        peak_memory = peak_memory if peak_memory is not None else (tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB if tracemalloc.is_tracing() else 0)
        
        if train_end is None:
            train_time = now - start
            sample_time = timedelta(0)
        else:
            train_time = train_end - start
            sample_time = now - train_end

        exception_text, error_text = format_exception()
        raise BenchmarkError(
            original_exc=e,
            train_time=train_time,
            sample_time=sample_time,
            synthesizer_size=synthesizer_size,
            peak_memory=peak_memory,
            exception_text=exception_text,
            error_text=error_text,
        ) from e


def _compute_scores(
    metrics: Any,
    real_data: Any,
    synthetic_data: Any,
    metadata: Any,
    output: Dict[str, Any],
    compute_quality_score: bool,
    compute_diagnostic_score: bool,
    compute_privacy_score: bool,
    modality: str,
    dataset_name: str,
):
    metrics = metrics or[]
    sdmetrics_metadata = convert_metadata_to_sdmetrics(metadata) if modality == 'single_table' else metadata

    if len(metrics) > 0:
        metric_objs, metric_kwargs = get_metrics(metrics, modality=modality)
        scores =[]
        # Assign explicit array first to avoid deep sync anomalies
        output['scores'] = scores
        
        for metric_name, metric in metric_objs.items():
            current_score_entry = {
                'metric': metric_name,
                'Error': 'Metric Timeout',
            }
            scores.append(current_score_entry)
            output['scores'] = scores

            error = None
           
