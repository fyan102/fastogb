"""Thread controls shared by compiled CPU search kernels."""

from contextlib import contextmanager

from numba import config, get_num_threads, set_num_threads


def resolve_numba_jobs(n_jobs):
    jobs = config.NUMBA_NUM_THREADS if n_jobs == -1 else int(n_jobs)
    if jobs < 1:
        raise ValueError('n_jobs must be a positive integer or -1')
    if jobs > config.NUMBA_NUM_THREADS:
        raise ValueError(f'n_jobs={jobs} exceeds the Numba thread limit of {config.NUMBA_NUM_THREADS}; restart '
                         'Python with NUMBA_NUM_THREADS unset or set to at least n_jobs before importing fastogb')
    return jobs


@contextmanager
def numba_threads(n_jobs):
    """Temporarily apply an estimator or search thread limit."""
    jobs = resolve_numba_jobs(n_jobs)
    previous = get_num_threads()
    if jobs != previous:
        set_num_threads(jobs)
    try:
        yield jobs
    finally:
        if jobs != previous:
            set_num_threads(previous)
