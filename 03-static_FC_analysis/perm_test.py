#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:40:45 2021

@author: alado
"""
import logging
import os
import inspect
from math import sqrt
import numpy as np

# helpers to get function arguments
def _get_args(function, varargs=False):
    params = inspect.signature(function).parameters
    args = [key for key, param in params.items()
            if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)]
    if varargs:
        varargs = [param.name for param in params.values()
                   if param.kind == param.VAR_POSITIONAL]
        if len(varargs) == 0:
            varargs = None
        return args, varargs
    else:
        return args

def get_config(key=None, default=None, raise_error=False, home_dir=None,
               use_env=True):
    """Read MNE-Python preferences from environment or config file.
    Parameters
    ----------
    key : None | str
        The preference key to look for. The os environment is searched first,
        then the mne-python config file is parsed.
        If None, all the config parameters present in environment variables or
        the path are returned. If key is an empty string, a list of all valid
        keys (but not values) is returned.
    default : str | None
        Value to return if the key is not found.
    raise_error : bool
        If True, raise an error if the key is not found (instead of returning
        default).
    home_dir : str | None
        The folder that contains the .mne config folder.
        If None, it is found automatically.
    use_env : bool
        If True, consider env vars, if available.
        If False, only use MNE-Python configuration file values.
        .. versionadded:: 0.18
    Returns
    -------
    value : dict | str | None
        The preference key value.
    See Also
    --------
    set_config
    """
    _validate_type(key, (str, type(None)), "key", 'string or None')

    if key == '':
        return known_config_types

    # first, check to see if key is in env
    if use_env and key is not None and key in os.environ:
        return os.environ[key]

    # second, look for it in mne-python config file
    config_path = get_config_path(home_dir=home_dir)
    if not op.isfile(config_path):
        config = {}
    else:
        config = _load_config(config_path)

    if key is None:
        # update config with environment variables
        if use_env:
            env_keys = (set(config).union(known_config_types).
                        intersection(os.environ))
            config.update({key: os.environ[key] for key in env_keys})
        return config
    elif raise_error is True and key not in config:
        loc_env = 'the environment or in the ' if use_env else ''
        meth_env = ('either os.environ["%s"] = VALUE for a temporary '
                    'solution, or ' % key) if use_env else ''
        extra_env = (' You can also set the environment variable before '
                     'running python.' if use_env else '')
        meth_file = ('mne.utils.set_config("%s", VALUE, set_env=True) '
                     'for a permanent one' % key)
        raise KeyError('Key "%s" not found in %s'
                       'the mne-python config file (%s). '
                       'Try %s%s.%s'
                       % (key, loc_env, config_path, meth_env, meth_file,
                          extra_env))
    else:
        return config.get(key, default)


# adapted from scikit-learn utils/validation.py
def check_random_state(seed):
    """Turn seed into a numpy.random.mtrand.RandomState instance.
    If seed is None, return the RandomState singleton used by np.random.mtrand.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.mtrand.RandomState(seed)
    if isinstance(seed, np.random.mtrand.RandomState):
        return seed
    try:
        # Generator is only available in numpy >= 1.17
        if isinstance(seed, np.random.Generator):
            return seed
    except AttributeError:
        pass
    raise ValueError('%r cannot be used to seed a '
                     'numpy.random.mtrand.RandomState instance' % seed)

def bin_perm_rep(ndim, a=0, b=1):
    """Ndim permutations with repetitions of (a,b).
    Returns an array with all the possible permutations with repetitions of
    (0,1) in ndim dimensions.  The array is shaped as (2**ndim,ndim), and is
    ordered with the last index changing fastest.  For examble, for ndim=3:
    Examples
    --------
    >>> bin_perm_rep(3)
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [1, 0, 0],
           [1, 0, 1],
           [1, 1, 0],
           [1, 1, 1]])
    """
    # Create the leftmost column as 0,0,...,1,1,...
    nperms = 2 ** ndim
    perms = np.empty((nperms, ndim), type(a))
    perms.fill(a)
    half_point = nperms // 2
    perms[half_point:, 0] = b
    # Fill the rest of the table by sampling the previous column every 2 items
    for j in range(1, ndim):
        half_col = perms[::2, j - 1]
        perms[:half_point, j] = half_col
        perms[half_point:, j] = half_col
    # This is equivalent to something like:
    # orders = [np.fromiter(np.binary_repr(s + 1, ndim), dtype=int)
    #           for s in np.arange(2 ** ndim)]
    return perms


def _get_1samp_orders(n_samples, n_permutations, tail, rng):
    """Get the 1samp orders."""
    max_perms = 2 ** (n_samples - (tail == 0)) - 1
    extra = ''
    if isinstance(n_permutations, str):
        if n_permutations != 'all':
            raise ValueError('n_permutations as a string must be "all"')
        n_permutations = max_perms
    n_permutations = int(n_permutations)
    if max_perms < n_permutations:
        # omit first perm b/c accounted for in H0.append() later;
        # convert to binary array representation
        extra = ' (exact test)'
        orders = bin_perm_rep(n_samples)[1:max_perms + 1]
    elif n_samples <= 20:  # fast way to do it for small(ish) n_samples
        orders = rng.choice(max_perms, n_permutations - 1, replace=False)
        orders = [np.fromiter(np.binary_repr(s + 1, n_samples), dtype=int)
                  for s in orders]
    else:  # n_samples >= 64
        # Here we can just use the hash-table (w/collision detection)
        # functionality of a dict to ensure uniqueness
        orders = np.zeros((n_permutations - 1, n_samples), int)
        hashes = {}
        ii = 0
        # in the symmetric case, we should never flip one of the subjects
        # to prevent positive/negative equivalent collisions
        use_samples = n_samples - (tail == 0)
        while ii < n_permutations - 1:
            signs = tuple((rng.uniform(size=use_samples) < 0.5).astype(int))
            if signs not in hashes:
                orders[ii, :use_samples] = signs
                if tail == 0 and rng.uniform() < 0.5:
                    # To undo the non-flipping of the last subject in the
                    # tail == 0 case, half the time we use the positive
                    # last subject, half the time negative last subject
                    orders[ii] = 1 - orders[ii]
                hashes[signs] = None
                ii += 1
    return orders, n_permutations, extra

def check_n_jobs(n_jobs, allow_cuda=False):
    """Check n_jobs in particular for negative values.
    Parameters
    ----------
    n_jobs : int
        The number of jobs.
    allow_cuda : bool
        Allow n_jobs to be 'cuda'. Default: False.
    Returns
    -------
    n_jobs : int
        The checked number of jobs. Always positive (or 'cuda' if
        applicable).
    """
    if not isinstance(n_jobs, int_like):
        if not allow_cuda:
            raise ValueError('n_jobs must be an integer')
        elif not isinstance(n_jobs, str) or n_jobs != 'cuda':
            raise ValueError('n_jobs must be an integer, or "cuda"')
        # else, we have n_jobs='cuda' and this is okay, so do nothing
    elif _force_serial:
        n_jobs = 1
        logger.info('... MNE_FORCE_SERIAL set. Processing in forced '
                    'serial mode.')
    elif n_jobs <= 0:
        try:
            import multiprocessing
            n_cores = multiprocessing.cpu_count()
            n_jobs = min(n_cores + n_jobs + 1, n_cores)
            if n_jobs <= 0:
                raise ValueError('If n_jobs has a negative value it must not '
                                 'be less than the number of CPUs present. '
                                 'You\'ve got %s CPUs' % n_cores)
        except ImportError:
            # only warn if they tried to use something other than 1 job
            if n_jobs != 1:
                warn('multiprocessing not installed. Cannot run in parallel.')
                n_jobs = 1

    return n_jobs


def _check_wrapper(fun):
    def run(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except RuntimeError as err:
            msg = str(err.args[0]) if err.args else ''
            if msg.startswith('The task could not be sent to the workers'):
                raise RuntimeError(
                    msg + ' Consider using joblib memmap caching to get '
                    'around this problem. See mne.set_mmap_min_size, '
                    'mne.set_cache_dir, and buffer_size parallel function '
                    'arguments (if applicable).')
            raise
    return run

if 'MNE_FORCE_SERIAL' in os.environ:
    _force_serial = True
else:
    _force_serial = None

def parallel_func(func, n_jobs, max_nbytes='auto', pre_dispatch='n_jobs',
                  total=None, prefer=None, verbose=None):
    """Return parallel instance with delayed function.
    Util function to use joblib only if available
    Parameters
    ----------
    func : callable
        A function.
    n_jobs : int
        Number of jobs to run in parallel.
    max_nbytes : int, str, or None
        Threshold on the minimum size of arrays passed to the workers that
        triggers automated memory mapping. Can be an int in Bytes,
        or a human-readable string, e.g., '1M' for 1 megabyte.
        Use None to disable memmaping of large arrays. Use 'auto' to
        use the value set using mne.set_memmap_min_size.
    pre_dispatch : int, or str, optional
        See :class:`joblib.Parallel`.
    total : int | None
        If int, use a progress bar to display the progress of dispatched
        jobs. This should only be used when directly iterating, not when
        using ``split_list`` or :func:`np.array_split`.
        If None (default), do not add a progress bar.
    prefer : str | None
        If str, can be "processes" or "threads". See :class:`joblib.Parallel`.
        Ignored if the joblib version is too old to support this.
        .. versionadded:: 0.18
    %(verbose)s INFO or DEBUG
        will print parallel status, others will not.
    Returns
    -------
    parallel: instance of joblib.Parallel or list
        The parallel object.
    my_func: callable
        ``func`` if not parallel or delayed(func).
    n_jobs: int
        Number of jobs >= 0.
    """
    #should_print = (logger.level <= logging.INFO)
    # for a single job, we don't need joblib
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            try:
                from sklearn.externals.joblib import Parallel, delayed
            except ImportError:
                print('joblib not installed. Cannot run in parallel.')
                n_jobs = 1
    if n_jobs == 1:
        n_jobs = 1
        my_func = func
        parallel = list
    else:
        # check if joblib is recent enough to support memmaping
        p_args = _get_args(Parallel.__init__)
        joblib_mmap = ('temp_folder' in p_args and 'max_nbytes' in p_args)

        cache_dir = get_config('MNE_CACHE_DIR', None)
        if isinstance(max_nbytes, str) and max_nbytes == 'auto':
            max_nbytes = get_config('MNE_MEMMAP_MIN_SIZE', None)

        if max_nbytes is not None:
            if not joblib_mmap and cache_dir is not None:
                print('"MNE_CACHE_DIR" is set but a newer version of joblib is '
                     'needed to use the memmapping pool.')
            if joblib_mmap and cache_dir is None:
                print(
                    'joblib supports memapping pool but "MNE_CACHE_DIR" '
                    'is not set in MNE-Python config. To enable it, use, '
                    'e.g., mne.set_cache_dir(\'/tmp/shm\'). This will '
                    'store temporary files under /dev/shm and can result '
                    'in large memory savings.')

        # create keyword arguments for Parallel
        kwargs = {'verbose': 5 if total is None else 0}
        kwargs['pre_dispatch'] = pre_dispatch
        if 'prefer' in p_args:
            kwargs['prefer'] = prefer

        if joblib_mmap:
            if cache_dir is None:
                max_nbytes = None  # disable memmaping
            kwargs['temp_folder'] = cache_dir
            kwargs['max_nbytes'] = max_nbytes

        n_jobs = check_n_jobs(n_jobs)
        parallel = _check_wrapper(Parallel(n_jobs, **kwargs))
        my_func = delayed(func)

    if total is not None:
        def parallel_progress(op_iter):
            return parallel(ProgressBar(iterable=op_iter, max_value=total))
        parallel_out = parallel_progress
    else:
        parallel_out = parallel
    return parallel_out, my_func, n_jobs


def _max_stat(X, X2, perms, dof_scaling):
    """Aux function for permutation_t_test (for parallel comp)."""
    n_samples = len(X)
    mus = np.dot(perms, X) / float(n_samples)
    stds = np.sqrt(X2[None, :] - mus * mus) * dof_scaling  # std with splitting
    max_abs = np.max(np.abs(mus) / (stds / sqrt(n_samples)), axis=1)  # t-max
    return max_abs

def permutation_t_test(X, n_permutations=10000, tail=0, n_jobs=1,
                       seed=None, verbose=None):
    """One sample/paired sample permutation test based on a t-statistic.
    This function can perform the test on one variable or
    simultaneously on multiple variables. When applying the test to multiple
    variables, the "tmax" method is used for adjusting the p-values of each
    variable for multiple comparisons. Like Bonferroni correction, this method
    adjusts p-values in a way that controls the family-wise error rate.
    However, the permutation method will be more
    powerful than Bonferroni correction when different variables in the test
    are correlated (see [1]_).
    Parameters
    ----------
    X : array, shape (n_samples, n_tests)
        Samples (observations) by number of tests (variables).
    n_permutations : int | 'all'
        Number of permutations. If n_permutations is 'all' all possible
        permutations are tested. It's the exact test, that
        can be untractable when the number of samples is big (e.g. > 20).
        If n_permutations >= 2**n_samples then the exact test is performed.
    tail : -1 or 0 or 1 (default = 0)
        If tail is 1, the alternative hypothesis is that the
        mean of the data is greater than 0 (upper tailed test).  If tail is 0,
        the alternative hypothesis is that the mean of the data is different
        than 0 (two tailed test).  If tail is -1, the alternative hypothesis
        is that the mean of the data is less than 0 (lower tailed test).
    %(n_jobs)s
    %(seed)s
    %(verbose)s
    Returns
    -------
    T_obs : array of shape [n_tests]
        T-statistic observed for all variables.
    p_values : array of shape [n_tests]
        P-values for all the tests (a.k.a. variables).
    H0 : array of shape [n_permutations]
        T-statistic obtained by permutations and t-max trick for multiple
        comparison.
    Notes
    -----
    If ``n_permutations >= 2 ** (n_samples - (tail == 0))``,
    ``n_permutations`` and ``seed`` will be ignored since an exact test
    (full permutation test) will be performed.
    References
    ----------
    .. [1] Nichols, T. E. & Holmes, A. P. (2002). Nonparametric permutation
       tests for functional neuroimaging: a primer with examples.
       Human Brain Mapping, 15, 1-25.
    """
    n_samples, n_tests = X.shape
    X2 = np.mean(X ** 2, axis=0)  # precompute moments
    mu0 = np.mean(X, axis=0)
    dof_scaling = sqrt(n_samples / (n_samples - 1.0))
    std0 = np.sqrt(X2 - mu0 ** 2) * dof_scaling  # get std with var splitting
    T_obs = np.mean(X, axis=0) / (std0 / sqrt(n_samples))
    rng = check_random_state(seed)
    orders, _, extra = _get_1samp_orders(n_samples, n_permutations, tail, rng)
    perms = 2 * np.array(orders) - 1  # from 0, 1 -> 1, -1
    #logger.info('Permuting %d times%s...' % (len(orders), extra))
    print(f'Permuting {len(orders)} times {extra}...')
    parallel, my_max_stat, n_jobs = parallel_func(_max_stat, n_jobs)
    max_abs = np.concatenate(parallel(my_max_stat(X, X2, p, dof_scaling)
                                      for p in np.array_split(perms, n_jobs)))
    max_abs = np.concatenate((max_abs, [np.abs(T_obs).max()]))
    H0 = np.sort(max_abs)
    if tail == 0:
        p_values = (H0 >= np.abs(T_obs[:, np.newaxis])).mean(-1)
    elif tail == 1:
        p_values = (H0 >= T_obs[:, np.newaxis]).mean(-1)
    elif tail == -1:
        p_values = (-H0 <= T_obs[:, np.newaxis]).mean(-1)
    return T_obs, p_values, H0
