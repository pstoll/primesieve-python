# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdint cimport uint64_t, int64_t
from libcpp.vector cimport vector

cimport cpp_primesieve

import numpy as np
cimport numpy as np

cpdef np.ndarray get_out_array(np.ndarray out, uint64_t nn):
    cdef np.ndarray out_array
    if out is None:
        out_array = np.zeros([nn], dtype=np.uint64, order='c')
    else:
        out_array = np.ascontiguousarray(out)
    return out_array

cpdef np.ndarray generate_primes_numpy(uint64_t a, uint64_t b = 0, out = None) except +:
    """Generate a list of primes"""
    cdef vector[uint64_t] primes

    # if b == 0, then a is the upper bound, so swap the indexes
    if b == 0:
        (a,b) = (0,a)

    cpp_primesieve.generate_primes[uint64_t](a, b, &primes)
    cdef uint64_t nn = primes.size()

    cdef np.ndarray[np.uint64_t, ndim=1] out_array
    out_array = get_out_array(out, nn)
    cdef uint64_t ii = 0
    cdef uint64_t *data = <uint64_t *>(np.PyArray_DATA(out_array))
    for ii in range(0, nn):
        data[ii] = primes[ii]
    return out_array

cpdef np.ndarray generate_n_primes_numpy(uint64_t nn, uint64_t start = 0, out = None) except +:
    """Generate a list of primes"""
    cdef vector[uint64_t] primes
    cdef np.ndarray[np.uint64_t, ndim=1] out_array

    cpp_primesieve.generate_n_primes[uint64_t](nn, start, &primes)

    out_array = get_out_array(out, nn)
    cdef uint64_t ii = 0
    cdef uint64_t *data = <uint64_t *>np.PyArray_DATA(out_array)
    for ii in range(0, nn):
        data[ii] = primes[ii]
    return out_array


cpdef np.ndarray generate_n_primes_array_iter1(uint64_t nn,
                                               uint64_t start = 0,
                                               out = None) except +:
    """Generate a list of primes into a numpy array"""
    cdef np.ndarray[np.uint64_t, ndim=1] out_array
    out_array = get_out_array(out, nn)
    cdef cpp_primesieve.iterator iter
    iter = cpp_primesieve.iterator(start, start + nn)
    cdef uint64_t ii = 0
    for ii in range(0, nn):
        out_array[ii] = iter.next_prime()
    return out_array

cpdef np.ndarray generate_n_primes_array_iter2(uint64_t nn,
                                               uint64_t start = 0,
      					       out = None) except +:
    """Generate a list of primes into a numpy array"""
    cdef np.ndarray[np.uint64_t, ndim=1] out_array
    if out is None:
        out_array = np.zeros([nn], dtype=np.uint64, order='c')
    else:
        out_array = np.ascontiguousarray(out)
    cdef cpp_primesieve.iterator iter
    iter = cpp_primesieve.iterator(start, start + nn)
    cdef uint64_t ii = 0
    cdef uint64_t *data = <uint64_t *>np.PyArray_DATA(out_array)
    for ii in range(nn):
        data[ii] = iter.next_prime()
    return out_array

