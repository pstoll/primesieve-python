from libc.stdint cimport uint64_t, int64_t
from libcpp.vector cimport vector
cimport cpp_primesieve

include "config.pxi"
IF USE_NUMPY == 1:
    cimport numpy as np
    import numpy as np

    np.import_array()

    cpdef np.ndarray generate_primes_numpy(uint64_t a, uint64_t b = 0, out = None):
        """Generate a list of primes between A and B into the numpy array OUT. 
        If B is not specified or is zero, the interval (0,A) is used instead.

        OUT can be None, in which case a new 1d array is allocated.
        if provided, OUT must be a contiguous 1d array 
           && must be large enough to hold the primes between (A,B)
           IF either of thos conditions are not true, a new array will be allocated
           for the output array
        """
        cdef vector[uint64_t] primes

        # if b == 0, then a is the upper bound, so swap the indexes
        if b == 0:
            (a,b) = (0,a)

        cpp_primesieve.generate_primes[uint64_t](a, b, &primes)
        cdef uint64_t nn = primes.size()
        
        cdef np.ndarray[np.uint64_t, ndim=1, mode='c'] out_array
        if out is None or nn > out.shape[0]:
            out_array = np.zeros([nn], dtype=np.uint64, order='c')
        else:
            # this only makes a copy if the input array is not already contiguous
            out_array = np.ascontiguousarray(out)
        cdef uint64_t ii = 0
        
        # the following direct access to the data assumes/requires a contiguous block of memory
        # this is not always true for numpy arrays
        cdef uint64_t *data = <uint64_t *>(np.PyArray_DATA(out_array))
        for ii in range(0, nn):
            data[ii] = primes[ii]
        return out_array

    cpdef np.ndarray generate_n_primes_numpy(uint64_t nn,
                                             uint64_t start = 0,
                                             out = None):
        """Generate a list of primes into a numpy array"""
        cdef np.ndarray[np.uint64_t, ndim=1] out_array

        if out is None or nn > out.shape[0]:
            out_array = np.zeros([nn], dtype=np.uint64, order='c')
        else:
            out_array = np.ascontiguousarray(out)
            
        cdef cpp_primesieve.iterator iter
        iter = cpp_primesieve.iterator(start, start + nn)
        cdef uint64_t ii = 0
        for ii in range(0, nn):
            out_array[ii] = iter.next_prime()
        return out_array


cpdef vector[uint64_t] generate_primes(uint64_t a, uint64_t b = 0) except +:
    """Generate a list of primes"""
    cdef vector[uint64_t] primes
    if b == 0:
        (a,b) = (0,a)
    cpp_primesieve.generate_primes[uint64_t](a, b, &primes)
    return primes

cpdef vector[uint64_t] generate_n_primes(uint64_t n, uint64_t start = 0) except +:
    """List the first n primes"""
    cdef vector[uint64_t] primes
    cpp_primesieve.generate_n_primes[uint64_t](n, start, &primes)
    return primes

cpdef uint64_t nth_prime(int64_t n, uint64_t start = 0) except +:
    """Find the nth prime after start"""
    return cpp_primesieve.nth_prime(n, start)

cpdef uint64_t parallel_nth_prime(int64_t n, uint64_t start = 0) except +:
    """Find the nth prime after start using multi-threading"""
    return cpp_primesieve.parallel_nth_prime(n, start)

cpdef uint64_t count_primes(uint64_t a, uint64_t b = 0) except +:
    """Count prime numbers"""
    if b == 0:
        (a,b) = (0,a)
    return cpp_primesieve.count_primes(a, b)
 
cpdef uint64_t count_twins(uint64_t a, uint64_t b = 0) except +:
    """Count twin primes"""
    if b == 0:
        (a,b) = (0,a)
    return cpp_primesieve.count_twins(a, b)
 
cpdef uint64_t count_triplets(uint64_t a, uint64_t b = 0) except +:
    """Count prime triplets"""
    if b == 0:
        (a,b) = (0,a)
    return cpp_primesieve.count_triplets(a, b)

cpdef uint64_t count_quadruplets(uint64_t a, uint64_t b = 0) except +:
    """Count prime quadruplets"""
    if b == 0:
        (a,b) = (0,a)
    return cpp_primesieve.count_quadruplets(a, b)


cpdef uint64_t count_quintuplets(uint64_t a, uint64_t b = 0) except +:
    """Count prime quintuplets"""
    if b == 0:
        (a,b) = (0,a)
    return cpp_primesieve.count_quintuplets(a, b)

cpdef uint64_t count_sextuplets(uint64_t a, uint64_t b = 0) except +:
    """Count prime sextuplets"""
    if b == 0:
        (a,b) = (0,a)
    return cpp_primesieve.count_sextuplets(a, b)


cpdef uint64_t parallel_count_primes(uint64_t a, uint64_t b = 0) except +:
    """Count prime numbers using multi-threading"""
    if b == 0:
        (a,b) = (0,a)
    return cpp_primesieve.parallel_count_primes(a, b)
 
cpdef uint64_t parallel_count_twins(uint64_t a, uint64_t b = 0) except +:
    """Count twin primes using multi-threading"""
    if b == 0:
        (a,b) = (0,a)
    return cpp_primesieve.parallel_count_twins(a, b)

 
cpdef uint64_t parallel_count_triplets(uint64_t a, uint64_t b = 0) except +:
    """Count prime triplets using multi-threading"""
    if b == 0:
        (a,b) = (0,a)
    return cpp_primesieve.parallel_count_triplets(a, b)


cpdef uint64_t parallel_count_quadruplets(uint64_t a, uint64_t b = 0) except +:
    """Count prime quadruplets using multi-threading"""
    if b == 0:
        (a,b) = (0,a)
    return cpp_primesieve.parallel_count_quadruplets(a, b)

cpdef uint64_t parallel_count_quintuplets(uint64_t a, uint64_t b = 0) except +:
    """Count prime quintuplets using multi-threading"""
    if b == 0:
        (a,b) = (0,a)
    return cpp_primesieve.parallel_count_quintuplets(a, b)


cpdef uint64_t parallel_count_sextuplets(uint64_t a, uint64_t b = 0) except +:
    """Count prime sextuplets using multi-threading"""
    if b == 0:
        (a,b) = (0,a)
    return cpp_primesieve.parallel_count_sextuplets(a, b)

cpdef void print_primes(uint64_t a, uint64_t b = 0) except +:
    """Print prime numbers to stdout"""
    if b == 0:
        (a,b) = (0,a)
    cpp_primesieve.print_primes(a, b)
 
cpdef void print_twins(uint64_t a, uint64_t b = 0) except +:
    """Print twin primes to stdout"""
    if b == 0:
        (a,b) = (0,a)
    cpp_primesieve.print_twins(a, b)

 
cpdef void print_triplets(uint64_t a, uint64_t b = 0) except +:
    """Print prime triplets to stdout"""
    if b == 0:
        (a,b) = (0,a)
    cpp_primesieve.print_triplets(a, b)


cpdef void print_quadruplets(uint64_t a, uint64_t b = 0) except +:
    """Print prime quadruplets to stdout"""
    if b == 0:
        (a,b) = (0,a)
    cpp_primesieve.print_quadruplets(a, b)


cpdef void print_quintuplets(uint64_t a, uint64_t b = 0) except +:
    """Print prime quintuplets to stdout"""
    if b == 0:
        (a,b) = (0,a)
    cpp_primesieve.print_quintuplets(a, b)

cpdef void print_sextuplets(uint64_t a, uint64_t b = 0) except +:
    """Print prime sextuplets to stdout"""
    if b == 0:
        (a,b) = (0,a)
    cpp_primesieve.print_sextuplets(a, b)


cdef class Iterator:
    cdef cpp_primesieve.iterator _iterator
    def __cinit__(self):
        self._iterator = cpp_primesieve.iterator()
    cpdef void skipto(self, uint64_t start, uint64_t stop_hint = 2**62) except +:
        self._iterator.skipto(start, stop_hint)
    cpdef uint64_t next_prime(self) except +:
        return self._iterator.next_prime()
    cpdef uint64_t previous_prime(self) except +:
        return self._iterator.previous_prime()
