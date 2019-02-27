/*************************************************************************
ALGLIB 3.15.0 (source code generated 2019-02-20)
Copyright (c) Sergey Bochkanov (ALGLIB project).

>>> SOURCE LICENSE >>>
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation (www.fsf.org); either version 2 of the 
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License is available at
http://www.fsf.org/licensing/licenses
>>> END OF LICENSE >>>
*************************************************************************/
#ifndef _ap_h
#define _ap_h

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string>
#include <cstring>
#include <iostream>
#include <math.h>

#if defined(__CODEGEARC__)
#include <list>
#include <vector>
#elif defined(__BORLANDC__)
#include <list.h>
#include <vector.h>
#else
#include <list>
#include <vector>
#endif

#define AE_USE_CPP
/* Definitions */
#define AE_UNKNOWN 0
#define AE_INTEL 1
#define AE_SPARC 2

/* OS definitions */
#define AE_WINDOWS                    1
#define AE_POSIX                      2
#define AE_LINUX                    304
#if !defined(AE_OS)
#define AE_OS AE_UNKNOWN
#endif
#if AE_OS==AE_LINUX
#undef AE_OS
#define AE_OS AE_POSIX
#define _ALGLIB_USE_LINUX_EXTENSIONS
#endif

/* threading models for AE_THREADING */
#define AE_PARALLEL                 100
#define AE_SERIAL                   101
#define AE_SERIAL_UNSAFE            102
#if !defined(AE_THREADING)
#define AE_THREADING AE_PARALLEL
#endif

/* malloc types for AE_MALLOC */
#define AE_STDLIB_MALLOC            200
#define AE_BASIC_STATIC_MALLOC      201
#if !defined(AE_MALLOC)
#define AE_MALLOC AE_STDLIB_MALLOC
#endif

#define AE_LOCK_ALIGNMENT 16

/* automatically determine compiler */
#define AE_MSVC 1
#define AE_GNUC 2
#define AE_SUNC 3
#define AE_COMPILER AE_UNKNOWN
#ifdef __GNUC__
#undef AE_COMPILER
#define AE_COMPILER AE_GNUC
#endif
#if defined(__SUNPRO_C)||defined(__SUNPRO_CC)
#undef AE_COMPILER
#define AE_COMPILER AE_SUNC
#endif
#ifdef _MSC_VER
#undef AE_COMPILER
#define AE_COMPILER AE_MSVC
#endif

/* compiler-specific definitions */
#if AE_COMPILER==AE_MSVC
#define ALIGNED __declspec(align(8))
#elif AE_COMPILER==AE_GNUC
#define ALIGNED __attribute__((aligned(8)))
#else
#define ALIGNED
#endif

/* state flags */
#define _ALGLIB_FLG_THREADING_MASK          0x7
#define _ALGLIB_FLG_THREADING_SHIFT         0
#define _ALGLIB_FLG_THREADING_USE_GLOBAL    0x0
#define _ALGLIB_FLG_THREADING_SERIAL        0x1
#define _ALGLIB_FLG_THREADING_PARALLEL      0x2


/* now we are ready to include headers */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <setjmp.h>
#include <math.h>
#include <stddef.h>

#if defined(AE_HAVE_STDINT)
#include <stdint.h>
#endif

/*
 * SSE2 intrinsics
 *
 * Preprocessor directives below:
 * - include headers for SSE2 intrinsics
 * - define AE_HAS_SSE2_INTRINSICS definition
 *
 * These actions are performed when we have:
 * - x86 architecture definition (AE_CPU==AE_INTEL)
 * - compiler which supports intrinsics
 *
 * Presence of AE_HAS_SSE2_INTRINSICS does NOT mean that our CPU
 * actually supports SSE2 - such things should be determined at runtime
 * with ae_cpuid() call. It means that we are working under Intel and
 * out compiler can issue SSE2-capable code.
 *
 */
#if defined(AE_CPU)
#if AE_CPU==AE_INTEL
#if AE_COMPILER==AE_MSVC
#include <emmintrin.h>
#define AE_HAS_SSE2_INTRINSICS
#endif
#if AE_COMPILER==AE_GNUC
#include <xmmintrin.h>
#define AE_HAS_SSE2_INTRINSICS
#endif
#if AE_COMPILER==AE_SUNC
#include <xmmintrin.h>
#include <emmintrin.h>
#define AE_HAS_SSE2_INTRINSICS
#endif
#endif
#endif



/////////////////////////////////////////////////////////////////////////
//
// THIS SECTION CONTAINS DECLARATIONS FOR BASIC FUNCTIONALITY 
// LIKE MEMORY MANAGEMENT FOR VECTORS/MATRICES WHICH IS SHARED 
// BETWEEN C++ AND PURE C LIBRARIES
//
/////////////////////////////////////////////////////////////////////////
namespace alglib_impl
{

/* if we work under C++ environment, define several conditions */
#ifdef AE_USE_CPP
#define AE_USE_CPP_BOOL
#define AE_USE_CPP_SERIALIZATION
#include <iostream>
#endif

/*
 * define ae_int32_t, ae_int64_t, ae_int_t, ae_bool, ae_complex, ae_error_type and ae_datatype
 */

#if defined(AE_INT32_T)
typedef AE_INT32_T ae_int32_t;
#endif
#if defined(AE_HAVE_STDINT) && !defined(AE_INT32_T)
typedef int32_t ae_int32_t;
#endif
#if !defined(AE_HAVE_STDINT) && !defined(AE_INT32_T)
#if AE_COMPILER==AE_MSVC
typedef __int32 ae_int32_t;
#endif
#if (AE_COMPILER==AE_GNUC) || (AE_COMPILER==AE_SUNC) || (AE_COMPILER==AE_UNKNOWN)
typedef int ae_int32_t;
#endif
#endif

#if defined(AE_INT64_T)
typedef AE_INT64_T ae_int64_t;
#endif
#if defined(AE_HAVE_STDINT) && !defined(AE_INT64_T)
typedef int64_t ae_int64_t;
#endif
#if !defined(AE_HAVE_STDINT) && !defined(AE_INT64_T)
#if AE_COMPILER==AE_MSVC
typedef __int64 ae_int64_t;
#endif
#if (AE_COMPILER==AE_GNUC) || (AE_COMPILER==AE_SUNC) || (AE_COMPILER==AE_UNKNOWN)
typedef signed long long ae_int64_t;
#endif
#endif

#if defined(AE_UINT64_T)
typedef AE_UINT64_T ae_uint64_t;
#endif
#if defined(AE_HAVE_STDINT) && !defined(AE_UINT64_T)
typedef uint64_t ae_uint64_t;
#endif
#if !defined(AE_HAVE_STDINT) && !defined(AE_UINT64_T)
#if AE_COMPILER==AE_MSVC
typedef unsigned __int64 ae_uint64_t;
#endif
#if (AE_COMPILER==AE_GNUC) || (AE_COMPILER==AE_SUNC) || (AE_COMPILER==AE_UNKNOWN)
typedef unsigned long long ae_uint64_t;
#endif
#endif

#if !defined(AE_INT_T)
typedef ptrdiff_t ae_int_t;
#endif

#if !defined(AE_USE_CPP_BOOL)
#define ae_bool char
#define ae_true 1
#define ae_false 0
#else
#define ae_bool bool
#define ae_true true
#define ae_false false
#endif

typedef struct { double x, y; } ae_complex;

typedef enum
{
    ERR_OK = 0,
    ERR_OUT_OF_MEMORY = 1,
    ERR_XARRAY_TOO_LARGE = 2,
    ERR_ASSERTION_FAILED = 3
} ae_error_type;

typedef ae_int_t ae_datatype;

/*
 * other definitions
 */
enum { OWN_CALLER=1, OWN_AE=2 };
enum { ACT_UNCHANGED=1, ACT_SAME_LOCATION=2, ACT_NEW_LOCATION=3 };
enum { DT_BOOL=1, DT_INT=2, DT_REAL=3, DT_COMPLEX=4 };
enum { CPU_SSE2=1 };

/************************************************************************
x-string (zero-terminated):
    owner       OWN_CALLER or OWN_AE. Determines what to do on realloc().
                If vector is owned by caller, X-interface  will  just set
                ptr to NULL before realloc(). If it is  owned  by  X,  it
                will call ae_free/x_free/aligned_free family functions.

    last_action ACT_UNCHANGED, ACT_SAME_LOCATION, ACT_NEW_LOCATION
                contents is either: unchanged, stored at the same location,
                stored at the new location.
                this field is set on return from X.

    ptr         pointer to the actual data

Members of this structure are ae_int64_t to avoid alignment problems.
************************************************************************/
typedef struct
{
    ALIGNED ae_int64_t     owner;
    ALIGNED ae_int64_t     last_action;
    ALIGNED char *ptr;
} x_string;

/************************************************************************
x-vector:
    cnt         number of elements

    datatype    one of the DT_XXXX values

    owner       OWN_CALLER or OWN_AE. Determines what to do on realloc().
                If vector is owned by caller, X-interface  will  just set
                ptr to NULL before realloc(). If it is  owned  by  X,  it
                will call ae_free/x_free/aligned_free family functions.

    last_action ACT_UNCHANGED, ACT_SAME_LOCATION, ACT_NEW_LOCATION
                contents is either: unchanged, stored at the same location,
                stored at the new location.
                this field is set on return from X interface and may be
                used by caller as hint when deciding what to do with data
                (if it was ACT_UNCHANGED or ACT_SAME_LOCATION, no array
                reallocation or copying is required).

    ptr         pointer to the actual data

Members of this structure are ae_int64_t to avoid alignment problems.
************************************************************************/
typedef struct
{
    ae_int64_t     cnt;
    ae_int64_t     datatype;
    ae_int64_t     owner;
    ae_int64_t     last_action;
    union
    {
        void *p_ptr;
        ae_int64_t portable_alignment_enforcer;
    } x_ptr;
} x_vector;


/************************************************************************
x-matrix:
    rows        number of rows. may be zero only when cols is zero too.

    cols        number of columns. may be zero only when rows is zero too.

    stride      stride, i.e. distance between first elements of rows (in bytes)

    datatype    one of the DT_XXXX values

    owner       OWN_CALLER or OWN_AE. Determines what to do on realloc().
                If vector is owned by caller, X-interface  will  just set
                ptr to NULL before realloc(). If it is  owned  by  X,  it
                will call ae_free/x_free/aligned_free family functions.

    last_action ACT_UNCHANGED, ACT_SAME_LOCATION, ACT_NEW_LOCATION
                contents is either: unchanged, stored at the same location,
                stored at the new location.
                this field is set on return from X interface and may be
                used by caller as hint when deciding what to do with data
                (if it was ACT_UNCHANGED or ACT_SAME_LOCATION, no array
                reallocation or copying is required).

    ptr         pointer to the actual data, stored rowwise

Members of this structure are ae_int64_t to avoid alignment problems.
************************************************************************/
typedef struct
{
    ae_int64_t     rows;
    ae_int64_t     cols;
    ae_int64_t     stride;
    ae_int64_t     datatype;
    ae_int64_t     owner;
    ae_int64_t     last_action;
    union
    {
        void *p_ptr;
        ae_int64_t portable_alignment_enforcer;
    } x_ptr;
} x_matrix;


/************************************************************************
dynamic block which may be automatically deallocated during stack unwinding

p_next          next block in the stack unwinding list.
                NULL means that this block is not in the list
deallocator     deallocator function which should be used to deallocate block.
                NULL for "special" blocks (frame/stack boundaries)
ptr             pointer which should be passed to the deallocator.
                may be null (for zero-size block), DYN_BOTTOM or DYN_FRAME
                for "special" blocks (frame/stack boundaries).

valgrind_hint   is a special field which stores a special hint pointer for
                Valgrind and other similar memory checking tools.  ALGLIB
                manually aligns pointers obtained via malloc, so ptr usually
                points to location past the beginning  of  the  actuallly
                allocated memory. In such cases memory testing tools  may
                report "(possibly) lost" memory.
                
                This "hint" field stores  pointer  actually  returned  by
                malloc (or NULL, if for some reason  we  do  not  support
                this feature). This field is used merely as  a  hint  for
                Valgrind - it should NOT be used for anything else.

************************************************************************/
typedef struct ae_dyn_block
{
    struct ae_dyn_block * volatile p_next;
    /* void *deallocator; */
    void (*deallocator)(void*);
    void * volatile ptr;
    void* valgrind_hint;
} ae_dyn_block;

typedef void(*ae_deallocator)(void*);

/************************************************************************
frame marker
************************************************************************/
typedef struct ae_frame
{
    ae_dyn_block db_marker;
} ae_frame;

/************************************************************************
ALGLIB environment state
************************************************************************/
typedef struct ae_state
{
    /*
     * endianness type: AE_LITTLE_ENDIAN or AE_BIG_ENDIAN
     */
    ae_int_t endianness;
    
    /*
     * double value for NAN
     */
    double v_nan;
    
    /*
     * double value for +INF
     */
    double v_posinf;
    
    /*
     * double value for -INF
     */
    double v_neginf;
    
    /*
     * pointer to the top block in a stack of frames
     * which hold dynamically allocated objects
     */
    ae_dyn_block * volatile p_top_block;
    ae_dyn_block last_block;
    
    /*
     * jmp_buf pointer for internal C-style exception handling
     */
    jmp_buf * volatile break_jump;

    /*
     * ae_error_type of the last error (filled when exception is thrown)
     */
    ae_error_type volatile last_error;
    
    /*
     * human-readable message (filled when exception is thrown)
     */
    const char* volatile error_msg;
    
    /*
     * Flags: call-local settings for ALGLIB
     */
    ae_uint64_t flags;
    
    /*
     * threading information:
     * a) current thread pool
     * b) current worker thread
     * c) parent task (one we are solving right now)
     * d) thread exception handler (function which must be called
     *    by ae_assert before raising exception).
     *
     * NOTE: we use void* to store pointers in order to avoid explicit dependency on smp.h
     */
    void *worker_thread;
    void *parent_task;
    void (*thread_exception_handler)(void*);
    
} ae_state;

/************************************************************************
Serializer:

* ae_stream_writer type is a function pointer for stream  writer  method;
  this pointer is used by X-core for out-of-core serialization  (say,  to
  serialize ALGLIB structure directly to managed C# stream).
  
  This function accepts two parameters: pointer to  ANSI  (7-bit)  string
  and pointer-sized integer passed to serializer  during  initialization.
  String being passed is a part of the data stream; aux paramerer may  be
  arbitrary value intended to be used by actual implementation of  stream
  writer. String parameter may include spaces and  linefeed  symbols,  it
  should be written to stream as is.
  
  Return value must be zero for success or non-zero for failure.
  
* ae_stream_reader type is a function pointer for stream  reader  method;
  this pointer is used by X-core for out-of-core unserialization (say, to
  unserialize ALGLIB structure directly from managed C# stream).
  
  This function accepts three parameters: pointer-sized integer passed to
  serializer  during  initialization; number  of  symbols  to  read  from
  stream; pointer to buffer used to store next  token  read  from  stream
  (ANSI encoding is used, buffer is large enough to store all symbols and
  trailing zero symbol).
  
  Number of symbols to read is always positive.
  
  After being called by X-core, this function must:
  * skip all space and linefeed characters from the current  position  at
    the stream and until first non-space non-linefeed character is found
  * read exactly cnt symbols  from  stream  to  buffer;  check  that  all
    symbols being read are non-space non-linefeed ones
  * append trailing zero symbol to buffer
  * return value must be zero on success, non-zero if  even  one  of  the
    conditions above fails. When reader returns non-zero value,  contents
    of buf is not used.
************************************************************************/
typedef char(*ae_stream_writer)(const char *p_string, ae_int_t aux);
typedef char(*ae_stream_reader)(ae_int_t aux, ae_int_t cnt, char *p_buf);

typedef struct
{
    ae_int_t mode;
    ae_int_t entries_needed;
    ae_int_t entries_saved;
    ae_int_t bytes_asked;
    ae_int_t bytes_written;

#ifdef AE_USE_CPP_SERIALIZATION
    std::string     *out_cppstr;
#endif
    char            *out_str; /* pointer to the current position at the output buffer; advanced with each write operation */
    const char      *in_str;  /* pointer to the current position at the input  buffer; advanced with each read  operation */
    ae_int_t         stream_aux;
    ae_stream_writer stream_writer;
    ae_stream_reader stream_reader;
} ae_serializer;


typedef struct ae_vector
{
    /*
     * Number of elements in array, cnt>=0
     */
    ae_int_t cnt;
    
    /*
     * Either DT_BOOL, DT_INT, DT_REAL or DT_COMPLEX
     */
    ae_datatype datatype;
    
    /*
     * If ptr points to memory owned and managed by ae_vector itself,
     * this field is ae_false. If vector was attached to x_vector structure
     * with ae_vector_init_attach_to_x(), this field is ae_true.
     */
    ae_bool is_attached;
    
    /*
     * ae_dyn_block structure which manages data in ptr. This structure
     * is responsible for automatic deletion of object when its frame
     * is destroyed.
     */
    ae_dyn_block data;
    
    /*
     * Pointer to data.
     * User usually works with this field.
     */
    union
    {
        void *p_ptr;
        ae_bool *p_bool;
        ae_int_t *p_int;
        double *p_double;
        ae_complex *p_complex;
    } ptr;
} ae_vector;

typedef struct ae_matrix
{
    ae_int_t rows;
    ae_int_t cols;
    ae_int_t stride;
    ae_datatype datatype;
    
    /*
     * If ptr points to memory owned and managed by ae_vector itself,
     * this field is ae_false. If vector was attached to x_vector structure
     * with ae_vector_init_attach_to_x(), this field is ae_true.
     */
    ae_bool is_attached;
    
    ae_dyn_block data;
    union
    {
        void *p_ptr;
        void **pp_void;
        ae_bool **pp_bool;
        ae_int_t **pp_int;
        double **pp_double;
        ae_complex **pp_complex;
    } ptr;
} ae_matrix;

typedef struct ae_smart_ptr
{
    /* pointer to subscriber; all changes in ptr are translated to subscriber */
    void **subscriber;
    
    /* pointer to object */
    void *ptr;
    
    /* whether smart pointer owns ptr */
    ae_bool is_owner;
    
    /* whether object pointed by ptr is dynamic - clearing such object requires BOTH
       calling destructor function AND calling ae_free for memory occupied by object. */
    ae_bool is_dynamic;
    
    /* destructor function for pointer; clears all dynamically allocated memory */
    void (*destroy)(void*);
    
    /* frame entry; used to ensure automatic deallocation of smart pointer in case of exception/exit */
    ae_dyn_block frame_entry;
} ae_smart_ptr;


/*************************************************************************
Lock.

This structure provides OS-independent non-reentrant lock:
* under Windows/Posix systems it uses system-provided locks
* under Boost it uses OS-independent lock provided by Boost package
* when no OS is defined, it uses "fake lock" (just stub which is not thread-safe):
  a) "fake lock" can be in locked or free mode
  b) "fake lock" can be used only from one thread - one which created lock
  c) when thread acquires free lock, it immediately returns
  d) when thread acquires busy lock, program is terminated
     (because lock is already acquired and no one else can free it)
*************************************************************************/
typedef struct
{
    /*
     * Pointer to _lock structure. This pointer has type void* in order to
     * make header file OS-independent (lock declaration depends on OS).
     */
    void *lock_ptr;
    
    /*
     * For eternal=false this field manages pointer to _lock structure.
     *
     * ae_dyn_block structure is responsible for automatic deletion of
     * the memory allocated for the pointer when its frame is destroyed.
     */
    ae_dyn_block db;
    
    /*
     * Whether we have eternal lock object (used by thread pool) or
     * transient lock. Eternal locks are allocated without using ae_dyn_block
     * structure and do not allow deallocation.
     */
    ae_bool eternal;
} ae_lock;


/*************************************************************************
Shared pool: data structure used to provide thread-safe access to pool  of
temporary variables.
*************************************************************************/
typedef struct ae_shared_pool_entry
{
    void * volatile obj;
    void * volatile next_entry;
} ae_shared_pool_entry;

typedef struct ae_shared_pool
{
    /* lock object which protects pool */
    ae_lock pool_lock;
    
    /* seed object (used to create new instances of temporaries) */
    void                    * volatile seed_object;
    
    /*
     * list of recycled OBJECTS:
     * 1. entries in this list store pointers to recycled objects
     * 2. every time we retrieve object, we retrieve first entry from this list,
     *    move it to recycled_entries and return its obj field to caller/
     */
    ae_shared_pool_entry    * volatile recycled_objects;
    
    /* 
     * list of recycled ENTRIES:
     * 1. this list holds entries which are not used to store recycled objects;
     *    every time recycled object is retrieved, its entry is moved to this list.
     * 2. every time object is recycled, we try to fetch entry for him from this list
     *    before allocating it with malloc()
     */
    ae_shared_pool_entry    * volatile recycled_entries;
    
    /* enumeration pointer, points to current recycled object*/
    ae_shared_pool_entry    * volatile enumeration_counter;
    
    /* size of object; this field is used when we call malloc() for new objects */
    ae_int_t                size_of_object;
    
    /* initializer function; accepts pointer to malloc'ed object, initializes its fields */
    void (*init)(void* dst, ae_state* state, ae_bool make_automatic);
    
    /* copy constructor; accepts pointer to malloc'ed, but not initialized object */
    void (*init_copy)(void* dst, void* src, ae_state* state, ae_bool make_automatic);
    
    /* destructor function; */
    void (*destroy)(void* ptr);
    
    /* frame entry; contains pointer to the pool object itself */
    ae_dyn_block frame_entry;
} ae_shared_pool;

void ae_never_call_it();
void ae_set_dbg_flag(ae_int64_t flag_id, ae_int64_t flag_val);
ae_int64_t ae_get_dbg_value(ae_int64_t id);
void ae_set_global_threading(ae_uint64_t flg_value);
ae_uint64_t ae_get_global_threading();

void ae_set_error_flag(ae_bool *p_flag, ae_bool cond, const char *filename, int lineno, const char *xdesc);
const char * ae_get_last_error_file();
int          ae_get_last_error_line();
const char * ae_get_last_error_xdesc();

ae_int_t ae_misalignment(const void *ptr, size_t alignment);
void* ae_align(void *ptr, size_t alignment);
ae_int_t ae_get_effective_workers(ae_int_t nworkers);
void  ae_optional_atomic_add_i(ae_int_t *p, ae_int_t v);
void  ae_optional_atomic_sub_i(ae_int_t *p, ae_int_t v);

void* aligned_malloc(size_t size, size_t alignment);
void* aligned_extract_ptr(void *block);
void  aligned_free(void *block);
void* eternal_malloc(size_t size);
#if AE_MALLOC==AE_BASIC_STATIC_MALLOC
void set_memory_pool(void *ptr, size_t size);
void memory_pool_stats(ae_int_t *bytes_used, ae_int_t *bytes_free);
#endif

void* ae_malloc(size_t size, ae_state *state);
void  ae_free(void *p);
ae_int_t ae_sizeof(ae_datatype datatype);
ae_bool ae_check_zeros(const void *ptr, ae_int_t n);
void ae_touch_ptr(void *p);

void ae_state_init(ae_state *state);
void ae_state_clear(ae_state *state);
void ae_state_set_break_jump(ae_state *state, jmp_buf *buf);
void ae_state_set_flags(ae_state *state, ae_uint64_t flags);
void ae_break(ae_state *state, ae_error_type error_type, const char *msg);

void ae_frame_make(ae_state *state, ae_frame *tmp);
void ae_frame_leave(ae_state *state);

void ae_db_attach(ae_dyn_block *block, ae_state *state);
void ae_db_init(ae_dyn_block *block, ae_int_t size, ae_state *state, ae_bool make_automatic);
void ae_db_realloc(ae_dyn_block *block, ae_int_t size, ae_state *state);
void ae_db_free(ae_dyn_block *block);
void ae_db_swap(ae_dyn_block *block1, ae_dyn_block *block2);

void ae_vector_init(ae_vector *dst, ae_int_t size, ae_datatype datatype, ae_state *state, ae_bool make_automatic);
void ae_vector_init_copy(ae_vector *dst, ae_vector *src, ae_state *state, ae_bool make_automatic);
void ae_vector_init_from_x(ae_vector *dst, x_vector *src, ae_state *state, ae_bool make_automatic);
void ae_vector_init_attach_to_x(ae_vector *dst, x_vector *src, ae_state *state, ae_bool make_automatic);
void ae_vector_set_length(ae_vector *dst, ae_int_t newsize, ae_state *state);
void ae_vector_clear(ae_vector *dst);
void ae_vector_destroy(ae_vector *dst);
void ae_swap_vectors(ae_vector *vec1, ae_vector *vec2);

void ae_matrix_init(ae_matrix *dst, ae_int_t rows, ae_int_t cols, ae_datatype datatype, ae_state *state, ae_bool make_automatic);
void ae_matrix_init_copy(ae_matrix *dst, ae_matrix *src, ae_state *state, ae_bool make_automatic);
void ae_matrix_init_from_x(ae_matrix *dst, x_matrix *src, ae_state *state, ae_bool make_automatic);
void ae_matrix_init_attach_to_x(ae_matrix *dst, x_matrix *src, ae_state *state, ae_bool make_automatic);
void ae_matrix_set_length(ae_matrix *dst, ae_int_t rows, ae_int_t cols, ae_state *state);
void ae_matrix_clear(ae_matrix *dst);
void ae_matrix_destroy(ae_matrix *dst);
void ae_swap_matrices(ae_matrix *mat1, ae_matrix *mat2);

void ae_smart_ptr_init(ae_smart_ptr *dst, void **subscriber, ae_state *state, ae_bool make_automatic);
void ae_smart_ptr_clear(void *_dst); /* accepts ae_smart_ptr* */
void ae_smart_ptr_destroy(void *_dst);
void ae_smart_ptr_assign(ae_smart_ptr *dst, void *new_ptr, ae_bool is_owner, ae_bool is_dynamic, void (*destroy)(void*));
void ae_smart_ptr_release(ae_smart_ptr *dst);

void ae_yield();
void ae_init_lock(ae_lock *lock, ae_state *state, ae_bool make_automatic);
void ae_init_lock_eternal(ae_lock *lock);
void ae_acquire_lock(ae_lock *lock);
void ae_release_lock(ae_lock *lock);
void ae_free_lock(ae_lock *lock);

void ae_shared_pool_init(void *_dst, ae_state *state, ae_bool make_automatic);
void ae_shared_pool_init_copy(void *_dst, void *_src, ae_state *state, ae_bool make_automatic);
void ae_shared_pool_clear(void *dst);
void ae_shared_pool_destroy(void *dst);
ae_bool ae_shared_pool_is_initialized(void *_dst);
void ae_shared_pool_set_seed(
    ae_shared_pool  *dst,
    void            *seed_object,
    ae_int_t        size_of_object,
    void            (*init)(void* dst, ae_state* state, ae_bool make_automatic),
    void            (*init_copy)(void* dst, void* src, ae_state* state, ae_bool make_automatic),
    void            (*destroy)(void* ptr),
    ae_state        *state);
void ae_shared_pool_retrieve(
    ae_shared_pool  *pool,
    ae_smart_ptr    *pptr,
    ae_state        *state);
void ae_shared_pool_recycle(
    ae_shared_pool  *pool,
    ae_smart_ptr    *pptr,
    ae_state        *state);
void ae_shared_pool_clear_recycled(
    ae_shared_pool  *pool,
    ae_state        *state);
void ae_shared_pool_first_recycled(
    ae_shared_pool  *pool,
    ae_smart_ptr    *pptr,
    ae_state        *state);
void ae_shared_pool_next_recycled(
    ae_shared_pool  *pool,
    ae_smart_ptr    *pptr,
    ae_state        *state);
void ae_shared_pool_reset(
    ae_shared_pool  *pool,
    ae_state        *state);

void ae_x_set_vector(x_vector *dst, ae_vector *src, ae_state *state);
void ae_x_set_matrix(x_matrix *dst, ae_matrix *src, ae_state *state);
void ae_x_attach_to_vector(x_vector *dst, ae_vector *src);
void ae_x_attach_to_matrix(x_matrix *dst, ae_matrix *src);

void x_vector_clear(x_vector *dst);

ae_bool x_is_symmetric(x_matrix *a);
ae_bool x_is_hermitian(x_matrix *a);
ae_bool x_force_symmetric(x_matrix *a);
ae_bool x_force_hermitian(x_matrix *a);
ae_bool ae_is_symmetric(ae_matrix *a);
ae_bool ae_is_hermitian(ae_matrix *a);
ae_bool ae_force_symmetric(ae_matrix *a);
ae_bool ae_force_hermitian(ae_matrix *a);

void ae_serializer_init(ae_serializer *serializer);
void ae_serializer_clear(ae_serializer *serializer);

void ae_serializer_alloc_start(ae_serializer *serializer);
void ae_serializer_alloc_entry(ae_serializer *serializer);
ae_int_t ae_serializer_get_alloc_size(ae_serializer *serializer);

#ifdef AE_USE_CPP_SERIALIZATION
void ae_serializer_sstart_str(ae_serializer *serializer, std::string *buf);
void ae_serializer_ustart_str(ae_serializer *serializer, const std::string *buf);
void ae_serializer_sstart_stream(ae_serializer *serializer, std::ostream *stream);
void ae_serializer_ustart_stream(ae_serializer *serializer, const std::istream *stream);
#endif
void ae_serializer_sstart_str(ae_serializer *serializer, char *buf);
void ae_serializer_ustart_str(ae_serializer *serializer, const char *buf);
void ae_serializer_sstart_stream(ae_serializer *serializer, ae_stream_writer writer, ae_int_t aux);
void ae_serializer_ustart_stream(ae_serializer *serializer, ae_stream_reader reader, ae_int_t aux);

void ae_serializer_serialize_bool(ae_serializer *serializer, ae_bool v, ae_state *state);
void ae_serializer_serialize_int(ae_serializer *serializer, ae_int_t v, ae_state *state);
void ae_serializer_serialize_double(ae_serializer *serializer, double v, ae_state *state);
void ae_serializer_unserialize_bool(ae_serializer *serializer, ae_bool *v, ae_state *state);
void ae_serializer_unserialize_int(ae_serializer *serializer, ae_int_t *v, ae_state *state);
void ae_serializer_unserialize_double(ae_serializer *serializer, double *v, ae_state *state);

void ae_serializer_stop(ae_serializer *serializer, ae_state *state);

/************************************************************************
Service functions
************************************************************************/
void ae_assert(ae_bool cond, const char *msg, ae_state *state);
ae_int_t ae_cpuid();

/************************************************************************
Real math functions:
* IEEE-compliant floating point comparisons
* standard functions
************************************************************************/
ae_bool ae_fp_eq(double v1, double v2);
ae_bool ae_fp_neq(double v1, double v2);
ae_bool ae_fp_less(double v1, double v2);
ae_bool ae_fp_less_eq(double v1, double v2);
ae_bool ae_fp_greater(double v1, double v2);
ae_bool ae_fp_greater_eq(double v1, double v2);

ae_bool ae_isfinite_stateless(double x, ae_int_t endianness);
ae_bool ae_isnan_stateless(double x,    ae_int_t endianness);
ae_bool ae_isinf_stateless(double x,    ae_int_t endianness);
ae_bool ae_isposinf_stateless(double x, ae_int_t endianness);
ae_bool ae_isneginf_stateless(double x, ae_int_t endianness);

ae_int_t ae_get_endianness();

ae_bool ae_isfinite(double x,ae_state *state);
ae_bool ae_isnan(double x,   ae_state *state);
ae_bool ae_isinf(double x,   ae_state *state);
ae_bool ae_isposinf(double x,ae_state *state);
ae_bool ae_isneginf(double x,ae_state *state);

double   ae_fabs(double x,   ae_state *state);
ae_int_t ae_iabs(ae_int_t x, ae_state *state);
double   ae_sqr(double x,    ae_state *state);
double   ae_sqrt(double x,   ae_state *state);

ae_int_t ae_sign(double x,   ae_state *state);
ae_int_t ae_round(double x,  ae_state *state);
ae_int_t ae_trunc(double x,  ae_state *state);
ae_int_t ae_ifloor(double x, ae_state *state);
ae_int_t ae_iceil(double x,  ae_state *state);

ae_int_t ae_maxint(ae_int_t m1, ae_int_t m2, ae_state *state);
ae_int_t ae_minint(ae_int_t m1, ae_int_t m2, ae_state *state);
double   ae_maxreal(double m1, double m2, ae_state *state);
double   ae_minreal(double m1, double m2, ae_state *state);
double   ae_randomreal(ae_state *state);
ae_int_t ae_randominteger(ae_int_t maxv, ae_state *state);

double   ae_sin(double x, ae_state *state);
double   ae_cos(double x, ae_state *state);
double   ae_tan(double x, ae_state *state);
double   ae_sinh(double x, ae_state *state);
double   ae_cosh(double x, ae_state *state);
double   ae_tanh(double x, ae_state *state);
double   ae_asin(double x, ae_state *state);
double   ae_acos(double x, ae_state *state);
double   ae_atan(double x, ae_state *state);
double   ae_atan2(double y, double x, ae_state *state);

double   ae_log(double x, ae_state *state);
double   ae_pow(double x, double y, ae_state *state);
double   ae_exp(double x, ae_state *state);

/************************************************************************
Complex math functions:
* basic arithmetic operations
* standard functions
************************************************************************/
ae_complex ae_complex_from_i(ae_int_t v);
ae_complex ae_complex_from_d(double v);

ae_complex ae_c_neg(ae_complex lhs);
ae_bool ae_c_eq(ae_complex lhs,       ae_complex rhs);
ae_bool ae_c_neq(ae_complex lhs,      ae_complex rhs);
ae_complex ae_c_add(ae_complex lhs,   ae_complex rhs);
ae_complex ae_c_mul(ae_complex lhs,   ae_complex rhs);
ae_complex ae_c_sub(ae_complex lhs,   ae_complex rhs);
ae_complex ae_c_div(ae_complex lhs,   ae_complex rhs);
ae_bool ae_c_eq_d(ae_complex lhs,     double rhs);
ae_bool ae_c_neq_d(ae_complex lhs,    double rhs);
ae_complex ae_c_add_d(ae_complex lhs, double rhs);
ae_complex ae_c_mul_d(ae_complex lhs, double rhs);
ae_complex ae_c_sub_d(ae_complex lhs, double rhs);
ae_complex ae_c_d_sub(double lhs,     ae_complex rhs);
ae_complex ae_c_div_d(ae_complex lhs, double rhs);
ae_complex ae_c_d_div(double lhs,   ae_complex rhs);

ae_complex ae_c_conj(ae_complex lhs, ae_state *state);
ae_complex ae_c_sqr(ae_complex lhs, ae_state *state);
double     ae_c_abs(ae_complex z, ae_state *state);

/************************************************************************
Complex BLAS operations
************************************************************************/
ae_complex ae_v_cdotproduct(const ae_complex *v0, ae_int_t stride0, const char *conj0, const ae_complex *v1, ae_int_t stride1, const char *conj1, ae_int_t n);
void ae_v_cmove(ae_complex *vdst,    ae_int_t stride_dst, const ae_complex* vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n);
void ae_v_cmoveneg(ae_complex *vdst, ae_int_t stride_dst, const ae_complex* vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n);
void ae_v_cmoved(ae_complex *vdst,   ae_int_t stride_dst, const ae_complex* vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n, double alpha);
void ae_v_cmovec(ae_complex *vdst,   ae_int_t stride_dst, const ae_complex* vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n, ae_complex alpha);
void ae_v_cadd(ae_complex *vdst,     ae_int_t stride_dst, const ae_complex *vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n);
void ae_v_caddd(ae_complex *vdst,    ae_int_t stride_dst, const ae_complex *vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n, double alpha);
void ae_v_caddc(ae_complex *vdst,    ae_int_t stride_dst, const ae_complex *vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n, ae_complex alpha);
void ae_v_csub(ae_complex *vdst,     ae_int_t stride_dst, const ae_complex *vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n);
void ae_v_csubd(ae_complex *vdst, ae_int_t stride_dst, const ae_complex *vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n, double alpha);
void ae_v_csubc(ae_complex *vdst, ae_int_t stride_dst, const ae_complex *vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n, ae_complex alpha);
void ae_v_cmuld(ae_complex *vdst, ae_int_t stride_dst, ae_int_t n, double alpha);
void ae_v_cmulc(ae_complex *vdst, ae_int_t stride_dst, ae_int_t n, ae_complex alpha);

/************************************************************************
Real BLAS operations
************************************************************************/
double ae_v_dotproduct(const double *v0, ae_int_t stride0, const double *v1, ae_int_t stride1, ae_int_t n);
void ae_v_move(double *vdst,    ae_int_t stride_dst, const double* vsrc,  ae_int_t stride_src, ae_int_t n);
void ae_v_moveneg(double *vdst, ae_int_t stride_dst, const double* vsrc,  ae_int_t stride_src, ae_int_t n);
void ae_v_moved(double *vdst,   ae_int_t stride_dst, const double* vsrc,  ae_int_t stride_src, ae_int_t n, double alpha);
void ae_v_add(double *vdst,     ae_int_t stride_dst, const double *vsrc,  ae_int_t stride_src, ae_int_t n);
void ae_v_addd(double *vdst,    ae_int_t stride_dst, const double *vsrc,  ae_int_t stride_src, ae_int_t n, double alpha);
void ae_v_sub(double *vdst,     ae_int_t stride_dst, const double *vsrc,  ae_int_t stride_src, ae_int_t n);
void ae_v_subd(double *vdst,    ae_int_t stride_dst, const double *vsrc,  ae_int_t stride_src, ae_int_t n, double alpha);
void ae_v_muld(double *vdst,  ae_int_t stride_dst, ae_int_t n, double alpha);

/************************************************************************
Other functions
************************************************************************/
ae_int_t ae_v_len(ae_int_t a, ae_int_t b);

/*
extern const double ae_machineepsilon;
extern const double ae_maxrealnumber;
extern const double ae_minrealnumber;
extern const double ae_pi;
*/
#define ae_machineepsilon 5E-16
#define ae_maxrealnumber  1E300
#define ae_minrealnumber  1E-300
#define ae_pi 3.1415926535897932384626433832795


/************************************************************************
RComm functions
************************************************************************/
typedef struct rcommstate
{
    int stage;
    ae_vector ia;
    ae_vector ba;
    ae_vector ra;
    ae_vector ca;
} rcommstate;
void _rcommstate_init(rcommstate* p, ae_state *_state, ae_bool make_automatic);
void _rcommstate_init_copy(rcommstate* dst, rcommstate* src, ae_state *_state, ae_bool make_automatic);
void _rcommstate_clear(rcommstate* p);
void _rcommstate_destroy(rcommstate* p);


/************************************************************************
Allocation counters, inactive by default.
Turned on when needed for debugging purposes.

_alloc_counter is incremented by 1 on malloc(), decremented on free().
_alloc_counter_total is only incremented by 1.
************************************************************************/
extern ae_int_t   _alloc_counter;
extern ae_int_t   _alloc_counter_total;
extern ae_bool    _use_alloc_counter;


/************************************************************************
Malloc debugging:

* _force_malloc_failure - set this flag to ae_true in  order  to  enforce
  failure of ALGLIB malloc(). Useful to debug handling of  errors  during
  memory allocation. As long as this flag is set, ALGLIB malloc will fail.
* _malloc_failure_after - set it to non-zero value in  order  to  enforce
  malloc failure as soon as _alloc_counter_total increases above value of
  this variable. This value has no effect if  _use_alloc_counter  is  not
  set.
************************************************************************/
extern ae_bool    _force_malloc_failure;
extern ae_int_t   _malloc_failure_after;


/************************************************************************
debug functions (must be turned on by preprocessor definitions):
* tickcount(), which is wrapper around GetTickCount()
* flushconsole(), fluches console
* ae_debugrng(), returns random number generated with high-quality random numbers generator
* ae_set_seed(), sets seed of the debug RNG (NON-THREAD-SAFE!!!)
* ae_get_seed(), returns two seed values of the debug RNG (NON-THREAD-SAFE!!!)
************************************************************************/
#ifdef AE_DEBUG4WINDOWS
#define flushconsole(s) fflush(stdout)
#define tickcount(s) _tickcount()
int _tickcount();
#endif
#ifdef AE_DEBUG4POSIX
#define flushconsole(s) fflush(stdout)
#define tickcount(s) _tickcount()
int _tickcount();
#endif


}


/////////////////////////////////////////////////////////////////////////
//
// THIS SECTION CONTAINS DECLARATIONS FOR C++ RELATED FUNCTIONALITY
//
/////////////////////////////////////////////////////////////////////////

namespace alglib
{

typedef alglib_impl::ae_int_t ae_int_t;

/********************************************************************
Class forwards
********************************************************************/
class complex;

ae_int_t vlen(ae_int_t n1, ae_int_t n2);

/********************************************************************
Exception class.
********************************************************************/
#if !defined(AE_NO_EXCEPTIONS)
class ap_error
{
public:
    std::string msg;
    
    ap_error();
    ap_error(const char *s);
    static void make_assertion(bool bClause);
    static void make_assertion(bool bClause, const char *p_msg);
private:
};
#endif

/********************************************************************
Complex number with double precision.
********************************************************************/
class complex
{
public:
    complex();
    complex(const double &_x);
    complex(const double &_x, const double &_y);
    complex(const complex &z);

    complex& operator= (const double& v);
    complex& operator+=(const double& v);
    complex& operator-=(const double& v);
    complex& operator*=(const double& v);
    complex& operator/=(const double& v);

    complex& operator= (const complex& z);
    complex& operator+=(const complex& z);
    complex& operator-=(const complex& z);
    complex& operator*=(const complex& z);
    complex& operator/=(const complex& z);

    alglib_impl::ae_complex*       c_ptr();
    const alglib_impl::ae_complex* c_ptr() const;
    
#if !defined(AE_NO_EXCEPTIONS)
    std::string tostring(int dps) const;
#endif

    double x, y;
};

const alglib::complex operator/(const alglib::complex& lhs, const alglib::complex& rhs);
bool operator==(const alglib::complex& lhs, const alglib::complex& rhs);
bool operator!=(const alglib::complex& lhs, const alglib::complex& rhs);
const alglib::complex operator+(const alglib::complex& lhs);
const alglib::complex operator-(const alglib::complex& lhs);
const alglib::complex operator+(const alglib::complex& lhs, const alglib::complex& rhs);
const alglib::complex operator+(const alglib::complex& lhs, const double& rhs);
const alglib::complex operator+(const double& lhs, const alglib::complex& rhs);
const alglib::complex operator-(const alglib::complex& lhs, const alglib::complex& rhs);
const alglib::complex operator-(const alglib::complex& lhs, const double& rhs);
const alglib::complex operator-(const double& lhs, const alglib::complex& rhs);
const alglib::complex operator*(const alglib::complex& lhs, const alglib::complex& rhs);
const alglib::complex operator*(const alglib::complex& lhs, const double& rhs);
const alglib::complex operator*(const double& lhs, const alglib::complex& rhs);
const alglib::complex operator/(const alglib::complex& lhs, const alglib::complex& rhs);
const alglib::complex operator/(const double& lhs, const alglib::complex& rhs);
const alglib::complex operator/(const alglib::complex& lhs, const double& rhs);
double abscomplex(const alglib::complex &z);
alglib::complex conj(const alglib::complex &z);
alglib::complex csqr(const alglib::complex &z);

/********************************************************************
Level 1 BLAS functions

NOTES:
* destination and source should NOT overlap
* stride is assumed to be positive, but it is not 
  assert'ed within function
* conj_src parameter specifies whether complex source is conjugated 
  before processing or not. Pass string which starts with 'N' or 'n'
  ("No conj", for example) to use unmodified parameter. All other
  values will result in conjugation of input, but it is recommended
  to use "Conj" in such cases.
********************************************************************/
double vdotproduct(const double *v0, ae_int_t stride0, const double *v1, ae_int_t stride1, ae_int_t n);
double vdotproduct(const double *v1, const double *v2, ae_int_t N);

alglib::complex vdotproduct(const alglib::complex *v0, ae_int_t stride0, const char *conj0, const alglib::complex *v1, ae_int_t stride1, const char *conj1, ae_int_t n);
alglib::complex vdotproduct(const alglib::complex *v1, const alglib::complex *v2, ae_int_t N);

void vmove(double *vdst,  ae_int_t stride_dst, const double* vsrc,  ae_int_t stride_src, ae_int_t n);
void vmove(double *vdst, const double* vsrc, ae_int_t N);

void vmove(alglib::complex *vdst, ae_int_t stride_dst, const alglib::complex* vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n);
void vmove(alglib::complex *vdst, const alglib::complex* vsrc, ae_int_t N);

void vmoveneg(double *vdst,  ae_int_t stride_dst, const double* vsrc,  ae_int_t stride_src, ae_int_t n);
void vmoveneg(double *vdst, const double *vsrc, ae_int_t N);

void vmoveneg(alglib::complex *vdst, ae_int_t stride_dst, const alglib::complex* vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n);
void vmoveneg(alglib::complex *vdst, const alglib::complex *vsrc, ae_int_t N);

void vmove(double *vdst,  ae_int_t stride_dst, const double* vsrc,  ae_int_t stride_src, ae_int_t n, double alpha);
void vmove(double *vdst, const double *vsrc, ae_int_t N, double alpha);

void vmove(alglib::complex *vdst, ae_int_t stride_dst, const alglib::complex* vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n, double alpha);
void vmove(alglib::complex *vdst, const alglib::complex *vsrc, ae_int_t N, double alpha);

void vmove(alglib::complex *vdst, ae_int_t stride_dst, const alglib::complex* vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n, alglib::complex alpha);
void vmove(alglib::complex *vdst, const alglib::complex *vsrc, ae_int_t N, alglib::complex alpha);

void vadd(double *vdst,  ae_int_t stride_dst, const double *vsrc,  ae_int_t stride_src, ae_int_t n);
void vadd(double *vdst, const double *vsrc, ae_int_t N);

void vadd(alglib::complex *vdst, ae_int_t stride_dst, const alglib::complex *vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n);
void vadd(alglib::complex *vdst, const alglib::complex *vsrc, ae_int_t N);

void vadd(double *vdst,  ae_int_t stride_dst, const double *vsrc,  ae_int_t stride_src, ae_int_t n, double alpha);
void vadd(double *vdst, const double *vsrc, ae_int_t N, double alpha);

void vadd(alglib::complex *vdst, ae_int_t stride_dst, const alglib::complex *vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n, double alpha);
void vadd(alglib::complex *vdst, const alglib::complex *vsrc, ae_int_t N, double alpha);

void vadd(alglib::complex *vdst, ae_int_t stride_dst, const alglib::complex *vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n, alglib::complex alpha);
void vadd(alglib::complex *vdst, const alglib::complex *vsrc, ae_int_t N, alglib::complex alpha);

void vsub(double *vdst,  ae_int_t stride_dst, const double *vsrc,  ae_int_t stride_src, ae_int_t n);
void vsub(double *vdst, const double *vsrc, ae_int_t N);

void vsub(alglib::complex *vdst, ae_int_t stride_dst, const alglib::complex *vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n);
void vsub(alglib::complex *vdst, const alglib::complex *vsrc, ae_int_t N);

void vsub(double *vdst,  ae_int_t stride_dst, const double *vsrc,  ae_int_t stride_src, ae_int_t n, double alpha);
void vsub(double *vdst, const double *vsrc, ae_int_t N, double alpha);

void vsub(alglib::complex *vdst, ae_int_t stride_dst, const alglib::complex *vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n, double alpha);
void vsub(alglib::complex *vdst, const alglib::complex *vsrc, ae_int_t N, double alpha);

void vsub(alglib::complex *vdst, ae_int_t stride_dst, const alglib::complex *vsrc, ae_int_t stride_src, const char *conj_src, ae_int_t n, alglib::complex alpha);
void vsub(alglib::complex *vdst, const alglib::complex *vsrc, ae_int_t N, alglib::complex alpha);

void vmul(double *vdst,  ae_int_t stride_dst, ae_int_t n, double alpha);
void vmul(double *vdst, ae_int_t N, double alpha);

void vmul(alglib::complex *vdst, ae_int_t stride_dst, ae_int_t n, double alpha);
void vmul(alglib::complex *vdst, ae_int_t N, double alpha);

void vmul(alglib::complex *vdst, ae_int_t stride_dst, ae_int_t n, alglib::complex alpha);
void vmul(alglib::complex *vdst, ae_int_t N, alglib::complex alpha);


/********************************************************************
xparams type and several predefined constants
********************************************************************/
struct xparams
{
    alglib_impl::ae_uint64_t flags;
};

extern const xparams &xdefault;
extern const xparams &serial;
extern const xparams &parallel;

/********************************************************************
Threading functions
********************************************************************/
// nworkers can be 1, 2, ... ; or 0 for auto; or -1/-2/... for all except for one/two/...
void setnworkers(alglib::ae_int_t nworkers);

// sets global threading settings to alglib::serial or alglib::parallel
void setglobalthreading(const xparams settings);

// nworkers can be 1, 2, ... ; or 0 for auto; or -1/-2/... for all except for one/two/...
alglib::ae_int_t getnworkers();

/********************************************************************
internal functions used by test_x.cpp, interfaces for functions present
in commercial ALGLIB but lacking in free edition.
********************************************************************/
ae_int_t _ae_cores_count();
void _ae_set_global_threading(alglib_impl::ae_uint64_t flg_value);
alglib_impl::ae_uint64_t _ae_get_global_threading();

/********************************************************************
1- and 2-dimensional arrays
********************************************************************/
class ae_vector_wrapper
{
public:
    //
    // Creates object attached to external ae_vector structure.
    //
    // NOTE: this function also checks that source ae_vector* has
    //       required datatype. An exception is generated otherwise.
    //
    ae_vector_wrapper(alglib_impl::ae_vector *e_ptr, alglib_impl::ae_datatype datatype);
    
    //
    // Creates zero-size vector of specific datatype
    //
    ae_vector_wrapper(alglib_impl::ae_datatype datatype);
    
    //
    // Creates a copy of another vector (can be reference to one of the derived classes)
    //
    // NOTE: this function also checks that source ae_vector* has
    //       required datatype. An exception is generated otherwise.
    //
    ae_vector_wrapper(const ae_vector_wrapper &rhs, alglib_impl::ae_datatype datatype);
    
    //
    // Well, it is destructor...
    //
    virtual ~ae_vector_wrapper();

    //
    // For wrapper object allocated with allocate_own() this function
    // changes length, completely dropping previous contents.
    //
    // It does not work (throws exception) for frozen proxy objects.
    //
    void setlength(ae_int_t iLen);
    
    //
    // Element count
    //
    ae_int_t length() const;
    
    //
    // Access to internal C-structure used by C-core.
    // Not intended for external use.
    //
    const alglib_impl::ae_vector* c_ptr() const;
    alglib_impl::ae_vector* c_ptr();
private:
    ae_vector_wrapper();
    ae_vector_wrapper(const ae_vector_wrapper &rhs);
    const ae_vector_wrapper& operator=(const ae_vector_wrapper &rhs);
protected:
#if !defined(AE_NO_EXCEPTIONS)
    //
    // Copies array given by string into current object. Additional
    // parameter DATATYPE contains information about type of the data
    // in S and type of the array to create.
    //
    // NOTE: this function is not supported in exception-free mode.
    //
    ae_vector_wrapper(const char *s, alglib_impl::ae_datatype datatype);
#endif

    //
    // This function attaches wrapper object to external x_vector structure;
    // "frozen proxy" mode is activated (you can read/write, but can not reallocate
    // and do not own memory of the vector).
    //
    // NOTE: initial state of wrapper object is assumed to be initialized;
    //       all previously allocated memory is properly deallocated.
    //
    // NOTE: x_vector structure pointed by new_ptr is used only once; after
    //       we fetch pointer to memory and its size, this structure is ignored
    //       and not referenced anymore. So, you can pass pointers to temporary
    //       x-structures which are deallocated immediately after you call attach_to()
    //
    // NOTE: state structure is used for error reporting purposes (longjmp on errors).
    //
    void attach_to(alglib_impl::x_vector *new_ptr, alglib_impl::ae_state *_state);

    //
    // Assigns RHS to current object. Returns *this.
    //
    // It has several branches depending on target object status:
    // * in case it is proxy object, data are copied into memory pointed by
    //   proxy. Function checks that source has exactly same size as target
    //   (exception is thrown on failure).
    // * in case it is non-proxy object, data allocated by object are cleared
    //   and a copy of RHS is created in target.
    //
    // NOTE: this function correctly handles assignments of the object to itself.
    //
    const ae_vector_wrapper& assign(const ae_vector_wrapper &rhs);
    
    //
    // Pointer to ae_vector structure:
    // * ptr==&inner_vec means that wrapper object owns ae_vector structure and
    //   is responsible for proper deallocation of its memory
    // * ptr!=&inner_vec means that wrapper object works with someone's other
    //   ae_vector record and is not responsible for its memory; in this case
    //   inner_vec is assumed to be uninitialized.
    //
    alglib_impl::ae_vector *ptr;
    
    //
    // Inner ae_vector record.
    // Ignored for ptr!=&inner_rec.
    //
    alglib_impl::ae_vector inner_vec;
    
    //
    // Whether this wrapper object is frozen proxy (you may read array, may
    // modify its value, but can not deallocate its memory or resize it) or not.
    //
    // If is_frozen_proxy==true and if:
    // * ptr==&inner_vec, it means that wrapper works with its own ae_vector
    //   structure, but this structure points to externally allocated memory.
    //   This memory is NOT owned by ae_vector object.
    // * ptr!=&inner_vec, it means that wrapper works with externally allocated
    //   and managed ae_vector structure. Both memory pointed by ae_vector and
    //   ae_vector structure itself are not owned by wrapper object.
    //
    bool                   is_frozen_proxy;
};

class boolean_1d_array : public ae_vector_wrapper
{
public:
    boolean_1d_array();
    boolean_1d_array(const boolean_1d_array &rhs);
    boolean_1d_array(alglib_impl::ae_vector *p);
    const boolean_1d_array& operator=(const boolean_1d_array &rhs);
    virtual ~boolean_1d_array() ;

    const ae_bool& operator()(ae_int_t i) const;
    ae_bool& operator()(ae_int_t i);

    const ae_bool& operator[](ae_int_t i) const;
    ae_bool& operator[](ae_int_t i);

    //
    // This function allocates array[iLen] and copies data
    // pointed by pContent to its memory. Completely independent
    // copy of data is created.
    //
    void setcontent(ae_int_t iLen, const bool *pContent );
    
    //
    // This function returns pointer to internal memory
    //
    ae_bool* getcontent();
    const ae_bool* getcontent() const;

#if !defined(AE_NO_EXCEPTIONS)
    boolean_1d_array(const char *s);
    std::string tostring() const;
#endif
};

class integer_1d_array : public ae_vector_wrapper
{
public:
    integer_1d_array();
    integer_1d_array(const integer_1d_array &rhs);
    integer_1d_array(alglib_impl::ae_vector *p);
    const integer_1d_array& operator=(const integer_1d_array &rhs);
    virtual ~integer_1d_array();

    const ae_int_t& operator()(ae_int_t i) const;
    ae_int_t& operator()(ae_int_t i);

    const ae_int_t& operator[](ae_int_t i) const;
    ae_int_t& operator[](ae_int_t i);

    //
    // This function allocates array[iLen] and copies data
    // pointed by pContent to its memory. Completely independent
    // copy of data is created.
    //
    void setcontent(ae_int_t iLen, const ae_int_t *pContent );
    
    //
    // This function returns pointer to internal memory
    //
    ae_int_t* getcontent();
    const ae_int_t* getcontent() const;

#if !defined(AE_NO_EXCEPTIONS)
    integer_1d_array(const char *s);
    std::string tostring() const;
#endif
};

class real_1d_array : public ae_vector_wrapper
{
public:
    real_1d_array();
    real_1d_array(const real_1d_array &rhs);
    real_1d_array(alglib_impl::ae_vector *p);
    const real_1d_array& operator=(const real_1d_array &rhs);
    virtual ~real_1d_array();

    const double& operator()(ae_int_t i) const;
    double& operator()(ae_int_t i);

    const double& operator[](ae_int_t i) const;
    double& operator[](ae_int_t i);

    //
    // This function allocates array[iLen] and copies data
    // pointed by pContent to its memory. Completely independent
    // copy of data is created.
    //
    void setcontent(ae_int_t iLen, const double *pContent);
    
    //
    // This function attaches array to memory pointed by pContent.
    // No own memory is allocated, no copying of data is performed,
    // so pContent pointer should be valid as long as we work with
    // array.
    //
    // After you attach array object to external memory, it becomes
    // "frozen": it is possible to read/write array elements, but
    // it is not allowed to resize it (no setlength() calls).
    //
    void attach_to_ptr(ae_int_t iLen, double *pContent);
    
    //
    // This function returns pointer to internal memory
    //
    double* getcontent();
    const double* getcontent() const;

#if !defined(AE_NO_EXCEPTIONS)
    real_1d_array(const char *s);
    std::string tostring(int dps) const;
#endif
};

class complex_1d_array : public ae_vector_wrapper
{
public:
    complex_1d_array();
    complex_1d_array(const complex_1d_array &rhs);
    complex_1d_array(alglib_impl::ae_vector *p);
    const complex_1d_array& operator=(const complex_1d_array &rhs);
    virtual ~complex_1d_array();

    const alglib::complex& operator()(ae_int_t i) const;
    alglib::complex& operator()(ae_int_t i);

    const alglib::complex& operator[](ae_int_t i) const;
    alglib::complex& operator[](ae_int_t i);

    //
    // This function allocates array[iLen] and copies data
    // pointed by pContent to its memory. Completely independent
    // copy of data is created.
    //
    void setcontent(ae_int_t iLen, const alglib::complex *pContent );
    alglib::complex* getcontent();
    const alglib::complex* getcontent() const;

#if !defined(AE_NO_EXCEPTIONS)
    complex_1d_array(const char *s);
    std::string tostring(int dps) const;
#endif
};

class ae_matrix_wrapper
{
public:
    //
    // Creates object attached to external ae_vector structure, with additional
    // check for matching datatypes (e_ptr->datatype==datatype is required).
    //
    ae_matrix_wrapper(alglib_impl::ae_matrix *e_ptr, alglib_impl::ae_datatype datatype);
    
    //
    // Creates zero-sized matrix of specified datatype.
    //
    ae_matrix_wrapper(alglib_impl::ae_datatype datatype);
    
    //
    // Creates copy of rhs, with additional check for matching datatypes
    // (rhs.datatype==datatype is required).
    //
    ae_matrix_wrapper(const ae_matrix_wrapper &rhs, alglib_impl::ae_datatype datatype);
    
    //
    // Destructor
    //
    virtual ~ae_matrix_wrapper();
    

    void setlength(ae_int_t rows, ae_int_t cols);
    ae_int_t rows() const;
    ae_int_t cols() const;
    bool isempty() const;
	ae_int_t getstride() const;

    const alglib_impl::ae_matrix* c_ptr() const;
    alglib_impl::ae_matrix* c_ptr();
private:
    ae_matrix_wrapper();
    ae_matrix_wrapper(const ae_matrix_wrapper &rhs);
    const ae_matrix_wrapper& operator=(const ae_matrix_wrapper &rhs);
protected:
#if !defined(AE_NO_EXCEPTIONS)
    //
    // Copies array given by string into current object. Additional
    // parameter DATATYPE contains information about type of the data
    // in S and type of the array to create.
    //
    // Current object is considered empty (this function should be
    // called from copy constructor).
    //
    ae_matrix_wrapper(const char *s, alglib_impl::ae_datatype datatype);
#endif
    
    //
    // This function attaches wrapper object to external x_vector structure;
    // "frozen proxy" mode is activated (you can read/write, but can not reallocate
    // and do not own memory of the vector).
    //
    // NOTE: initial state of wrapper object is assumed to be initialized;
    //       all previously allocated memory is properly deallocated.
    //
    // NOTE: x_vector structure pointed by new_ptr is used only once; after
    //       we fetch pointer to memory and its size, this structure is ignored
    //       and not referenced anymore. So, you can pass pointers to temporary
    //       x-structures which are deallocated immediately after you call attach_to()
    //
    // NOTE: state structure is used for error-handling (a longjmp is performed
    //       on allocation error). All previously allocated memory is correctly
    //       freed on error.
    //
    void attach_to(alglib_impl::x_matrix *new_ptr, alglib_impl::ae_state *_state);

    //
    // This function initializes matrix and allocates own memory storage.
    //
    // NOTE: initial state of wrapper object is assumed to be uninitialized;
    //       if ptr!=NULL on entry, it is considered critical error (abort is called).
    //
    void init(ae_int_t rows, ae_int_t cols, alglib_impl::ae_datatype datatype, alglib_impl::ae_state *_state);
    
    //
    // Assigns RHS to current object.
    //
    // It has several branches depending on target object status:
    // * in case it is proxy object, data are copied into memory pointed by
    //   proxy. Function checks that source has exactly same size as target
    //   (exception is thrown on failure).
    // * in case it is non-proxy object, data allocated by object are cleared
    //   and a copy of RHS is created in target.
    //
    // NOTE: this function correctly handles assignments of the object to itself.
    //
    const ae_matrix_wrapper & assign(const ae_matrix_wrapper &rhs);
    
    
    //
    // Pointer to ae_matrix structure:
    // * ptr==&inner_mat means that wrapper object owns ae_matrix structure and
    //   is responsible for proper deallocation of its memory
    // * ptr!=&inner_mat means that wrapper object works with someone's other
    //   ae_matrix record and is not responsible for its memory; in this case
    //   inner_mat is assumed to be uninitialized.
    //
    alglib_impl::ae_matrix *ptr;
    
    //
    // Inner ae_matrix record.
    // Ignored for ptr!=&inner_mat.
    //
    alglib_impl::ae_matrix inner_mat;
    
    //
    // Whether this wrapper object is frozen proxy (you may read array, may
    // modify its value, but can not deallocate its memory or resize it) or not.
    //
    // If is_frozen_proxy==true and if:
    // * ptr==&inner_vec, it means that wrapper works with its own ae_vector
    //   structure, but this structure points to externally allocated memory.
    //   This memory is NOT owned by ae_vector object.
    // * ptr!=&inner_vec, it means that wrapper works with externally allocated
    //   and managed ae_vector structure. Both memory pointed by ae_vector and
    //   ae_vector structure itself are not owned by wrapper object.
    //
    bool                   is_frozen_proxy;
};

class boolean_2d_array : public ae_matrix_wrapper
{
public:
    boolean_2d_array();
    boolean_2d_array(const boolean_2d_array &rhs);
    boolean_2d_array(alglib_impl::ae_matrix *p);
    virtual ~boolean_2d_array();
    
    const boolean_2d_array& operator=(const boolean_2d_array &rhs);

    const ae_bool& operator()(ae_int_t i, ae_int_t j) const;
    ae_bool& operator()(ae_int_t i, ae_int_t j);

    const ae_bool* operator[](ae_int_t i) const;
    ae_bool* operator[](ae_int_t i);

    //
    // This function allocates array[irows,icols] and copies data
    // pointed by pContent to its memory. Completely independent
    // copy of data is created.
    //
    void setcontent(ae_int_t irows, ae_int_t icols, const bool *pContent );
    
#if !defined(AE_NO_EXCEPTIONS)
    boolean_2d_array(const char *s);
    std::string tostring() const ;
#endif
};

class integer_2d_array : public ae_matrix_wrapper
{
public:
    integer_2d_array();
    integer_2d_array(const integer_2d_array &rhs);
    integer_2d_array(alglib_impl::ae_matrix *p);
    virtual ~integer_2d_array();
    
    const integer_2d_array& operator=(const integer_2d_array &rhs);

    const ae_int_t& operator()(ae_int_t i, ae_int_t j) const;
    ae_int_t& operator()(ae_int_t i, ae_int_t j);

    const ae_int_t* operator[](ae_int_t i) const;
    ae_int_t* operator[](ae_int_t i);

    //
    // This function allocates array[irows,icols] and copies data
    // pointed by pContent to its memory. Completely independent
    // copy of data is created.
    //
    void setcontent(ae_int_t irows, ae_int_t icols, const ae_int_t *pContent );
    
    
#if !defined(AE_NO_EXCEPTIONS)
    integer_2d_array(const char *s);
    std::string tostring() const;
#endif
};

class real_2d_array : public ae_matrix_wrapper
{
public:
    real_2d_array();
    real_2d_array(const real_2d_array &rhs);
    real_2d_array(alglib_impl::ae_matrix *p);
    virtual ~real_2d_array();
    
    const real_2d_array& operator=(const real_2d_array &rhs);

    const double& operator()(ae_int_t i, ae_int_t j) const;
    double& operator()(ae_int_t i, ae_int_t j);

    const double* operator[](ae_int_t i) const;
    double* operator[](ae_int_t i);

    //
    // This function allocates array[irows,icols] and copies data
    // pointed by pContent to its memory. Completely independent
    // copy of data is created.
    //
    void setcontent(ae_int_t irows, ae_int_t icols, const double *pContent);
    
    //
    // This function attaches array to memory pointed by pContent:
    // * only minor amount of own memory is allocated - O(irows) bytes to
    //   store precomputed pointers; but no costly copying of O(rows*cols)
    //   data is performed.
    // * pContent pointer should be valid as long as we work with array
    //
    // After you attach array object to external memory, it becomes
    // "frozen": it is possible to read/write array elements, but
    // it is not allowed to resize it (no setlength() calls).
    //
    void attach_to_ptr(ae_int_t irows, ae_int_t icols, double *pContent);

#if !defined(AE_NO_EXCEPTIONS)
    real_2d_array(const char *s);
    std::string tostring(int dps) const;
#endif
};

class complex_2d_array : public ae_matrix_wrapper
{
public:
    complex_2d_array();
    complex_2d_array(const complex_2d_array &rhs);
    complex_2d_array(alglib_impl::ae_matrix *p);
    virtual ~complex_2d_array();
    
    const complex_2d_array& operator=(const complex_2d_array &rhs);

    const alglib::complex& operator()(ae_int_t i, ae_int_t j) const;
    alglib::complex& operator()(ae_int_t i, ae_int_t j);

    const alglib::complex* operator[](ae_int_t i) const;
    alglib::complex* operator[](ae_int_t i);

    //
    // This function allocates array[irows,icols] and copies data
    // pointed by pContent to its memory. Completely independent
    // copy of data is created.
    //
    void setcontent(ae_int_t irows, ae_int_t icols, const alglib::complex *pContent );

#if !defined(AE_NO_EXCEPTIONS)
    complex_2d_array(const char *s);
    std::string tostring(int dps) const;
#endif
};

/********************************************************************
CSV operations: reading CSV file to real matrix.

This function reads CSV  file  and  stores  its  contents  to  double
precision 2D array. Format of the data file must conform to RFC  4180
specification, with additional notes:
* file size should be less than 2GB
* ASCI encoding, UTF-8 without BOM (in header names) are supported
* any character (comma/tab/space) may be used as field separator,  as
  long as it is distinct from one used for decimal point
* multiple subsequent field separators (say, two  spaces) are treated
  as MULTIPLE separators, not one big separator
* both comma and full stop may be used as decimal point. Parser  will
  automatically determine specific character being used.  Both  fixed
  and exponential number formats are  allowed.   Thousand  separators
  are NOT allowed.
* line may end with \n (Unix style) or \r\n (Windows  style),  parser
  will automatically adapt to chosen convention
* escaped fields (ones in double quotes) are not supported

INPUT PARAMETERS:
    filename        relative/absolute path
    separator       character used to separate fields.  May  be  ' ',
                    ',', '\t'. Other separators are possible too.
    flags           several values combined with bitwise OR:
                    * alglib::CSV_SKIP_HEADERS -  if present, first row
                      contains headers  and  will  be  skipped.   Its
                      contents is used to determine fields count, and
                      that's all.
                    If no flags are specified, default value 0x0  (or
                    alglib::CSV_DEFAULT, which is same) should be used.
                    
OUTPUT PARAMETERS:
    out             2D matrix, CSV file parsed with atof()
    
HANDLING OF SPECIAL CASES:
* file does not exist - alglib::ap_error exception is thrown
* empty file - empty array is returned (no exception)
* skip_first_row=true, only one row in file - empty array is returned
* field contents is not recognized by atof() - field value is replaced
  by 0.0
********************************************************************/
#if !defined(AE_NO_EXCEPTIONS)
void read_csv(const char *filename, char separator, int flags, alglib::real_2d_array &out);
#endif


/********************************************************************
Constants and functions introduced for compatibility with AlgoPascal
********************************************************************/
extern const double machineepsilon;
extern const double maxrealnumber;
extern const double minrealnumber;
extern const double fp_nan;
extern const double fp_posinf;
extern const double fp_neginf;
extern const ae_int_t endianness;
static const int CSV_DEFAULT = 0x0;
static const int CSV_SKIP_HEADERS = 0x1;

int sign(double x);
double randomreal();
ae_int_t randominteger(ae_int_t maxv);
int round(double x);
int trunc(double x);
int ifloor(double x);
int iceil(double x);
double pi();
double sqr(double x);
int maxint(int m1, int m2);
int minint(int m1, int m2);
double maxreal(double m1, double m2);
double minreal(double m1, double m2);

bool fp_eq(double v1, double v2);
bool fp_neq(double v1, double v2);
bool fp_less(double v1, double v2);
bool fp_less_eq(double v1, double v2);
bool fp_greater(double v1, double v2);
bool fp_greater_eq(double v1, double v2);

bool fp_isnan(double x);
bool fp_isposinf(double x);
bool fp_isneginf(double x);
bool fp_isinf(double x);
bool fp_isfinite(double x);

/********************************************************************
Exception handling macros
********************************************************************/
#if !defined(AE_NO_EXCEPTIONS)
///////////////////////////////////////
// exception-based code
//////////////////////////////
#define _ALGLIB_CPP_EXCEPTION(msg) throw alglib::ap_error(msg)
#define _ALGLIB_CALLBACK_EXCEPTION_GUARD_BEGIN          try{
#define _ALGLIB_CALLBACK_EXCEPTION_GUARD_END            }catch(...){ goto lbl_user_exception; }

#else
    
///////////////////////////////////////
// Exception-free version
//////////////////////////////
#if AE_OS!=AE_UNKNOWN
#error Exception-free mode can not be combined with AE_OS definition
#endif
#if AE_THREADING!=AE_SERIAL_UNSAFE
#error Exception-free mode is thread-unsafe; define AE_THREADING=AE_SERIAL_UNSAFE to prove that you know it
#endif
#define _ALGLIB_CALLBACK_EXCEPTION_GUARD_BEGIN
#define _ALGLIB_CALLBACK_EXCEPTION_GUARD_END
#define _ALGLIB_SET_ERROR_FLAG(s) set_error_flag(s)

// sets eror flag and (optionally) sets error message
void set_error_flag(const char *s = NULL);

// returns error flag and optionally returns error message (loaded to *p_msg);
// if error flag is not set (or p_msg is NULL) *p_msg is not changed.
bool get_error_flag(const char **p_msg = NULL);

// clears error flag (it is not cleared until explicit call to this function)
void clear_error_flag();
#endif

}//namespace alglib



/////////////////////////////////////////////////////////////////////////
//
// THIS SECTIONS CONTAINS DECLARATIONS FOR OPTIMIZED LINEAR ALGEBRA CODES
// IT IS SHARED BETWEEN C++ AND PURE C LIBRARIES
//
/////////////////////////////////////////////////////////////////////////

namespace alglib_impl
{
#define ALGLIB_INTERCEPTS_ABLAS
void _ialglib_vzero(ae_int_t n, double *p, ae_int_t stride);
void _ialglib_vzero_complex(ae_int_t n, ae_complex *p, ae_int_t stride);
void _ialglib_vcopy(ae_int_t n, const double *a, ae_int_t stridea, double *b, ae_int_t strideb);
void _ialglib_vcopy_complex(ae_int_t n, const ae_complex *a, ae_int_t stridea, double *b, ae_int_t strideb, const char *conj);
void _ialglib_vcopy_dcomplex(ae_int_t n, const double *a, ae_int_t stridea, double *b, ae_int_t strideb, const char *conj);
void _ialglib_mcopyblock(ae_int_t m, ae_int_t n, const double *a, ae_int_t op, ae_int_t stride, double *b);
void _ialglib_mcopyunblock(ae_int_t m, ae_int_t n, const double *a, ae_int_t op, double *b, ae_int_t stride);
void _ialglib_mcopyblock_complex(ae_int_t m, ae_int_t n, const ae_complex *a, ae_int_t op, ae_int_t stride, double *b);
void _ialglib_mcopyunblock_complex(ae_int_t m, ae_int_t n, const double *a, ae_int_t op, ae_complex* b, ae_int_t stride);

ae_bool _ialglib_i_rmatrixgemmf(ae_int_t m,
     ae_int_t n,
     ae_int_t k,
     double alpha,
     ae_matrix *a,
     ae_int_t ia,
     ae_int_t ja,
     ae_int_t optypea,
     ae_matrix *b,
     ae_int_t ib,
     ae_int_t jb,
     ae_int_t optypeb,
     double beta,
     ae_matrix *c,
     ae_int_t ic,
     ae_int_t jc);
ae_bool _ialglib_i_cmatrixgemmf(ae_int_t m,
     ae_int_t n,
     ae_int_t k,
     ae_complex alpha,
     ae_matrix *a,
     ae_int_t ia,
     ae_int_t ja,
     ae_int_t optypea,
     ae_matrix *b,
     ae_int_t ib,
     ae_int_t jb,
     ae_int_t optypeb,
     ae_complex beta,
     ae_matrix *c,
     ae_int_t ic,
     ae_int_t jc);
ae_bool _ialglib_i_cmatrixrighttrsmf(ae_int_t m,
     ae_int_t n,
     ae_matrix *a,
     ae_int_t i1,
     ae_int_t j1,
     ae_bool isupper,
     ae_bool isunit,
     ae_int_t optype,
     ae_matrix *x,
     ae_int_t i2,
     ae_int_t j2);
ae_bool _ialglib_i_rmatrixrighttrsmf(ae_int_t m,
     ae_int_t n,
     ae_matrix *a,
     ae_int_t i1,
     ae_int_t j1,
     ae_bool isupper,
     ae_bool isunit,
     ae_int_t optype,
     ae_matrix *x,
     ae_int_t i2,
     ae_int_t j2);
ae_bool _ialglib_i_cmatrixlefttrsmf(ae_int_t m,
     ae_int_t n,
     ae_matrix *a,
     ae_int_t i1,
     ae_int_t j1,
     ae_bool isupper,
     ae_bool isunit,
     ae_int_t optype,
     ae_matrix *x,
     ae_int_t i2,
     ae_int_t j2);
ae_bool _ialglib_i_rmatrixlefttrsmf(ae_int_t m,
     ae_int_t n,
     ae_matrix *a,
     ae_int_t i1,
     ae_int_t j1,
     ae_bool isupper,
     ae_bool isunit,
     ae_int_t optype,
     ae_matrix *x,
     ae_int_t i2,
     ae_int_t j2);
ae_bool _ialglib_i_cmatrixherkf(ae_int_t n,
     ae_int_t k,
     double alpha,
     ae_matrix *a,
     ae_int_t ia,
     ae_int_t ja,
     ae_int_t optypea,
     double beta,
     ae_matrix *c,
     ae_int_t ic,
     ae_int_t jc,
     ae_bool isupper);
ae_bool _ialglib_i_rmatrixsyrkf(ae_int_t n,
     ae_int_t k,
     double alpha,
     ae_matrix *a,
     ae_int_t ia,
     ae_int_t ja,
     ae_int_t optypea,
     double beta,
     ae_matrix *c,
     ae_int_t ic,
     ae_int_t jc,
     ae_bool isupper);
ae_bool _ialglib_i_cmatrixrank1f(ae_int_t m,
     ae_int_t n,
     ae_matrix *a,
     ae_int_t ia,
     ae_int_t ja,
     ae_vector *u,
     ae_int_t uoffs,
     ae_vector *v,
     ae_int_t voffs);
ae_bool _ialglib_i_rmatrixrank1f(ae_int_t m,
     ae_int_t n,
     ae_matrix *a,
     ae_int_t ia,
     ae_int_t ja,
     ae_vector *u,
     ae_int_t uoffs,
     ae_vector *v,
     ae_int_t voffs);
ae_bool _ialglib_i_rmatrixgerf(ae_int_t m,
     ae_int_t n,
     ae_matrix *a,
     ae_int_t ia,
     ae_int_t ja,
     double alpha,
     ae_vector *u,
     ae_int_t uoffs,
     ae_vector *v,
     ae_int_t voffs);



}


/////////////////////////////////////////////////////////////////////////
//
// THIS SECTION CONTAINS PARALLEL SUBROUTINES
//
/////////////////////////////////////////////////////////////////////////

namespace alglib_impl
{

}


/////////////////////////////////////////////////////////////////////////
//
// THIS SECTION CONTAINS DEFINITIONS FOR PARTIAL COMPILATION
//
/////////////////////////////////////////////////////////////////////////
#ifdef AE_COMPILE_SCODES
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_APSERV
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_TSORT
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#endif

#ifdef AE_COMPILE_NEARESTNEIGHBOR
#define AE_PARTIAL_BUILD
#define AE_COMPILE_SCODES
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#endif

#ifdef AE_COMPILE_HQRND
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#endif

#ifdef AE_COMPILE_XDEBUG
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_ODESOLVER
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#endif

#ifdef AE_COMPILE_ABLASMKL
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_SPARSE
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#endif

#ifdef AE_COMPILE_ABLASF
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_ABLAS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#endif

#ifdef AE_COMPILE_DLU
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#endif

#ifdef AE_COMPILE_SPTRF
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#endif

#ifdef AE_COMPILE_CREFLECTIONS
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_MATGEN
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#endif

#ifdef AE_COMPILE_ROTATIONS
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_TRFAC
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#endif

#ifdef AE_COMPILE_TRLINSOLVE
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_SAFESOLVE
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_RCOND
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#endif

#ifdef AE_COMPILE_MATINV
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#endif

#ifdef AE_COMPILE_HBLAS
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_SBLAS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#endif

#ifdef AE_COMPILE_ORTFAC
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#endif

#ifdef AE_COMPILE_FBLS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#endif

#ifdef AE_COMPILE_CQMODELS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_FBLS
#endif

#ifdef AE_COMPILE_OPTGUARDAPI
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#endif

#ifdef AE_COMPILE_BLAS
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_BDSVD
#define AE_PARTIAL_BUILD
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#endif

#ifdef AE_COMPILE_SVD
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_BDSVD
#endif

#ifdef AE_COMPILE_OPTSERV
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#endif

#ifdef AE_COMPILE_SNNLS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_FBLS
#endif

#ifdef AE_COMPILE_SACTIVESETS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_FBLS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#endif

#ifdef AE_COMPILE_QQPSOLVER
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLAS
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_FBLS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#endif

#ifdef AE_COMPILE_LINMIN
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_MINLBFGS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_LINMIN
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_FBLS
#endif

#ifdef AE_COMPILE_XBLAS
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_DIRECTDENSESOLVERS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_MATGEN
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_XBLAS
#endif

#ifdef AE_COMPILE_NORMESTIMATOR
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#endif

#ifdef AE_COMPILE_LINLSQR
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_NORMESTIMATOR
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#endif

#ifdef AE_COMPILE_QPDENSEAULSOLVER
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLAS
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_XBLAS
#define AE_COMPILE_DIRECTDENSESOLVERS
#define AE_COMPILE_NORMESTIMATOR
#define AE_COMPILE_LINLSQR
#define AE_COMPILE_LINMIN
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_FBLS
#define AE_COMPILE_MINLBFGS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_QQPSOLVER
#endif

#ifdef AE_COMPILE_MINBLEIC
#define AE_PARTIAL_BUILD
#define AE_COMPILE_LINMIN
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_FBLS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#endif

#ifdef AE_COMPILE_QPBLEICSOLVER
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLAS
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_LINMIN
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_FBLS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_MINBLEIC
#endif

#ifdef AE_COMPILE_MINQP
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLAS
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_XBLAS
#define AE_COMPILE_DIRECTDENSESOLVERS
#define AE_COMPILE_NORMESTIMATOR
#define AE_COMPILE_LINLSQR
#define AE_COMPILE_LINMIN
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_FBLS
#define AE_COMPILE_MINLBFGS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_QQPSOLVER
#define AE_COMPILE_QPDENSEAULSOLVER
#define AE_COMPILE_MINBLEIC
#define AE_COMPILE_QPBLEICSOLVER
#endif

#ifdef AE_COMPILE_REVISEDDUALSIMPLEX
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#endif

#ifdef AE_COMPILE_MINLP
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLAS
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_REVISEDDUALSIMPLEX
#endif

#ifdef AE_COMPILE_NLCSLP
#define AE_PARTIAL_BUILD
#define AE_COMPILE_LINMIN
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_REVISEDDUALSIMPLEX
#endif

#ifdef AE_COMPILE_MINNLC
#define AE_PARTIAL_BUILD
#define AE_COMPILE_LINMIN
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_FBLS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_MINLBFGS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_MINBLEIC
#define AE_COMPILE_REVISEDDUALSIMPLEX
#define AE_COMPILE_NLCSLP
#endif

#ifdef AE_COMPILE_MINBC
#define AE_PARTIAL_BUILD
#define AE_COMPILE_LINMIN
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#endif

#ifdef AE_COMPILE_MINNS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_LINMIN
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_FBLS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_MINBLEIC
#endif

#ifdef AE_COMPILE_MINCOMP
#define AE_PARTIAL_BUILD
#define AE_COMPILE_LINMIN
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_FBLS
#define AE_COMPILE_MINLBFGS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_MINBLEIC
#endif

#ifdef AE_COMPILE_MINCG
#define AE_PARTIAL_BUILD
#define AE_COMPILE_LINMIN
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#endif

#ifdef AE_COMPILE_MINLM
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_XBLAS
#define AE_COMPILE_DIRECTDENSESOLVERS
#define AE_COMPILE_NORMESTIMATOR
#define AE_COMPILE_LINLSQR
#define AE_COMPILE_LINMIN
#define AE_COMPILE_FBLS
#define AE_COMPILE_MINLBFGS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_QQPSOLVER
#define AE_COMPILE_QPDENSEAULSOLVER
#define AE_COMPILE_MINBLEIC
#define AE_COMPILE_QPBLEICSOLVER
#define AE_COMPILE_MINQP
#endif

#ifdef AE_COMPILE_HSSCHUR
#define AE_PARTIAL_BUILD
#define AE_COMPILE_BLAS
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLAS
#endif

#ifdef AE_COMPILE_BASICSTATOPS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#endif

#ifdef AE_COMPILE_EVD
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_MATGEN
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_HSSCHUR
#define AE_COMPILE_BASICSTATOPS
#endif

#ifdef AE_COMPILE_BASESTAT
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#endif

#ifdef AE_COMPILE_PCA
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_MATGEN
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_HSSCHUR
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_EVD
#define AE_COMPILE_BASESTAT
#endif

#ifdef AE_COMPILE_BDSS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_BASESTAT
#endif

#ifdef AE_COMPILE_HPCCORES
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_MLPBASE
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_BASESTAT
#define AE_COMPILE_BDSS
#define AE_COMPILE_HPCCORES
#define AE_COMPILE_SCODES
#define AE_COMPILE_HQRND
#define AE_COMPILE_SPARSE
#endif

#ifdef AE_COMPILE_LDA
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_MATGEN
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_HSSCHUR
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_EVD
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#endif

#ifdef AE_COMPILE_SSA
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_MATGEN
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_HSSCHUR
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_EVD
#endif

#ifdef AE_COMPILE_GAMMAFUNC
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_NORMALDISTR
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_IGAMMAF
#define AE_PARTIAL_BUILD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#endif

#ifdef AE_COMPILE_LINREG
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_BASESTAT
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#define AE_COMPILE_IGAMMAF
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#endif

#ifdef AE_COMPILE_FILTERS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_BASESTAT
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#define AE_COMPILE_IGAMMAF
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_LINREG
#endif

#ifdef AE_COMPILE_LOGIT
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_BASESTAT
#define AE_COMPILE_BDSS
#define AE_COMPILE_HPCCORES
#define AE_COMPILE_SCODES
#define AE_COMPILE_HQRND
#define AE_COMPILE_SPARSE
#define AE_COMPILE_MLPBASE
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_MATGEN
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_XBLAS
#define AE_COMPILE_DIRECTDENSESOLVERS
#endif

#ifdef AE_COMPILE_MCPD
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_LINMIN
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_FBLS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_MINBLEIC
#endif

#ifdef AE_COMPILE_MLPE
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_BASESTAT
#define AE_COMPILE_BDSS
#define AE_COMPILE_HPCCORES
#define AE_COMPILE_SCODES
#define AE_COMPILE_HQRND
#define AE_COMPILE_SPARSE
#define AE_COMPILE_MLPBASE
#endif

#ifdef AE_COMPILE_MLPTRAIN
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_BASESTAT
#define AE_COMPILE_BDSS
#define AE_COMPILE_HPCCORES
#define AE_COMPILE_SCODES
#define AE_COMPILE_HQRND
#define AE_COMPILE_SPARSE
#define AE_COMPILE_MLPBASE
#define AE_COMPILE_MLPE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_LINMIN
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_FBLS
#define AE_COMPILE_MINLBFGS
#define AE_COMPILE_XBLAS
#define AE_COMPILE_DIRECTDENSESOLVERS
#endif

#ifdef AE_COMPILE_CLUSTERING
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_BLAS
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_BASESTAT
#endif

#ifdef AE_COMPILE_DFOREST
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_SCODES
#define AE_COMPILE_TSORT
#define AE_COMPILE_HQRND
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_BASESTAT
#define AE_COMPILE_BDSS
#endif

#ifdef AE_COMPILE_KNN
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_SCODES
#define AE_COMPILE_TSORT
#define AE_COMPILE_HQRND
#define AE_COMPILE_NEARESTNEIGHBOR
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_BASESTAT
#define AE_COMPILE_BDSS
#endif

#ifdef AE_COMPILE_DATACOMP
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_BLAS
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_BASESTAT
#define AE_COMPILE_CLUSTERING
#endif

#ifdef AE_COMPILE_GQ
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_MATGEN
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_HSSCHUR
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_EVD
#define AE_COMPILE_GAMMAFUNC
#endif

#ifdef AE_COMPILE_GKQ
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_HSSCHUR
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_EVD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_GQ
#endif

#ifdef AE_COMPILE_AUTOGK
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_HSSCHUR
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_EVD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_GQ
#define AE_COMPILE_GKQ
#endif

#ifdef AE_COMPILE_NTHEORY
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_FTBASE
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_NTHEORY
#endif

#ifdef AE_COMPILE_FFT
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_NTHEORY
#define AE_COMPILE_FTBASE
#endif

#ifdef AE_COMPILE_FHT
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_NTHEORY
#define AE_COMPILE_FTBASE
#define AE_COMPILE_FFT
#endif

#ifdef AE_COMPILE_CONV
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_NTHEORY
#define AE_COMPILE_FTBASE
#define AE_COMPILE_FFT
#endif

#ifdef AE_COMPILE_CORR
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_NTHEORY
#define AE_COMPILE_FTBASE
#define AE_COMPILE_FFT
#define AE_COMPILE_CONV
#endif

#ifdef AE_COMPILE_IDW
#define AE_PARTIAL_BUILD
#define AE_COMPILE_SCODES
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_NEARESTNEIGHBOR
#define AE_COMPILE_HQRND
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#endif

#ifdef AE_COMPILE_RATINT
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#endif

#ifdef AE_COMPILE_FITSPHERE
#define AE_PARTIAL_BUILD
#define AE_COMPILE_LINMIN
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_FBLS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_MINBLEIC
#define AE_COMPILE_XBLAS
#define AE_COMPILE_DIRECTDENSESOLVERS
#define AE_COMPILE_NORMESTIMATOR
#define AE_COMPILE_LINLSQR
#define AE_COMPILE_MINLBFGS
#define AE_COMPILE_QQPSOLVER
#define AE_COMPILE_QPDENSEAULSOLVER
#define AE_COMPILE_QPBLEICSOLVER
#define AE_COMPILE_MINQP
#define AE_COMPILE_MINLM
#define AE_COMPILE_REVISEDDUALSIMPLEX
#define AE_COMPILE_NLCSLP
#define AE_COMPILE_MINNLC
#endif

#ifdef AE_COMPILE_INTFITSERV
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#endif

#ifdef AE_COMPILE_SPLINE1D
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_INTFITSERV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_FBLS
#endif

#ifdef AE_COMPILE_PARAMETRIC
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_INTFITSERV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_FBLS
#define AE_COMPILE_SPLINE1D
#define AE_COMPILE_BLAS
#define AE_COMPILE_HSSCHUR
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_EVD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_GQ
#define AE_COMPILE_GKQ
#define AE_COMPILE_AUTOGK
#endif

#ifdef AE_COMPILE_SPLINE3D
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_INTFITSERV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_FBLS
#define AE_COMPILE_SPLINE1D
#endif

#ifdef AE_COMPILE_POLINT
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_RATINT
#endif

#ifdef AE_COMPILE_LSFIT
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_INTFITSERV
#define AE_COMPILE_RATINT
#define AE_COMPILE_POLINT
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_FBLS
#define AE_COMPILE_SPLINE1D
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_XBLAS
#define AE_COMPILE_DIRECTDENSESOLVERS
#define AE_COMPILE_NORMESTIMATOR
#define AE_COMPILE_LINLSQR
#define AE_COMPILE_LINMIN
#define AE_COMPILE_MINLBFGS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_QQPSOLVER
#define AE_COMPILE_QPDENSEAULSOLVER
#define AE_COMPILE_MINBLEIC
#define AE_COMPILE_QPBLEICSOLVER
#define AE_COMPILE_MINQP
#define AE_COMPILE_MINLM
#endif

#ifdef AE_COMPILE_RBFV2
#define AE_PARTIAL_BUILD
#define AE_COMPILE_SCODES
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_NEARESTNEIGHBOR
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_INTFITSERV
#define AE_COMPILE_RATINT
#define AE_COMPILE_POLINT
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_FBLS
#define AE_COMPILE_SPLINE1D
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_XBLAS
#define AE_COMPILE_DIRECTDENSESOLVERS
#define AE_COMPILE_NORMESTIMATOR
#define AE_COMPILE_LINLSQR
#define AE_COMPILE_LINMIN
#define AE_COMPILE_MINLBFGS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_QQPSOLVER
#define AE_COMPILE_QPDENSEAULSOLVER
#define AE_COMPILE_MINBLEIC
#define AE_COMPILE_QPBLEICSOLVER
#define AE_COMPILE_MINQP
#define AE_COMPILE_MINLM
#define AE_COMPILE_LSFIT
#endif

#ifdef AE_COMPILE_SPLINE2D
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_SCODES
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_INTFITSERV
#define AE_COMPILE_NORMESTIMATOR
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_LINLSQR
#define AE_COMPILE_FBLS
#define AE_COMPILE_SPLINE1D
#endif

#ifdef AE_COMPILE_RBFV1
#define AE_PARTIAL_BUILD
#define AE_COMPILE_SCODES
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_NEARESTNEIGHBOR
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_INTFITSERV
#define AE_COMPILE_RATINT
#define AE_COMPILE_POLINT
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_FBLS
#define AE_COMPILE_SPLINE1D
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_XBLAS
#define AE_COMPILE_DIRECTDENSESOLVERS
#define AE_COMPILE_NORMESTIMATOR
#define AE_COMPILE_LINLSQR
#define AE_COMPILE_LINMIN
#define AE_COMPILE_MINLBFGS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_QQPSOLVER
#define AE_COMPILE_QPDENSEAULSOLVER
#define AE_COMPILE_MINBLEIC
#define AE_COMPILE_QPBLEICSOLVER
#define AE_COMPILE_MINQP
#define AE_COMPILE_MINLM
#define AE_COMPILE_LSFIT
#endif

#ifdef AE_COMPILE_RBF
#define AE_PARTIAL_BUILD
#define AE_COMPILE_SCODES
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_NEARESTNEIGHBOR
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_INTFITSERV
#define AE_COMPILE_RATINT
#define AE_COMPILE_POLINT
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_FBLS
#define AE_COMPILE_SPLINE1D
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_XBLAS
#define AE_COMPILE_DIRECTDENSESOLVERS
#define AE_COMPILE_NORMESTIMATOR
#define AE_COMPILE_LINLSQR
#define AE_COMPILE_LINMIN
#define AE_COMPILE_MINLBFGS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_QQPSOLVER
#define AE_COMPILE_QPDENSEAULSOLVER
#define AE_COMPILE_MINBLEIC
#define AE_COMPILE_QPBLEICSOLVER
#define AE_COMPILE_MINQP
#define AE_COMPILE_MINLM
#define AE_COMPILE_LSFIT
#define AE_COMPILE_RBFV1
#define AE_COMPILE_RBFV2
#endif

#ifdef AE_COMPILE_INTCOMP
#define AE_PARTIAL_BUILD
#define AE_COMPILE_LINMIN
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_OPTGUARDAPI
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_HQRND
#define AE_COMPILE_MATGEN
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_BDSVD
#define AE_COMPILE_SVD
#define AE_COMPILE_OPTSERV
#define AE_COMPILE_FBLS
#define AE_COMPILE_CQMODELS
#define AE_COMPILE_SNNLS
#define AE_COMPILE_SACTIVESETS
#define AE_COMPILE_MINBLEIC
#define AE_COMPILE_XBLAS
#define AE_COMPILE_DIRECTDENSESOLVERS
#define AE_COMPILE_NORMESTIMATOR
#define AE_COMPILE_LINLSQR
#define AE_COMPILE_MINLBFGS
#define AE_COMPILE_QQPSOLVER
#define AE_COMPILE_QPDENSEAULSOLVER
#define AE_COMPILE_QPBLEICSOLVER
#define AE_COMPILE_MINQP
#define AE_COMPILE_MINLM
#define AE_COMPILE_REVISEDDUALSIMPLEX
#define AE_COMPILE_NLCSLP
#define AE_COMPILE_MINNLC
#define AE_COMPILE_FITSPHERE
#endif

#ifdef AE_COMPILE_ELLIPTIC
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_HERMITE
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_DAWSON
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_TRIGINTEGRALS
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_POISSONDISTR
#define AE_PARTIAL_BUILD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#define AE_COMPILE_IGAMMAF
#endif

#ifdef AE_COMPILE_BESSEL
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_IBETAF
#define AE_PARTIAL_BUILD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#endif

#ifdef AE_COMPILE_FDISTR
#define AE_PARTIAL_BUILD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#define AE_COMPILE_IBETAF
#endif

#ifdef AE_COMPILE_FRESNEL
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_JACOBIANELLIPTIC
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_PSIF
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_EXPINTEGRALS
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_LAGUERRE
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_CHISQUAREDISTR
#define AE_PARTIAL_BUILD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#define AE_COMPILE_IGAMMAF
#endif

#ifdef AE_COMPILE_LEGENDRE
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_BETAF
#define AE_PARTIAL_BUILD
#define AE_COMPILE_GAMMAFUNC
#endif

#ifdef AE_COMPILE_CHEBYSHEV
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_STUDENTTDISTR
#define AE_PARTIAL_BUILD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#define AE_COMPILE_IBETAF
#endif

#ifdef AE_COMPILE_NEARUNITYUNIT
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_BINOMIALDISTR
#define AE_PARTIAL_BUILD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#define AE_COMPILE_IBETAF
#define AE_COMPILE_NEARUNITYUNIT
#endif

#ifdef AE_COMPILE_AIRYF
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_WSR
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#endif

#ifdef AE_COMPILE_STEST
#define AE_PARTIAL_BUILD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#define AE_COMPILE_IBETAF
#define AE_COMPILE_NEARUNITYUNIT
#define AE_COMPILE_BINOMIALDISTR
#endif

#ifdef AE_COMPILE_CORRELATIONTESTS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#define AE_COMPILE_IBETAF
#define AE_COMPILE_STUDENTTDISTR
#define AE_COMPILE_APSERV
#define AE_COMPILE_TSORT
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_BASESTAT
#endif

#ifdef AE_COMPILE_STUDENTTTESTS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#define AE_COMPILE_IBETAF
#define AE_COMPILE_STUDENTTDISTR
#endif

#ifdef AE_COMPILE_MANNWHITNEYU
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#endif

#ifdef AE_COMPILE_JARQUEBERA
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_VARIANCETESTS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_GAMMAFUNC
#define AE_COMPILE_NORMALDISTR
#define AE_COMPILE_IBETAF
#define AE_COMPILE_FDISTR
#define AE_COMPILE_IGAMMAF
#define AE_COMPILE_CHISQUAREDISTR
#endif

#ifdef AE_COMPILE_SCHUR
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_HSSCHUR
#endif

#ifdef AE_COMPILE_SPDGEVD
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#define AE_COMPILE_SBLAS
#define AE_COMPILE_BLAS
#define AE_COMPILE_TRLINSOLVE
#define AE_COMPILE_SAFESOLVE
#define AE_COMPILE_RCOND
#define AE_COMPILE_MATINV
#define AE_COMPILE_HBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_HSSCHUR
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_EVD
#endif

#ifdef AE_COMPILE_INVERSEUPDATE
#define AE_PARTIAL_BUILD
#endif

#ifdef AE_COMPILE_MATDET
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#endif

#ifdef AE_COMPILE_POLYNOMIALSOLVER
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_MATGEN
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_BLAS
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_HSSCHUR
#define AE_COMPILE_BASICSTATOPS
#define AE_COMPILE_EVD
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_TRFAC
#endif

#ifdef AE_COMPILE_NLEQ
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_LINMIN
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_ABLAS
#define AE_COMPILE_HQRND
#define AE_COMPILE_HBLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_SBLAS
#define AE_COMPILE_ORTFAC
#define AE_COMPILE_FBLS
#endif

#ifdef AE_COMPILE_DIRECTSPARSESOLVERS
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLAS
#define AE_COMPILE_DLU
#define AE_COMPILE_SPTRF
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#define AE_COMPILE_ROTATIONS
#define AE_COMPILE_TRFAC
#endif

#ifdef AE_COMPILE_LINCG
#define AE_PARTIAL_BUILD
#define AE_COMPILE_APSERV
#define AE_COMPILE_ABLASMKL
#define AE_COMPILE_HQRND
#define AE_COMPILE_TSORT
#define AE_COMPILE_SPARSE
#define AE_COMPILE_ABLASF
#define AE_COMPILE_ABLAS
#define AE_COMPILE_CREFLECTIONS
#define AE_COMPILE_MATGEN
#endif

#ifdef AE_COMPILE_ALGLIBBASICS
#define AE_PARTIAL_BUILD
#endif



#endif

