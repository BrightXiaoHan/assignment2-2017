/*!
 *  Copyright (c) 2017 by Contributors
 * \file dlarray.h
 * \brief Header that defines array struct.
 */
#ifndef DLSYS_H_
#define DLSYS_H_

#ifdef __cplusplus
#define DLSYS_EXTERN_C extern "C"
#else
#define DLSYS_EXTERN_C
#endif

#include <stddef.h>
#include <stdint.h>

typedef int64_t array_size_t;

DLSYS_EXTERN_C {
  /*!
   * \brief The device type in DLContext.
   */
  typedef enum {
    kCPU = 1,
    kGPU = 2,
  } DLDeviceType;

  /*!
   * \brief A Device context for array.
   */
  typedef struct {
    /*! \brief The device index */
    int device_id;
    /*! \brief The device type used in the device. */
    DLDeviceType device_type;
  } DLContext;

  /*!
   * \brief Plain C Array object, does not manage memory.
   */
  typedef struct {
    /*!
     * \brief The opaque data pointer points to the allocated data.
     *  This will be CUDA device pointer or cl_mem handle in OpenCL.
     *  This pointer is always aligns to 256 bytes as in CUDA.
     */
    void *data;
    /*! \brief The device context of the tensor */
    DLContext ctx;
    /*! \brief Number of dimensions */
    int ndim;
    /*! \brief The shape of the tensor */
    array_size_t *shape;

    array_size_t size(){
      array_size_t s = 1;
      for (int i = 0; i < ndim; i++){
        s *= shape[i];
      }
      return s;
    }
  } DLArray;

} // DLSYS_EXTERN_C
#endif // DLSYS_H_
