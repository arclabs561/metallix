/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_METAL_H
#define MLX_METAL_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup metal Metal specific operations
 */
/**@{*/

int mlx_metal_get_metallib_path(mlx_string* res);
int mlx_metal_is_available(bool* res);
/**
 * Whether MLX will use the Neural Accelerator (NAX) kernels: the OS is at
 * least 26.2, the GPU is generation 17 or later (18 for phone GPUs), and
 * the library was built with them (deployment target >= 26.2).
 */
int mlx_metal_is_nax_available(bool* res);
int mlx_metal_set_metallib_path(const char* path);
int mlx_metal_start_capture(const char* path);
int mlx_metal_stop_capture(void);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
