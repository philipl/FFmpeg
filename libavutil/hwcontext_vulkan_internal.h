/*
 * Copyright (c) Lynne
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef AVUTIL_HWCONTEXT_VULKAN_INTERNAL_H
#define AVUTIL_HWCONTEXT_VULKAN_INTERNAL_H

/**
 * @file
 * FFmpeg internal API for CUDA.
 */

#include "vulkan.h"

typedef struct VulkanFramesPriv {
    /* Image conversions */
    FFVkExecPool compute_exec;

    /* Image transfers */
    FFVkExecPool upload_exec;
    FFVkExecPool download_exec;

    /* Modifier info list to free at uninit */
    VkImageDrmFormatModifierListCreateInfoEXT *modifier_info;

    /* Used by the decoder in case there's no contex */
    void *video_profile_data;
} VulkanFramesPriv;

#endif /* AVUTIL_HWCONTEXT_VULKAN_INTERNAL_H */
