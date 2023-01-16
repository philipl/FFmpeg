/*
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

#ifndef AVCODEC_VULKAN_DECODE_H
#define AVCODEC_VULKAN_DECODE_H

#include "decode.h"
#include "hwconfig.h"
#include "internal.h"

#include "vulkan_video.h"

typedef struct FFVulkanDecodeContext {
    FFVulkanContext s;
    FFVkVideoCommon common;

    int dedicated_dpb; /* Oddity  #1 - separate DPB images */
    int layered_dpb;   /* Madness #1 - layered  DPB images */

    AVBufferRef *dpb_hwfc_ref;  /* Only used for dedicated_dpb */

    AVFrame *layered_frame;     /* Only used for layered_dpb   */
    VkImageView layered_view;
    VkImageAspectFlags layered_aspect;

    VkVideoSessionParametersKHR empty_session_params;

    VkSamplerYcbcrConversion yuv_sampler;
    VkVideoDecodeCapabilitiesKHR dec_caps;
    int init;

    AVBufferRef *session_params;

    FFVkQueueFamilyCtx qf_dec;
    FFVkExecPool exec_pool;

    AVBufferPool *tmp_pool; /* Pool for temporary data, if needed (HEVC) */
    size_t tmp_pool_ele_size;

    uint16_t last_ref_frames_in_use;
} FFVulkanDecodeContext;

typedef struct FFVulkanDecodePicture {
    AVFrame                        *dpb_frame;      /* Only used for out-of-place decoding. */

    VkImageView                     img_view_ref;   /* Image representation view (reference) */
    VkImageView                     img_view_out;   /* Image representation view (output-only) */
    VkImageAspectFlags              img_aspect;     /* Image plane mask bits */
    VkImageAspectFlags              img_aspect_ref; /* Only used for out-of-place decoding */

    VkSemaphore                     sem;
    uint64_t                        sem_value;

    /* State */
    int                             update_params;
    AVBufferRef                    *session_params;

    /* Current picture */
    VkVideoPictureResourceInfoKHR   ref;
    VkVideoReferenceSlotInfoKHR     ref_slot;

    /* Picture refs. H264 has the maximum number of refs (36) of any supported codec. */
    VkVideoPictureResourceInfoKHR   refs     [36];
    VkVideoReferenceSlotInfoKHR     ref_slots[36];

    /* Main decoding struct */
    AVBufferRef                    *params_buf;
    VkVideoDecodeInfoKHR            decode_info;

    /* Slice data */
    AVBufferRef                    *slices_buf;
    size_t                          slices_size;
    uint32_t                       *slice_off;
    unsigned int                    slice_off_max;
    uint32_t                        nb_slices;
} FFVulkanDecodePicture;

/**
 * Initialize decoder.
 */
int ff_vk_decode_init(AVCodecContext *avctx);

/**
 * Initialize hw_frames_ctx with the parameters needed to decode the stream
 * using the parameters from avctx.
 *
 * NOTE: if avctx->internal->hwaccel_priv_data exists, will partially initialize
 * the context.
 */
int ff_vk_frame_params(AVCodecContext *avctx, AVBufferRef *hw_frames_ctx);

/**
 * Prepare a frame, creates the image view, and sets up the dpb fields.
 */
int ff_vk_decode_prepare_frame(FFVulkanDecodeContext *ctx, AVFrame *pic,
                               FFVulkanDecodePicture *vkpic, int is_current,
                               int alloc_dpb);

/**
 * Add slice data to frame.
 */
int ff_vk_decode_add_slice(AVCodecContext *avctx, FFVulkanDecodePicture *vp,
                           const uint8_t *data, size_t size, int add_startcode,
                           uint32_t *nb_slices, const uint32_t **offsets);

/**
 * Decode a frame.
 */
int ff_vk_decode_frame(AVCodecContext *avctx,
                       AVFrame *pic,    FFVulkanDecodePicture *vp,
                       AVFrame *rpic[], FFVulkanDecodePicture *rvkp[]);

/**
 * Free a frame and its state.
 */
void ff_vk_decode_free_frame(FFVulkanDecodeContext *ctx, FFVulkanDecodePicture *vp);

/**
 * Get an FFVkBuffer suitable for decoding from.
 */
int ff_vk_get_decode_buffer(FFVulkanDecodeContext *ctx, AVBufferRef **buf,
                            void *create_pNext, size_t size);

/**
 * Free VkVideoSessionParametersKHR.
 */
void ff_vk_decode_free_params(void *opaque, uint8_t *data);

/**
 * Flush decoder.
 */
void ff_vk_decode_flush(AVCodecContext *avctx);

/**
 * Free decoder.
 */
int ff_vk_decode_uninit(AVCodecContext *avctx);

#endif /* AVCODEC_VULKAN_DECODE_H */
