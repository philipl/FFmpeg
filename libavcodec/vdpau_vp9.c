/*
 * VC-1 decode acceleration through VDPAU
 *
 * Copyright (c) 2008 NVIDIA
 * Copyright (c) 2013 Rémi Denis-Courmont
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
 * License along with FFmpeg; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <vdpau/vdpau.h>

#include "libavutil/pixdesc.h"

#include "avcodec.h"
#include "hwaccel.h"
#include "vp9dec.h"
#include "vdpau.h"
#include "vdpau_internal.h"

static int vdpau_vp9_start_frame(AVCodecContext *avctx,
                                 const uint8_t *buffer, uint32_t size)
{
    VP9Context * const v  = avctx->priv_data;
    VP9SharedContext * const h = &v->s;
    VP9Frame *f = &h->frames[CUR_FRAME];
    struct vdpau_picture_context *pic_ctx = f->hwaccel_picture_private;
    VdpPictureInfoVP9 *info = &pic_ctx->info.vp9;
    const AVPixFmtDescriptor *pixdesc = av_pix_fmt_desc_get(avctx->sw_pix_fmt);
    int i = 0;

    *info = (VdpPictureInfoVP9) {
            .width                    = avctx->width,
            .height                   = avctx->height,

            .lastReference            = VDP_INVALID_HANDLE,
            .goldenReference          = VDP_INVALID_HANDLE,
            .altReference             = VDP_INVALID_HANDLE,

            .profile                  = h->h.profile,
            .frameContextIdx          = h->h.framectxid,
            .keyFrame                 = h->h.keyframe,
            .showFrame                = !h->h.invisible,
            .errorResilient           = h->h.errorres,
            .frameParallelDecoding    = h->h.parallelmode,
            .subSamplingX             = pixdesc->log2_chroma_w,
            .subSamplingY             = pixdesc->log2_chroma_h,
            .intraOnly                = h->h.intraonly,
            .allowHighPrecisionMv     = h->h.keyframe ? 0 : h->h.highprecisionmvs,
            .refreshEntropyProbs      = h->h.refreshctx,

            .refFrameSignBias[0]      = 0,

            .bitDepthMinus8Luma       = pixdesc->comp[0].depth - 8,
            .bitDepthMinus8Chroma     = pixdesc->comp[1].depth - 8,
            .loopFilterLevel          = h->h.filter.level,
            .loopFilterSharpness      = h->h.filter.sharpness,

            .modeRefLfEnabled         = h->h.lf_delta.enabled,
            .log2TileColumns          = h->h.tiling.log2_tile_cols,
            .log2TileRows             = h->h.tiling.log2_tile_rows,

            .segmentEnabled           = h->h.segmentation.enabled,
            .segmentMapUpdate         = h->h.segmentation.update_map,
            .segmentMapTemporalUpdate = h->h.segmentation.temporal,
            .segmentFeatureMode       = h->h.segmentation.absolute_vals,

            .qpYAc                    = h->h.yac_qi,
            .qpYDc                    = h->h.ydc_qdelta,
            .qpChDc                   = h->h.uvdc_qdelta,
            .qpChAc                   = h->h.uvac_qdelta,

            .resetFrameContext        = h->h.resetctx,
            .mcompFilterType          = h->h.filtermode ^ (h->h.filtermode <= 1),

            .uncompressedHeaderSize   = h->h.uncompressed_header_size,
            .compressedHeaderSize     = h->h.compressed_header_size,
    };

    if (h->refs[h->h.refidx[0]].f->buf[0])
        info->lastReference = ff_vdpau_get_surface_id(h->refs[h->h.refidx[0]].f);
    if (h->refs[h->h.refidx[1]].f->buf[0])
        info->goldenReference = ff_vdpau_get_surface_id(h->refs[h->h.refidx[1]].f);
    if (h->refs[h->h.refidx[2]].f->buf[0])
        info->altReference = ff_vdpau_get_surface_id(h->refs[h->h.refidx[2]].f);

    switch (avctx->colorspace) {
    default:
    case AVCOL_SPC_UNSPECIFIED:
        info->colorSpace = 0;
        break;
    case AVCOL_SPC_BT470BG:
        info->colorSpace = 1;
        break;
    case AVCOL_SPC_BT709:
        info->colorSpace = 2;
        break;
    case AVCOL_SPC_SMPTE170M:
        info->colorSpace = 3;
        break;
    case AVCOL_SPC_SMPTE240M:
        info->colorSpace = 4;
        break;
    case AVCOL_SPC_BT2020_NCL:
        info->colorSpace = 5;
        break;
    case AVCOL_SPC_RESERVED:
        info->colorSpace = 6;
        break;
    case AVCOL_SPC_RGB:
        info->colorSpace = 7;
        break;
    }

    for (i = 0; i < 8; i++) {
        info->segmentFeatureEnable[i][0] = h->h.segmentation.feat[i].q_enabled;
        info->segmentFeatureEnable[i][1] = h->h.segmentation.feat[i].lf_enabled;
        info->segmentFeatureEnable[i][2] = h->h.segmentation.feat[i].ref_enabled;
        info->segmentFeatureEnable[i][3] = h->h.segmentation.feat[i].skip_enabled;

        info->segmentFeatureData[i][0] = h->h.segmentation.feat[i].q_val;
        info->segmentFeatureData[i][1] = h->h.segmentation.feat[i].lf_val;
        info->segmentFeatureData[i][2] = h->h.segmentation.feat[i].ref_val;
        info->segmentFeatureData[i][3] = 0;
    }

    for (i = 0; i < 7; i++)
        info->mbSegmentTreeProbs[i] = h->h.segmentation.prob[i];

    for (i = 0; i < 3; i++) {
        info->activeRefIdx[i] = h->h.refidx[i];
        info->segmentPredProbs[i] = h->h.segmentation.pred_prob[i];
        info->refFrameSignBias[i + 1] = h->h.signbias[i];
    }

    for (i = 0; i < 4; i++)
        info->mbRefLfDelta[i] = h->h.lf_delta.ref[i];

    for (i = 0; i < 2; i++)
        info->mbModeLfDelta[i] = h->h.lf_delta.mode[i];

    return ff_vdpau_common_start_frame(pic_ctx, buffer, size);
}

static int vdpau_vp9_decode_slice(AVCodecContext *avctx,
                                  const uint8_t *buffer, uint32_t size)
{
    VP9Context * const v  = avctx->priv_data;
    VP9SharedContext * const h = &v->s;
    VP9Frame *f = &h->frames[CUR_FRAME];
    struct vdpau_picture_context *pic_ctx = f->hwaccel_picture_private;
    int val;

    val = ff_vdpau_add_buffer(pic_ctx, buffer, size);
    if (val)
        return val;

    return 0;
}

static int vdpau_vp9_init(AVCodecContext *avctx)
{
    VdpDecoderProfile profile;

    switch (avctx->profile) {
    case FF_PROFILE_VP9_0:
        profile = VDP_DECODER_PROFILE_VP9_PROFILE_0;
        break;
    case FF_PROFILE_VP9_1:
        profile = VDP_DECODER_PROFILE_VP9_PROFILE_1;
        break;
    case FF_PROFILE_VP9_2:
        profile = VDP_DECODER_PROFILE_VP9_PROFILE_2;
        break;
    case FF_PROFILE_VP9_3:
        profile = VDP_DECODER_PROFILE_VP9_PROFILE_3;
        break;
    default:
        return AVERROR(ENOTSUP);
    }

    return ff_vdpau_common_init(avctx, profile, avctx->level);
}

static int vdpau_vp9_end_frame(AVCodecContext *avctx)
{
    VP9Context * const v  = avctx->priv_data;
    VP9SharedContext * const h = &v->s;
    VP9Frame *f = &h->frames[CUR_FRAME];
    struct vdpau_picture_context *pic_ctx = f->hwaccel_picture_private;
    int val;

    val = ff_vdpau_common_end_frame(avctx, f->tf.f, pic_ctx);
    if (val < 0)
        return val;

    return 0;
}

const AVHWAccel ff_vp9_vdpau_hwaccel = {
    .name           = "vp9_vdpau",
    .type           = AVMEDIA_TYPE_VIDEO,
    .id             = AV_CODEC_ID_VP9,
    .pix_fmt        = AV_PIX_FMT_VDPAU,
    .start_frame    = vdpau_vp9_start_frame,
    .end_frame      = vdpau_vp9_end_frame,
    .decode_slice   = vdpau_vp9_decode_slice,
    .frame_priv_data_size = sizeof(struct vdpau_picture_context),
    .init           = vdpau_vp9_init,
    .uninit         = ff_vdpau_common_uninit,
    .frame_params   = ff_vdpau_common_frame_params,
    .priv_data_size = sizeof(VDPAUContext),
    .caps_internal  = HWACCEL_CAP_ASYNC_SAFE,
};
