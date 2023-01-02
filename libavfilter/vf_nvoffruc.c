/*
 * Copyright (C) 2022 Philip Langdale <philipl@overt.org>
 * Based on vf_framerate - Copyright (C) 2012 Mark Himsley
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

/**
 * @file
 * filter upsamples the frame rate of a source using the nvidia Optical Flow
 * FRUC library.
 */

#include <dlfcn.h>
#include "libavutil/avassert.h"
#include "libavutil/cuda_check.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "filters.h"
#include "internal.h"
/*
 * This cannot be distributed with the filter due to licensing. If you want to
 * compile this filter, you will need to obtain it from nvidia and then fix it
 * to work in a pure C environment:
 * * Remove the `using namespace std;`
 * * Replace the `bool *` with `void *`
 */
#include "NvOFFRUC.h"

typedef struct FRUCContext {
    const AVClass *class;

    AVCUDADeviceContext *hwctx;
    AVBufferRef         *device_ref;

    CUcontext cu_ctx;
    CUstream  stream;
    CUarray   c0;                       ///< CUarray for f0
    CUarray   c1;                       ///< CUarray for f1
    CUarray   cw;                       ///< CUarray for work

    AVRational dest_frame_rate;
    int interp_start;                   ///< start of range to apply interpolation
    int interp_end;                     ///< end of range to apply interpolation

    AVRational srce_time_base;          ///< timebase of source
    AVRational dest_time_base;          ///< timebase of destination

    int blend_factor_max;
    AVFrame *work;
    enum AVPixelFormat format;

    AVFrame *f0;                        ///< last frame
    AVFrame *f1;                        ///< current frame
    int64_t pts0;                       ///< last frame pts in dest_time_base
    int64_t pts1;                       ///< current frame pts in dest_time_base
    int64_t delta;                      ///< pts1 to pts0 delta
    int flush;                          ///< 1 if the filter is being flushed
    int64_t start_pts;                  ///< pts of the first output frame
    int64_t n;                          ///< output frame counter

    void *fruc_dl;
    PtrToFuncNvOFFRUCCreate NvOFFRUCCreate;
    PtrToFuncNvOFFRUCRegisterResource NvOFFRUCRegisterResource;
    PtrToFuncNvOFFRUCUnregisterResource NvOFFRUCUnregisterResource;
    PtrToFuncNvOFFRUCProcess NvOFFRUCProcess;
    PtrToFuncNvOFFRUCDestroy NvOFFRUCDestroy;
    NvOFFRUCHandle fruc;
} FRUCContext;

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)
#define OFFSET(x) offsetof(FRUCContext, x)
#define V AV_OPT_FLAG_VIDEO_PARAM
#define F AV_OPT_FLAG_FILTERING_PARAM
#define FRAMERATE_FLAG_SCD 01

static const AVOption nvoffruc_options[] = {
    {"fps",                 "required output frames per second rate", OFFSET(dest_frame_rate), AV_OPT_TYPE_VIDEO_RATE, {.str="50"},             0,       INT_MAX, V|F },

    {"interp_start",        "point to start linear interpolation",    OFFSET(interp_start),    AV_OPT_TYPE_INT,      {.i64=15},                 0,       255,     V|F },
    {"interp_end",          "point to end linear interpolation",      OFFSET(interp_end),      AV_OPT_TYPE_INT,      {.i64=240},                0,       255,     V|F },

    {NULL}
};

AVFILTER_DEFINE_CLASS(nvoffruc);

static int blend_frames(AVFilterContext *ctx, int64_t work_pts)
{
    FRUCContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];

    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUDA_MEMCPY2D cpy_params = {0,};
    NvOFFRUC_PROCESS_IN_PARAMS in = {0,};
    NvOFFRUC_PROCESS_OUT_PARAMS out = {0,};
    NvOFFRUC_STATUS status;

    int num_channels = s->format == AV_PIX_FMT_NV12 ? 1 : 4;
    int ret;
    uint64_t ignored;

    // get work-space for output frame
    s->work = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!s->work)
        return AVERROR(ENOMEM);

    av_frame_copy_props(s->work, s->f0);

    cpy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE,
    cpy_params.srcDevice = (CUdeviceptr)s->f0->data[0],
    cpy_params.srcPitch = s->f0->linesize[0],
    cpy_params.srcY = 0,
    cpy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY,
    cpy_params.dstArray = s->c0,
    cpy_params.dstY = 0,
    cpy_params.WidthInBytes = s->f0->width * num_channels,
    cpy_params.Height = s->f0->height,
    ret = CHECK_CU(cu->cuMemcpy2DAsync(&cpy_params, s->stream));
    if (ret < 0)
        return ret;

    if (s->f0->data[1]) {
        cpy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE,
        cpy_params.srcDevice = (CUdeviceptr)s->f0->data[1],
        cpy_params.srcPitch = s->f0->linesize[1],
        cpy_params.srcY = 0,
        cpy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY,
        cpy_params.dstArray = s->c0,
        cpy_params.dstY = s->f0->height,
        cpy_params.WidthInBytes = s->f0->width * num_channels,
        cpy_params.Height = s->f0->height * 0.5,
        CHECK_CU(cu->cuMemcpy2DAsync(&cpy_params, s->stream));
        if (ret < 0)
            return ret;
    }

    cpy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE,
    cpy_params.srcDevice = (CUdeviceptr)s->f1->data[0],
    cpy_params.srcPitch = s->f1->linesize[0],
    cpy_params.srcY = 0,
    cpy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY,
    cpy_params.dstArray = s->c1,
    cpy_params.dstY = 0,
    cpy_params.WidthInBytes = s->f1->width * num_channels,
    cpy_params.Height = s->f1->height,
    CHECK_CU(cu->cuMemcpy2DAsync(&cpy_params, s->stream));
    if (ret < 0)
        return ret;

    if (s->f1->data[1]) {
        cpy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE,
        cpy_params.srcDevice = (CUdeviceptr)s->f1->data[1],
        cpy_params.srcPitch = s->f1->linesize[1],
        cpy_params.srcY = 0,
        cpy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY,
        cpy_params.dstArray = s->c1,
        cpy_params.dstY = s->f1->height,
        cpy_params.WidthInBytes = s->f1->width * num_channels,
        cpy_params.Height = s->f1->height * 0.5,
        CHECK_CU(cu->cuMemcpy2DAsync(&cpy_params, s->stream));
        if (ret < 0)
            return ret;
    }

    in.stFrameDataInput.pFrame = s->c0;
    in.stFrameDataInput.nTimeStamp = s->pts0;
    out.stFrameDataOutput.pFrame = s->cw,
    out.stFrameDataOutput.nTimeStamp = s->pts0;
    out.stFrameDataOutput.bHasFrameRepetitionOccurred = &ignored;
    status = s->NvOFFRUCProcess(s->fruc, &in, &out);
    if (status) {
        av_log(ctx, AV_LOG_ERROR, "FRUC: Process failure: %d\n", status);
        return AVERROR(ENOSYS);
    }

    in.stFrameDataInput.pFrame = s->c1;
    in.stFrameDataInput.nTimeStamp = s->pts1;
    out.stFrameDataOutput.pFrame = s->cw,
    out.stFrameDataOutput.nTimeStamp = work_pts;
    out.stFrameDataOutput.bHasFrameRepetitionOccurred = &ignored;
    status = s->NvOFFRUCProcess(s->fruc, &in, &out);
    if (status) {
        av_log(ctx, AV_LOG_ERROR, "FRUC: Process failure: %d\n", status);
        return AVERROR(ENOSYS);
    }

    cpy_params.srcMemoryType = CU_MEMORYTYPE_ARRAY,
    cpy_params.srcArray = s->cw,
    cpy_params.srcY = 0,
    cpy_params.dstMemoryType = CU_MEMORYTYPE_DEVICE,
    cpy_params.dstDevice = (CUdeviceptr)s->work->data[0],
    cpy_params.dstPitch = s->work->linesize[0],
    cpy_params.dstY = 0,
    cpy_params.WidthInBytes = s->work->width * num_channels,
    cpy_params.Height = s->work->height,
    CHECK_CU(cu->cuMemcpy2DAsync(&cpy_params, s->stream));
    if (ret < 0)
        return ret;

    if (s->work->data[1]) {
        cpy_params.srcMemoryType = CU_MEMORYTYPE_ARRAY,
        cpy_params.srcArray = s->cw,
        cpy_params.srcY = s->work->height,
        cpy_params.dstMemoryType = CU_MEMORYTYPE_DEVICE,
        cpy_params.dstDevice = (CUdeviceptr)s->work->data[1],
        cpy_params.dstPitch = s->work->linesize[1],
        cpy_params.dstY = 0,
        cpy_params.WidthInBytes = s->work->width * num_channels,
        cpy_params.Height = s->work->height * 0.5,
        CHECK_CU(cu->cuMemcpy2DAsync(&cpy_params, s->stream));
        if (ret < 0)
            return ret;
    }

    return 0;
}

static int process_work_frame(AVFilterContext *ctx)
{
    FRUCContext *s = ctx->priv;
    int64_t work_pts;
    int64_t interpolate, interpolate8;
    int ret;

    if (!s->f1)
        return 0;
    if (!s->f0 && !s->flush)
        return 0;

    work_pts = s->start_pts + av_rescale_q(s->n, av_inv_q(s->dest_frame_rate), s->dest_time_base);

    if (work_pts >= s->pts1 && !s->flush)
        return 0;

    if (!s->f0) {
        av_assert1(s->flush);
        s->work = s->f1;
        s->f1 = NULL;
    } else {
        if (work_pts >= s->pts1 + s->delta && s->flush)
            return 0;

        interpolate = av_rescale(work_pts - s->pts0, s->blend_factor_max, s->delta);
        interpolate8 = av_rescale(work_pts - s->pts0, 256, s->delta);
        ff_dlog(ctx, "process_work_frame() interpolate: %"PRId64"/256\n", interpolate8);
        if (interpolate >= s->blend_factor_max || interpolate8 > s->interp_end) {
            av_log(ctx, AV_LOG_DEBUG, "Matched f0: pts %lu\n", work_pts);
            s->work = av_frame_clone(s->f1);
        } else if (interpolate <= 0 || interpolate8 < s->interp_start) {
            av_log(ctx, AV_LOG_DEBUG, "Matched f1: pts %lu\n", work_pts);
            s->work = av_frame_clone(s->f0);
        } else {
            av_log(ctx, AV_LOG_DEBUG, "Unmatched pts: %lu\n", work_pts);
            ret = blend_frames(ctx, work_pts);
            if (ret < 0)
                return ret;
        }
    }

    if (!s->work)
        return AVERROR(ENOMEM);

    s->work->pts = work_pts;
    s->n++;

    return 1;
}

static av_cold int init(AVFilterContext *ctx)
{
    FRUCContext *s = ctx->priv;
    s->start_pts = AV_NOPTS_VALUE;

    // TODO: Need windows equivalent symbol loading
    s->fruc_dl = dlopen("libNvOFFRUC.so", RTLD_LAZY);
    if (!s->fruc_dl) {
        av_log(ctx, AV_LOG_ERROR, "Failed to load FRUC: %s\n", dlerror());
        return AVERROR(EINVAL);
    }

    s->NvOFFRUCCreate = (PtrToFuncNvOFFRUCCreate)
        dlsym(s->fruc_dl, "NvOFFRUCCreate");
    s->NvOFFRUCRegisterResource = (PtrToFuncNvOFFRUCRegisterResource)
        dlsym(s->fruc_dl, "NvOFFRUCRegisterResource");
    s->NvOFFRUCUnregisterResource = (PtrToFuncNvOFFRUCUnregisterResource)
        dlsym(s->fruc_dl, "NvOFFRUCUnregisterResource");
    s->NvOFFRUCProcess = (PtrToFuncNvOFFRUCProcess)
        dlsym(s->fruc_dl, "NvOFFRUCProcess");
    s->NvOFFRUCDestroy = (PtrToFuncNvOFFRUCDestroy)
        dlsym(s->fruc_dl, "NvOFFRUCDestroy");
    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    FRUCContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy;

    CHECK_CU(cu->cuCtxPushCurrent(s->cu_ctx));

    if (s->fruc) {
        NvOFFRUC_UNREGISTER_RESOURCE_PARAM in_param = {
            .pArrResource = {s->c0, s->c1, s->cw},
            .uiCount = 1,
        };
        NvOFFRUC_STATUS nv_status = s->NvOFFRUCUnregisterResource(s->fruc, &in_param);
        if (nv_status) {
            av_log(ctx, AV_LOG_WARNING, "Could not unregister: %d\n", nv_status);
        }
        s->NvOFFRUCDestroy(s->fruc);
    }
    if (s->c0)
        CHECK_CU(cu->cuArrayDestroy(s->c0));
    if (s->c1)
        CHECK_CU(cu->cuArrayDestroy(s->c1));
    if (s->cw)
        CHECK_CU(cu->cuArrayDestroy(s->cw));

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    if (s->fruc_dl)
        dlclose(s->fruc_dl);
    av_frame_free(&s->f0);
    av_frame_free(&s->f1);
    av_buffer_unref(&s->device_ref);
}

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_NV12,
    // Actually any single plane, four channel, 8bit format will work.
    AV_PIX_FMT_ARGB,
    AV_PIX_FMT_ABGR,
    AV_PIX_FMT_RGBA,
    AV_PIX_FMT_BGRA,
    AV_PIX_FMT_NONE
};

static int format_is_supported(enum AVPixelFormat fmt)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static int activate(AVFilterContext *ctx)
{
    int ret, status;
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    FRUCContext *s = ctx->priv;
    AVFrame *inpicref;
    int64_t pts;

    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    CHECK_CU(cu->cuCtxPushCurrent(s->cu_ctx));

retry:
    ret = process_work_frame(ctx);
    if (ret < 0) {
        goto exit;
    } else if (ret == 1) {
        ret = ff_filter_frame(outlink, s->work);
        goto exit;
    }

    ret = ff_inlink_consume_frame(inlink, &inpicref);
    if (ret < 0)
        goto exit;

    if (inpicref) {
        if (inpicref->interlaced_frame)
            av_log(ctx, AV_LOG_WARNING, "Interlaced frame found - the output will not be correct.\n");

        if (inpicref->pts == AV_NOPTS_VALUE) {
            av_log(ctx, AV_LOG_WARNING, "Ignoring frame without PTS.\n");
            av_frame_free(&inpicref);
        }
    }

    if (inpicref) {
        pts = av_rescale_q(inpicref->pts, s->srce_time_base, s->dest_time_base);

        if (s->f1 && pts == s->pts1) {
            av_log(ctx, AV_LOG_WARNING, "Ignoring frame with same PTS.\n");
            av_frame_free(&inpicref);
        }
    }

    if (inpicref) {
        av_frame_free(&s->f0);
        s->f0 = s->f1;
        s->pts0 = s->pts1;

        s->f1 = inpicref;
        s->pts1 = pts;
        s->delta = s->pts1 - s->pts0;

        if (s->delta < 0) {
            av_log(ctx, AV_LOG_WARNING, "PTS discontinuity.\n");
            s->start_pts = s->pts1;
            s->n = 0;
            av_frame_free(&s->f0);
        }

        if (s->start_pts == AV_NOPTS_VALUE)
            s->start_pts = s->pts1;

        goto retry;
    }

    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (!s->flush) {
            s->flush = 1;
            goto retry;
        }
        ff_outlink_set_status(outlink, status, pts);
        ret = 0;
        goto exit;
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return FFERROR_NOT_READY;

exit:
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return ret;
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    FRUCContext *s = ctx->priv;

    s->srce_time_base = inlink->time_base;
    s->blend_factor_max = 1 << (8 -1);

    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    AVHWFramesContext *in_frames_ctx;
    AVHWFramesContext *output_frames;
    FRUCContext *s = ctx->priv;
    CudaFunctions *cu;
    CUcontext dummy;
    CUDA_ARRAY_DESCRIPTOR desc = {0,};
    NvOFFRUC_CREATE_PARAM create_param = {0,};
    NvOFFRUC_REGISTER_RESOURCE_PARAM register_param = {0,};
    NvOFFRUC_STATUS status;
    int exact;
    int ret;

    ff_dlog(ctx, "config_output()\n");

    ff_dlog(ctx,
           "config_output() input time base:%u/%u (%f)\n",
           ctx->inputs[0]->time_base.num,ctx->inputs[0]->time_base.den,
           av_q2d(ctx->inputs[0]->time_base));

    // make sure timebase is small enough to hold the framerate

    exact = av_reduce(&s->dest_time_base.num, &s->dest_time_base.den,
                      av_gcd((int64_t)s->srce_time_base.num * s->dest_frame_rate.num,
                             (int64_t)s->srce_time_base.den * s->dest_frame_rate.den ),
                      (int64_t)s->srce_time_base.den * s->dest_frame_rate.num, INT_MAX);

    av_log(ctx, AV_LOG_INFO,
           "time base:%u/%u -> %u/%u exact:%d\n",
           s->srce_time_base.num, s->srce_time_base.den,
           s->dest_time_base.num, s->dest_time_base.den, exact);
    if (!exact) {
        av_log(ctx, AV_LOG_WARNING, "Timebase conversion is not exact\n");
    }

    outlink->frame_rate = s->dest_frame_rate;
    outlink->time_base = s->dest_time_base;

    ff_dlog(ctx,
           "config_output() output time base:%u/%u (%f) w:%d h:%d\n",
           outlink->time_base.num, outlink->time_base.den,
           av_q2d(outlink->time_base),
           outlink->w, outlink->h);


    av_log(ctx, AV_LOG_INFO, "fps -> fps:%u/%u\n",
            s->dest_frame_rate.num, s->dest_frame_rate.den);

    /* check that we have a hw context */
    if (!inlink->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    s->format = in_frames_ctx->sw_format;

    if (!format_is_supported(s->format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n",
               av_get_pix_fmt_name(s->format));
        return AVERROR(ENOSYS);
    }

    s->device_ref = av_buffer_ref(in_frames_ctx->device_ref);
    if (!s->device_ref)
        return AVERROR(ENOMEM);

    s->hwctx = ((AVHWDeviceContext*)s->device_ref->data)->hwctx;
    s->cu_ctx = s->hwctx->cuda_ctx;
    s->stream = s->hwctx->stream;
    cu = s->hwctx->internal->cuda_dl;
    outlink->hw_frames_ctx = av_hwframe_ctx_alloc(s->device_ref);
    if (!inlink->hw_frames_ctx)
        return AVERROR(ENOMEM);

    output_frames = (AVHWFramesContext*)outlink->hw_frames_ctx->data;

    output_frames->format    = AV_PIX_FMT_CUDA;
    output_frames->sw_format = s->format;
    output_frames->width     = ctx->inputs[0]->w;
    output_frames->height    = ctx->inputs[0]->h;

    output_frames->initial_pool_size = 3;

    ret = ff_filter_init_hw_frames(ctx, outlink, 0);
    if (ret < 0)
        return ret;

    ret = av_hwframe_ctx_init(outlink->hw_frames_ctx);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to initialise CUDA frame "
               "context for output: %d\n", ret);
        return ret;
    }

    outlink->w = inlink->w;
    outlink->h = inlink->h;

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->cu_ctx));
    if (ret < 0)
        return ret;

    desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    desc.Height = inlink->h * (s->format == AV_PIX_FMT_NV12 ? 1.5 : 1);
    desc.Width = inlink->w;
    desc.NumChannels = s->format == AV_PIX_FMT_NV12 ? 1 : 4;
    ret = CHECK_CU(cu->cuArrayCreate(&s->c0, &desc));
    if (ret < 0)
        goto exit;
    ret = CHECK_CU(cu->cuArrayCreate(&s->c1, &desc));
    if (ret < 0)
        goto exit;
    ret = CHECK_CU(cu->cuArrayCreate(&s->cw, &desc));
    if (ret < 0)
        goto exit;

    create_param.uiWidth = inlink->w;
    create_param.uiHeight = inlink->h;
    create_param.pDevice = NULL;
    create_param.eResourceType = CudaResource;
    create_param.eSurfaceFormat = s->format == AV_PIX_FMT_NV12 ? NV12Surface : ARGBSurface;
    create_param.eCUDAResourceType = CudaResourceCuArray;
    status = s->NvOFFRUCCreate(&create_param, &s->fruc);
    if (status) {
        av_log(ctx, AV_LOG_ERROR, "FRUC: Failed to create: %d\n", status);
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    register_param.pArrResource[0] = s->c0;
    register_param.pArrResource[1] = s->c1;
    register_param.pArrResource[2] = s->cw;
    register_param.uiCount = 3;
    status = s->NvOFFRUCRegisterResource(s->fruc, &register_param);
    if (status) {
        av_log(ctx, AV_LOG_ERROR, "FRUC: Failed to register: %d\n", status);
        ret = AVERROR(ENOSYS);
        goto exit;
    }

    ret = 0;
exit:
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return ret;
}

static const AVFilterPad framerate_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
    },
};

static const AVFilterPad framerate_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
};

const AVFilter ff_vf_nvoffruc = {
    .name          = "nvoffruc",
    .description   = NULL_IF_CONFIG_SMALL("Upsamples progressive source to specified frame rates with nvidia FRUC"),
    .priv_size     = sizeof(FRUCContext),
    .priv_class    = &nvoffruc_class,
    .init          = init,
    .uninit        = uninit,
    .activate      = activate,
    FILTER_INPUTS(framerate_inputs),
    FILTER_OUTPUTS(framerate_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_CUDA),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
