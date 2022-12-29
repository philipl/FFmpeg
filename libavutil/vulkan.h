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

#ifndef AVUTIL_VULKAN_H
#define AVUTIL_VULKAN_H

#define VK_NO_PROTOTYPES

#include <stdatomic.h>

#include "pixdesc.h"
#include "bprint.h"
#include "hwcontext.h"
#include "vulkan_functions.h"
#include "hwcontext_vulkan.h"
#include "vulkan_loader.h"

#define FF_VK_DEFAULT_USAGE_FLAGS (VK_IMAGE_USAGE_SAMPLED_BIT      |           \
                                   VK_IMAGE_USAGE_STORAGE_BIT      |           \
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT |           \
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT)

/* GLSL management macros */
#define INDENT(N) INDENT_##N
#define INDENT_0
#define INDENT_1 INDENT_0 "    "
#define INDENT_2 INDENT_1 INDENT_1
#define INDENT_3 INDENT_2 INDENT_1
#define INDENT_4 INDENT_3 INDENT_1
#define INDENT_5 INDENT_4 INDENT_1
#define INDENT_6 INDENT_5 INDENT_1
#define C(N, S)          INDENT(N) #S "\n"
#define GLSLC(N, S)      av_bprintf(&shd->src, C(N, S))
#define GLSLA(...)       av_bprintf(&shd->src, __VA_ARGS__)
#define GLSLF(N, S, ...) av_bprintf(&shd->src, C(N, S), __VA_ARGS__)
#define GLSLD(D)         GLSLC(0, );                                           \
                         av_bprint_append_data(&shd->src, D, strlen(D));       \
                         GLSLC(0, )

/* Helper, pretty much every Vulkan return value needs to be checked */
#define RET(x)                                                                 \
    do {                                                                       \
        if ((err = (x)) < 0)                                                   \
            goto fail;                                                         \
    } while (0)

typedef struct FFVkSPIRVShader {
    const char *name;                       /* Name for id/debugging purposes */
    AVBPrint src;
    int local_size[3];                      /* Compute shader workgroup sizes */
    VkPipelineShaderStageCreateInfo shader;
} FFVkSPIRVShader;

typedef struct FFVkSPIRVCompiler {
    void *priv;
    int (*compile_shader)(struct FFVkSPIRVCompiler *ctx, void *avctx,
                          struct FFVkSPIRVShader *shd, uint8_t **data,
                          size_t *size, const char *entrypoint, void **opaque);
    void (*free_shader)(struct FFVkSPIRVCompiler *ctx, void **opaque);
    void (*uninit)(struct FFVkSPIRVCompiler **ctx);
} FFVkSPIRVCompiler;

typedef struct FFVkSampler {
    VkSampler sampler[4];
} FFVkSampler;

typedef struct FFVulkanDescriptorSetBinding {
    const char         *name;
    VkDescriptorType    type;
    const char         *mem_layout;  /* Storage images (rgba8, etc.) and buffers (std430, etc.) */
    const char         *mem_quali;   /* readonly, writeonly, etc. */
    const char         *buf_content; /* For buffers */
    uint32_t            dimensions;  /* Needed for e.g. sampler%iD */
    uint32_t            elems;       /* 0 - scalar, 1 or more - vector */
    VkShaderStageFlags  stages;
    FFVkSampler        *sampler;     /* Sampler to use for all elems */
    void               *updater;     /* Pointer to VkDescriptor*Info */
} FFVulkanDescriptorSetBinding;

typedef struct FFVkBuffer {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkMemoryPropertyFlagBits flags;
    size_t size;
} FFVkBuffer;

typedef struct FFVkQueueFamilyCtx {
    int queue_family;
    int nb_queues;
} FFVkQueueFamilyCtx;

typedef struct FFVulkanPipeline {
    FFVkQueueFamilyCtx *qf;

    VkPipelineBindPoint bind_point;

    /* Contexts */
    VkPipelineLayout pipeline_layout;
    VkPipeline       pipeline;

    /* Shaders */
    FFVkSPIRVShader **shaders;
    int shaders_num;

    /* Push consts */
    VkPushConstantRange *push_consts;
    int push_consts_num;

    /* Descriptors */
    VkDescriptorSetLayout         *desc_layout;
    VkDescriptorPool               desc_pool;
    VkDescriptorSet               *desc_set;
#if VK_USE_64_BIT_PTR_DEFINES == 1
    void                         **desc_staging;
#else
    uint64_t                      *desc_staging;
#endif
    VkDescriptorSetLayoutBinding **desc_binding;
    VkDescriptorUpdateTemplate    *desc_template;
    int                           *desc_set_initialized;
    int                            desc_layout_num;
    int                            descriptor_sets_num;
    int                            total_descriptor_sets;
    int                            pool_size_desc_num;

    /* Temporary, used to store data in between initialization stages */
    VkDescriptorUpdateTemplateCreateInfo *desc_template_info;
    VkDescriptorPoolSize *pool_size_desc;
} FFVulkanPipeline;

typedef struct FFVkExecContext {
    const struct FFVkExecPool *parent;

    /* Queue for the execution context */
    VkQueue queue;
    int qf;
    int qi;

    /* Command buffer for the context */
    VkCommandBuffer buf;

    /* Fence for the command buffer */
    VkFence fence;

    void *query_data;
    int query_idx;

    /* Buffer dependencies */
    AVBufferRef **buf_deps;
    int nb_buf_deps;
    unsigned int buf_deps_alloc_size;

    /* Frame dependencies */
    AVBufferRef **frame_deps;
    unsigned int frame_deps_alloc_size;
    int nb_frame_deps;

    VkSemaphore *sem_wait;
    unsigned int sem_wait_alloc; /* Allocated sem_wait */
    int sem_wait_cnt;

    uint64_t *sem_wait_val;
    unsigned int sem_wait_val_alloc;

    VkPipelineStageFlagBits *sem_wait_dst;
    unsigned int sem_wait_dst_alloc; /* Allocated sem_wait_dst */

    VkSemaphore *sem_sig;
    unsigned int sem_sig_alloc; /* Allocated sem_sig */
    int sem_sig_cnt;

    uint64_t *sem_sig_val;
    unsigned int sem_sig_val_alloc;

    uint64_t **sem_sig_val_dst;
    unsigned int sem_sig_val_dst_alloc;

    uint8_t *frame_locked;
    unsigned int frame_locked_alloc_size;

    VkAccessFlagBits *access_dst;
    unsigned int access_dst_alloc;

    VkImageLayout *layout_dst;
    unsigned int layout_dst_alloc;

    uint32_t *queue_family_dst;
    unsigned int queue_family_dst_alloc;

    uint8_t *frame_update;
    unsigned int frame_update_alloc_size;
} FFVkExecContext;

typedef struct FFVkExecPool {
    FFVkQueueFamilyCtx *qf;
    FFVkExecContext *contexts;
    atomic_int_least64_t idx;

    VkCommandPool cmd_buf_pool;
    VkCommandBuffer *cmd_bufs;
    int pool_size;

    VkQueryPool query_pool;
    void *query_data;
    int query_results;
    int query_statuses;
    int query_64bit;
    int query_status_stride;
    int nb_queries;
    size_t qd_size;
} FFVkExecPool;

typedef struct FFVulkanContext {
    const AVClass *class; /* Filters and encoders use this */

    FFVulkanFunctions     vkfn;
    FFVulkanExtensions    extensions;
    VkPhysicalDeviceProperties2 props;
    VkPhysicalDeviceDriverProperties driver_props;
    VkPhysicalDeviceMemoryProperties mprops;
    VkQueueFamilyQueryResultStatusPropertiesKHR *query_props;
    VkQueueFamilyVideoPropertiesKHR *video_props;
    VkQueueFamilyProperties2 *qf_props;

    AVBufferRef           *device_ref;
    AVHWDeviceContext     *device;
    AVVulkanDeviceContext *hwctx;

    AVBufferRef           *frames_ref;
    AVHWFramesContext     *frames;
    AVVulkanFramesContext *hwfc;

    uint32_t               qfs[5];
    int                    nb_qfs;

    FFVkSPIRVCompiler     *spirv_compiler;

    /* Properties */
    int                 output_width;
    int                output_height;
    enum AVPixelFormat output_format;
    enum AVPixelFormat  input_format;
} FFVulkanContext;

/* Identity mapping - r = r, b = b, g = g, a = a */
extern const VkComponentMapping ff_comp_identity_map;

/**
 * Converts Vulkan return values to strings
 */
const char *ff_vk_ret2str(VkResult res);

/**
 * Returns 1 if pixfmt is a usable RGB format.
 */
int ff_vk_mt_is_np_rgb(enum AVPixelFormat pix_fmt);

/**
 * Returns the format to use for images in shaders.
 */
const char *ff_vk_shader_rep_fmt(enum AVPixelFormat pixfmt);

/**
 * Loads props/mprops/driver_props
 */
int ff_vk_load_props(FFVulkanContext *s);

/**
 * Loads queue families into the main context.
 * Chooses a QF and loads it into a context.
 */
void ff_vk_qf_fill(FFVulkanContext *s);
int ff_vk_qf_init(FFVulkanContext *s, FFVkQueueFamilyCtx *qf,
                  VkQueueFlagBits dev_family);

/**
 * Allocates/frees an execution pool.
 */
int ff_vk_exec_pool_init(FFVulkanContext *s, FFVkQueueFamilyCtx *qf,
                         FFVkExecPool *pool, int nb_contexts,
                         int nb_queries, VkQueryType query_type, int query_64bit,
                         const void *query_create_pnext);
void ff_vk_exec_pool_free(FFVulkanContext *s, FFVkExecPool *pool);

/**
 * Retrieve an execution pool. Threadsafe.
 */
FFVkExecContext *ff_vk_exec_get(FFVkExecPool *pool);

/**
 * Explicitly wait on an execution to be finished.
 * Starting via ff_vk_exec_start() also waits on it.
 */

/**
 * Performs nb_queries queries and returns their results and statuses.
 * Execution must have been waited on to produce valid results.
 */
VkResult ff_vk_exec_get_query(FFVulkanContext *s, FFVkExecContext *e,
                              void **data, int64_t *status);

/**
 * Start/submit/wait an execution.
 * ff_vk_exec_start() always waits on a submission, so using ff_vk_exec_wait()
 * is not necessary (unless using it is just better).
 */
int ff_vk_exec_start(FFVulkanContext *s, FFVkExecContext *e);
int ff_vk_exec_submit(FFVulkanContext *s, FFVkExecContext *e);
void ff_vk_exec_wait(FFVulkanContext *s, FFVkExecContext *e);

/**
 * Execution dependency management.
 * Can attach buffers to executions that will only be unref'd once the
 * buffer has finished executing.
 * Adding a frame dep will *lock the frame*, until either the dependencies
 * are discarded, the execution is submitted, or a failure happens.
 * update_frame will update the frame's properties before it is unlocked,
 * only if submission was successful.
 */
int ff_vk_exec_add_dep_buf(FFVulkanContext *s, FFVkExecContext *e,
                           AVBufferRef **deps, int nb_deps, int ref);
int ff_vk_exec_add_dep_frame(FFVulkanContext *s, FFVkExecContext *e,
                             AVBufferRef *vkfb, VkPipelineStageFlagBits in_wait_dst_flag);
void ff_vk_exec_update_frame(FFVulkanContext *s, FFVkExecContext *e, AVBufferRef *vkfb,
                             VkImageMemoryBarrier2 *bar);
void ff_vk_exec_discard_deps(FFVulkanContext *s, FFVkExecContext *e);

/**
 * Create an imageview and add it as a dependency to an execution.
 */
int ff_vk_create_imageview(FFVulkanContext *s, FFVkExecContext *e,
                           VkImageView *v, VkImage img, VkFormat fmt,
                           const VkComponentMapping map);

/**
 * Memory/buffer/image allocation helpers.
 */
int ff_vk_alloc_mem(FFVulkanContext *s, VkMemoryRequirements *req,
                    VkMemoryPropertyFlagBits req_flags, void *alloc_extension,
                    VkMemoryPropertyFlagBits *mem_flags, VkDeviceMemory *mem);
int ff_vk_create_buf(FFVulkanContext *s, FFVkBuffer *buf, size_t size,
                     void *pNext, void *alloc_pNext,
                     VkBufferUsageFlags usage, VkMemoryPropertyFlagBits flags);

/**
 * Buffer management code.
 */
int ff_vk_map_buffers(FFVulkanContext *s, FFVkBuffer *buf, uint8_t *mem[],
                      int nb_buffers, int invalidate);
int ff_vk_unmap_buffers(FFVulkanContext *s, FFVkBuffer *buf, int nb_buffers,
                        int flush);
void ff_vk_free_buf(FFVulkanContext *s, FFVkBuffer *buf);

typedef struct FFVkPooledBuffer {
    FFVkBuffer buf;
    uint8_t *mem;
} FFVkPooledBuffer;

/** Initialize a pool and create AVBufferRefs containing FFVkPooledBuffer.
 * Threadsafe to use. Buffers are automatically mapped on creation if
 * VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT is set in mem_props. Users should
 * synchronize access themselvesd. Mainly meant for device-local buffers. */
int ff_vk_get_pooled_buffer(FFVulkanContext *ctx, AVBufferPool **buf_pool,
                            AVBufferRef **buf, VkBufferUsageFlags usage,
                            void *create_pNext, size_t size,
                            VkMemoryPropertyFlagBits mem_props);

/**
 * Sampler management.
 */
FFVkSampler *ff_vk_init_sampler(FFVulkanContext *s, FFVkSampler *sctx,
                                int unnorm_coords, VkFilter filt);
void ff_vk_sampler_free(FFVulkanContext *s, FFVkSampler *sctx);

/**
 * Shader management.
 */
int ff_vk_shader_init(FFVulkanPipeline *pl, FFVkSPIRVShader *shd, const char *name,
                      VkShaderStageFlags stage);
void ff_vk_shader_set_compute_sizes(FFVkSPIRVShader *shd, int local_size[3]);
void ff_vk_shader_print(void *ctx, FFVkSPIRVShader *shd, int prio);
int ff_vk_shader_compile(FFVulkanContext *s, FFVkSPIRVShader *shd,
                         const char *entrypoint);
void ff_vk_shader_free(FFVulkanContext *s, FFVkSPIRVShader *shd);

/**
 * Register a descriptor set.
 * Update a descriptor set for execution.
 */
int ff_vk_add_descriptor_set(FFVulkanContext *s, FFVulkanPipeline *pl,
                             FFVkSPIRVShader *shd, FFVulkanDescriptorSetBinding *desc,
                             int num, int only_print_to_shader);
void ff_vk_update_descriptor_set(FFVulkanContext *s, FFVulkanPipeline *pl,
                                 int set_id);

/**
 * Add/update push constants for execution.
 */
int ff_vk_add_push_constant(FFVulkanPipeline *pl, int offset, int size,
                            VkShaderStageFlagBits stage);
void ff_vk_update_push_exec(FFVulkanContext *s, FFVkExecContext *e,
                            FFVulkanPipeline *pl,
                            VkShaderStageFlagBits stage,
                            int offset, size_t size, void *src);

/**
 * Pipeline management.
 */
int ff_vk_init_compute_pipeline(FFVulkanContext *s, FFVulkanPipeline *pl,
                                FFVkQueueFamilyCtx *qf);
int ff_vk_init_pipeline_layout(FFVulkanContext *s, FFVulkanPipeline *pl);
void ff_vk_pipeline_bind_exec(FFVulkanContext *s, FFVkExecContext *e,
                              FFVulkanPipeline *pl);
void ff_vk_pipeline_free(FFVulkanContext *s, FFVulkanPipeline *pl);

/**
 * Frees main context.
 */
void ff_vk_uninit(FFVulkanContext *s);

#endif /* AVUTIL_VULKAN_H */
