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

#include "libavutil/random_seed.h"
#include "libavutil/opt.h"
#include "vulkan_filter.h"
#include "vulkan_spirv.h"
#include "internal.h"

typedef struct NLMeansVulkanContext {
    FFVulkanContext vkctx;

    int initialized;
    FFVkExecPool e;
    FFVkQueueFamilyCtx qf;
    VkSampler sampler;

    AVBufferPool *integral_buf_pool;
    AVBufferPool *line_buf_pool;
    AVBufferPool *state_buf_pool;
    AVBufferPool *weights_buf_pool;
    AVBufferPool *sums_buf_pool;

    int pl_int_hor_n_rows;
    FFVulkanPipeline pl_int_hor;
    FFVkSPIRVShader shd_int_hor;

    int pl_int_ver_n_rows;
    FFVulkanPipeline pl_int_ver;
    FFVkSPIRVShader shd_int_ver;

    FFVulkanPipeline pl_weights;
    FFVkSPIRVShader shd_weights;

    FFVulkanPipeline pl_denoise;
    FFVkSPIRVShader shd_denoise;

    int *xoffsets;
    int *yoffsets;
    int nb_offsets;

    double sigma;
    uint32_t patch_size[4];
    uint32_t research_size[4];
} NLMeansVulkanContext;

typedef struct HorizontalPushData {
    uint32_t dimensions[4];
    uint32_t int_stride[4];
    int32_t  xoffs[4];
    int32_t  yoffs[4];
    VkDeviceAddress line_data;
    VkDeviceAddress state_data;
    VkDeviceAddress integral_data[4];
} HorizontalPushData;

extern const char *ff_source_prefix_sum_comp;

static av_cold int init_hor_pipeline(FFVulkanContext *vkctx, FFVkExecPool *exec,
                                     FFVulkanPipeline *pl, FFVkSPIRVShader *shd,
                                     VkSampler sampler, FFVkSPIRVCompiler *spv,
                                     int width, int planes, int *nb_rows)
{
    int err;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque;
    FFVulkanDescriptorSetBinding *desc;

    RET(ff_vk_shader_init(pl, shd, "nlmeans_integral_hor", VK_SHADER_STAGE_COMPUTE_BIT, 0));

    ff_vk_shader_set_compute_sizes(shd, 512, 1, 1);
    *nb_rows = 8;

    GLSLC(0, #extension GL_ARB_gpu_shader_int64 : require                               );
    GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require         );
    GLSLC(0, #pragma use_vulkan_memory_model                                            );
    GLSLC(0, #extension GL_KHR_memory_scope_semantics : enable                          );
    GLSLC(0,                                                                            );
    GLSLF(0, #define N_ROWS %i                                                 ,*nb_rows);
    GLSLC(0, #define WG_SIZE (gl_WorkGroupSize.x)                                       );
    GLSLF(0, #define LG_WG_SIZE %i                          ,ff_log2(shd->local_size[0]));
    GLSLC(0, #define PARTITION_SIZE (N_ROWS*WG_SIZE)                                    );
    GLSLC(0, #define DTYPE vec4                                                         );
    GLSLC(0, #define ITYPE ivec4                                                        );
    GLSLC(0,                                                                            );
    GLSLC(0, layout(buffer_reference, buffer_reference_align = 16) buffer DataBuffer {  );
    GLSLC(1,     DTYPE v[];                                                             );
    GLSLC(0, };                                                                         );
    GLSLC(0,                                                                            );
    GLSLC(0, layout(buffer_reference) buffer StateData;                                 );
    GLSLC(0,                                                                            );
    GLSLC(0, layout(push_constant, std430) uniform pushConstants {                      );
    GLSLC(1,     uvec4 dimension;                                                       );
    GLSLC(1,     uvec4 int_stride;                                                      );
    GLSLC(1,     ITYPE xoffs;                                                           );
    GLSLC(1,     ITYPE yoffs;                                                           );
    GLSLC(1,     DataBuffer line_data;                                                  );
    GLSLC(1,     StateData  state;                                                      );
    GLSLC(1,     DataBuffer integral_data[4];                                           );
    GLSLC(0, };                                                                         );

    ff_vk_add_push_constant(pl, 0, sizeof(HorizontalPushData), VK_SHADER_STAGE_COMPUTE_BIT);

    desc = (FFVulkanDescriptorSetBinding []) {
        {
            .name       = "input_img",
            .type       = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .dimensions = 2,
            .elems      = planes,
            .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
            .samplers   = DUP_SAMPLER(sampler),
        },
    };
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 1, 0, 0));

    GLSLD(   ff_source_prefix_sum_comp                                                                    );
    GLSLC(0,                                                                                              );
    GLSLC(0, void main()                                                                                  );
    GLSLC(0, {                                                                                            );
    GLSLC(1,     uint64_t offset;                                                                         );
    GLSLC(1,     DataBuffer dst;                                                                          );
    GLSLC(1,     float s1;                                                                                );
    GLSLC(1,     DTYPE s2;                                                                                );
    GLSLF(1,     uint x = gl_GlobalInvocationID.x * %i;                                          ,*nb_rows);
    GLSLC(0,                                                                                              );
    GLSLC(1,     for (int y = 0; y < dimension[0]; y++) {                                                 );
    for (int r = 0; r < *nb_rows; r++) {
        GLSLF(2,         s1    = texture(input_img[0], ivec2(x + %i, y)).x;                             ,r);
        GLSLF(2,         s2[0] = texture(input_img[0], ivec2(x + %i + xoffs[0], y + yoffs[0])).x;       ,r);
        GLSLF(2,         s2[1] = texture(input_img[0], ivec2(x + %i + xoffs[1], y + yoffs[1])).x;       ,r);
        GLSLF(2,         s2[2] = texture(input_img[0], ivec2(x + %i + xoffs[2], y + yoffs[2])).x;       ,r);
        GLSLF(2,         s2[3] = texture(input_img[0], ivec2(x + %i + xoffs[3], y + yoffs[3])).x;       ,r);
        GLSLF(2,         line_data.v[x + %i] = (s1 - s2) * (s1 - s2);                                   ,r);
        GLSLC(0,                                                                                          );
    }
    GLSLC(2,         offset = uint64_t(int_stride[0])*y*4*4;                                              );
    GLSLC(2,         dst = DataBuffer(uint64_t(integral_data[0]) + offset);                               );
    GLSLC(2,         barrier();                                                                           );
    GLSLC(2,         prefix_sum(dst, line_data);                                                          );
    GLSLC(1,     }                                                                                        );
    GLSLC(0, }                                                                                            );

    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));

    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));

    return 0;

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

typedef struct VerticalPushData {
    uint32_t dimensions[4];
    uint32_t int_stride[4];
    VkDeviceAddress line_data;
    VkDeviceAddress state_data;
    VkDeviceAddress integral_data[4];
} VerticalPushData;

static av_cold int init_ver_pipeline(FFVulkanContext *vkctx, FFVkExecPool *exec,
                                     FFVulkanPipeline *pl, FFVkSPIRVShader *shd,
                                     VkSampler sampler, FFVkSPIRVCompiler *spv,
                                     int height, int planes, int *nb_rows)
{
    int err;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque;
    FFVulkanDescriptorSetBinding *desc;

    RET(ff_vk_shader_init(pl, shd, "nlmeans_integral_ver", VK_SHADER_STAGE_COMPUTE_BIT, 0));

    ff_vk_shader_set_compute_sizes(shd, 512, 1, 1);
    *nb_rows = 4;

    GLSLC(0, #extension GL_ARB_gpu_shader_int64 : require                               );
    GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require         );
    GLSLC(0, #pragma use_vulkan_memory_model                                            );
    GLSLC(0, #extension GL_KHR_memory_scope_semantics : enable                          );
    GLSLC(0,                                                                            );
    GLSLF(0, #define N_ROWS %i                                                 ,*nb_rows);
    GLSLC(0, #define WG_SIZE (gl_WorkGroupSize.x)                                       );
    GLSLF(0, #define LG_WG_SIZE %i                          ,ff_log2(shd->local_size[0]));
    GLSLC(0, #define PARTITION_SIZE (N_ROWS*WG_SIZE)                                    );
    GLSLC(0, #define DTYPE vec4                                                         );
    GLSLC(0,                                                                            );
    GLSLC(0, layout(buffer_reference, buffer_reference_align = 16) buffer DataBuffer {  );
    GLSLC(1,     DTYPE v[];                                                             );
    GLSLC(0, };                                                                         );
    GLSLC(0,                                                                            );
    GLSLC(0, layout(buffer_reference) buffer StateData;                                 );
    GLSLC(0,                                                                            );
    GLSLC(0, layout(push_constant, std430) uniform pushConstants {                      );
    GLSLC(1,     uvec4 dimension;                                                       );
    GLSLC(1,     uvec4 int_stride;                                                      );
    GLSLC(1,     DataBuffer line_data;                                                  );
    GLSLC(1,     StateData  state;                                                      );
    GLSLC(1,     DataBuffer integral_data[4];                                           );
    GLSLC(0, };                                                                         );

    ff_vk_add_push_constant(pl, 0, sizeof(VerticalPushData), VK_SHADER_STAGE_COMPUTE_BIT);

    GLSLD(   ff_source_prefix_sum_comp                                                                    );
    GLSLC(0,                                                                                              );
    GLSLC(0, void main()                                                                                  );
    GLSLC(0, {                                                                                            );
    GLSLC(1,     uint64_t offset;                                                                         );
    GLSLC(1,     DataBuffer dst;                                                                          );
    GLSLC(1,     DataBuffer src;                                                                          );
    GLSLF(1,     uint y = gl_GlobalInvocationID.x * %i;                                          ,*nb_rows);
    GLSLC(0,                                                                                              );
    GLSLC(1,     for (int x = 0; x < dimension[0]; x++) {                                                 );
    for (int r = 0; r < *nb_rows; r++) {
        GLSLF(2,         offset = uint64_t((y + %i)*int_stride[0]);                                     ,r);
        GLSLC(2,         src = DataBuffer(uint64_t(integral_data[0]) + 4*4*offset);                       );
        GLSLF(2,         line_data.v[y + %i] = src.v[x];                                                ,r);
    }
    GLSLC(0,                                                                                              );
    GLSLC(2,         barrier();                                                                           );
    GLSLC(2,         prefix_sum(line_data, line_data);                                                    );
    GLSLC(2,         barrier();                                                                           );
    GLSLC(0,                                                                                              );
    for (int r = 0; r < *nb_rows; r++) {
        GLSLF(2,         offset = uint64_t((y + %i)*int_stride[0]);                                     ,r);
        GLSLC(2,         dst = DataBuffer(uint64_t(integral_data[0]) + 4*4*offset);                       );
        GLSLF(2,         src.v[x] = line_data.v[y + %i];                                                ,r);
    }
    GLSLC(1,     }                                                                                        );
    GLSLC(0, }                                                                                            );

    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));

    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));

    return 0;

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

typedef struct WeightsPushData {
    uint32_t int_stride[4];
    int32_t  xoffs[4];
    int32_t  yoffs[4];
    uint32_t patch[4];
    VkDeviceAddress integral_data[4];
    uint32_t width[4];
    uint32_t height[4];
    float strength[4];
} WeightsPushData;

static av_cold int init_weights_pipeline(FFVulkanContext *vkctx, FFVkExecPool *exec,
                                         FFVulkanPipeline *pl, FFVkSPIRVShader *shd,
                                         VkSampler sampler, FFVkSPIRVCompiler *spv,
                                         int planes)
{
    int err;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque;
    FFVulkanDescriptorSetBinding *desc;

    RET(ff_vk_shader_init(pl, shd, "nlmeans_weights", VK_SHADER_STAGE_COMPUTE_BIT, 0));

    ff_vk_shader_set_compute_sizes(shd, 32, 32, 1);

    GLSLC(0, #extension GL_ARB_gpu_shader_int64 : require                                  );
    GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require            );
    GLSLC(0,                                                                               );
    GLSLC(0, #define DTYPE vec4                                                            );
    GLSLC(0, #define ITYPE ivec4                                                           );
    GLSLC(0,                                                                               );
    GLSLC(0, layout(buffer_reference, buffer_reference_align = 16) buffer DataBuffer {     );
    GLSLC(1,     DTYPE v[];                                                                );
    GLSLC(0, };                                                                            );
    GLSLC(0,                                                                               );
    GLSLC(0, layout(push_constant, std430) uniform pushConstants {                         );
    GLSLC(1,     uvec4 int_stride;                                                         );
    GLSLC(1,     ITYPE xoffs;                                                              );
    GLSLC(1,     ITYPE yoffs;                                                              );
    GLSLC(1,     uvec4 patch_size;                                                         );
    GLSLC(1,     DataBuffer integral_data[4];                                              );
    GLSLC(1,     uvec4 width;                                                              );
    GLSLC(1,     uvec4 height;                                                             );
    GLSLC(1,     vec4 strength;                                                            );
    GLSLC(0, };                                                                            );

    ff_vk_add_push_constant(pl, 0, sizeof(WeightsPushData), VK_SHADER_STAGE_COMPUTE_BIT);

    desc = (FFVulkanDescriptorSetBinding []) {
        {
            .name       = "input_img",
            .type       = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .dimensions = 2,
            .elems      = planes,
            .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
            .samplers   = DUP_SAMPLER(sampler),
        },
        {
            .name       = "output_img",
            .type       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .mem_layout = ff_vk_shader_rep_fmt(vkctx->output_format),
            .mem_quali  = "writeonly",
            .dimensions = 2,
            .elems      = planes,
            .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .name        = "weights_buffer",
            .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "vec4 weights[];",
        },
        {
            .name        = "sums_buffer",
            .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "vec4 sums[];",
        },
    };
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 4, 0, 0));

    GLSLC(0,                                                                                              );
    GLSLC(0, void main()                                                                                  );
    GLSLC(0, {                                                                                            );
    GLSLC(1,     uint64_t offset;                                                                         );
    GLSLC(1,     DataBuffer int_src;                                                                      );
    GLSLC(0,                                                                                              );
    GLSLC(1,     ivec2 pos = ivec2(gl_GlobalInvocationID.xy);                                             );
    GLSLC(1,     ITYPE xoff = pos.x + xoffs;                                                              );
    GLSLC(1,     ITYPE yoff = pos.y + yoffs;                                                              );
    GLSLC(0,                                                                                              );
    GLSLC(1,     DTYPE src;                                                                               );
    GLSLC(0,                                                                                              );
    GLSLC(1,     DTYPE patch_diff;                                                                        );
    GLSLC(1,     DTYPE w;                                                                                 );
    GLSLC(1,     vec4 w_sum;                                                                              );
    GLSLC(1,     vec4 sum;                                                                                );
    GLSLC(0,                                                                                              );
    GLSLC(1,     DTYPE a = DTYPE(0);                                                                      );
    GLSLC(1,     DTYPE b = DTYPE(0);                                                                      );
    GLSLC(1,     DTYPE c = DTYPE(0);                                                                      );
    GLSLC(1,     DTYPE d = DTYPE(0);                                                                      );
    GLSLC(0,                                                                                              );
    GLSLC(1,     bool lt = (pos.x - patch_size[0]) < 0 || (pos.y - patch_size[0]) < 0;                    );
    GLSLC(1,     bool gt = (pos.x + patch_size[0]) >= width[0] || (pos.y + patch_size[0]) >= height[0];   );
    GLSLC(1,     bool oobb = lt || gt;                                                                    );
    GLSLC(0,                                                                                              );
    GLSLC(1,     src[0] = texture(input_img[0], ivec2(pos.x + xoffs[0], pos.y + yoffs[0])).x;             );
    GLSLC(1,     src[1] = texture(input_img[0], ivec2(pos.x + xoffs[1], pos.y + yoffs[1])).x;             );
    GLSLC(1,     src[2] = texture(input_img[0], ivec2(pos.x + xoffs[2], pos.y + yoffs[2])).x;             );
    GLSLC(1,     src[3] = texture(input_img[0], ivec2(pos.x + xoffs[3], pos.y + yoffs[3])).x;             );
    GLSLC(0,                                                                                              );
    GLSLC(1,     if (oobb == false) {                                                                     );
    GLSLC(2,         offset = uint64_t((pos.y - patch_size[0])*int_stride[0]);                            );
    GLSLC(2,         int_src = DataBuffer(uint64_t(integral_data[0]) + 4*4*offset);                       );
    GLSLC(2,         a = int_src.v[-patch_size[0]];                                                       );
    GLSLC(2,         c = int_src.v[+patch_size[0]];                                                       );
    GLSLC(2,         offset = uint64_t((pos.y + patch_size[0])*int_stride[0]);                            );
    GLSLC(2,         int_src = DataBuffer(uint64_t(integral_data[0]) + 4*4*offset);                       );
    GLSLC(2,         b = int_src.v[-patch_size[0]];                                                       );
    GLSLC(2,         d = int_src.v[+patch_size[0]];                                                       );
    GLSLC(1,     }                                                                                        );
    GLSLC(0,                                                                                              );
    GLSLC(1,     patch_diff = (d + a - c - b);                                                            );
    GLSLC(1,     w = exp(-patch_diff / strength[0]);                                                      );
    GLSLC(1,     w_sum[0] = w[0] + w[1] + w[2] + w[3];                                                    );
    GLSLC(1,     sum[0] = dot(w, src);                                                                    );
    GLSLC(0,                                                                                              );
    GLSLC(1,     weights[pos.y*int_stride[0] + pos.x] += w_sum;                                           );
//    GLSLC(1,     sums[pos.y*int_stride[0] + pos.x] += sum;                                                );
    GLSLC(0, }                                                                                            );

    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));

    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));

    return 0;

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static av_cold int init_denoise_pipeline(FFVulkanContext *vkctx, FFVkExecPool *exec,
                                         FFVulkanPipeline *pl, FFVkSPIRVShader *shd,
                                         VkSampler sampler, int planes,
                                         FFVkSPIRVCompiler *spv)
{
    int err;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque;
    FFVulkanDescriptorSetBinding *desc;

    RET(ff_vk_shader_init(pl, shd, "nlmeans_denoise",
                          VK_SHADER_STAGE_COMPUTE_BIT, 0));

    ff_vk_shader_set_compute_sizes(shd, 32, 32, 1);

    GLSLC(0, layout(push_constant, std430) uniform pushConstants {        );
    GLSLC(1,    uvec4 int_stride;                                         );
    GLSLC(1,    uvec4 buffer_stride;                                      );
    GLSLC(1,    uvec4 patch_size;                                         );
    GLSLC(1,    vec4 sigma;                                               );
    GLSLC(0, };                                                           );

//    ff_vk_add_push_constant(pl, 0, sizeof(WeightsPushData),
    //                          VK_SHADER_STAGE_COMPUTE_BIT);

    desc = (FFVulkanDescriptorSetBinding []) {
        {
            .name       = "input_img",
            .type       = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .dimensions = 2,
            .elems      = planes,
            .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
            .samplers   = DUP_SAMPLER(sampler),
        },
        {
            .name       = "output_img",
            .type       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .mem_layout = ff_vk_shader_rep_fmt(vkctx->output_format),
            .mem_quali  = "writeonly",
            .dimensions = 2,
            .elems      = planes,
            .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));

    GLSLC(0, layout(buffer_reference, buffer_reference_align = 32) readonly buffer WeightData {         );
    GLSLC(1,     vec4 weight;                                                                           );
    GLSLC(1,     vec4 sum;                                                                              );
    GLSLC(0, };                                                                                         );

    desc = (FFVulkanDescriptorSetBinding []) {
        {
            .name        = "weights_data",
            .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .mem_quali   = "readonly",
            .mem_layout  = "std430",
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "WeightData weights[4];",
        },
    };

    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 1, 0, 0));

    GLSLC(0, void main()                                                                        );
    GLSLC(0, {                                                                                  );
    GLSLC(1,     ivec2 size;                                                                    );
    GLSLC(1,     const ivec2 pos = ivec2(gl_GlobalInvocationID.xy);                             );
    GLSLF(1,  size = imageSize(output_img[%i]);                                               ,0);
    GLSLC(1,  if (IS_WITHIN(pos, size)) {                                                       );
    GLSLF(2,      vec4 weight = weights[%i][pos.y * buffer_stride[%i] + pos.x].weight;     ,0, 0);
    GLSLC(1,  }                                                                                 );
    GLSLC(0, }                                                                                  );

    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));

    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));

    return 0;

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}


static av_cold int init_filter(AVFilterContext *ctx)
{
    int err;
    int xcnt = 0, ycnt = 0;
    NLMeansVulkanContext *s = ctx->priv;
    FFVulkanContext *vkctx = &s->vkctx;
    const int planes = av_pix_fmt_count_planes(s->vkctx.output_format);
    FFVkSPIRVCompiler *spv;

    spv = ff_vk_spirv_init();
    if (!spv) {
        av_log(ctx, AV_LOG_ERROR, "Unable to initialize SPIR-V compiler!\n");
        return AVERROR_EXTERNAL;
    }

    ff_vk_qf_init(vkctx, &s->qf, VK_QUEUE_COMPUTE_BIT);
    RET(ff_vk_exec_pool_init(vkctx, &s->qf, &s->e, 1, 0, 0, 0, NULL));
    RET(ff_vk_init_sampler(vkctx, &s->sampler, 1, VK_FILTER_LINEAR));

    RET(init_hor_pipeline(vkctx, &s->e, &s->pl_int_hor, &s->shd_int_hor, s->sampler,
                          spv, s->vkctx.output_width, planes, &s->pl_int_hor_n_rows));

    RET(init_ver_pipeline(vkctx, &s->e, &s->pl_int_ver, &s->shd_int_ver, s->sampler,
                          spv, s->vkctx.output_height, planes, &s->pl_int_ver_n_rows));

    RET(init_weights_pipeline(vkctx, &s->e, &s->pl_weights, &s->shd_weights, s->sampler,
                              spv, planes));

#if 0

    RET(init_denoise_pipeline(vkctx, &s->e, &s->pl_denoise, &s->shd_denoise, s->sampler,
                              planes, spv));
#endif

    {
        int radius = 3;
        s->nb_offsets = (2*radius + 1)*(2*radius + 1) - 1;
        s->xoffsets = av_malloc(s->nb_offsets*sizeof(*s->xoffsets));
        s->yoffsets = av_malloc(s->nb_offsets*sizeof(*s->yoffsets));

        for (int x = -radius; x <= radius; x++) {
            for (int y = -radius; y <= radius; y++) {
                if (!x && !y)
                    continue;

                s->xoffsets[xcnt++] = x;
                s->yoffsets[ycnt++] = y;
            }
        }

        av_log(ctx, AV_LOG_VERBOSE, "Filter initialized, %i x/y offsets, %i dispatches\n",
               s->nb_offsets, ((3*s->nb_offsets) / 4) + 1);
    }

    s->initialized = 1;

    return 0;

fail:
    if (spv)
        spv->uninit(&spv);

    return err;
}

static int horizontal_pass(NLMeansVulkanContext *s, FFVkExecContext *exec,
                           int xoffs[4], int yoffs[4],
                           FFVkBuffer *integral_vk, FFVkBuffer *line_vk, FFVkBuffer *state_vk)
{
    int err;
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanFunctions *vk = &vkctx->vkfn;
    VkBufferMemoryBarrier2 buf_bar[8];
    int nb_buf_bar = 0;

    /* Horizontal pipeline */
    ff_vk_exec_bind_pipeline(vkctx, exec, &s->pl_int_hor);

    /* Push data */
    ff_vk_update_push_exec(vkctx, exec, &s->pl_int_hor, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(HorizontalPushData), &(HorizontalPushData) {
                               { vkctx->output_height, 0, 0, 0, },
                               { vkctx->output_width, 0, 0, 0, },
                               { xoffs[0], xoffs[1], xoffs[2], xoffs[3] },
                               { yoffs[0], yoffs[1], yoffs[2], yoffs[3] },
                               line_vk->address,
                               state_vk->address,
                               { integral_vk->address },
                           });

    /* Buffer prep/sync */
    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = integral_vk->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = integral_vk->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = integral_vk->buf,
        .size = integral_vk->size,
        .offset = 0,
    };
    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = line_vk->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = line_vk->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = line_vk->buf,
        .size = line_vk->size,
        .offset = 0,
    };
    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = state_vk->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = state_vk->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = state_vk->buf,
        .size = state_vk->size,
        .offset = 0,
    };

    vk->CmdPipelineBarrier2KHR(exec->buf, &(VkDependencyInfo) {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
            .pBufferMemoryBarriers = buf_bar,
            .bufferMemoryBarrierCount = nb_buf_bar,
        });

    integral_vk->stage = buf_bar[0].dstStageMask;
    integral_vk->access = buf_bar[0].dstAccessMask;
    line_vk->stage = buf_bar[1].dstStageMask;
    line_vk->access = buf_bar[1].dstAccessMask;
    state_vk->stage = buf_bar[2].dstStageMask;
    state_vk->access = buf_bar[2].dstAccessMask;

    /* End of horizontal pass */
    vk->CmdDispatch(exec->buf,
                    FFALIGN(vkctx->output_width,
                            s->pl_int_hor.wg_size[0]*s->pl_int_hor_n_rows)/
                           (s->pl_int_hor.wg_size[0]*s->pl_int_hor_n_rows),
                    1,
                    s->pl_int_hor.wg_size[2]);

    return 0;
}

static int vertical_pass(NLMeansVulkanContext *s, FFVkExecContext *exec,
                         FFVkBuffer *integral_vk, FFVkBuffer *line_vk, FFVkBuffer *state_vk)
{
    int err;
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanFunctions *vk = &vkctx->vkfn;
    VkBufferMemoryBarrier2 buf_bar[8];
    int nb_buf_bar = 0;

    /* Vertical pass pipeline */
    ff_vk_exec_bind_pipeline(vkctx, exec, &s->pl_weights);

    /* Push data */
    ff_vk_update_push_exec(vkctx, exec, &s->pl_weights, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(VerticalPushData), &(VerticalPushData) {
                               { vkctx->output_width, 0, 0, 0, },
                               { vkctx->output_width, 0, 0, 0, },
                               line_vk->address,
                               state_vk->address,
                               { integral_vk->address },
                           });

    /* Buffer prep/sync */
    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = integral_vk->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = integral_vk->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = integral_vk->buf,
        .size = integral_vk->size,
        .offset = 0,
    };
    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = line_vk->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = line_vk->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = line_vk->buf,
        .size = line_vk->size,
        .offset = 0,
    };
    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = state_vk->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = state_vk->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = state_vk->buf,
        .size = state_vk->size,
        .offset = 0,
    };

    vk->CmdPipelineBarrier2KHR(exec->buf, &(VkDependencyInfo) {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
            .pBufferMemoryBarriers = buf_bar,
            .bufferMemoryBarrierCount = nb_buf_bar,
        });

    integral_vk->stage = buf_bar[0].dstStageMask;
    integral_vk->access = buf_bar[0].dstAccessMask;
    line_vk->stage = buf_bar[1].dstStageMask;
    line_vk->access = buf_bar[1].dstAccessMask;
    state_vk->stage = buf_bar[2].dstStageMask;
    state_vk->access = buf_bar[2].dstAccessMask;

    /* End of vertical/weights pass */
    vk->CmdDispatch(exec->buf,
                    FFALIGN(vkctx->output_height,
                            s->pl_int_ver.wg_size[0]*s->pl_int_ver_n_rows)/
                           (s->pl_int_ver.wg_size[0]*s->pl_int_ver_n_rows),
                    1,
                    s->pl_int_ver.wg_size[2]);

    return 0;
}

static int weights_pass(NLMeansVulkanContext *s, FFVkExecContext *exec,
                        int xoffs[4], int yoffs[4], int patch[4],
                        FFVkBuffer *integral_vk, FFVkBuffer *weights_vk)
{
    int err;
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanFunctions *vk = &vkctx->vkfn;
    VkBufferMemoryBarrier2 buf_bar[8];
    int nb_buf_bar = 0;

    /* Weights pass pipeline */
    ff_vk_exec_bind_pipeline(vkctx, exec, &s->pl_weights);

    /* Push data */
    ff_vk_update_push_exec(vkctx, exec, &s->pl_weights, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(WeightsPushData), &(WeightsPushData) {
                               { vkctx->output_width, 0, 0, 0, },
                               { xoffs[0], xoffs[1], xoffs[2], xoffs[3] },
                               { yoffs[0], yoffs[1], yoffs[2], yoffs[3] },
                               { patch[0], patch[1], patch[2], patch[3] },
                               { integral_vk->address },
                           });

    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = integral_vk->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = integral_vk->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = integral_vk->buf,
        .size = integral_vk->size,
        .offset = 0,
    };
    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = weights_vk->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = weights_vk->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = weights_vk->buf,
        .size = weights_vk->size,
        .offset = 0,
    };

    vk->CmdPipelineBarrier2KHR(exec->buf, &(VkDependencyInfo) {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
            .pBufferMemoryBarriers = buf_bar,
            .bufferMemoryBarrierCount = nb_buf_bar,
        });
    integral_vk->stage = buf_bar[0].dstStageMask;
    integral_vk->access = buf_bar[0].dstAccessMask;
    weights_vk->stage = buf_bar[2].dstStageMask;
    weights_vk->access = buf_bar[2].dstAccessMask;

    /* End of weights pass */
    vk->CmdDispatch(exec->buf,
                    FFALIGN(vkctx->output_width,  s->pl_weights.wg_size[0])/s->pl_weights.wg_size[0],
                    FFALIGN(vkctx->output_height, s->pl_weights.wg_size[1])/s->pl_weights.wg_size[1],
                    s->pl_weights.wg_size[2]);

    return 0;
}

static int nlmeans_vulkan_filter_frame(AVFilterLink *link, AVFrame *in)
{
    int err;
    AVFrame *out = NULL;
    AVFilterContext *ctx = link->dst;
    NLMeansVulkanContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];

    AVBufferRef *weights_buf;
    AVBufferRef *sums_buf;
    AVBufferRef *integral_buf;
    AVBufferRef *line_buf;
    AVBufferRef *state_buf;

    FFVkBuffer *weights_vk;
    FFVkBuffer *sums_vk;
    FFVkBuffer *integral_vk;
    FFVkBuffer *line_vk;
    FFVkBuffer *state_vk;

    int *xoffs, *yoffs;

    FFVkExecContext *exec;
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanFunctions *vk = &vkctx->vkfn;
    VkImageView in_views[AV_NUM_DATA_POINTERS];
    VkImageView out_views[AV_NUM_DATA_POINTERS];
    VkImageMemoryBarrier2 img_bar[8];
    int nb_img_bar = 0;

    if (!s->initialized)
        RET(init_filter(ctx));

    /* Buffers */
    err = ff_vk_get_pooled_buffer(&s->vkctx, &s->integral_buf_pool, &integral_buf,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  NULL, outlink->w * outlink->h * 4 * 4 * 2,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0)
        return err;
    integral_vk = (FFVkBuffer *)integral_buf->data;

    err = ff_vk_get_pooled_buffer(&s->vkctx, &s->line_buf_pool, &line_buf,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  NULL, outlink->w * 4 * 4 * 2,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0)
        return err;
    line_vk = (FFVkBuffer *)line_buf->data;

    err = ff_vk_get_pooled_buffer(&s->vkctx, &s->state_buf_pool, &state_buf,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  NULL, outlink->w * outlink->h * 4,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0)
        return err;
    state_vk = (FFVkBuffer *)state_buf->data;

    err = ff_vk_get_pooled_buffer(&s->vkctx, &s->weights_buf_pool, &weights_buf,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  NULL, outlink->w * outlink->h * 4,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0)
        return err;
    weights_vk = (FFVkBuffer *)weights_buf->data;

    err = ff_vk_get_pooled_buffer(&s->vkctx, &s->sums_buf_pool, &sums_buf,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  NULL, outlink->w * outlink->h * 4,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0)
        return err;
    sums_vk = (FFVkBuffer *)sums_buf->data;

    /* Output frame */
    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    /* Execution context */
    exec = ff_vk_exec_get(&s->e);
    ff_vk_exec_start(vkctx, exec);

    /* Dependencies */
    RET(ff_vk_exec_add_dep_frame(vkctx, exec, in,  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT));
    RET(ff_vk_exec_add_dep_frame(vkctx, exec, out, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT));
    RET(ff_vk_exec_add_dep_buf(vkctx, exec, &integral_buf, 1, 0));
    RET(ff_vk_exec_add_dep_buf(vkctx, exec, &line_buf,     1, 0));
    RET(ff_vk_exec_add_dep_buf(vkctx, exec, &state_buf,    1, 0));
    RET(ff_vk_exec_add_dep_buf(vkctx, exec, &weights_buf,  1, 0));
    RET(ff_vk_exec_add_dep_buf(vkctx, exec, &sums_buf,     1, 0));

    /* Input frame prep */
    RET(ff_vk_create_imageviews(vkctx, exec, in_views, in));
    ff_vk_update_descriptor_img_array(vkctx, &s->pl_int_hor, exec, in, in_views, 0, 0,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      s->sampler);
    ff_vk_frame_barrier(vkctx, exec, in, img_bar, &nb_img_bar,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_QUEUE_FAMILY_IGNORED);

    /* Output frame prep */
    RET(ff_vk_exec_add_dep_frame(vkctx, exec, out, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT));
    RET(ff_vk_create_imageviews(vkctx, exec, out_views, out));
    ff_vk_frame_barrier(vkctx, exec, out, img_bar, &nb_img_bar,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_QUEUE_FAMILY_IGNORED);

    vk->CmdPipelineBarrier2KHR(exec->buf, &(VkDependencyInfo) {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
            .pImageMemoryBarriers = img_bar,
            .imageMemoryBarrierCount = nb_img_bar,
        });

    /* Update horizontal descriptors */
    ff_vk_update_descriptor_img_array(vkctx, &s->pl_int_hor, exec, in, in_views, 0, 0,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      s->sampler);

    /* Update weights descriptors */
    ff_vk_update_descriptor_img_array(vkctx, &s->pl_weights, exec, in, in_views, 0, 0,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      s->sampler);
    ff_vk_update_descriptor_img_array(vkctx, &s->pl_weights, exec, out, out_views, 0, 1,
                                      VK_IMAGE_LAYOUT_GENERAL, s->sampler);

    RET(ff_vk_set_descriptor_buffer(&s->vkctx, &s->pl_weights, exec, 0, 2, 0,
                                    weights_vk->address, weights_buf->size,
                                    VK_FORMAT_UNDEFINED));
    RET(ff_vk_set_descriptor_buffer(&s->vkctx, &s->pl_weights, exec, 0, 3, 0,
                                    sums_vk->address, sums_buf->size,
                                    VK_FORMAT_UNDEFINED));

    xoffs = s->xoffsets;
    yoffs = s->yoffsets;

    RET(horizontal_pass(s, exec, xoffs, yoffs, integral_vk, line_vk, state_vk));
    RET(vertical_pass(s, exec, integral_vk, line_vk, state_vk));
    RET(weights_pass(s, exec, xoffs, yoffs, s->patch_size, integral_vk, weights_vk));

    err = ff_vk_exec_submit(vkctx, exec);
    if (err < 0)
        return err;
    ff_vk_exec_wait(vkctx, exec);

    err = av_frame_copy_props(out, in);
    if (err < 0)
        goto fail;

    av_frame_free(&in);

    return ff_filter_frame(outlink, out);

fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return err;
}

static void nlmeans_vulkan_uninit(AVFilterContext *avctx)
{
    NLMeansVulkanContext *s = avctx->priv;
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanFunctions *vk = &vkctx->vkfn;

    ff_vk_exec_pool_free(vkctx, &s->e);
    ff_vk_pipeline_free(vkctx, &s->pl_int_hor);
    ff_vk_shader_free(vkctx, &s->shd_int_hor);
    ff_vk_pipeline_free(vkctx, &s->pl_weights);
    ff_vk_shader_free(vkctx, &s->shd_weights);
    ff_vk_pipeline_free(vkctx, &s->pl_denoise);
    ff_vk_shader_free(vkctx, &s->shd_denoise);

    if (s->sampler)
        vk->DestroySampler(vkctx->hwctx->act_dev, s->sampler,
                           vkctx->hwctx->alloc);

    ff_vk_uninit(&s->vkctx);

    s->initialized = 0;
}

#define OFFSET(x) offsetof(NLMeansVulkanContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption nlmeans_vulkan_options[] = {
    { "s",  "denoising strength",                OFFSET(sigma),            AV_OPT_TYPE_DOUBLE, { .dbl = 1.0   }, 1.0, 30.0, FLAGS },
//    { "p",  "patch size",                        OFFSET(patch_size),       AV_OPT_TYPE_INT,    { .i64 = 2*3+1 },   0,   99, FLAGS },
//    { "pc", "patch size for chroma planes",      OFFSET(patch_size_uv),    AV_OPT_TYPE_INT,    { .i64 = 0     },   0,   99, FLAGS },
//    { "r",  "research window",                   OFFSET(research_size),    AV_OPT_TYPE_INT,    { .i64 = 7*2+1 },   0,   99, FLAGS },
//    { "rc", "research window for chroma planes", OFFSET(research_size_uv), AV_OPT_TYPE_INT,    { .i64 = 0     },   0,   99, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(nlmeans_vulkan);

static const AVFilterPad nlmeans_vulkan_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = &nlmeans_vulkan_filter_frame,
        .config_props = &ff_vk_filter_config_input,
    },
};

static const AVFilterPad nlmeans_vulkan_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_vk_filter_config_output,
    },
};

const AVFilter ff_vf_nlmeans_vulkan = {
    .name           = "nlmeans_vulkan",
    .description    = NULL_IF_CONFIG_SMALL("Non-local means denoiser (Vulkan)"),
    .priv_size      = sizeof(NLMeansVulkanContext),
    .init           = &ff_vk_filter_init,
    .uninit         = &nlmeans_vulkan_uninit,
    FILTER_INPUTS(nlmeans_vulkan_inputs),
    FILTER_OUTPUTS(nlmeans_vulkan_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_VULKAN),
    .priv_class     = &nlmeans_vulkan_class,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
