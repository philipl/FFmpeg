/*
 * Copyright (C) 2018 Philip Langdale <philipl@overt.org>
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

__device__ static const unsigned short coef_lf[2] = { 4309, 213 };
__device__ static const unsigned short coef_hf[3] = { 5570, 3801, 1016 };
__device__ static const unsigned short coef_sp[2] = { 5077, 981 };

template<typename T>
__inline__ __device__ T max3(T a, T b, T c)
{
    T x = max(a, b);
    return max(x, c);
}

template<typename T>
__inline__ __device__ T min3(T a, T b, T c)
{
    T x = min(a, b);
    return min(x, c);
}

template<typename T>
__inline__ __device__ T clip(T a, T min, T max)
{
    if (a < min) {
        return min;
    } else if (a > max) {
        return max;
    } else {
        return a;
    }
}

template<typename T>
__inline__ __device__ T filter(T A, T B, T C, T D,
                               T a, T b, T c, T d, T e, T f, T g,
                               T h, T i, T j, T k, T l, T m, T n,
                               int clip_max)
{
    T final;

    int fc = C;
    int fd = (c + l) >> 1;
    int fe = B;

    int temporal_diff0 = abs(c - l);
    int temporal_diff1 = (abs(g - fc) + abs(f - fe)) >> 1;
    int temporal_diff2 = (abs(i - fc) + abs(h - fe)) >> 1;
    int diff = max3(temporal_diff0 >> 1, temporal_diff1, temporal_diff2);

    if (!diff) {
        final = fd;
    } else {
        int fb = ((d + m) >> 1) - fc;
        int ff = ((c + l) >> 1) - fe;
        int dc = fd - fc;
        int de = fd - fe;
        int mmax = max3(de, dc, min(fb, ff));
        int mmin = min3(de, dc, max(fb, ff));
        diff = max3(diff, mmin, -mmax);

        int interpol;
        if (abs(fc - fe) > temporal_diff0) {
            interpol = (((coef_hf[0] * (c + l)
                - coef_hf[1] * (d + m + b + k)
                + coef_hf[2] * (e + n + a + j)) >> 2)
                + coef_lf[0] * (C + B) - coef_lf[1] * (D + A)) >> 13;
        } else {
            interpol = (coef_sp[0] * (C + B) - coef_sp[1] * (D + A)) >> 13;
        }
        if (interpol > fd + diff) {
            interpol = fd + diff;
        } else if (interpol < fd - diff) {
            interpol = fd - diff;
        }
        final = clip(interpol, 0, clip_max);
    }

    return final;
}

template<typename T>
__inline__ __device__ void bwdif_single(T *dst,
                                        cudaTextureObject_t prev,
                                        cudaTextureObject_t cur,
                                        cudaTextureObject_t next,
                                        int dst_width, int dst_height, int dst_pitch,
                                        int src_width, int src_height,
                                        int parity, int tff, bool skip_spatial_check,
                                        int clip_max)
{
    // Identify location
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

    if (xo >= dst_width || yo >= dst_height) {
        return;
    }

    // Don't modify the primary field
    if (yo % 2 == parity) {
      dst[yo*dst_pitch+xo] = tex2D<T>(cur, xo, yo);
      return;
    }

    T A = tex2D<T>(cur, xo, yo + 3);
    T B = tex2D<T>(cur, xo, yo + 1);
    T C = tex2D<T>(cur, xo, yo - 1);
    T D = tex2D<T>(cur, xo, yo - 3);

    // Calculate temporal prediction
    int is_second_field = !(parity ^ tff);

    cudaTextureObject_t prev2 = prev;
    cudaTextureObject_t prev1 = is_second_field ? cur : prev;
    cudaTextureObject_t next1 = is_second_field ? next : cur;
    cudaTextureObject_t next2 = next;

    T a = tex2D<T>(prev2, xo,  yo + 4);
    T b = tex2D<T>(prev2, xo,  yo + 2);
    T c = tex2D<T>(prev2, xo,  yo + 0);
    T d = tex2D<T>(prev2, xo,  yo - 2);
    T e = tex2D<T>(prev2, xo,  yo - 4);
    T f = tex2D<T>(prev1, xo,  yo + 1);
    T g = tex2D<T>(prev1, xo,  yo - 1);
    T h = tex2D<T>(next1, xo,  yo + 1);
    T i = tex2D<T>(next1, xo,  yo - 1);
    T j = tex2D<T>(next2, xo,  yo + 4);
    T k = tex2D<T>(next2, xo,  yo + 2);
    T l = tex2D<T>(next2, xo,  yo + 0);
    T m = tex2D<T>(next2, xo,  yo - 2);
    T n = tex2D<T>(next2, xo,  yo - 4);

    dst[yo*dst_pitch+xo] = filter(A, B, C, D,
                                  a, b, c, d, e, f, g,
                                  h, i, j, k, l, m, n,
                                  clip_max);
}

template <typename T>
__inline__ __device__ void bwdif_double(T *dst,
                                        cudaTextureObject_t prev,
                                        cudaTextureObject_t cur,
                                        cudaTextureObject_t next,
                                        int dst_width, int dst_height, int dst_pitch,
                                        int src_width, int src_height,
                                        int parity, int tff, bool skip_spatial_check,
                                        int clip_max)
{
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

    if (xo >= dst_width || yo >= dst_height) {
        return;
    }

    if (yo % 2 == parity) {
      // Don't modify the primary field
      dst[yo*dst_pitch+xo] = tex2D<T>(cur, xo, yo);
      return;
    }

    T A = tex2D<T>(cur, xo, yo + 3);
    T B = tex2D<T>(cur, xo, yo + 1);
    T C = tex2D<T>(cur, xo, yo - 1);
    T D = tex2D<T>(cur, xo, yo - 3);

    // Calculate temporal prediction
    int is_second_field = !(parity ^ tff);

    cudaTextureObject_t prev2 = prev;
    cudaTextureObject_t prev1 = is_second_field ? cur : prev;
    cudaTextureObject_t next1 = is_second_field ? next : cur;
    cudaTextureObject_t next2 = next;

    T a = tex2D<T>(prev2, xo,  yo + 4);
    T b = tex2D<T>(prev2, xo,  yo + 2);
    T c = tex2D<T>(prev2, xo,  yo + 0);
    T d = tex2D<T>(prev2, xo,  yo - 2);
    T e = tex2D<T>(prev2, xo,  yo - 4);
    T f = tex2D<T>(prev1, xo,  yo + 1);
    T g = tex2D<T>(prev1, xo,  yo - 1);
    T h = tex2D<T>(next1, xo,  yo + 1);
    T i = tex2D<T>(next1, xo,  yo - 1);
    T j = tex2D<T>(next2, xo,  yo + 4);
    T k = tex2D<T>(next2, xo,  yo + 2);
    T l = tex2D<T>(next2, xo,  yo + 0);
    T m = tex2D<T>(next2, xo,  yo - 2);
    T n = tex2D<T>(next2, xo,  yo - 4);

    T final;
    final.x = filter(A.x, B.x, C.x, D.x,
                     a.x, b.x, c.x, d.x, e.x, f.x, g.x,
                     h.x, i.x, j.x, k.x, l.x, m.x, n.x,
                     clip_max);
    final.y = filter(A.y, B.y, C.y, D.y,
                     a.y, b.y, c.y, d.y, e.y, f.y, g.y,
                     h.y, i.y, j.y, k.y, l.y, m.y, n.y,
                     clip_max);




    dst[yo*dst_pitch+xo] = final;
}

extern "C" {

__global__ void bwdif_uchar(unsigned char *dst,
                            cudaTextureObject_t prev,
                            cudaTextureObject_t cur,
                            cudaTextureObject_t next,
                            int dst_width, int dst_height, int dst_pitch,
                            int src_width, int src_height,
                            int parity, int tff, bool skip_spatial_check,
                            int clip_max)
{
    bwdif_single(dst, prev, cur, next,
                 dst_width, dst_height, dst_pitch,
                 src_width, src_height,
                 parity, tff, skip_spatial_check,
                 clip_max);
}

__global__ void bwdif_ushort(unsigned short *dst,
                            cudaTextureObject_t prev,
                            cudaTextureObject_t cur,
                            cudaTextureObject_t next,
                            int dst_width, int dst_height, int dst_pitch,
                            int src_width, int src_height,
                            int parity, int tff, bool skip_spatial_check,
                            int clip_max)
{
    bwdif_single(dst, prev, cur, next,
                 dst_width, dst_height, dst_pitch,
                 src_width, src_height,
                 parity, tff, skip_spatial_check,
                 clip_max);
}

__global__ void bwdif_uchar2(uchar2 *dst,
                            cudaTextureObject_t prev,
                            cudaTextureObject_t cur,
                            cudaTextureObject_t next,
                            int dst_width, int dst_height, int dst_pitch,
                            int src_width, int src_height,
                            int parity, int tff, bool skip_spatial_check,
                            int clip_max)
{
    bwdif_double(dst, prev, cur, next,
                 dst_width, dst_height, dst_pitch,
                 src_width, src_height,
                 parity, tff, skip_spatial_check,
                 clip_max);
}

__global__ void bwdif_ushort2(ushort2 *dst,
                            cudaTextureObject_t prev,
                            cudaTextureObject_t cur,
                            cudaTextureObject_t next,
                            int dst_width, int dst_height, int dst_pitch,
                            int src_width, int src_height,
                            int parity, int tff, bool skip_spatial_check,
                            int clip_max)
{
    bwdif_double(dst, prev, cur, next,
                 dst_width, dst_height, dst_pitch,
                 src_width, src_height,
                 parity, tff, skip_spatial_check,
                 clip_max);
}

} /* extern "C" */
