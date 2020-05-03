#ifndef OPENMM_VECTORIZE8_NEON_H_
#define OPENMM_VECTORIZE8_NEON_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2013-2014 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "vectorize.h"
#include <sys/auxv.h>
#include <asm/hwcap.h>
#include <arm_neon.h>
#include <cmath>

typedef int int32_t;

// This file defines classes and functions to simplify vectorizing code with NEON.

/**
 * Determine whether ivec8 and fvec8 are supported on this processor.
 */
bool isVec8Supported() {
    // Should also work on ARM32, but untested
    unsigned long features = getauxval(AT_HWCAP);
    return (features & HWCAP_ASIMD) != 0;
}

class ivec8;

/**
 * An eight element vector of floats.
 */
class fvec8 {
public:
    float32x4x2_t val;

    fvec8() {}
    fvec8(float v) {
        val.val[0] = vdupq_n_f32(v);
        val.val[1] = vdupq_n_f32(v);
    }
    fvec8(float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8)
        : val {{{v1, v2, v3, v4}, {v5, v6, v7, v8}}} {}
    fvec8(float32x4x2_t v) : val(v) {}
    fvec8(const float* v) {
        // Only present in GCC version >= 8
        //val = vld1q_f32_x2(v);
        val.val[0] = vld1q_f32(v);
        val.val[1] = vld1q_f32(v + 4);
    }

    /** Create a vector by gathering individual indexes of data from a table. Element i of the vector will
     * be loaded from table[idx[i]].
     * @param table The table from which to do a lookup.
     * @param indexes The indexes to gather.
     */
    fvec8(const float* table, const int idx[8]) {
        val = fvec8(table[idx[0]], table[idx[1]], table[idx[2]], table[idx[3]], table[idx[4]], table[idx[5]], table[idx[6]], table[idx[7]]);
    }

    operator float32x4x2_t() const {
        return val;
    }
    fvec4 lowerVec() const {
        return val.val[0];
    }
    fvec4 upperVec() const {
        return val.val[1];
    }
    void store(float* v) const {
        // Only present in GCC version >= 9
        // vst1q_f32_x2(v, val);
        vst1q_f32(v, val.val[0]);
        vst1q_f32(v + 4, val.val[1]);
    }
    fvec8 operator+(const fvec8& other) const {
        float32x4x2_t t;
        t.val[0] = vaddq_f32(val.val[0], other.lowerVec());
        t.val[1] = vaddq_f32(val.val[1], other.upperVec());
        return t;
    }
    fvec8 operator-(const fvec8& other) const {
        float32x4x2_t t;
        t.val[0] = vsubq_f32(val.val[0], other.lowerVec());
        t.val[1] = vsubq_f32(val.val[1], other.upperVec());
        return t;
    }
    fvec8 operator*(const fvec8& other) const {
        float32x4x2_t t;
        t.val[0] = vmulq_f32(val.val[0], other.lowerVec());
        t.val[1] = vmulq_f32(val.val[1], other.upperVec());
        return t;
    }
    fvec8 operator/(const fvec8& other) const {
        // NEON does not have a divide float-point operator, so we get the reciprocal and multiply.

        float32x4x2_t rec;
        rec.val[0] = vrecpeq_f32(other.lowerVec());
        rec.val[0] = vmulq_f32(vrecpsq_f32(other.lowerVec(), rec.val[0]), rec.val[0]);
        rec.val[0] = vmulq_f32(vrecpsq_f32(other.lowerVec(), rec.val[0]), rec.val[0]);

        rec.val[1] = vrecpeq_f32(other.upperVec());
        rec.val[1] = vmulq_f32(vrecpsq_f32(other.upperVec(), rec.val[1]), rec.val[1]);
        rec.val[1] = vmulq_f32(vrecpsq_f32(other.upperVec(), rec.val[1]), rec.val[1]);

        float32x4x2_t result;
        result.val[0] = vmulq_f32(val.val[0], rec.val[0]);
        result.val[1] = vmulq_f32(val.val[1], rec.val[1]);
        return result;
    }
    void operator+=(const fvec8& other) {
        val.val[0] = vaddq_f32(val.val[0], other.lowerVec());
        val.val[1] = vaddq_f32(val.val[1], other.upperVec());
    }
    void operator-=(const fvec8& other) {
        val.val[0] = vsubq_f32(val.val[0], other.lowerVec());
        val.val[1] = vsubq_f32(val.val[1], other.upperVec());
    }
    void operator*=(const fvec8& other) {
        val.val[0] = vmulq_f32(val.val[0], other.lowerVec());
        val.val[1] = vmulq_f32(val.val[1], other.upperVec());
    }
    void operator/=(const fvec8& other) {
        val = *this / other;
    }
    fvec8 operator-() const {
        float32x4x2_t t;
        t.val[0] = vnegq_f32(val.val[0]);
        t.val[1] = vnegq_f32(val.val[1]);
        return t;
    }
    fvec8 operator&(const fvec8& other) const {
        float32x4x2_t t;
        t.val[0] = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(val.val[0]), vreinterpretq_u32_f32(other.lowerVec())));
        t.val[1] = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(val.val[1]), vreinterpretq_u32_f32(other.upperVec())));
        return t;
    }
    fvec8 operator|(const fvec8& other) const {
        float32x4x2_t t;
        t.val[0] = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(val.val[0]), vreinterpretq_u32_f32(other.lowerVec())));
        t.val[1] = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(val.val[1]), vreinterpretq_u32_f32(other.upperVec())));
        return t;
    }

    ivec8 operator==(const fvec8& other) const;
    ivec8 operator!=(const fvec8& other) const;
    ivec8 operator>(const fvec8& other) const;
    ivec8 operator<(const fvec8& other) const;
    ivec8 operator>=(const fvec8& other) const;
    ivec8 operator<=(const fvec8& other) const;
    operator ivec8() const;

    /**
     * Convert an integer bitmask into a full vector of elements which can be used
     * by the blend function.
     */
    static ivec8 expandBitsToMask(int bitmask);
};

/**
 * An eight element vector of ints.
 */
class ivec8 {
public:

    int32x4x2_t val;

    ivec8() {}
    ivec8(int v) {
        val.val[0] = vdupq_n_s32(v);
        val.val[1] = vdupq_n_s32(v);
    }
    ivec8(int v1, int v2, int v3, int v4, int v5, int v6, int v7, int v8)
        : val {{{v1, v2, v3, v4}, {v5, v6, v7, v8}}} {}
    ivec8(int32x4x2_t v) : val(v) {}
    ivec8(const int* v) {
        // Only present in GCC version >= 8
        //val = vld1q_s32_x2(v);
        val.val[0] = vld1q_s32(v);
        val.val[1] = vld1q_s32(v + 4);
    }
    operator int32x4x2_t() const {
        return val;
    }
    ivec4 lowerVec() const {
        return val.val[0];
    }
    ivec4 upperVec() const {
        return val.val[1];
    }
    void store(int* v) const {
        // Only present in GCC version >= 9
        //vst1q_s32_x2(v, val);
        vst1q_s32(v, val.val[0]);
        vst1q_s32(v + 4, val.val[1]);
    }
    ivec8 operator&(const ivec8& other) const {
        int32x4x2_t t;
        t.val[0] = vandq_s32(val.val[0], other.lowerVec());
        t.val[1] = vandq_s32(val.val[1], other.upperVec());
        return t;
    }
    ivec8 operator|(const ivec8& other) const {
        int32x4x2_t t;
        t.val[0] = vorrq_s32(val.val[0], other.lowerVec());
        t.val[1] = vorrq_s32(val.val[1], other.upperVec());
        return t;
    }
    operator fvec8() const;
};

// Conversion operators.

inline fvec8::operator ivec8() const {
    int32x4x2_t t;
    t.val[0] = vcvtq_s32_f32(val.val[0]);
    t.val[1] = vcvtq_s32_f32(val.val[1]);
    return t;
}

inline ivec8::operator fvec8() const {
    float32x4x2_t t;
    t.val[0] = vcvtq_f32_s32(val.val[0]);
    t.val[1] = vcvtq_f32_s32(val.val[1]);
    return t;
}

inline ivec8 fvec8::expandBitsToMask(int bitmask) {
    return ivec8(bitmask & 1 ? -1 : 0,
                 bitmask & 2 ? -1 : 0,
                 bitmask & 4 ? -1 : 0,
                 bitmask & 8 ? -1 : 0,
                 bitmask & 16 ? -1 : 0,
                 bitmask & 32 ? -1 : 0,
                 bitmask & 64 ? -1 : 0,
                 bitmask & 128 ? -1 : 0);
}

// Comparison operations
ivec8 fvec8::operator==(const fvec8& other) const {
    int32x4x2_t t;
    t.val[0] = vreinterpretq_s32_u32(vceqq_f32(val.val[0], other.lowerVec()));
    t.val[1] = vreinterpretq_s32_u32(vceqq_f32(val.val[1], other.upperVec()));
    return t;
}

ivec8 fvec8::operator!=(const fvec8& other) const {
    // not(equals(val, other))
    int32x4x2_t t;
    t.val[0] = vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(val.val[0], other.lowerVec())));
    t.val[1] = vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(val.val[1], other.upperVec())));
    return t;
}

ivec8 fvec8::operator>(const fvec8& other) const {
    int32x4x2_t t;
    t.val[0] = vreinterpretq_s32_u32(vcgtq_f32(val.val[0], other.lowerVec()));
    t.val[1] = vreinterpretq_s32_u32(vcgtq_f32(val.val[1], other.upperVec()));
    return t;
}

ivec8 fvec8::operator<(const fvec8& other) const {
    int32x4x2_t t;
    t.val[0] = vreinterpretq_s32_u32(vcltq_f32(val.val[0], other.lowerVec()));
    t.val[1] = vreinterpretq_s32_u32(vcltq_f32(val.val[1], other.upperVec()));
    return t;
}

ivec8 fvec8::operator>=(const fvec8& other) const {
    int32x4x2_t t;
    t.val[0] = vreinterpretq_s32_u32(vcgeq_f32(val.val[0], other.lowerVec()));
    t.val[1] = vreinterpretq_s32_u32(vcgeq_f32(val.val[1], other.upperVec()));
    return t;
}

ivec8 fvec8::operator<=(const fvec8& other) const {
    int32x4x2_t t;
    t.val[0] = vreinterpretq_s32_u32(vcleq_f32(val.val[0], other.lowerVec()));
    t.val[1] = vreinterpretq_s32_u32(vcleq_f32(val.val[1], other.upperVec()));
    return t;
}

// Functions that operate on fvec8s.

static inline fvec8 min(const fvec8& v1, const fvec8& v2) {
    float32x4x2_t t;
    t.val[0] = vminq_f32(v1.lowerVec(), v2.lowerVec());
    t.val[1] = vminq_f32(v1.upperVec(), v2.upperVec());
    return t;
}

static inline fvec8 max(const fvec8& v1, const fvec8& v2) {
    float32x4x2_t t;
    t.val[0] = vmaxq_f32(v1.lowerVec(), v2.lowerVec());
    t.val[1] = vmaxq_f32(v1.upperVec(), v2.upperVec());
    return t;
}

static inline fvec8 abs(const fvec8& v) {
    float32x4x2_t t;
    t.val[0] = vabsq_f32(v.lowerVec());
    t.val[1] = vabsq_f32(v.upperVec());
    return t;
}

static inline fvec8 rsqrt(const fvec8& v) {
    float32x4x2_t recipSqrt;
    recipSqrt.val[0] = vrsqrteq_f32(v.lowerVec());
    recipSqrt.val[0] = vmulq_f32(recipSqrt.val[0], vrsqrtsq_f32(vmulq_f32(recipSqrt.val[0], v.lowerVec()), recipSqrt.val[0]));
    recipSqrt.val[0] = vmulq_f32(recipSqrt.val[0], vrsqrtsq_f32(vmulq_f32(recipSqrt.val[0], v.lowerVec()), recipSqrt.val[0]));

    recipSqrt.val[1] = vrsqrteq_f32(v.upperVec());
    recipSqrt.val[1] = vmulq_f32(recipSqrt.val[1], vrsqrtsq_f32(vmulq_f32(recipSqrt.val[1], v.upperVec()), recipSqrt.val[1]));
    recipSqrt.val[1] = vmulq_f32(recipSqrt.val[1], vrsqrtsq_f32(vmulq_f32(recipSqrt.val[1], v.upperVec()), recipSqrt.val[1]));
    return recipSqrt;
}

static inline fvec8 sqrt(const fvec8& v) {
    return rsqrt(v) * v;
}

static inline float dot8(const fvec8& v1, const fvec8& v2) {
    fvec8 result = v1 * v2;
    return (vaddvq_f32(result.lowerVec()) + vaddvq_f32(result.upperVec()));
}

static inline float reduceAdd(const fvec8& v) {
    return (vaddvq_f32(v.lowerVec()) + vaddvq_f32(v.upperVec()));
}

static inline void transpose(const fvec4& in1, const fvec4& in2, const fvec4& in3, const fvec4& in4, const fvec4& in5, const fvec4& in6, const fvec4& in7, const fvec4& in8, fvec8& out1, fvec8& out2, fvec8& out3, fvec8& out4) {
    float32x4x2_t u1, u2;
    u1 = vuzpq_f32(in1, in3);
    u2 = vuzpq_f32(in2, in4);
    float32x4x2_t t1 = vtrnq_f32(u1.val[0], u2.val[0]);
    float32x4x2_t t2 = vtrnq_f32(u1.val[1], u2.val[1]);

    u1 = vuzpq_f32(in5, in7);
    u2 = vuzpq_f32(in6, in8);
    float32x4x2_t t3 = vtrnq_f32(u1.val[0], u2.val[0]);
    float32x4x2_t t4 = vtrnq_f32(u1.val[1], u2.val[1]);

    out1.val.val[0] = t1.val[0];
    out1.val.val[1] = t3.val[0];
    out2.val.val[0] = t2.val[0];
    out2.val.val[1] = t4.val[0];
    out3.val.val[0] = t1.val[1];
    out3.val.val[1] = t3.val[1];
    out4.val.val[0] = t2.val[1];
    out4.val.val[1] = t4.val[1];
}

/** Given a vec4[8] input array, generate 4 vec8 outputs. The first output contains all the first elements
 * the second output the second elements, and so on. Note that the prototype is essentially differing only
 * in output type so it can be overloaded in other SIMD fvec types.
 */
static inline void transpose(const fvec4 in[8], fvec8& out1, fvec8& out2, fvec8& out3, fvec8& out4) {
    transpose(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], out1, out2, out3, out4);
}

static inline void transpose(const fvec8& in1, const fvec8& in2, const fvec8& in3, const fvec8& in4, fvec4& out1, fvec4& out2, fvec4& out3, fvec4& out4, fvec4& out5, fvec4& out6, fvec4& out7, fvec4& out8) {
    float32x4x2_t t1, t2, t3, t4;
    t1 = vuzpq_f32(in1.lowerVec(), in3.lowerVec());
    t2 = vuzpq_f32(in2.lowerVec(), in4.lowerVec());
    t3 = vtrnq_f32(t1.val[0], t2.val[0]);
    t4 = vtrnq_f32(t1.val[1], t2.val[1]);
    out1 = t3.val[0];
    out2 = t4.val[0];
    out3 = t3.val[1];
    out4 = t4.val[1];

    t1 = vuzpq_f32(in1.upperVec(), in3.upperVec());
    t2 = vuzpq_f32(in2.upperVec(), in4.upperVec());
    t3 = vtrnq_f32(t1.val[0], t2.val[0]);
    t4 = vtrnq_f32(t1.val[1], t2.val[1]);
    out5 = t3.val[0];
    out6 = t4.val[0];
    out7 = t3.val[1];
    out8 = t4.val[1];
}

/**
 * Given 4 input vectors of 8 elements, transpose them to form 8 output vectors of 4 elements.
 */
static inline void transpose(const fvec8& in1, const fvec8& in2, const fvec8& in3, const fvec8& in4, fvec4 out[8]) {
    transpose(in1, in2, in3, in4, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
}

// Functions that operate on ivec8s.

static inline bool any(const ivec8& v) {
    return (vmaxvq_u32(vreinterpretq_u32_s32(v.lowerVec())) != 0) || (vmaxvq_u32(vreinterpretq_u32_s32(v.upperVec())) != 0);
}

// Mathematical operators involving a scalar and a vector.

static inline fvec8 operator+(float v1, const fvec8& v2) {
    return fvec8(v1)+v2;
}

static inline fvec8 operator-(float v1, const fvec8& v2) {
    return fvec8(v1)-v2;
}

static inline fvec8 operator*(float v1, const fvec8& v2) {
    return fvec8(v1)*v2;
}

static inline fvec8 operator/(float v1, const fvec8& v2) {
    return fvec8(v1) / v2;
}

// Operations for blending fvec8s based on an ivec8.

static inline fvec8 blend(const fvec8& v1, const fvec8& v2, const ivec8& mask) {
    float32x4x2_t t;
    t.val[0] = vbslq_f32(vreinterpretq_u32_s32(mask.lowerVec()), v2.lowerVec(), v1.lowerVec());
    t.val[1] = vbslq_f32(vreinterpretq_u32_s32(mask.upperVec()), v2.upperVec(), v1.upperVec());
    return t;
}

static inline fvec8 blendZero(const fvec8& v, const ivec8& mask) {
    float32x4x2_t t;
    t.val[0] = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(v.lowerVec()), mask.lowerVec()));
    t.val[1] = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(v.upperVec()), mask.upperVec()));
    return t;
}

static inline ivec8 blendZero(const ivec8& v, const ivec8& mask) {
    return v & mask;
}

static inline fvec8 round(const fvec8& v) {
    fvec8 shift(0x1.0p23f);
    fvec8 absResult = (abs(v) + shift) - shift;
    return blend(v, absResult, ivec8(0x7FFFFFFF));
}

static inline fvec8 floor(const fvec8& v) {
    fvec8 rounded = round(v);
    return rounded + blend(0.0f, -1.0f, rounded > v);
}

static inline fvec8 ceil(const fvec8& v) {
    fvec8 rounded = round(v);
    return rounded + blend(0.0f, 1.0f, rounded < v);
}

/**
 * Given a table of floating-point values and a set of indexes, perform a gather read into a pair
 * of vectors. The first result vector contains the values at the given indexes, and the second
 * result vector contains the values from each respective index+1.
 */
static inline void gatherVecPair(const float* table, const ivec8& index, fvec8& out0, fvec8& out1) {

    // Gather all the separate memory data together. Each vector will have two values
    // which get used, and two which are ultimately discarded.
    fvec4 t0(table + index.lowerVec()[0]);
    fvec4 t1(table + index.lowerVec()[1]);
    fvec4 t2(table + index.lowerVec()[2]);
    fvec4 t3(table + index.lowerVec()[3]);
    fvec4 t4(table + index.upperVec()[0]);
    fvec4 t5(table + index.upperVec()[1]);
    fvec4 t6(table + index.upperVec()[2]);
    fvec4 t7(table + index.upperVec()[3]);

    // Tranposing the 8 vectors above will put all the first elements into one output
    // vector, all the second elements into the next vector and so on.
    fvec8 discard0, discard1;
    transpose(t0, t1, t2, t3, t4, t5, t6, t7, out0, out1, discard0, discard1);
}

/**
 * Given 3 vectors of floating-point data, reduce them to a single 3-element position
 * value by adding all the elements in each vector. Given inputs of:
 *   X0 X1 X2 X3 X4 X5 X6 X7
 *   Y0 Y1 Y2 Y3 Y4 Y5 Y6 Y7
 *   Z0 Z1 Z2 Z3 Z4 Z5 Z6 Z7
 * Each vector of values needs to be summed into a single value, and then stored into
 * the output vector:
 *   output[0] = (X0 + X1 + X2 + ...)
 *   output[1] = (Y0 + Y1 + Y2 + ...)
 *   output[2] = (Z0 + Z1 + Z2 + ...)
 *   output[3] = undefined
 */
static inline fvec4 reduceToVec3(const fvec8& x, const fvec8& y, const fvec8& z) {
    const auto nx = reduceAdd(x);
    const auto ny = reduceAdd(y);
    const auto nz = reduceAdd(z);
    return fvec4(nx, ny, nz, 0.0);
}

#endif /*OPENMM_VECTORIZE_NEON_H_*/
