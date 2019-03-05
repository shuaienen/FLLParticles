/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
This file contains simple wrapper functions that call the CUDA kernels
*/

#include <helper_cuda.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "helper_math.h"
#include "math_constants.h"

//cuda的库
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "ParticleSystem_cuda.cuh"

texture<float4, 3, cudaReadModeElementType> noiseTex;

// simulation parameters
__constant__ SimParams params;

// look up in 3D noise texture
__device__ float3 noise3D(float3 p)
{
	float4 n = tex3D(noiseTex, p.x, p.y, p.z);
	return make_float3(n.x, n.y, n.z);
}

// integrate particle attributes
// 更新粒子
struct integrate_functor
{
	float deltaTime;
	float d_offset;
	float lifetime;



	__host__ __device__
		integrate_functor(float delta_time, float offset, float life_time) : deltaTime(delta_time), d_offset(offset), lifetime(life_time) {}

	template <typename Tuple>
	__device__
		void operator()(Tuple t)
	{
		volatile float4 posData = thrust::get<2>(t);

		//volatile float4 velData1 = thrust::get<4>(t);
		//volatile float4 velData2 = thrust::get<5>(t);
		volatile float4 colorData1 = thrust::get<3>(t);
		volatile float4 colorData2 = thrust::get<4>(t);

		float3 pos = make_float3(posData.x, posData.y, posData.z);

		//float3 vel1 = make_float3(velData1.x, velData1.y, velData1.z);
		//float3 vel2 = make_float3(velData2.x, velData2.y, velData2.z);
		//float3 color1 = make_float3(colorData1.x, colorData1.y, colorData1.z);
		//float3 color2 = make_float3(colorData2.x, colorData2.y, colorData2.z);

		float4 color1 = make_float4(colorData1.x, colorData1.y, colorData1.z, colorData1.w);
		float4 color2 = make_float4(colorData2.x, colorData2.y, colorData2.z, colorData2.w);

		

		//根据Offset设置渐变
		//float3 del_vel = vel2 - vel1;
		//float3 vel = vel1 + del_vel * d_offset;
		//float3 vel = vel2;
		//float3 del_color = color2 - color1;
		//float3 color = color1 + del_color * d_offset;
		float4 del_color = color2 - color1;
		float4 color = color1 + del_color * d_offset;

		color.w = color.w * 0.1f;
		// update particle age
		//float age = posData.w;
		//age = 0;
		//float lifetime = velData.w;

		//if (age < lifetime)
		//{
		//	age += deltaTime;
		//}
		//else
		//{
		//	age = lifetime;
		//}

		//float phase = (lifetime > 0.0) ? (age / lifetime) : 1.0; // [0, 1]
		//float fade = 1.0 - phase;
		//if (vel.y <0)
		//{
		//	fade = phase;
		//}

		// apply accelerations
		//vel += params.gravity * deltaTime;
		//vel += g * deltaTime;

		// apply procedural noise
		//float3 noise = noise3D(pos*params.noiseFreq + params.time*params.noiseSpeed);
		//vel += noise * params.noiseAmp;

		// new position = old position + velocity * deltaTime
		//pos += vel * deltaTime;

		//vel *= params.globalDamping;

		// store new position and velocity
		thrust::get<0>(t) = make_float4(pos,1);
		//thrust::get<1>(t) = make_float4(vel, lifetime);
		//thrust::get<2>(t) = make_float4(color, fade);
		thrust::get<1>(t) = color;
	}
};

struct calcDepth_functor
{
	float3 sortVector;

	__host__ __device__
		calcDepth_functor(float3 sort_vector) : sortVector(sort_vector) {}

	template <typename Tuple>
	__host__ __device__
		void operator()(Tuple t)
	{
		volatile float4 p = thrust::get<0>(t);
		float key = -dot(make_float3(p.x, p.y, p.z), sortVector); // project onto sort vector
		thrust::get<1>(t) = key;
	}
};


extern "C"
{

    cudaArray *noiseArray;

    void initCuda(bool bUseGL)
    {
        if (bUseGL)
        {
            cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
        }
        else
        {
            cudaSetDevice(gpuGetMaxGflopsDeviceId());
        }
    }

	// copy parameters to constant memory
    void setParameters(SimParams *hostParams)
    {
        
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    //Round a / b to nearest higher integer value
    int iDivUp(int a, int b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(int n, int blockSize, int &numBlocks, int &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    inline float frand()
    {
        return rand() / (float) RAND_MAX;
    }

    // create 3D texture containing random values
    void createNoiseTexture(int w, int h, int d)
    {
        cudaExtent size = make_cudaExtent(w, h, d);
        uint elements = (uint) size.width*size.height*size.depth;

        float *volumeData = (float *)malloc(elements*4*sizeof(float));
        float *ptr = volumeData;

        for (uint i=0; i<elements; i++)
        {
            *ptr++ = frand()*2.0f-1.0f;
            *ptr++ = frand()*2.0f-1.0f;
            *ptr++ = frand()*2.0f-1.0f;
            *ptr++ = frand()*2.0f-1.0f;
        }


        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        checkCudaErrors(cudaMalloc3DArray(&noiseArray, &channelDesc, size));

        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr   = make_cudaPitchedPtr((void *)volumeData, size.width*sizeof(float4), size.width, size.height);
        copyParams.dstArray = noiseArray;
        copyParams.extent   = size;
        copyParams.kind     = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));

        free(volumeData);

        // set texture parameters
        noiseTex.normalized = true;                      // access with normalized texture coordinates
        noiseTex.filterMode = cudaFilterModeLinear;      // linear interpolation
        noiseTex.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
        noiseTex.addressMode[1] = cudaAddressModeWrap;
        noiseTex.addressMode[2] = cudaAddressModeWrap;

        // bind array to 3D texture
        checkCudaErrors(cudaBindTextureToArray(noiseTex, noiseArray, channelDesc));
    }

    void
    integrateSystem(float4 *oldPos, float4 *newPos,
                    //float4 *newVel,
					//float4 *Vel1, float4 *Vel2, 
					float4 *newColor,
					float4 *Color1, float4 *Color2, 
                    float deltaTime, float offset,
                    int numParticles, float lifetime)
    {
        thrust::device_ptr<float4> d_newPos(newPos);
       // thrust::device_ptr<float4> d_newVel(newVel);
		thrust::device_ptr<float4> d_newColor(newColor);
        thrust::device_ptr<float4> d_oldPos(oldPos);

		//thrust::device_ptr<float4> d_Vel1(Vel1);
		//thrust::device_ptr<float4> d_Vel2(Vel2);
		thrust::device_ptr<float4> d_Color1(Color1);
		thrust::device_ptr<float4> d_Color2(Color2);

        thrust::for_each(
			//begin
            thrust::make_zip_iterator(thrust::make_tuple(d_newPos, /*d_newVel,*/ d_newColor, d_oldPos, /*d_Vel1, d_Vel2,*/ d_Color1, d_Color2)),
			//end
            thrust::make_zip_iterator(thrust::make_tuple(d_newPos+numParticles, /*d_newVel+numParticles,*/ d_newColor+numParticles, d_oldPos+numParticles, /*d_Vel1+numParticles, d_Vel2+numParticles,*/ d_Color1+numParticles, d_Color2+numParticles)),
			//operation
            integrate_functor(deltaTime, offset, lifetime));
    }

	// 根据位置和sortVector（half_vector），得到所有numParticles的排序keys和index
    void
    calcDepth(float4  *pos,
              float   *keys,        // output key为各个pos和half_vector dot后的结果
              uint    *indices,     // output
              float3   sortVector,
              int      numParticles)
    {
        thrust::device_ptr<float4> d_pos(pos);
        thrust::device_ptr<float> d_keys(keys);
        thrust::device_ptr<uint> d_indices(indices);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_pos, d_keys)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos+numParticles, d_keys+numParticles)),
            calcDepth_functor(sortVector));

        thrust::sequence(d_indices, d_indices + numParticles);
    }

	//根据sortKeys排序indices
    void sortParticles(float *sortKeys, uint *indices, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<float>(sortKeys),
                            thrust::device_ptr<float>(sortKeys + numParticles),
                            thrust::device_ptr<uint>(indices));
    }

}   // extern "C"
