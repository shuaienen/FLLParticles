#ifndef PARTICLESYSTEM_CUDA_H
#define PARTICLESYSTEM_CUDA_H

#include "vector_types.h"

typedef unsigned int uint;

struct SimParams
{
	float3 gravity;
	float globalDamping;
	float noiseFreq;
	float noiseAmp;
	float3 cursorPos;

	float time;
	float3 noiseSpeed;
};

struct float4x4
{
	float m[16];
};

extern "C"
{
    void initCuda(bool bUseGL);
    void setParameters(SimParams *hostParams);
    void createNoiseTexture(int w, int h, int d);

    void
    integrateSystem(float4 *oldPos, float4 *newPos,
                    //float4 *newVel,
					//float4 *Vel1, float4 *Vel2, 
					float4 *newColor,
					float4 *Color1, float4 *Color2, 
                    float deltaTime, float offset,
                    int numParticles, float lifetime);

    void
    calcDepth(float4  *pos,
              float   *keys,        // output
              uint    *indices,     // output
              float3   sortVector,
              int      numParticles);

    void sortParticles(float *sortKeys, uint *indices, uint numParticles);
}

#endif
