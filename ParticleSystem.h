#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#include <helper_functions.h>
#include "ParticleSystem_cuda.cuh"
#include "vector_functions.h"
#include "GpuArray.h"
#include "nvMath.h"
#include "BmpReader.h"
#include "TiffReader.h"
#include "Hdf4Reader.h"

using namespace nv;


typedef struct
{
	//float4 *vel;
	float4 *color;
} ParticleInfo;

// CUDA BodySystem: runs on the GPU
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, float particleLiftTime, bool bUseVBO, bool bUseGL, vec2f startLonLat, float spatialRes, vec2f imageSize, float scaleFactor, float alpha, int depthTimes) ;
        ~ParticleSystem();

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            _NUM_CONFIGS
        };

        void step(float deltaTime, float *currentTime, int *currentMonth);
        void depthSort();
        void reset(ParticleConfig config);

        uint getNumParticles()
        {
            return m_numParticles;
        }

        uint getPosBuffer()
        {
            return m_pos.getVbo();
        }
		uint getColorBuffer()
        {
            return m_color.getVbo();
        }
        uint getSortedIndexBuffer()
        {
            return m_indices.getVbo();
        }
        uint *getSortedIndices();

        float getParticleRadius()
        {
            return m_particleRadius;
        }
        SimParams &getParams()
        {
            return m_params;
        }

        void setSorting(bool x)
        {
            m_doDepthSort = x;
        }
        void setModelView(float *m);
        void setSortVector(float3 v)
        {
            m_sortVector = v;
        }

		
		void createColorRamp(GDALColorEntry* colorBegin,GDALColorEntry* colorEnd, GDALColorTable** colorRamp);

		vec3f getColorFromColorRamp(float val,float max,float min);

    protected: // methods
        ParticleSystem() {}

        void _initialize(int numParticlesm, bool bUseGL=true);
        void _free();

        void initGrid(vec2f startLonLat, float spatialRes, vec2f imageSize, float sampleFactor, float jitter,  uint numParticles, float lifetime);

		void initArgoData(int argoCount, uint numParticles);

		void initDepthData();

    protected: // data
        bool m_bInitialized;
        bool m_bUseVBO;
        uint m_numParticles;
		float m_particleLifetime;
		float m_particleAlpha;

		float m_scaleFactor;
		vec2f m_startLonLat;
		float m_spatialRes;
		vec2f m_imageSize;

        float m_particleRadius;

        GpuArray<float4> m_pos;
        //GpuArray<float4> m_vel, m_vel1, m_vel2;
		GpuArray<float4> m_color, m_color1, m_color2;
		
		int m_depthTimes;
        // params
        SimParams m_params;

        float4x4 m_modelView;
        float3 m_sortVector;
        bool m_doDepthSort;

        GpuArray<float> m_sortKeys;
        GpuArray<uint> m_indices;   // sorted indices for rendering

		std::vector<TiffReader *> fluxData;
		std::vector<Hdf4Reader *> argoData;
		std::vector<ParticleInfo *> particleInfos;
		int argoDataNum;
		float depthData[16][326][400];

        StopWatchInterface *m_timer;
        float m_time;
		int d;

		GDALColorTable* m_particleColorRamp;
};

#endif // __PARTICLESYSTEM_H__
