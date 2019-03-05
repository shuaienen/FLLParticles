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


#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>
#include <iostream>
#include <string>
#include <fstream>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "ParticleSystem.h"
#include "ParticleSystem_cuda.cuh"
#include "Hdf4Reader.h"

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

#define FLUX_DATA_NUM  12
#define CYCLE_TIME  200.0

/*
    This handles the particle simulation using CUDA
*/

ParticleSystem::ParticleSystem(uint numParticles, float particleLiftTime, bool bUseVBO, bool bUseGL, vec2f startLonLat, float spatialRes, vec2f imageSize, float scaleFactor, float alpha, int depthTimes) :
m_bInitialized(false),
	m_bUseVBO(bUseVBO),
	m_numParticles(numParticles),
	m_particleRadius(0.02),
	m_doDepthSort(false),
	m_timer(NULL),
	m_time(0.0f),
	m_depthTimes(depthTimes)
	
{
	m_params.gravity = make_float3(0.0f, 0.0f, 0.0f);
	m_params.globalDamping = 1.0f;
	m_params.noiseSpeed = make_float3(0.0f, 0.0f, 0.0f);

	m_particleLifetime = particleLiftTime;
	m_scaleFactor = scaleFactor;
	m_startLonLat = startLonLat;
	m_spatialRes = spatialRes;
	m_imageSize = imageSize;

	m_particleAlpha = alpha;
	m_particleColorRamp = new GDALColorTable();

	_initialize(numParticles, bUseGL);
}

ParticleSystem::~ParticleSystem()
{
    _free();
    m_numParticles = 0;
}

void
ParticleSystem::_initialize(int numParticles, bool bUseGL)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate GPU arrays
    m_pos.alloc(m_numParticles, m_bUseVBO, true);    // create as VBO
	m_color.alloc(m_numParticles, m_bUseVBO, true);    // create as VBO

	// 交替速度与颜色，只读不写，单缓存即可

	m_color1.alloc(m_numParticles, false, false);
	m_color2.alloc(m_numParticles, false, false);

    m_sortKeys.alloc(m_numParticles);
    m_indices.alloc(m_numParticles, m_bUseVBO, false, true); // create as index buffer

    sdkCreateTimer(&m_timer);
    setParameters(&m_params);

    m_bInitialized = true;
	d = 0;
}

void
ParticleSystem::_free()
{
    assert(m_bInitialized);
}

// step the simulation 
void
ParticleSystem::step(float deltaTime, float *currentTime, int *currentMonth)
{
    assert(m_bInitialized);

    m_params.time = m_time;
	//更新cuda中的参数（GPU），其实也就时间变了而已
    setParameters(&m_params);

	if (*currentTime > CYCLE_TIME)
	{
		*currentTime -= CYCLE_TIME;
		*currentMonth = ++d + 1;

		if (d + 1 >= argoDataNum)
		{
			d=0;
			*currentMonth = 1;
		}


		m_color1.setHostPtr(particleInfos[d]->color);
		m_color2.setHostPtr(particleInfos[(d+1) ]->color);
		//复制到显存
		m_color1.copy(GpuArray<float4>::HOST_TO_DEVICE);
		m_color2.copy(GpuArray<float4>::HOST_TO_DEVICE);
	}

	float offset = *currentTime / CYCLE_TIME;

    m_pos.map();
	//m_vel.map();
	m_color.map();

    // integrate particles
	// 在GPU中更新粒子的状态
    integrateSystem(m_pos.getDevicePtr(), m_pos.getDeviceWritePtr(),
                    //m_vel.getDeviceWritePtr(),
					//m_vel1.getDevicePtr(), m_vel2.getDevicePtr(), 
					m_color.getDeviceWritePtr(),
					m_color1.getDevicePtr(), m_color2.getDevicePtr(), 
					deltaTime, offset, m_numParticles, m_particleLifetime);

	
    m_pos.unmap();
	//m_vel.unmap();
	m_color.unmap();

	// double buffer
    m_pos.swap();
	//m_vel.swap();
	m_color.swap();

    m_time += deltaTime;
}

// depth sort the particles //for every step
//SORT到halfvector上，用于slice显示时粒子的调用
//如共32000个粒子，32个slice,则每个slice上显示1000个粒子（按排序先后）
void
ParticleSystem::depthSort()
{
    if (!m_doDepthSort)
    {
        return;
    }

    m_pos.map();
    m_indices.map();

    // calculate depth
	// 根据位置和m_sortVector（half_vector），得到所有m_numParticles的排序keys和index
    calcDepth(m_pos.getDevicePtr(), m_sortKeys.getDevicePtr(), m_indices.getDevicePtr(), m_sortVector, m_numParticles);

    // radix sort
	//根据keys排序index
    sortParticles(m_sortKeys.getDevicePtr(), m_indices.getDevicePtr(), m_numParticles);

    m_pos.unmap();
    m_indices.unmap();
}

uint *
ParticleSystem::getSortedIndices()
{
    // copy sorted indices back to CPU
    m_indices.copy(GpuArray<uint>::DEVICE_TO_HOST);
    return m_indices.getHostPtr();
}

// random float [0, 1]
inline float frand()
{
    return rand() / (float) RAND_MAX;
}

// signed random float [-1, 1]
inline float sfrand()
{
    return frand()*2.0f-1.0f;
}

// random signed vector
inline vec3f svrand()
{
    return vec3f(sfrand(), sfrand(), sfrand());
}

// random point in circle
inline vec2f randCircle()
{
    vec2f r;

    do
    {
        r = vec2f(sfrand(), sfrand());
    }
    while (length(r) > 1.0f);

    return r;
}

// random point in sphere
inline vec3f randSphere()
{
    vec3f r;

    do
    {
        r = svrand();
    }
    while (length(r) > 1.0f);

    return r;
}


void ParticleSystem::initArgoData(int argoCount, uint numParticles)
{
	std::string dataPath = "ncData/roms_avg_2000";
	std::string dataName = "";
	argoDataNum = argoCount*6;
	char numChar[3];
	for (int i = 0; i < argoCount; i++)
	{		
		sprintf(numChar,"%02d",i+1);
		std::string s = numChar;
		dataName = dataPath + s + ".nc";

		for(int j = 0; j < 6; j++)
		{
			Hdf4Reader* hdf4Reader = new Hdf4Reader();
			hdf4Reader->OpenFile(dataName.c_str(),"temp",j);
			argoData.push_back(hdf4Reader);
			ParticleInfo *curParticleInfo = new ParticleInfo();
			curParticleInfo->color = (float4 *) new float4[numParticles];
			//curParticleInfo->vel = (float4 *) new float4[numParticles];
			particleInfos.push_back(curParticleInfo);
		}
	}
	unsigned int height = argoData[0]->GetDataHeight();
	unsigned int width = argoData[0]->GetDataWidth();
	unsigned int depth = argoData[0]->GetDataDepth();

}

void ParticleSystem::initDepthData()
{
	for(int z=0;z<16;z++)
	{
		std::stringstream ss;
		ss<<std::fixed;
		ss<<z+1;
		std::string output;
		ss>>output;ss.clear();
		
		output = output + ".txt";
		output = "ncdata/"+output;
		std::ifstream file(output);
		for(int y=0;y<401;y++)  
			for(int x=0;x<326;x++)  
			{   
				file>>depthData[z][x][y]; 
				//std::cout<<a[i][j]<<endl;  
			} 
		file.close();
	}
}


// initialize in regular grid
// 增加到60个flux +++++
void
ParticleSystem::initGrid(vec2f startLonLat, float spatialRes, vec2f imageSize, float sampleFactor, float jitter, uint numParticles, float lifetime)
{
	srand(1973);

	initDepthData();

	initArgoData(12, numParticles);

	if (argoData.size() == 0)
	{
		return;
	}

	//读取Argo数据宽度、高度、深度
	unsigned int height = argoData[0]->GetDataHeight();
	unsigned int width = argoData[0]->GetDataWidth();
	unsigned int depth = argoData[0]->GetDataDepth();

	//每帧状态
	float4 *posPtr = m_pos.getHostPtr();

	//乱序创建
	std::vector<uint> randomVector;
	for (uint i=0; i<m_numParticles; i++) 
		randomVector.push_back(i); 
	//随机打乱
	std::random_shuffle ( randomVector.begin(), randomVector.end() );
	uint index = 0;
	uint n =0;	uint d = 0;	uint w = 0; uint h = 0;

	int availNum = 0;

	GDALColorEntry* colorBegin = new GDALColorEntry();
	GDALColorEntry* colorEnd = new GDALColorEntry();

	colorBegin->c1 = 0;
	colorBegin->c2 = 0;
	colorBegin->c3 = 255;
	colorBegin->c4 = 255;

	colorEnd->c1 = 255;
	colorEnd->c2 = 0;
	colorEnd->c3 = 0;
	colorEnd->c4 = 255;

	createColorRamp(colorBegin,colorEnd,&m_particleColorRamp);


	float minValue = argoData[0]->GetMinValue();
	float maxValue = argoData[0]->GetMaxValue();

	for(int i=0;i<argoData.size();i++)
	{
		minValue = minValue > argoData[i]->GetMinValue() ? argoData[i]->GetMinValue():minValue;
		maxValue = maxValue < argoData[i]->GetMaxValue() ? argoData[i]->GetMaxValue():maxValue;
	}


	//有一个问题，粒子总数和网格总数不对齐
	//这里仅针对8位bmp
	std::vector<uint>::iterator it=randomVector.begin();
	int resamplePixel = 1;
	//float minVector = 12;
	for (; it!=randomVector.end(); ++it)
	{
		//外部增加一个循环，合理利用所有粒子
		for (n = 0; n< 10000; n++)
		{
			for (d = 0; d <= (depth-1)*m_depthTimes*m_depthTimes; d+=1)
			{
				for (h = 0; h < height/m_depthTimes; h+=resamplePixel)
				{
					for (w = 0; w < width/m_depthTimes; w+=resamplePixel)
					{
						if(it==randomVector.end())
							break;

						//有多少组数据，就要放到多少个info里面。
						for (int argoIndex = 0; argoIndex < argoData.size(); argoIndex++)
						{
							float value;
							if(d%(m_depthTimes*m_depthTimes) == 0)
								value = argoData[argoIndex]->GetValue(w*m_depthTimes,h*m_depthTimes,15-(d/m_depthTimes/m_depthTimes));
							else
							{
								
								float value1 = argoData[argoIndex]->GetValue(w*m_depthTimes,h*m_depthTimes,15-(d/m_depthTimes/m_depthTimes));
								float value2 = argoData[argoIndex]->GetValue(w*m_depthTimes,h*m_depthTimes,15-(d/m_depthTimes/m_depthTimes+1));
								value = value1 + (value2 - value1) * ((d % (m_depthTimes*m_depthTimes))*1.0 / m_depthTimes/m_depthTimes*1.0);
							}
							index = *it;
							vec3f color;
							float alpha;
							//alpha = 1.0;
							if(value == 0.0)
							{
								alpha = 0.0;
								color = vec3f(0.0,0.0,0.0);
							}
							else
							{
								color = getColorFromColorRamp(value,maxValue,minValue);
								alpha = (value-minValue)/(maxValue-minValue);
							}
							//首尾状态
							particleInfos[argoIndex]->color[index] = make_float4(color.x, color.y, color.z, alpha);
							//particleInfos[argoIndex]->vel[index] = make_float4(0,0,0,lifetime);
						}
						
						//设定初始位置。
						//vec3f LonLat = vec3f(startLonLat.x, 0, startLonLat.y) + 0.1 * vec3f(w, -(float)d, -(float)h);
						//LonLat.y *= 1.5;

						vec3f LonLat = vec3f(-startLonLat.x, 0, -startLonLat.y-0.00790475 * (float)(height) ) + vec3f(-0.00923313 * w * m_depthTimes, 0.08 * (float)d / m_depthTimes / m_depthTimes/*-0.03*depthData[d][w][h]*/, 0.00790475 * (float)(h) * m_depthTimes);
						vec3f pos = m_scaleFactor * (LonLat);

						posPtr[index] = make_float4(pos.x, pos.y, pos.z, 0.0f);

						//准备下一组循环
						++it;
						availNum++;

						if (availNum == numParticles-1)
						{
							break;
						}
					}
					if (availNum == numParticles-1)
					{
						break;
					}
				}
				if (availNum == numParticles-1)
				{
					break;
				}
			}
			if (availNum == numParticles-1)
			{
				break;
			}
		}
	}

	/*fluxData.*/
	std::vector<Hdf4Reader *>().swap(argoData);  //清除容器并最小化它的容量
	std::vector<uint>().swap(randomVector);  //清除容器并最小化它的容量
}

// 初始化
void
ParticleSystem::reset(ParticleConfig config)
{
    switch (config)
    {
        default:
        case CONFIG_GRID:
            {
				//暂时设置这个大小
				float jitter = m_particleRadius*0.001f;
				//vec3f vel = vec3f(0.0f, 0.01f, 0.0f);
				initGrid(m_startLonLat, m_spatialRes, m_imageSize, 4.0f, jitter, m_numParticles, m_particleLifetime);
			}
            break;
    }

	//内存到显存
    m_pos.copy(GpuArray<float4>::HOST_TO_DEVICE);


	//首尾状态
	//m_vel1.setHostPtr(particleInfos[0]->vel);
	//m_vel2.setHostPtr(particleInfos[1]->vel);
	m_color1.setHostPtr(particleInfos[0]->color);
	m_color2.setHostPtr(particleInfos[1]->color);

    //m_vel1.copy(GpuArray<float4>::HOST_TO_DEVICE);
	//m_vel2.copy(GpuArray<float4>::HOST_TO_DEVICE);
	m_color1.copy(GpuArray<float4>::HOST_TO_DEVICE);
	m_color2.copy(GpuArray<float4>::HOST_TO_DEVICE);
}


vec3f ParticleSystem::getColorFromColorRamp(float val,float max,float min)
{

	if (val == 99999)
	{
		return vec3f(0.0f,0.0f,0.0f);
	}

	int index = int ((val -min) / (max - min) * 255);
	const GDALColorEntry* color_rgb = m_particleColorRamp->GetColorEntry(index);
	vec3f color = vec3f(color_rgb->c1 * 1.0f / 255,color_rgb->c2 * 1.0f / 255,color_rgb->c3 * 1.0f / 255);
	return color;
}




void ParticleSystem::createColorRamp(GDALColorEntry* colorBegin,GDALColorEntry* colorEnd, GDALColorTable** colorRamp)
{
	GDALRGBtoHSL(colorBegin);
	GDALRGBtoHSL(colorEnd);
	(*colorRamp) = new GDALColorTable(GPI_HLS);
	(*colorRamp)->CreateColorRamp(0,colorBegin,255,colorEnd);
	(*colorRamp)->ConvertPaletteInterp(GPI_RGB);

}



void
ParticleSystem::setModelView(float *m)
{
    for (int i=0; i<16; i++)
    {
        m_modelView.m[i] = m[i];
    }
}