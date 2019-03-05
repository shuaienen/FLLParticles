
#ifndef __HDF4READER_H__
#define __HDF4READER_H__

#include "gdal_priv.h"
#include "nvMath.h"

class Hdf4Reader
{
public:
	Hdf4Reader();
	~Hdf4Reader();

	unsigned int GetDataWidth()
	{
		return dataWidth;
	}

	unsigned int GetDataHeight()
	{
		return dataHeight;
	}

	unsigned int GetDataDepth()
	{
		return dataDepth;
	}

	bool OpenFile(const char *dataName,const char* varName,int timeIndex);


	float GetValue(unsigned int x,unsigned int y, unsigned int z);

	float GetMaxValue()
	{

		return maxValue;
	}


	float GetMinValue()
	{
		return minValue;
	}



protected: 
	float* pData;
	
	float maxValue;
	float minValue;
	
	unsigned int dataWidth;//���
	unsigned int dataHeight;//�߶�
	unsigned int dataDepth;//���

};

#endif // __TIFFREADER_H__
