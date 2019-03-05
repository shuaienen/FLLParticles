
#include "Hdf4Reader.h"
#include <iostream>
#include <math.h>
#include <iomanip> 
#include <stdlib.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>



Hdf4Reader::Hdf4Reader()
{
}

Hdf4Reader::~Hdf4Reader()
{
}

bool Hdf4Reader::OpenFile(const char *dataName,const char* varName,int timeIndex) 
{
	GDALAllRegister();

	GDALDataset *pDataset = (GDALDataset *) GDALOpen(dataName, GA_ReadOnly);
	if( pDataset == NULL )
	{
		return false;
	}

	char** subdatasets = GDALGetMetadata((GDALDatasetH)pDataset,"SUBDATASETS");
	if (CSLCount(subdatasets) > 0)
	{
		for (int i = 0; subdatasets[i] != NULL; i+=2)
		{
			std::string tmpstr = std::string(subdatasets[i]);
			tmpstr = tmpstr.substr(tmpstr.find_first_of("=")+1);
			std::string tempVarName = tmpstr.substr(tmpstr.find_last_of(":")+1);
	
			if(varName != tempVarName)
				continue;
			const char* tmpfileName = tmpstr.c_str();
			GDALDataset* tmpdt = (GDALDataset*)GDALOpen(tmpfileName,GA_ReadOnly); 

			dataWidth = tmpdt->GetRasterXSize();
			dataHeight = tmpdt->GetRasterYSize();
			dataDepth = 16;//tmpdt->GetRasterCount();
			

			pData = (float *) malloc(sizeof(float)*(dataWidth)*(dataHeight)*dataDepth);
			GDALRasterBand* tmprb = NULL;
			for(int j=1;j<=dataDepth;j++)
			{
				tmprb = tmpdt->GetRasterBand(timeIndex*dataDepth+j);

				double* ignoreValues = new double[1];
				ignoreValues[0] = 0.0;
				//ignoreValues[1] = 10;
				//ignoreValues[2] = 10.000001;
				tmprb->SetStatIngoreValues(&ignoreValues,1);
				tmprb->ComputeStatistics(0,NULL,NULL,NULL,NULL,NULL,NULL);

				if (j == 1)
				{
					minValue = tmprb->GetMinimum();
					maxValue = tmprb->GetMaximum();
				}
				else
				{
					if (minValue > tmprb->GetMinimum())
					{
						minValue = tmprb->GetMinimum();
					}
					if (maxValue < tmprb->GetMaximum())
					{
						maxValue = tmprb->GetMaximum();
					}
				}

				tmprb->RasterIO(GF_Read, 0, 0, dataWidth, dataHeight, pData + (j-1) * dataWidth * dataHeight, dataWidth, dataHeight, GDT_Float32, 0, 0);
			}
			
			//nv::vec3f* tmpGrad = (nv::vec3f*) malloc(sizeof(nv::vec3f)*(dataWidth)*(dataHeight)*16);
			//float* tmpGradM = (float *) malloc(sizeof(float)*(dataWidth)*(dataHeight)*16);
			//float tmpMaxGradM = -1,tmpMinGradM = -1;

			////float tmp;
			//for (int j  = 0; j < 16; j++)
			//{
			//	for (int k = 0; k < dataHeight; k++)
			//	{
			//		for (int l = 0; l < dataWidth; l++)
			//		{
			//			int l2 = l + 1 >= dataWidth ? l : l + 1;
			//			int k2 = k + 1 >= dataHeight ? k : k + 1;
			//			int j2 = j + 1 >= 16 ? j : j + 1;

			//			int l1 = l - 1 < 0 ? l : l - 1;
			//			int k1 = k - 1 < 0 ? k : k - 1;
			//			int j1 = j - 1 < 0 ? j : j - 1;

			//			float w2 = tmpData[l2 + k * dataWidth + j * dataWidth * dataHeight];
			//			float w1 = tmpData[l1 + k * dataWidth + j * dataWidth * dataHeight];

			//			float h2 = tmpData[l + k2 * dataWidth + j * dataWidth * dataHeight];
			//			float h1 = tmpData[l + k1 * dataWidth + j * dataWidth * dataHeight];

			//			float d2 = tmpData[l + k * dataWidth + j1 * dataWidth * dataHeight];
			//			float d1 = tmpData[l + k * dataWidth + j2 * dataWidth * dataHeight];

			//			float w,h,d;

			//			//当前点无值的情况
			//			if (tmpData[l + k * dataWidth + j * dataWidth * dataHeight] == 99999)
			//			{
			//				continue;
			//			}

			//			if (w2 == 99999 || w1 == 99999)
			//			{
			//				w = 0;
			//			}
			//			else
			//			{
			//				w = w2 - w1;
			//			}

			//			if (h2 == 99999 || h1 == 99999)
			//			{
			//				h = 0;
			//			}
			//			else
			//			{
			//				h = h2 - h1;
			//			}

			//			if (d2 == 99999 || d1 == 99999)
			//			{
			//				d = 0;
			//			}
			//			else
			//			{
			//				d = d2 - d1;
			//			}

			//			tmpGrad[l + k * dataWidth + j * dataWidth * dataHeight] = nv::vec3f(w,d,h) * 0.5f;

			//			tmpGradM[l + k * dataWidth + j * dataWidth * dataHeight] = sqrt(
			//				tmpGrad[l + k * dataWidth + j * dataWidth * dataHeight].x * tmpGrad[l + k * dataWidth + j * dataWidth * dataHeight].x +
			//				tmpGrad[l + k * dataWidth + j * dataWidth * dataHeight].y * tmpGrad[l + k * dataWidth + j * dataWidth * dataHeight].y +
			//				tmpGrad[l + k * dataWidth + j * dataWidth * dataHeight].z * tmpGrad[l + k * dataWidth + j * dataWidth * dataHeight].z);

			//			if (tmpMaxGradM == -1)
			//			{
			//				tmpMaxGradM = tmpMinGradM = tmpGradM[l + k * dataWidth + j * dataWidth * dataHeight];
			//			}
			//			else
			//			{
			//				if (tmpMaxGradM < tmpGradM[l + k * dataWidth + j * dataWidth * dataHeight])
			//				{
			//					tmpMaxGradM = tmpGradM[l + k * dataWidth + j * dataWidth * dataHeight];
			//				}
			//				if (tmpMinGradM > tmpGradM[l + k * dataWidth + j * dataWidth * dataHeight])
			//				{
			//					tmpMinGradM = tmpGradM[l + k * dataWidth + j * dataWidth * dataHeight];
			//				}
			//			}
			//		}
			//	}
			//}

			/*pData->push_back(tmpData);
			minValues->push_back(minValue);
			maxValues->push_back(maxValue);
			pDataGradient->push_back(tmpGrad);
			pDataGradM->push_back(tmpGradM);
			minGradM->push_back(tmpMinGradM);
			maxGradM->push_back(tmpMaxGradM);*/
		}
	}
	GDALClose(pDataset);
	return true;
}


float Hdf4Reader::GetValue( unsigned int x,unsigned int y, unsigned int z)
{

		return (pData)[z*dataHeight*dataWidth + y*dataWidth + x];
}