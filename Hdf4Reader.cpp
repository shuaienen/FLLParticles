
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
		}
	}
	GDALClose(pDataset);
	return true;
}


float Hdf4Reader::GetValue( unsigned int x,unsigned int y, unsigned int z)
{

		return (pData)[z*dataHeight*dataWidth + y*dataWidth + x];
}
