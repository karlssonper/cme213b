/*
 * DeviceArraysNoSwap.cpp
 *
 *  Created on: Apr 14, 2012
 *      Author: per
 */

#include "DeviceArraysNoSwap.h"

unsigned char * DeviceArraysNoSwap::mask()
{
    return thrust::raw_pointer_cast(&mask_[0]);
}

const unsigned char * DeviceArraysNoSwap::mask() const
{
    return thrust::raw_pointer_cast(&mask_[0]);
}

thrust::device_vector<unsigned char>& DeviceArraysNoSwap::maskVec()
{
    return mask_;
}

float * DeviceArraysNoSwap::velocityMag()
{
    return thrust::raw_pointer_cast(&velMag_[0]);
}

const float * DeviceArraysNoSwap::velocityMag() const
{
    return thrust::raw_pointer_cast(&velMag_[0]);
}

thrust::device_vector<float> & DeviceArraysNoSwap::velocityMagVec()
{
    return velMag_;
}

float2 * DeviceArraysNoSwap::surfacePoints()
{
    return thrust::raw_pointer_cast(&surfacePoints_[0]);
}

const float2 * DeviceArraysNoSwap::surfacePoints() const
{
    return thrust::raw_pointer_cast(&surfacePoints_[0]);
}

thrust::device_vector<float2> & DeviceArraysNoSwap::surfacePointsVec()
{
    return surfacePoints_;
}
