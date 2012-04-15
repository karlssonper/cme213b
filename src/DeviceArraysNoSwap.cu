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

void DeviceArraysNoSwap::setZero()
{
    thrust::fill(mask_.begin(), mask_.end(), 0);
    thrust::fill(velMag_.begin(), velMag_.end(), 0.0f);
    thrust::fill(surfacePoints_.begin(), 
                 surfacePoints_.end(), 
                 make_float2(0.0f,0.0f));   
}

void DeviceArraysNoSwap::resize(unsigned int size)
{
    mask_.resize(size);
    velMag_.resize(size);
    surfacePoints_.resize(size);
}
