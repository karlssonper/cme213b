/*
 * DeviceArraysNoSwap.h
 *
 *  Created on: Apr 14, 2012
 *      Author: per
 */

#ifndef DEVICEARRAYSNOSWAP_H_
#define DEVICEARRAYSNOSWAP_H_

#include <thrust/device_vector.h>

class DeviceArraysNoSwap
{
public:
    unsigned char * mask();
    const unsigned char * mask() const;
    thrust::device_vector<unsigned char>& maskVec();

    float * velocityMag();
    const float * velocityMag() const;
    thrust::device_vector<float> & velocityMagVec();

    float2 * surfacePoints();
    const float2 * surfacePoints() const;
    thrust::device_vector<float2> & surfacePointsVec();
private:
    thrust::device_vector<unsigned char> mask_;
    thrust::device_vector<float> velMag_;
    thrust::device_vector<float2> surfacePoints_;
};
#endif /* DEVICEARRAYSNOSWAP_H_ */
