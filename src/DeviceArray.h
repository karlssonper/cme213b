/*
 * DeviceArray.h
 *
 *  Created on: Apr 10, 2012
 *      Author: per
 */

#ifndef DEVICEARRAY_H_
#define DEVICEARRAY_H_

#include <thrust/device_vector.h>

class DeviceArray
{
private:
	enum State{FIRST = 0, SECOND = 1};
	State in_;
	State out_;
	thrust::device_vector<float> vec_[2];
public:
	DeviceArray();
	DeviceArray(unsigned int size);
	const float * inPtr() const;
	const thrust::device_vector<float> & inVec() const;
	float * outPtr();
	void swap();
	void setZero();
	void resize(unsigned int size);
};

class DeviceArraysNoSwap
{
public:
    unsigned char * mask();
    const unsigned char * mask() const;
    void maskResize(unsigned int size);
    thrust::device_vector<unsigned char>& maskVec();

    float * velocityMag();
    const float * velocityMag() const;
    void velocityMagResize(unsigned int size);
    thrust::device_vector<float> & velocityMagVec();

    float2 * surfacePoints();
    const float2 * surfacePoints() const;
    void surfacePointsResize(unsigned int size);
    thrust::device_vector<float2> & surfacePointsVec();

    void setZero();
private:
    thrust::device_vector<unsigned char> mask_;
    thrust::device_vector<float> velMag_;
    thrust::device_vector<float2> surfacePoints_;
};

#endif /* DEVICEARRAY_H_ */
