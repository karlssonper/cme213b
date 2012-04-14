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

	void resize(unsigned int size);
};

#endif /* DEVICEARRAY_H_ */
