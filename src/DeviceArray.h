/*
 * DeviceArray.h
 *
 *  Created on: Apr 10, 2012
 *      Author: per
 */

#ifndef DEVICEARRAY_H_
#define DEVICEARRAY_H_

#include <thrust/device_vector.h>

template <typename T>
class DeviceArray
{
public:
	DeviceArray();
	DeviceArray(unsigned int size);
	const T * inPtr() const;
	T * outPtr() const;
	void swap();
	void resize(unsigned int size);
private:
	thrust::device_vector<T> vec_[2];
};

#endif /* DEVICEARRAY_H_ */
