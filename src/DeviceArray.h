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
private:
	enum State{FIRST = 0, SECOND = 1};
	State in_;
	State out_;
	thrust::device_vector<T> vec_[2];
public:
	DeviceArray() : in_(FIRST), out_(SECOND)
	{
		resize(0);
	};

	DeviceArray(unsigned int size) : in_(FIRST), out_(SECOND)
	{
		resize(size);
	};

	const T * inPtr() const
    {
		return thrust::raw_pointer_cast(&vec_[in_][0]);
	};

	const thrust::device_vector<T> & inVec() const
	{
		return vec_[in_];
	};

	T * outPtr()
    {
		return thrust::raw_pointer_cast(&vec_[out_][0]);
	};

	void swap()
	{
		State temp = in_;
		in_ = out_;
		out_ = temp;
	};

	void resize(unsigned int size)
	{
		vec_[FIRST].resize(size);
		vec_[SECOND].resize(size);
	};
};

#endif /* DEVICEARRAY_H_ */
