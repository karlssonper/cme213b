#include "DeviceArray.h"
#include <thrust/fill.h>

DeviceArray::DeviceArray() : in_(FIRST), out_(SECOND)
{
    resize(0);
};

DeviceArray::DeviceArray(unsigned int size) : in_(FIRST), out_(SECOND)
{
    resize(size);
};

const float * DeviceArray::inPtr() const
{
    return thrust::raw_pointer_cast(&vec_[in_][0]);
};

const thrust::device_vector<float> & DeviceArray::inVec() const
{
    return vec_[in_];
};

float * DeviceArray::outPtr()
{
    return thrust::raw_pointer_cast(&vec_[out_][0]);
};

void DeviceArray::swap()
{
    State temp = in_;
    in_ = out_;
    out_ = temp;
};

void DeviceArray::setZero()
{
    thrust::fill(vec_[FIRST].begin(), vec_[FIRST].end(), 0.0f);
    thrust::fill(vec_[SECOND].begin(), vec_[SECOND].end(), 0.0f);   
}

void DeviceArray::resize(unsigned int size)
{
    vec_[FIRST].resize(size);
    vec_[SECOND].resize(size);
};