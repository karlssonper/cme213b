#include "DeviceArray.h"

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

void DeviceArray::resize(unsigned int size)
{
    vec_[FIRST].resize(size);
    vec_[SECOND].resize(size);
};