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

/////////////////////////////////////////////////////////////////////////////

unsigned char * DeviceArraysNoSwap::mask()
{
    return thrust::raw_pointer_cast(&mask_[0]);
}

const unsigned char * DeviceArraysNoSwap::mask() const
{
    return thrust::raw_pointer_cast(&mask_[0]);
}

void DeviceArraysNoSwap::maskResize(unsigned int size)
{
    mask_.resize(size);
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

void DeviceArraysNoSwap::velocityMagResize(unsigned int size)
{
    velMag_.resize(size);
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

void DeviceArraysNoSwap::surfacePointsResize(unsigned int size)
{
    surfacePoints_.resize(size);
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
