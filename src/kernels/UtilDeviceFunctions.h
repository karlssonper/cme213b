/*
 * UtilDeviceFunctions.h
 *
 *  Created on: Apr 17, 2012
 *      Author: per
 */

#ifndef UTILDEVICEFUNCTIONS_H_
#define UTILDEVICEFUNCTIONS_H_

enum Direction
{
    DIR_LEFT = 0,
    DIR_RIGHT = 1,
    DIR_UP = 2,
    DIR_DOWN = 3,
    DIR_UNDEFINED = 4
};

__forceinline__ __device__
float2 indexToWorld(const int i, const int j, const float dx)
{
    return make_float2(dx * (i + 0.5), dx * (j + 0.5));
}

__forceinline__ __device__
int2 worldToIndex(const float x, const float y, const float invDx)
{
    return make_int2(x * invDx -0.5, y * invDx -0.5);
}


#endif /* UTILDEVICEFUNCTIONS_H_ */
