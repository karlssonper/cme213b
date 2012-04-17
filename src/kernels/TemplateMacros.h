/*
 * ExpInitMacro.h
 *
 *  Created on: Apr 16, 2012
 *      Author: per
 */

#ifndef TEMPLATE_MACROS_H
#define TEMPLATE_MACROS_H

#define TEMPLATE_ARGS() int T_THREADS_X, int T_THREADS_Y
#define TEMPLATE_ARGS_RUN() T_THREADS_X,     T_THREADS_Y

#define EXPLICIT_TEMPLATE_FUNCTION_INSTANTIATION(func, args...)  \
    template void func<8,8>(args);                            \
    template void func<8,16>(args);                           \
    template void func<8,32>(args);                           \
    template void func<16,8>(args);                           \
    template void func<16,16>(args);                          \
    template void func<16,32>(args);                          \
    template void func<32,8>(args);                           \
    template void func<32,16>(args);                          \
    template void func<32,32>(args);                          \

#endif /* TEMPLATE_MACROS_H */
