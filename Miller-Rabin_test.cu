/*
    Miller-Rabin Test Kernal
*/

#include <stdint.h>
#include <stdlib.h>
#include "Miller_Rabin_Test.h"

__device__ uint32_t modular_exponent_32(uint32_t base, uint32_t power, uint32_t modulus) 
{
    uint64_t result = 1;
    int i; 
    for (i = 32; i > 0; i--) 
    {
        result = (result*result) % modulus;
        if (power & (1 << i)) 
        {
            result = (result*base) % modulus;
        }
    }
    return (uint32_t)result; /* Will not truncate since modulus is a uint32_t */
}


__global__ void Miller_Rabin_Kernal(Test_Result *results)
{
    int index = threadIdx.x % 32;
    int test_num = results[index].num;

    //mod random number so that a < n

    //do test
    uint32_t a_to_power, s, d, i;

    //16-bit compute s and d
    s = 0;
    d = test_num - 1;

    while ((d % 2) == 0) 
    {
        d /= 2;
        s++;
    }

    a_to_power = modular_exponent_32(a, d, test_num);

    if (a_to_power == 1)
        return;

    for(i=0; i < s-1; i++) 
    {
        if (a_to_power == n-1) 
            return;

        a_to_power = modular_exponent_32(a_to_power, 2, n);
    }

    if (a_to_power == n-1)
        return;

    results[index].passed++;
}
