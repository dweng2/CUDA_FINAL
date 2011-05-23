/*
    Miller-Rabin Test Kernal
*/

#include <stdint.h>
#include <stdlib.h>
#include <curand_kernel.h>

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

__global__ void setup_kernel ( curandState *state )
{
    int id = threadIdx . x + blockIdx .x * 64;
    /* Each thread gets same seed , a different sequence number  -
    ,
    no offset */
    curand_init (1234 , id , 0 , & state [ id ]) ;
}


__global__ void Miller_Rabin_Kernal(Test_Result *results, curandState *state)
{
    int index = threadIdx.x % 32;
    int test_num = results[index].num;

    //mod random number so that a < n
    uint32_t a = curand(state) % test_num;
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
        if (a_to_power == test_num - 1) 
            return;

        a_to_power = modular_exponent_32(a_to_power, 2, test_num);
    }

    if (a_to_power == test_num - 1)
        return;

    results[index].passed++;
}

int main()
{
    curandState *dev_rand_state;
    
    //Allocate mem for RNG states
    cudaMalloc((void **) &dev_rand_state, sizeof(curandState) * 512);
    
    //Init RNG states
    setup_kernel<<<1, 512>>>(dev_rand_state);
}

