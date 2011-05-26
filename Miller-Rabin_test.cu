/*
 *  Miller-Rabin Test Kernal
 *  Darrin Weng
*/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <curand_kernel.h>

#include "Miller_Rabin_Test.h"

#define GRID_SIZE 4
#define BLOCK_SIZE 512
#define THREADS_PER_NUM 32

__device__ uint32_t modular_exponent_32(uint32_t base, uint32_t power, uint32_t modulus) 
{
    uint64_t result = 1;
    int i; 
    for (i = 32; i > 0; i--) 
    {
        result = (result * result) % modulus;
        if (power & (1 << i)) 
        {
            result = (result * base) % modulus;
        }
    }
    return (uint32_t)result; /* Will not truncate since modulus is a uint32_t */
}

__global__ void setup_kernel ( curandState *state , int seed)
{
    int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;

    curand_init (seed, id, 0, &state[id]) ;
}


__global__ void Miller_Rabin_Kernal(Test_Result *results, curandState *state)
{
    int index = (threadIdx.x / THREADS_PER_NUM) + blockIdx.x * (BLOCK_SIZE / THREADS_PER_NUM);
    int test_num = results[index].num;

    //mod random number so that a < n
    uint32_t a = curand(&state[threadIdx.x]) % test_num;

    //do test
    uint32_t a_to_power, s, d, i;

    //16-bit compute s and d
    s = 0;
    d = test_num - 1;

    while ((d % 2) == 0) 
    {
        d >>= 1;
        s++;
    }
    
    if(s == 0) //Even number so cannot be prime
    {
        results[index].passed = 1;
        return;
    }

    a_to_power = modular_exponent_32(a, d, test_num);

    if (a_to_power == 1)
    {
        printf("Thread #%d %d Return 1\n", threadIdx.x, test_num);
        return;
    }

    for(i = 0; i < s - 1; i++) 
    {
        if (a_to_power == test_num - 1)
        {
            printf("Thread #%d %d Return 2\n", threadIdx.x, test_num);
            return;
        }

        a_to_power = modular_exponent_32(a_to_power, 2, test_num);
    }

    if (a_to_power == test_num - 1)
    {
        printf("Thread #%d %d Return 3\n", threadIdx.x, test_num);
        return;
    }
    
    printf("Thread #%d %d a pow %u Return NOT\n", threadIdx.x, test_num, a_to_power);
    results[index].passed = 1;
}

int main()
{
    curandState *dev_rand_state;
    Test_Result *results, *dev_results;
    int num_results = (BLOCK_SIZE / THREADS_PER_NUM) * GRID_SIZE;

    results = (Test_Result *) malloc(sizeof(Test_Result) * num_results);

    //Generate or get from a file the test numbers NOTE dont test even numbers or 2,5,7
    for(int i = 0; i < num_results; i++)
    {
        results[i].num = i + 719;
        results[i].passed = 0;
    }

    //Allocate mem for RNG states, numbers to be tested, and results
    cudaMalloc((void **) &dev_rand_state, sizeof(curandState) * BLOCK_SIZE * GRID_SIZE);
    cudaMalloc((void **) &dev_results, sizeof(Test_Result) * num_results);
    
    //set up grid and blocksize
    dim3 dimBlock(BLOCK_SIZE, 1);
    dim3 dimGrid(GRID_SIZE, 1);
    
    //seed RND with cpu rand
    srand(52);
    int seed = rand();
    
    //Init RNG states and transfer data to GPU
    setup_kernel<<<dimGrid, dimBlock>>>(dev_rand_state, seed);
    
    cudaMemcpy(dev_results, results, sizeof(Test_Result) * num_results, cudaMemcpyHostToDevice);
    
    //Run Tests
    Miller_Rabin_Kernal<<<dimGrid, dimBlock>>>(dev_results, dev_rand_state);
    
    //Transfer results back to cpu
    cudaMemcpy(results, dev_results, sizeof(Test_Result) * num_results, cudaMemcpyDeviceToHost);
            
    //Clean up memory
    cudaFree(dev_results);
    cudaFree(dev_rand_state);
    
    //Print results
    for(int i = 0; i < num_results; i++)
    {
        printf("%u is ", results[i].num);
        
        if(results[i].passed == PASSED)
            printf("PRIME\n");
        else
            printf("COMPOSITE\n");
    }
    
    free(results);
}

