/*
 *  Miller-Rabin Test Kernal
 *  Darrin Weng
*/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "Miller_Rabin_Test.h"

#define GRID_SIZE 500 //DONT MAKE TOO BIG WILL KILL MACHINES
#define BLOCK_SIZE 256
#define CUDA_ERROR()  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

__device__ uint32_t modular_exponent_32(uint32_t base, uint32_t power, uint32_t modulus) 
{
    uint64_t result = 1;
    
    while(power > 0)
    {
        if((power & 1) == 1)
        {
            result = (result * base) % modulus;
        }
        
        power >>= 1;
        uint64_t temp = (uint64_t) base * base;
        base = temp % modulus;
    }
    
    return (uint32_t) result; /* Will not truncate since modulus is a uint32_t */
}

__global__ void Miller_Rabin_Kernal(Test_Result *results, int num_results)
{
    int index = threadIdx.x + blockIdx.x * BLOCK_SIZE;

    if(index >= num_results)
        return;

    uint32_t test_num = results[index].num;
    uint32_t a;

    if(test_num % 2 == 0) //Even number so cannot be prime
    {
        results[index].passed = 1;
        return;
    }

    switch(threadIdx.y)
    {
        case 0:
            a = 2;
        break;
        case 1:
            a = 7;
        break;
        case 2:
            a = 61;
        break;
    }

    //make sure a is in range
    if( a >= test_num)
        return;
    
    //do test
    uint32_t a_to_power, s, d, i;

    //compute s and d
    s = 0;
    d = test_num - 1;

    while ((d % 2) == 0) 
    {
        d >>= 1;
        s++;
    }
    
    a_to_power = modular_exponent_32(a, d, test_num);

    if (a_to_power == 1)
    {
        return;
    }

    for(i = 0; i < s - 1; i++) 
    {
        if (a_to_power == test_num - 1)
        {
            return;
        }

        a_to_power = modular_exponent_32(a_to_power, 2, test_num);
    }

    if (a_to_power == test_num - 1)
    {
        return;
    }
    
    results[index].passed = 1;
}

void run_kernel(Test_Result *results, int num_results)
{
    printf("Running CUDA\n");
    Test_Result *dev_results;
    
    cudaMalloc((void **) &dev_results, sizeof(Test_Result) * num_results);
        
    //set up grid and blocksize
    dim3 dimBlock(BLOCK_SIZE, 3);
    dim3 dimGrid(GRID_SIZE, 1);
    
    cudaMemcpy(dev_results, results, sizeof(Test_Result) * num_results, cudaMemcpyHostToDevice);
    
    int done = 0;
    while(done < num_results)
    {
        Miller_Rabin_Kernal<<<dimGrid, dimBlock>>>(dev_results + done, num_results);
        done += GRID_SIZE * BLOCK_SIZE;
    }

    CUDA_ERROR();
    //Transfer results back to cpu
    cudaMemcpy(results, dev_results, sizeof(Test_Result) * num_results, cudaMemcpyDeviceToHost);
            
    //Clean up memory
    cudaFree(dev_results);
}

uint32_t serial_modular_exponent_32(uint32_t base, uint32_t power, uint32_t modulus) 
{
    uint64_t result = 1;
    
    while(power > 0)
    {
        if((power & 1) == 1)
        {
            result = (result * base) % modulus;
        }
        
        power >>= 1;
        uint64_t temp = (uint64_t) base * base;
        base = temp % modulus;
    }
    
    return (uint32_t) result; /* Will not truncate since modulus is a uint32_t */
}

void Miller_Rabin_Serial(Test_Result *results, int num_results)
{
    printf("Running Serial on %d numbers\n", num_results);
    for(int index = 0; index < num_results; index++)
    {
        uint32_t test_num = results[index].num;
        for(int j = 0; j < 32; j++)
        {
            //mod random number so that a < n
            uint32_t a = 0;
            while(a < 1 || a > test_num - 1)
                a = rand() % test_num;

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
                continue;
            }

            a_to_power = serial_modular_exponent_32(a, d, test_num);
            
            if (a_to_power == 1)
            {
                continue;
            }

            for(i = 0; i < s - 1; i++) 
            {
                if (a_to_power == test_num - 1)
                {
                    continue;
                }

                a_to_power = serial_modular_exponent_32(a_to_power, 2, test_num);
            }

            if (a_to_power == test_num - 1)
            {
                continue;
            }
            
            results[index].passed = 1;
        }      
    }
}

int main(int argc, char **argv)
{
    Test_Result *results;
    int num_results = atoi(argv[1]);        
    int runCUDA = 1;
     
    results = (Test_Result *) malloc(sizeof(Test_Result) * num_results);

    //Generate or get from a file the test numbers NOTE dont test 2
    for(uint32_t i = 0; i < num_results; i++)
    {    
        results[i].num = i + 2;
        results[i].passed = 0;
    }
    
    if(argc >= 3)
    {
        if(strcmp(argv[2], "-s") == 0)
            runCUDA = 0; 
    }
    
    if(runCUDA)
        run_kernel(results, num_results);
    else
        Miller_Rabin_Serial(results, num_results);

    //Print results
    for(int i = 0; i < num_results; i++)
    {  
        if(results[i].passed == PASSED)
            printf("%d\n", results[i].num);      
    }
    
    printf("Tested %d numbers\n", num_results);
    free(results);
}

