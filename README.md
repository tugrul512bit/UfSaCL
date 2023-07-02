# UfSaCL

This simple simulated-annealing tool uses OpenCL to compute the simulation elements in parallel.

- uses all GPUs+CPUs in single computer
- 256 threads per state-clone (1 OpenCL work-group per state-clone)
- allows thousands of parameters per state-clone (up to local-memory limitations of OpenCL implementation of hardware)
- minimum state copies required = number of GPUs(and other accelerators)
- all parameter values given by solver are in normalized form (in range (0.0f, 1.0f)) and user maps them to their intended range in kernel

```C++
// Function to minimize energy, with 5 parameters and 1000 state-clone (that run in parallel)
// ```parameters``` array is in OpenCL-local memory and can be randomly accessed for any element fast (some GPUs have only several cycles latency in accessing this memory)
// NumParameters: a define macro in kernel code, equals to 5 here
// WorkGroupThreads: a define macro that is equal to number of threads per state-clone (256 currently)
// threadId: 0-255 ranged integer that points to id value of current thread in work-group in OpenCL kernel execution
// when number of parameters is greater than 256, this loop handles all extra iterations per thread
// finally the energy values from all threads are reduced into a single energy result (simply summed in parallel)
// this example uses only 5 threads per state-clone and the remaining 251 threads are idle
UFSACL::UltraFastSimulatedAnnealing<5, 1000> sim(R"(
        const int numLoopIter = (NumParameters / WorkGroupThreads) + 1;
        for(int i=0;i<numLoopIter;i++)
        {
            const int loopId = threadId + WorkGroupThreads * i;
            if(loopId < NumParameters)
            {
                float dif = (parameters[loopId] - 1.0f);

                // simulated annealing minimizes this
                energy += (test[0]?dif*dif:0.0f);
            }
        }
)");

// sample user-data 
std::vector<int> test = { 1,2 };

// adding user-data (this is broadcasted to all state-clone running in GPUs/CPUs)
sim.addUserInput("test", test);

// build all kernels & copy necessary data
sim.build();

// get solution parameters
// starting temperature = 1.0f (should be greater than 0.5 to span whole sarch-space in initial iteration)
// ending temperature = 0.001f (should be greater than zero to be able to finish computing)
// cooling rate = 1.1f (should be greater than 1.0 to be able to finish computing)
// debugging=true: just outputs performance per iteration
// device debugging=true: performance info for each device used
std::vector<float> prm = sim.run(1.0f, 0.001f, 1.1f, true,true);
for (auto& e : prm)
{
        std::cout << e << std::endl;
}
```

output:

```
...
iteration-71
computation-time=0.0003899 seconds
iteration-72
computation-time=0.0004613 seconds
total computation-time=0.0499326 seconds (this includes debugging console-output that is slow)
---------------
OpenCL device info:
GeForce GT 1030 computed 6.1% of total work
gfx1036 computed 0.1% of total work
AMD Ryzen 9 7900 12-Core Processor computed 93.8% of total work
---------------
0.999984
0.999819
0.999983
0.99997
0.999827
```
since this sample code minimizes the ```parameters[loopId] - 1.0f```, all parameters approach to 1.0f. Since there is too small work per work-group (256 threads for just computation of 3 parameters), CPU does more of the work than two GPUs. With more work, GPUs are given more of work automatically.
