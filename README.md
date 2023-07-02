# UfSaCL

This simple simulated-annealing tool uses OpenCL to compute the simulation elements in parallel.

- 256 threads per system
- thousands of parameters per system
- thousands of systems computed in parallel

```C++
        UFSACL::UltraFastSimulatedAnnealing<5, 1000> sim(R"(
                const int numLoopIter = (NumParameters / WorkGroupThreads) + 1;
                for(int i=0;i<numLoopIter;i++)
                {
                    const int loopId = threadId + WorkGroupThreads * i;
                    if(loopId < NumParameters)
                    {
                        float dif = (parameters[loopId] - 1.0f);
                        energy += (test[0]?dif*dif:0.0f);
                    }
                }
        )");

        // sample user-data 
        std::vector<int> test = { 1,2 };

        // adding user-data (this is broadcasted to all system clones running in GPUs/CPUs)
        sim.addUserInput("test", test);

        // build all kernels & copy necessary data
        sim.build();

        // get solution parameters
        // starting temperature = 1.0f (should be greater than 0.5 to span whole sarch-space in initial iteration)
        // ending temperature = 0.001f (should be greater than zero to be able to finish computing)
        // cooling rate = 1.1f (should be greater than 1.0 to be able to finish computing)
        // debugging=true: just outputs performance per iteration
        std::vector<float> prm = sim.run(1.0f, 0.001f, 1.1f, true);
```
