# Ultra Fast Simulated Annealing with OpenCL

This simple simulated-annealing tool uses OpenCL to compute the simulation elements in parallel.

- uses all GPUs+CPUs in single computer
- 256 threads per state-clone (1 OpenCL work-group per state-clone)
- allows thousands of parameters per state-clone (up to local-memory limitations of OpenCL implementation of hardware)
- minimum state copies required = number of GPUs(and other accelerators)
- all parameter values given by solver are in normalized form (in range (0.0f, 1.0f)) and user maps them to their intended range in kernel

Wiki: https://github.com/tugrul512bit/UfSaCL/wiki

# Dependencies

- Visual Studio (2022 community edition, etc) with vcpkg (that auto-installs OpenCL for the project) ![vcpkg](https://github.com/tugrul512bit/libGPGPU/assets/23708129/4a064dcb-b967-478d-a15f-fc69f4e3e9ee)
- - Maybe works in Ubuntu without vcpkg too, just need explicitly linking of OpenCL libraries and headers
- OpenCL 1.2 runtime (s) [Intel's runtime can find CPUs of AMD processors too & run AVX512 on Ryzen 7000 series CPU cores] (multiple platforms are scanned for all devices)
- OpenCL device(s) like GTX 1050 ti graphics card, a new CPU that has teraflops of performance, integrated GPU, all at the same time can be used as a big unified GPU.
- C++17

Polynomial curve-fitting sample: 

- std::sqrt(x) is approximated using 5 parameters: c0,c1,c2,c3,c4 and multiplication with powers of x
- 20000 data points are used for fitting (that would take too long to compute on single-thread)
- 100000 clones of states are computed in parallel (total of 25600000 workitems for GPUs)

```C++
#include"UfSaCL.h"
#include<vector>
#include<iostream>
// polynomial curve-fitting sample with 20000 data points & 5 polynomial coefficients
int main()
{
    try
    {
        // trying to approximate square-root algorithm in (0,1) input range by a polynomial
        // y = f(x) = y = c0 + x * c1 + x^2 * c2 + x^3 * c3 + x^4 * c4
        const int N = 20000;
        std::vector<float> dataPointsX;
        std::vector<float> dataPointsY;
        for (int i = 0; i < N; i++)
        {
            float x = i  / (float) N;
            float y = std::sqrt(x);
            dataPointsX.push_back(x);
            dataPointsY.push_back(y);
        }

        // 5 parameters: c0,c1,c2,c3,c4 of polynomial y = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4
        // 100000 clones in parallel
        UFSACL::UltraFastSimulatedAnnealing<5, 100000> sim(
            std::string("#define NUM_POINTS ") + std::to_string(N) + 
            std::string(
            R"(
                // 256 threads looping "nIter" times to compute NUM_POINTS data points error (energy)
                const int nIter = (NUM_POINTS / WorkGroupThreads) + 1;
                for(int i=0;i<nIter;i++)
                {
                    const int loopId = threadId + WorkGroupThreads * i;
                    if(loopId < NUM_POINTS)
                    {

                        // building the polynomial y = c0 + x * c1 + x^2 * c2 + x^3 * c3 + x^4 * c4

                        // powers of x
                        float x = dataPointsX[loopId];


                        // coefficients, after scaling of normalized parameters
                        float c0 = (parameters[0] - 0.5f)*1000.0f; // (-500,+500) range
                        float c1 = (parameters[1] - 0.5f)*1000.0f; // (-500,+500) range
                        float c2 = (parameters[2] - 0.5f)*1000.0f; // (-500,+500) range
                        float c3 = (parameters[3] - 0.5f)*1000.0f; // (-500,+500) range
                        float c4 = (parameters[4] - 0.5f)*1000.0f; // (-500,+500) range

                        // approximation polynomial
                        float yApproximation = ((((c4*x+c3) * x) + c2) * x + c1) * x + c0;

                        // data point value
                        float yReal = dataPointsY[loopId];

                        // the higher the difference, the higher the energy
                        float diff = yApproximation - yReal;

                        energy += diff * diff;
                    }
                }
        )"));

        sim.addUserInput("dataPointsX", dataPointsX);
        sim.addUserInput("dataPointsY", dataPointsY);
        sim.build();
        float startTemperature = 1.0f; // good if its between 0.5f and 1.0f
        float stopTemperature = 0.0001f; // the closer to zero, the higher the accuracy, the slower to solution
        float coolingRate = 1.05f;
        int numReHeating = 5; // when single cooling is not enough, re-start the process multiple times while keeping the best solution
        std::vector<float> prm = sim.run(startTemperature, stopTemperature, coolingRate, numReHeating,false,false,true);
        
        std::cout << "y = " << (prm[0]-0.5f)*1000.0f << " + (" << (prm[1] - 0.5f) * 1000.0f << " * x) + " << " (" << (prm[2] - 0.5f) * 1000.0f << " * x^2) + " << " (" << (prm[3] - 0.5f) * 1000.0f << " * x^3) + " << " (" << (prm[4] - 0.5f) * 1000.0f << " * x^4) "<< std::endl;

    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }
    return 0;
}
```

output:

```
...
lower energy found: 1.08967
reheating. num reheats left=1
lower energy found: 1.08605
lower energy found: 1.0735
lower energy found: 1.0656
total computation-time=55.7984 seconds (this includes debugging console-output that is slow)
---------------
OpenCL device info:
GeForce GT 1030 computed 24.981% of total work
gfx1036 computed 30.06% of total work
AMD Ryzen 9 7900 12-Core Processor computed 44.959% of total work
---------------
y = 0.097096 + (2.51061 * x) +  (-4.62151 * x^2) +  (5.10132 * x^3) +  (-2.1019 * x^4)
```
![graph](https://i.snipboard.io/LyHKOU.jpg)

Parallel computation loop can be simplified by using in-kernel define macro (that OpenCL implementation allows) ```parallelFor(iters,{ codes(); })```:

```C++
        UFSACL::UltraFastSimulatedAnnealing<5, 100000> sim(
            std::string("#define NUM_POINTS ") + std::to_string(N) +
            std::string(
                R"(
                parallelFor(NUM_POINTS,
                    {

                        // building the polynomial y = c0 + x * c1 + x^2 * c2 + x^3 * c3 + x^4 * c4

                        // powers of x
                        float x = dataPointsX[loopId];


                        // coefficients, after scaling of normalized parameters
                        float c0 = (parameters[0] - 0.5f)*1000.0f; // (-500,+500) range
                        float c1 = (parameters[1] - 0.5f)*1000.0f; // (-500,+500) range
                        float c2 = (parameters[2] - 0.5f)*1000.0f; // (-500,+500) range
                        float c3 = (parameters[3] - 0.5f)*1000.0f; // (-500,+500) range
                        float c4 = (parameters[4] - 0.5f)*1000.0f; // (-500,+500) range

                        // approximation polynomial
                        float yApproximation = ((((c4*x+c3) * x) + c2) * x + c1) * x + c0;

                        // data point value
                        float yReal = dataPointsY[loopId];

                        // the higher the difference, the higher the energy
                        float diff = yApproximation - yReal;

                        energy += diff * diff;
                    });
                
        )"));
```

The function body in parallelFor has branching so user should not use barriers inside. Barriers have to be called by all participating threads. There is ```parallelForWithBarrier(iters,{  });``` for this:

```C++
        UFSACL::UltraFastSimulatedAnnealing<5, 100000> sim(R"(
                    // applies barrier(CLK_LOCAL_MEM_FENCE); between thread-wave iterations (not individual iterations)
                    parallelForWithBarrier(NUM_POINTS,
                    {
                        someLocalArray[loopId]=newValue; // changing a local array on a unique index between threads
                    });           
        )"));
```
