# UfSaCL

This simple simulated-annealing tool uses OpenCL to compute the simulation elements in parallel.

- uses all GPUs+CPUs in single computer
- 256 threads per state-clone (1 OpenCL work-group per state-clone)
- allows thousands of parameters per state-clone (up to local-memory limitations of OpenCL implementation of hardware)
- minimum state copies required = number of GPUs(and other accelerators)
- all parameter values given by solver are in normalized form (in range (0.0f, 1.0f)) and user maps them to their intended range in kernel

# Dependencies

- Visual Studio (2022 community edition, etc) with vcpkg (that auto-installs OpenCL for the project) ![vcpkg](https://github.com/tugrul512bit/libGPGPU/assets/23708129/4a064dcb-b967-478d-a15f-fc69f4e3e9ee)
- - Maybe works in Ubuntu without vcpkg too, just need explicitly linking of OpenCL libraries and headers
- OpenCL 1.2 runtime (s) [Intel's runtime can find CPUs of AMD processors too & run AVX512 on Ryzen 7000 series CPU cores] (multiple platforms are scanned for all devices)
- OpenCL device(s) like GTX 1050 ti graphics card, a new CPU that has teraflops of performance, integrated GPU, all at the same time can be used as a big unified GPU.
- C++17

Polynomial curve-fitting sample: 

- std::sqrt(x) is approximated using only 4 parameters: c0,c1,c2,c3 and multiplication with powers of x
- 20000 data points are used for fitting (that would take too long to compute on single-thread)
- 100000 clones of states are computed in parallel (total of 25600000 workitems for GPUs)

```C++
#include"UfSaCL.h"
#include<vector>
#include<iostream>
// polynomial curve-fitting sample with 20000 data points & 5 polynomial coefficients
// normally this would take too much time on just single-thread version but it completes in 12 seconds with a low-end GPU and a high-end CPU
int main()
{
    try
    {
        // trying to approximate square-root algorithm in (0,1) input range by a polynomial
        // y = f(x) = y = c0 + x * c1 + x^2 * c2 + x^3 * c3
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

        // 4 parameters: c0,c1,c2,c3 of polynomial y = c0 + c1*x + c2*x^2 + c3*x^3
        // 100000 clones in parallel
        UFSACL::UltraFastSimulatedAnnealing<4, 100000> sim(
            std::string("#define NUM_POINTS ") + std::to_string(N) + 
            std::string(
            R"(
                // 256 threads looping "numLoopIter" times to compute NUM_POINTS data points error (energy)
                const int numLoopIter = (NUM_POINTS / WorkGroupThreads) + 1;
                for(int i=0;i<numLoopIter;i++)
                {
                    const int loopId = threadId + WorkGroupThreads * i;
                    if(loopId < NUM_POINTS)
                    {

                        // building the polynomial y = c0 + x * c1 + x^2 * c2 + x^3 * c3

                        // powers of x
                        float x = dataPointsX[loopId];


                        // coefficients, after scaling of normalized parameters
                        float c0 = (parameters[0] - 0.5f)*1000.0f; // (-500,+500) range
                        float c1 = (parameters[1] - 0.5f)*1000.0f; // (-500,+500) range
                        float c2 = (parameters[2] - 0.5f)*1000.0f; // (-500,+500) range
                        float c3 = (parameters[3] - 0.5f)*1000.0f; // (-500,+500) range

                        // approximation
                        float yApproximation = (((c3 * x) + c2) * x + c1) * x + c0;

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
        std::vector<float> prm = sim.run(1.0f, 0.001f, 2.0f,25,false,false,true);
        
        std::cout << "y = " << (prm[0]-0.5f)*1000.0f << " + (" << (prm[1] - 0.5f) * 1000.0f << " * x) + " << " (" << (prm[2] - 0.5f) * 1000.0f << " * x^2) + " << " (" << (prm[3] - 0.5f) * 1000.0f << " * x^3)" << std::endl;
        
        

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
lower energy found: 737634
lower energy found: 209079
lower energy found: 26445.2
lower energy found: 11496.1
lower energy found: 1097.19
lower energy found: 553.719
lower energy found: 30.4808
lower energy found: 24.536
lower energy found: 7.72375
lower energy found: 5.25135
lower energy found: 3.46719
lower energy found: 2.87768
lower energy found: 2.63414
lower energy found: 2.61885
y = 0.128508 + (1.92243 * x) +  (-1.96072 * x^2) +  (0.929654 * x^3)
```

