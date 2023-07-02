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
        std::vector<float> prm = sim.run(1.0f, 0.001f, 2.0f,25,true,true,true);
        
        std::cout << "y = " << (prm[0]-0.5f)*1000.0f << " + (" << (prm[1] - 0.5f) * 1000.0f << " * x) + " << " (" << (prm[2] - 0.5f) * 1000.0f << " * x^2) + " << " (" << (prm[3] - 0.5f) * 1000.0f << " * x^3)" << std::endl;

    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }
    return 0;
}