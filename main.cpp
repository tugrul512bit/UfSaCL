#include"UfSaCL.h"
#include<vector>
#include<iostream>

int main()
{
    try
    {

        // fit 106 circles into 1 square
        // r=0.5, d=10.0
        // 212 parameters: x,y data for 106 circles
        UFSACL::UltraFastSimulatedAnnealing<106*2, 100000> sim(R"(
                parallelFor(106,
                    {
                        const int id = loopId;
                        // select a circle
                        const float x = parameters[id*2] * 10.0f; // scaling the normalized parameter to the problem dimensions
                        const float y = parameters[id*2+1] * 10.0f; // scaling the normalized parameter to the problem dimensions

                        // brute-force checking collisions with other circles
                        for(int i=0;i<106;i++)
                        {
                            if(i == id) continue;
                            const float x2 = parameters[i*2] * 10.0f;
                            const float y2 = parameters[i*2+1] * 10.0f;
                            const float dx = x-x2;
                            const float dy = y-y2;
                            // no need for sqrt since its just comparison
                            const float r2 = dx*dx + dy*dy;
                            if(r2<1.0f)
                            {
                                energy += 1.0f / (r2+0.00001f); // with smoothing to counter division-by-zero
                            }
                        }

                        // checking collisions with square borders
                        if(x<0.5f)
                            energy += (x-0.5f)*(x-0.5f)*100.0f;

                        if(y<0.5f)
                            energy += (y-0.5f)*(y-0.5f)*100.0f;

                        if(x>9.5f)
                            energy += (x-9.5f)*(x-9.5f)*100.0f;

                        if(y>9.5f)
                            energy += (y-9.5f)*(y-9.5f)*100.0f;
                    });
                
        )");


        sim.build();
        float startTemperature = 1.0f; 
        float stopTemperature = 0.001f; 
        float coolingRate = 1.2f;
        bool debugPerformance = false;
        bool debugDevice = false;
        bool debugEnergy = true;
        int numReHeating = 100; 
        std::vector<float> prmInitialGuess(106*2,0.0f); // (optional) guessing all circles at 0.0
        std::vector<float> prm = sim.run(
            startTemperature, stopTemperature, coolingRate, numReHeating, 
            debugPerformance, debugDevice, debugEnergy,
            [](float * optimizedParameters) {
                // callback that is called whenever a better(lower) energy is found
                for (int i = 0; i < 106; i++)
                {
                    // do something with circle positions, render, etc
                }
                std::cout << "------" << std::endl;
            },
            prmInitialGuess
        );

        for (int i = 0; i < 106; i++)
        {
            // you can paste this into desmos graph page to see the circles
            std::cout << "(x-" << prm[i * 2]*10.0f << ")^2 + (y-" << prm[i * 2 + 1]*10.0f<<")^2 = 0.25" << std::endl;
           
        }

    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }
    return 0;
}