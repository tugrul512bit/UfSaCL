#include"UfSaCL.h"
#include<vector>
#include<iostream>

int main()
{
    try
    {

        // train a neural-network to compute square-root of x
        // single layer with 8 neurons + 1 output combining neuron with simple summation
        // 16 (1 bias + 1 multiplier before tanh per neuron) + 8 (multipliers of output's input) + 1 (bias of output)
        UFSACL::UltraFastSimulatedAnnealing<16 + 8 + 1, 5000> sim(R"(
     
                const int nData = numTrainingData[0];
                double energyLocal = 0.0f;

                // do same work for each pair of input-output & compute error (as energy for simulated annealing)
                // this parallel for loop is not per workitem but works on all workitems at once, so any local variables (energyLocal) only visible by themselves
                parallelFor(nData,
                {
                        int i=loopId;

                        float output = 0.0f;
                     
                        for(int neuronId=0;neuronId<8;neuronId++)
                        {
                            
                            // neuron bias
                            const float bias = parameters[neuronId*2]*2 - 1;

                            // neuron input multiplier
                            const float mult = parameters[neuronId*2+1]*2 - 1;

                            // neuron output multiplier (actually input of the last neuron on output)
                            const float multOut = parameters[16+neuronId]*2 - 1;

                            float result = tanh(mult * trainingInput[i] + bias) * multOut;  

                            output += result;
                        }

                        // bias of output
                        float result = tanh(output + parameters[16+8]*2 - 1);

                        // how much error did it have against training data?                            
                        float diff = result - trainingOutput[i]; 
                        energyLocal += pow(fabs(diff),0.1f);
                        
                });

                energy += energyLocal;
                
        )");

        const int nTrainingData = 250000;
        std::vector<float> trainingDataInput(nTrainingData);
        std::vector<float> trainingDataOutput(nTrainingData);
        std::vector<int> numTrainingData = { nTrainingData };
        for (int i = 0; i < nTrainingData; i++)
        {
            trainingDataInput[i] = i / (double)nTrainingData;
            trainingDataOutput[i] = std::sqrt((double)trainingDataInput[i]);
        }
        sim.addUserInput("trainingInput", trainingDataInput);
        sim.addUserInput("trainingOutput", trainingDataOutput);
        sim.addUserInput("numTrainingData", numTrainingData);
        sim.build();
        float startTemperature = 1.0f;
        float stopTemperature = 0.001f;
        float coolingRate = 1.5f;
        bool debugPerformance = false;
        bool debugDevice = false;
        bool debugEnergy = true;
        int numReHeating = 3;
        std::vector<float> prm = sim.run(
            startTemperature, stopTemperature, coolingRate, numReHeating,
            debugPerformance, debugDevice, debugEnergy,
            [](float* optimizedParameters) {
                // callback that is called whenever a better(lower) energy is found
                // do something with the optimized parameters
                float inp = 0.5f;
                float out = 0.0f;
                for (int i = 0; i < 8; i++)
                {
                    const float bias = optimizedParameters[i * 2] * 2 - 1;

                    // neuron input multiplier
                    const float mult = optimizedParameters[i * 2 + 1] * 2 - 1;

                    // neuron output multiplier (actually input of the last neuron on output)
                    const float multOut = optimizedParameters[16 + i] * 2 - 1;

                    out += std::tanh(mult * inp + bias) * multOut;
                }
                std::cout << "sqrt(0.5f)=" << std::tanh(out + (optimizedParameters[16 + 8] * 2 - 1)) << std::endl;
                std::cout << "------" << std::endl;
            }
        );

        for (float inp = 0.1515; inp < 0.9595; inp += 0.1)
        {

            float out = 0.0f;
            for (int i = 0; i < 8; i++)
            {
                const float bias = prm[i * 2] * 2 - 1;

                // neuron input multiplier
                const float mult = prm[i * 2 + 1] * 2 - 1;

                // neuron output multiplier (actually input of the last neuron on output)
                const float multOut = prm[16 + i] * 2 - 1;

                out += std::tanh(mult * inp + bias) * multOut;
            }
            std::cout << "sqrt(" << inp << ")=" << std::tanh(out + (prm[16 + 8] * 2 - 1)) << "  error = " << (std::tanh(out + (prm[16 + 8] * 2 - 1)) - std::sqrt(inp)) / std::sqrt(inp) * 100 << "%" << std::endl;
        }

    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }
    return 0;
}