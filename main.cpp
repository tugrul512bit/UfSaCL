#include"UfSaCL.h"
#include<vector>
#include<iostream>

int main()
{
    try
    {
        // neural network architecture to learn y = sqrt(x) [1 input --> 1 output]
        constexpr int nLayerInput = 1;
        constexpr int nLayerHidden1 = 10;
        constexpr int nLayerHidden2 = 20;
        constexpr int nLayerHidden3 = 10;
        constexpr int nLayerOutput = 1;
        constexpr int nParams = (nLayerInput * 2) + (nLayerInput * nLayerHidden1 + nLayerHidden1) + (nLayerHidden1 * nLayerHidden2 + nLayerHidden2) + (nLayerHidden2 * nLayerHidden3 + nLayerHidden3) + (nLayerHidden3 * nLayerOutput + nLayerOutput);
        std::vector<int> architecture = { nLayerInput,nLayerHidden1,nLayerHidden2,nLayerHidden3,nLayerOutput };

        
        // gpu-accelerated simulated-annealing that launches 1 block per simulation
        UFSACL::UltraFastSimulatedAnnealing<nParams, 500> sim(R"(
     
                const int nData = settings[0];
                const int nLayers = settings[1];
                float energyLocal = 0.0f;

                // do same work for each pair of input-output & compute error (as energy for simulated annealing)
                // this parallel for loop is not per workitem but works on all workitems at once, so any local variables (energyLocal) only visible by themselves
                parallelFor(nData,
                {
                        int i=loopId;
                        float trainingDataInputTmp[1];
                        float trainingDataOutputTmp[1];

                        trainingDataInputTmp[0] = trainingDataInput[i];
                        trainingDataOutputTmp[0] = 0.0f;
                        
                        Compute(architecture, trainingDataInputTmp, trainingDataOutputTmp, nLayers, parameters);
                        float diff = (trainingDataOutput[i] - trainingDataOutputTmp[0]);
                        energy += pow(fabs(diff),0.5f);
                        
                });

                energy += energyLocal;
                
        )");

        sim.addFunctionDefinition(R"(
    
            void Compute(global int * architecture, float * input, float * output, int numLayers, local float * parameters)
            {
                int parameterCtr = 0;
                float layerVal[256];
                float layerValTmp[256];
                for(int i=0;i<numLayers;i++)
                {
                    if(i==0)
                    {
                        // input layer
                        int n = architecture[i];
                        for(int j=0;j<n;j++)
                        {
                            const float bias = parameters[parameterCtr++]*2.0f - 1.0f;

                            // neuron input multiplier
                            const float mult = parameters[parameterCtr++]*2.0f - 1.0f;

                            // neuron output
                            layerVal[j] = tanh(mult * input[j] + bias);  
                        }                       

                    }
                    else if(i==numLayers-1)
                    {
                        // output layer
                        int n = architecture[i];
                        int n0 = architecture[i-1];
                        for(int j=0;j<n;j++)
                        {
                            const float bias = parameters[parameterCtr++]*2.0f - 1.0f;
                            float acc = 0.0f;
                            for(int k=0;k<n0;k++)
                            {
                                // neuron input multiplier
                                const float mult = parameters[parameterCtr++]*2.0f - 1.0f;

                                // neuron output
                                acc += mult * layerVal[k];  
                            }

                            output[j] = tanh(acc + bias);
                        }  
                    }
                    else
                    {
                        // hidden layer
                        int n = architecture[i];
                        int n0 = architecture[i-1];
                        for(int j=0;j<n;j++)
                        {
                            const float bias = parameters[parameterCtr++]*2.0f - 1.0f;
                            float acc = 0.0f;
                            for(int k=0;k<n0;k++)
                            {
                                // neuron input multiplier
                                const float mult = parameters[parameterCtr++]*2.0f - 1.0f;

                                // neuron output
                                acc += mult * layerVal[k];  
                            }

                            layerValTmp[j] = tanh(acc + bias);
                        }        

                        for(int j=0;j<n;j++)               
                            layerVal[j]=layerValTmp[j];

                    }
                }
    
            }

        )");

        const int nTrainingData = 25000;
        std::vector<float> trainingDataInput(nTrainingData);
        std::vector<float> trainingDataOutput(nTrainingData);
        std::vector<int> settings = { nTrainingData,(int)architecture.size()};
        for (int i = 0; i < nTrainingData; i++)
        {
            trainingDataInput[i] = (i)/(double) nTrainingData;
            trainingDataOutput[i] = std::sqrt((double)trainingDataInput[i]);
        }
        sim.addUserInput("architecture", architecture);
        sim.addUserInput("trainingDataInput", trainingDataInput);
        sim.addUserInput("trainingDataOutput", trainingDataOutput);
        sim.addUserInput("settings", settings);
        sim.build();
        float startTemperature = 1.0f;
        float stopTemperature = 0.0001f;
        float coolingRate = 1.1f;
        bool debugPerformance = false;
        bool debugDevice = false;
        bool debugEnergy = true;
        int numReHeating = 5;
        std::vector<float> prm = sim.run(
            startTemperature, stopTemperature, coolingRate, numReHeating,
            debugPerformance, debugDevice, debugEnergy,
            [&](float* optimizedParameters) {

                float input[1] = { 0.5f };
                float output[1] = { 0.0f };
                float* parameters = optimizedParameters;
                {
                    int parameterCtr = 0;
                    float layerVal[256];
                    float layerValTmp[256];
                    for (int i = 0; i < architecture.size(); i++)
                    {
                        if (i == 0)
                        {
                            // input layer
                            int n = architecture[i];
                            for (int j = 0; j < n; j++)
                            {
                                const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;

                                // neuron input multiplier
                                const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                // neuron output
                                layerVal[j] = tanh(mult * input[j] + bias);
                            }

                        }
                        else if (i == architecture.size() - 1)
                        {
                            // output layer
                            int n = architecture[i];
                            int n0 = architecture[i - 1];
                            for (int j = 0; j < n; j++)
                            {
                                const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;
                                float acc = 0.0f;
                                for (int k = 0; k < n0; k++)
                                {
                                    // neuron input multiplier
                                    const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                    // neuron output
                                    acc += mult * layerVal[k];
                                }

                                output[j] = tanh(acc + bias);
                            }
                        }
                        else
                        {
                            // hidden layer
                            int n = architecture[i];
                            int n0 = architecture[i - 1];
                            for (int j = 0; j < n; j++)
                            {
                                const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;
                                float acc = 0.0f;
                                for (int k = 0; k < n0; k++)
                                {
                                    // neuron input multiplier
                                    const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                    // neuron output
                                    acc += mult * layerVal[k];
                                }

                                layerValTmp[j] = tanh(acc + bias);
                            }

                            for (int j = 0; j < n; j++)
                                layerVal[j] = layerValTmp[j];

                        }
                    }


                    std::cout << "sqrt (0.5) =" << output[0] << std::endl;

                }
                std::cout << "------" << std::endl;
            }
        );

        for (float inp = 0.15; inp < 0.95; inp += 0.1)
        {


            float input[1] = { inp };
            float output[1] = { 0.0f };
            float* parameters = prm.data();
            {
                int parameterCtr = 0;
                float layerVal[256];
                float layerValTmp[256];
                for (int i = 0; i < architecture.size(); i++)
                {
                    if (i == 0)
                    {
                        // input layer
                        int n = architecture[i];
                        for (int j = 0; j < n; j++)
                        {
                            const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;

                            // neuron input multiplier
                            const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                            // neuron output
                            layerVal[j] = tanh(mult * input[j] + bias);
                        }

                    }
                    else if (i == architecture.size() - 1)
                    {
                        // output layer
                        int n = architecture[i];
                        int n0 = architecture[i - 1];
                        for (int j = 0; j < n; j++)
                        {
                            const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;
                            float acc = 0.0f;
                            for (int k = 0; k < n0; k++)
                            {
                                // neuron input multiplier
                                const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                // neuron output
                                acc += mult * layerVal[k];
                            }

                            output[j] = tanh(acc + bias);
                        }
                    }
                    else
                    {
                        // hidden layer
                        int n = architecture[i];
                        int n0 = architecture[i - 1];
                        for (int j = 0; j < n; j++)
                        {
                            const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;
                            float acc = 0.0f;
                            for (int k = 0; k < n0; k++)
                            {
                                // neuron input multiplier
                                const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                // neuron output
                                acc += mult * layerVal[k];
                            }

                            layerValTmp[j] = tanh(acc + bias);
                        }

                        for (int j = 0; j < n; j++)
                            layerVal[j] = layerValTmp[j];

                    }
                }


               

            }
            std::cout << "------" << std::endl;
            std::cout << "sqrt(" << input[0] << ")=" << output[0] << "  error = " << (output[0] - std::sqrt(input[0])) / std::sqrt(input[0]) * 100 << "%" << std::endl;
        }

    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }
    return 0;
}