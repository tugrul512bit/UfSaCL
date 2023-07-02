#pragma once

#include "libGPGPU/gpgpu.hpp"
#include<vector>
#include<string>
#include<cstdint>
#include<iostream>
namespace UFSACL
{

    // abstract solver that takes user algorithm into OpenCL kernel and runs on thousands of (GPU/CPU) threads
    // supports maximum 2 billion elements for (num paramters X num objects) [example: 1000 parameters for 2 million objects]
    // NumObjects = number of clones of state-machine (that are computed in parallel)
    // NumParameters = number of parameters to tune to minimize energy
    template<int NumParameters, int NumObjects>
    struct UltraFastSimulatedAnnealing
    {
    private:
        std::string kernel;
        GPGPU::Computer computer;
        GPGPU::HostParameter randomDataIn;
        GPGPU::HostParameter energyOut;
        GPGPU::HostParameter parameterIn;
        GPGPU::HostParameter parameterOut;
        GPGPU::HostParameter randomDataOut;
        GPGPU::HostParameter temperatureIn;
        std::vector<GPGPU::HostParameter> userInputFullAccess;
        int numWorkGroupsToRun;
        int workGroupThreads;
        int numParametersItersPerWorkgroupWithUnused;
        std::string constants;
        float currentEnergy;
        std::vector<float> currentParameters;
        std::string userInputs;
        std::string userInputsWithoutTypes;
        std::string funcMin;
    public:
        UltraFastSimulatedAnnealing(std::string funcToMinimize) :computer(GPGPU::Computer::DEVICE_ALL)
        {
            workGroupThreads = 256;
            numWorkGroupsToRun = NumObjects;

            if (NumParameters % workGroupThreads != 0)
                numParametersItersPerWorkgroupWithUnused = (NumParameters / workGroupThreads) + 1;
            else
                numParametersItersPerWorkgroupWithUnused = NumParameters / workGroupThreads;


            currentParameters.resize(NumParameters);
            funcMin = funcToMinimize;
        }

        void build()
        {

            constants = std::string(R"(
            #define NumItems )") + std::to_string(NumParameters * NumObjects) + std::string(R"(
        )");
            constants += std::string(R"(
            #define UIMAXFLOATINV (2.32830644e-10f)
        )");
            constants += std::string(R"(
            #define WorkGroupThreads )") + std::to_string(workGroupThreads) + std::string(R"(
        )");

            constants += std::string(R"(
            #define NumParameters )") + std::to_string(NumParameters) + std::string(R"(
        )");

            constants += std::string(R"(
            #define NumParamsPerThread )") + std::to_string(numParametersItersPerWorkgroupWithUnused) + std::string(R"(
        )");



            kernel = constants + std::string(R"(
   		    const unsigned int rnd(unsigned int seed)
		    {			
			    seed = (seed ^ 61) ^ (seed >> 16);
			    seed *= 9;
			    seed = seed ^ (seed >> 4);
			    seed *= 0x27d4eb2d;
			    seed = seed ^ (seed >> 15);
			    return seed;
		    }

            const float random(unsigned int seed)
            {
                return seed * UIMAXFLOATINV;
            }



            float funcToMinimize(const int threadId, const int objectId, local float * parameters )") + userInputs + std::string(R"()
            {
                float energy = 0.0f;
                )") + funcMin + std::string(R"(
                return energy;
            }


            kernel void kernelFunction(global unsigned int * seedIn, global unsigned int * seedOut, global float * tempIn, global float * energyOut, global float * parameterIn, global float * parameterOut )") + userInputs + std::string(R"()
            {
                const int id = get_global_id(0);
                const int groupId = id / WorkGroupThreads;
                const int localId = id % WorkGroupThreads;
                local float parameters[NumParameters];
                local float energies[WorkGroupThreads];
                const float temperature = tempIn[0];
                const int numLoopIter = (NumParameters / WorkGroupThreads) + 1;
                unsigned int tmpRnd = seedIn[id];
                for(int i=0;i<numLoopIter;i++)
                {
                    const int loopId = localId + WorkGroupThreads * i;
                    if(loopId < NumParameters)
                    {
                        tmpRnd = rnd(tmpRnd);
                        parameters[loopId] = parameterIn[loopId] + (random(tmpRnd) - 0.5f) * temperature;
                        while(parameters[loopId]<0.0f)
                            parameters[loopId]+=1.0f;
                        while(parameters[loopId]>1.0f)
                            parameters[loopId]-=1.0f;
                    }
                }
                seedOut[id]=tmpRnd;
                barrier(CLK_LOCAL_MEM_FENCE);
                energies[localId] = funcToMinimize(localId,groupId,parameters )") + userInputsWithoutTypes + std::string(R"();
                barrier(CLK_LOCAL_MEM_FENCE);
                for(unsigned int i=WorkGroupThreads/2;i>=1;i>>=1)
                {
                    unsigned int reduceId = i + localId;
                    if(localId<i)
                        energies[localId] += energies[reduceId]; 
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if(localId == 0)
                    energyOut[id]=energies[0];

                for(int i=0;i<numLoopIter;i++)
                {
                    const int loopId = localId + WorkGroupThreads * i;
                    const int arrayId = loopId + groupId*WorkGroupThreads*NumParamsPerThread;
                    if(loopId < NumParameters)
                    {
                        parameterOut[arrayId] = parameters[loopId];
                    }
                }
            }
        )");

            computer.compile(kernel, "kernelFunction");
            randomDataIn = computer.createArrayInputLoadBalanced<unsigned int>("rndIn", numWorkGroupsToRun * workGroupThreads);
            randomDataOut = computer.createArrayOutput<unsigned int>("rndOut", numWorkGroupsToRun * workGroupThreads);
            energyOut = computer.createArrayOutput<float>("energyOut", numWorkGroupsToRun * workGroupThreads);

            parameterIn = computer.createArrayInput<float>("parameterIn", NumParameters);
            temperatureIn = computer.createArrayInput<float>("tempIn", 1);
            parameterOut = computer.createArrayOutput<float>("parameterOut",
                numWorkGroupsToRun * numParametersItersPerWorkgroupWithUnused * workGroupThreads, numParametersItersPerWorkgroupWithUnused);

            for (int i = 0; i < numWorkGroupsToRun * workGroupThreads; i++)
                randomDataIn.access<unsigned int>(i) = i;
        }

        // use additional buffers from host-environment in simulated-annealing kernel
        // use same name with this in kernel when accessing data
        // only simulated-annealing (to be minimized) energy function parameters are cached inside local (in-chip) fast memory
        // any data added with addUserInput method is directly accessed from video-memory that is likely cached by hardware
        // if data with same name exists, it updates the data
        template<typename T>
        void addUserInput(std::string customInputName, std::vector<T> customInput)
        {
            const int sz = userInputFullAccess.size();
            for (int i = 0; i < sz; i++)
            {
                if (customInputName == userInputFullAccess[i].getName())
                {
                    userInputFullAccess[i].copyDataFromPtr(customInput.data());
                    return;
                }
            }

            // fully-copied array (for all GPGPU devices to have all-element random-access in kernel)
            userInputFullAccess.emplace_back(computer.createArrayInput<T>(customInputName, customInput.size(), 1));
            userInputFullAccess[sz].copyDataFromPtr(customInput.data());
            userInputsWithoutTypes += std::string(", ") + customInputName;
            if (typeid(T) == typeid(char))
            {
                userInputs += std::string(", global char * ") + customInputName;
            }
            else if (typeid(T) == typeid(unsigned char))
            {
                userInputs += std::string(", global unsigned char * ") + customInputName;
            }
            else if (typeid(T) == typeid(bool))
            {
                userInputs += std::string(", global unsigned char * ") + customInputName; // using char instead of bool since opencl not good at that
            }
            else if (typeid(T) == typeid(short))
            {
                userInputs += std::string(", global short * ") + customInputName;
            }
            else if (typeid(T) == typeid(unsigned short))
            {
                userInputs += std::string(", global unsigned short * ") + customInputName;
            }
            else if (typeid(T) == typeid(int))
            {
                userInputs += std::string(", global int * ") + customInputName;
            }
            else if (typeid(T) == typeid(unsigned int))
            {
                userInputs += std::string(", global unsigned int * ") + customInputName;
            }
            else if (typeid(T) == typeid(long long))
            {
                userInputs += std::string(", global long * ") + customInputName;
            }
            else if (typeid(T) == typeid(unsigned long long))
            {
                userInputs += std::string(", global unsigned long * ") + customInputName;
            }
            else if (typeid(T) == typeid(float))
            {
                userInputs += std::string(", global float * ") + customInputName;
            }
            else if (typeid(T) == typeid(double))
            {
                userInputs += std::string(", global double * ") + customInputName;
            }
            else if (typeid(T) == typeid(std::int8_t))
            {
                userInputs += std::string(", global char * ") + customInputName;
            }
            else if (typeid(T) == typeid(std::int16_t))
            {
                userInputs += std::string(", global short * ") + customInputName;
            }
            else if (typeid(T) == typeid(std::int32_t))
            {
                userInputs += std::string(", global int * ") + customInputName;
            }
            else if (typeid(T) == typeid(std::int64_t))
            {
                userInputs += std::string(", global long * ") + customInputName;
            }
            else if (typeid(T) == typeid(std::uint64_t))
            {
                userInputs += std::string(", global unsigned long * ") + customInputName;
            }

            return;
        }


        std::vector<float> run(
            const float temperatureStart = 1.0f, const float temperatureStop = 0.01f, const float temperatureDivider = 2.0f, 
            const int numReheats = 5,
            const bool debug = false, const bool deviceDebug = false, const bool energyDebug=false)
        {
            int reheat = numReheats;
            auto kernelParams = randomDataIn.next(randomDataOut).next(temperatureIn).next(energyOut).next(parameterIn).next(parameterOut);
            const int sz = userInputFullAccess.size();
            for (int i = 0; i < sz; i++)
            {
                auto kernelParamsNew = kernelParams.next(userInputFullAccess[i]);
                kernelParams = kernelParamsNew;
            }

            // initial guess for parameters (middle-points for all dimensions)
            for (int i = 0; i < NumParameters; i++)
            {
                parameterIn.access<float>(i) = 0.5f;
            }

            // initialize temperature
            temperatureIn.access<float>(0) = temperatureStart;

            float temp = temperatureStart;
            float foundEnergy = std::numeric_limits<float>::max();
            int foundId = -1;
            int iter = 0;
        
            std::vector<double> perf;
            size_t measuredNanoSecTot = 0;
            {
                GPGPU::Bench benchTot(&measuredNanoSecTot);
                while (temp > temperatureStop)
                {
                    if (debug)
                        std::cout << "iteration-" << iter++ << std::endl;
                    bool foundBetterEnergy = false;
                    size_t measuredNanoSec = 0;
                    {
                        GPGPU::Bench bench(&measuredNanoSec);
                        perf = computer.compute(kernelParams, "kernelFunction", 0, numWorkGroupsToRun * workGroupThreads, workGroupThreads);
                        randomDataIn.copyDataFromPtr(randomDataOut.accessPtr<unsigned int>(0));
                        for (int i = 0; i < NumObjects; i++)
                        {
                            const int index = i * workGroupThreads;

                            const float energy = energyOut.access<float>(index);
                            if (foundEnergy > energy)
                            {
                                foundEnergy = energy;
                                foundId = i;
                                foundBetterEnergy = true;
                            }

                        }
                    }
                    if (debug)
                        std::cout << "computation-time=" << measuredNanoSec * 0.000000001 << " seconds" << std::endl;
                    if (foundBetterEnergy)
                    {
                        for (int i = 0; i < NumParameters; i++)
                        {
                            currentParameters[i] = parameterOut.access<float>(i + foundId * numParametersItersPerWorkgroupWithUnused * workGroupThreads);
                        }

                        // new low-energy point becomes new guess for next iteration
                        for (int i = 0; i < NumParameters; i++)
                        {
                            parameterIn.access<float>(i) = currentParameters[i];
                        }

                        if (energyDebug)
                            std::cout << "lower energy found: " << foundEnergy << std::endl;
                    }

                    temp /= temperatureDivider;
                    temperatureIn.access<float>(0) = temp;

                    if (!(temp > temperatureStop))
                    {
                        reheat--;
                        if (reheat == 0)
                        {
                            break;
                        }
                        else
                        {
                            if (debug || energyDebug)
                                std::cout << "reheating. num reheats left="<< reheat << std::endl;

                            temp = temperatureStart;
                            iter = 0;
                        }
                    }
                }
            }
            if (debug || energyDebug)
                std::cout << "total computation-time=" << measuredNanoSecTot * 0.000000001 << " seconds (this includes debugging console-output that is slow)" << std::endl;

            if (deviceDebug || energyDebug)
            {
                std::cout << "---------------" << std::endl;
                std::cout << "OpenCL device info:" << std::endl;
                auto names = computer.deviceNames(false);
                for (int i = 0; i < names.size(); i++)
                {
                    std::cout << names[i] << " computed " << (perf[i] * 100.0) << "% of total work" << std::endl;
                }
                std::cout << "---------------" << std::endl;
            }
            return currentParameters;
        }
    };
}