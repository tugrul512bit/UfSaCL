#pragma once

#include "libGPGPU/gpgpu.hpp"
#include<vector>
#include<string>
#include<cstdint>
#include<iostream>
#include<random>
#include<limits>
namespace UFSACL
{

    // abstract solver that takes user algorithm into OpenCL kernel and runs on thousands of (GPU/CPU) threads
    // supports maximum 2 billion elements for (num paramters X num objects) [example: 1000 parameters for 2 million objects]
    // NumObjects = number of clones of state-machine (that are computed in parallel)
    // NumParameters = number of parameters to tune to minimize energy
    // ParameterType = float or double
    template<int NumParameters, int NumObjects, typename ParameterType = float>
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
        ParameterType currentEnergy;
        std::vector<ParameterType> currentParameters;
        std::vector<ParameterType> bestParameters;
        std::string userInputs;
        std::string userInputsWithoutTypes;
        std::string userFunction;
        std::string funcMin;
    public:
        UltraFastSimulatedAnnealing(std::string funcToMinimize, int gpuThreadsPerObject = 256, int numGPUsToUse = 16) :computer(GPGPU::Computer::DEVICE_ALL, -1, 1, true, numGPUsToUse)
        {
            workGroupThreads = gpuThreadsPerObject;
            numWorkGroupsToRun = NumObjects;

            if (NumParameters % workGroupThreads != 0)
                numParametersItersPerWorkgroupWithUnused = (NumParameters / workGroupThreads) + 1;
            else
                numParametersItersPerWorkgroupWithUnused = NumParameters / workGroupThreads;


            currentParameters.resize(NumParameters);
            bestParameters.resize(NumParameters);
            funcMin = funcToMinimize;
        }

        void build()
        {

            constants = std::string(R"(
            #define NumItems )") + std::to_string(NumParameters * NumObjects) + std::string(R"(
        )");
            if constexpr (std::is_floating_point_v<ParameterType> && sizeof(ParameterType) == 4)
                constants += std::string(R"(
                    #define GPGPU_REAL_VAL float
                    #define GPGPU_ZERO_REAL_VAL (0.0f)
                    #define UIMAXFLOATINV (2.32830644e-10f)
                )");
            else if (std::is_floating_point_v<ParameterType> && sizeof(ParameterType) == 8)
                constants += std::string(R"(
                    #define GPGPU_REAL_VAL double
                    #define GPGPU_ZERO_REAL_VAL (0.0)
                    #define UIMAXFLOATINV (2.32830644e-10)
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

            constants += userFunction;

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

            const GPGPU_REAL_VAL random(unsigned int seed)
            {
                return seed * UIMAXFLOATINV;
            }

#define parallelFor(ITERS,BODY)                                 \
{\
    const int numLoopIter = (ITERS / WorkGroupThreads) + 1;     \
        for(int iGPGPU=0;iGPGPU<numLoopIter;iGPGPU++)                          \
        {                                                       \
            const int loopId = threadId + WorkGroupThreads * iGPGPU; \
            if(loopId < ITERS)                                  \
            {                                                   \
                BODY                                           \
            }                                                   \
        }                                                       \
}

#define parallelForWithBarrier(ITERS,BODY)                                 \
{\
    const int numLoopIter = (ITERS / WorkGroupThreads) + 1;     \
        for(int iGPGPU=0;iGPGPU<numLoopIter;iGPGPU++)                          \
        {                                                       \
            const int loopId = threadId + WorkGroupThreads * iGPGPU; \
            if(loopId < ITERS)                                  \
            {                                                   \
                BODY                                           \
            }                                                   \
            barrier(CLK_LOCAL_MEM_FENCE);                       \
        }                                                       \
}



            kernel void kernelFunction(global unsigned int * seedIn, global unsigned int * seedOut, global GPGPU_REAL_VAL * tempIn, global GPGPU_REAL_VAL * energyOut, global GPGPU_REAL_VAL * parameterIn, global GPGPU_REAL_VAL * parameterOut )") + userInputs + std::string(R"()
            {
                const int id = get_global_id(0);
                const int groupId = id / WorkGroupThreads;
                const int localId = id % WorkGroupThreads;
                local GPGPU_REAL_VAL parameters[NumParameters];
                local GPGPU_REAL_VAL energies[WorkGroupThreads];
                const GPGPU_REAL_VAL temperature = tempIn[0];
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

                // objective function by user                
                GPGPU_REAL_VAL energy = GPGPU_ZERO_REAL_VAL;
                const int threadId = localId;
                const int objectId = groupId;
                )") + funcMin + std::string(R"(
                energies[localId] = energy;
                // objective function end

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
            energyOut = computer.createArrayOutput<ParameterType>("energyOut", numWorkGroupsToRun * workGroupThreads);

            parameterIn = computer.createArrayInput<ParameterType>("parameterIn", NumParameters);
            temperatureIn = computer.createArrayInput<ParameterType>("tempIn", 1);
            parameterOut = computer.createArrayOutput<ParameterType>("parameterOut",
                numWorkGroupsToRun * numParametersItersPerWorkgroupWithUnused * workGroupThreads, numParametersItersPerWorkgroupWithUnused);

            for (int i = 0; i < numWorkGroupsToRun * workGroupThreads; i++)
                randomDataIn.access<unsigned int>(i) = i;
        }

        // declare a function before simulated-annealing-kernel, to improve code reusability
        // can be called multiple times or once to add all user-functions
        void addFunctionDefinition(std::string userFunctionPrm)
        {
            userFunction += R"(
)";
            userFunction += userFunctionPrm;
            userFunction += R"(
)";
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


        std::vector<ParameterType> run(
            const ParameterType temperatureStart = 1.0f, const ParameterType temperatureStop = 0.01f, const ParameterType temperatureDivider = 2.0f,
            const int numReheats = 5,
            const bool debug = false, const bool deviceDebug = false, const bool energyDebug = false,
            std::function<void(ParameterType*)> callbackLowerEnergyFound = [](ParameterType*) {},
            std::vector<ParameterType> userHintForInitialParametersNormalized = std::vector<ParameterType>()
        )
        {

            std::random_device rd;
            std::mt19937 rng{ rd() };
            std::uniform_real_distribution<float> uid(0.0f, 1.0f);



            int reheat = numReheats;
            auto kernelParams = randomDataIn.next(randomDataOut).next(temperatureIn).next(energyOut).next(parameterIn).next(parameterOut);
            const int sz = userInputFullAccess.size();
            for (int i = 0; i < sz; i++)
            {
                auto kernelParamsNew = kernelParams.next(userInputFullAccess[i]);
                kernelParams = kernelParamsNew;
            }

            // initial guess for parameters (middle-points for all dimensions or user hint)
            if (userHintForInitialParametersNormalized.size() == NumParameters)
            {
                for (int i = 0; i < NumParameters; i++)
                {
                    parameterIn.access<ParameterType>(i) = userHintForInitialParametersNormalized[i];
                }
            }
            else
            {
                for (int i = 0; i < NumParameters; i++)
                {
                    parameterIn.access<ParameterType>(i) = 0.5f;
                }
            }

            ParameterType temp = temperatureStart;
            ParameterType foundEnergy = std::numeric_limits<ParameterType>::max();

            // compute user-hinted parameters first
            if (userHintForInitialParametersNormalized.size() == NumParameters)
            {
                // to compute with hint parameters exactly, set temperature to zero
                temperatureIn.access<ParameterType>(0) = 0;
                // run all GPUs to iterate random seeds
                computer.compute(kernelParams, "kernelFunction", 0, numWorkGroupsToRun * workGroupThreads, workGroupThreads);
                randomDataIn.copyDataFromPtr(randomDataOut.accessPtr<unsigned int>(0));
                // get energy of hint
                foundEnergy = energyOut.access<ParameterType>(0);
            }


            // initialize temperature
            temperatureIn.access<ParameterType>(0) = temperatureStart;


            int foundId = -1;
            int iter = 0;
            int foundIdBest = -1;

            std::vector<double> perf;
            size_t measuredNanoSecTot = 0;
            ParameterType bestEnergy = foundEnergy;
            {
                GPGPU::Bench benchTot(&measuredNanoSecTot);
                while (temp > temperatureStop)
                {
                    if (debug)
                        std::cout << "iteration-" << iter++ << std::endl;
                    bool foundBetterEnergy = false;
                    bool foundBestEnergy = false;
                    size_t measuredNanoSec = 0;
                    bool doNotHeat = false;
                    {
                        GPGPU::Bench bench(&measuredNanoSec);
                        perf = computer.compute(kernelParams, "kernelFunction", 0, numWorkGroupsToRun * workGroupThreads, workGroupThreads);
                        randomDataIn.copyDataFromPtr(randomDataOut.accessPtr<unsigned int>(0));

                        ParameterType tmpEn = std::numeric_limits<double>::max();
                        int tmpI = -1;
                      
                        for (int i = 0; i < NumObjects; i++)
                        {
                            const int index = i * workGroupThreads;

                            const ParameterType energy = energyOut.access<ParameterType>(index);
                            if (tmpEn > energy)
                            {
                                tmpEn = energy;
                                tmpI = i;
                            }
                        }

                        if (foundEnergy > tmpEn && tmpI>=0)
                        {
                            foundEnergy = tmpEn;
                            foundId = tmpI;
                            foundBetterEnergy = true;
                            if (bestEnergy > tmpEn)
                            {
                                bestEnergy = tmpEn;
                                foundIdBest = tmpI;
                                foundBestEnergy = true;
                            }
                        }
                        else if(false && tmpI >= 0)
                        {
                            doNotHeat = true;
                            double rnd0 = uid(rng);
                            double dE = std::abs(foundEnergy - tmpEn) / std::abs(std::numeric_limits<double>::min() + foundEnergy);
                            
                            if (rnd0 < std::exp(-dE / (temp*0.1)))
                            {
                                foundEnergy = tmpEn;
                                foundId = tmpI;
                                foundBetterEnergy = true;
                            }
                        }
                    }
                    if (debug)
                        std::cout << "computation-time=" << measuredNanoSec * 0.000000001 << " seconds" << std::endl;
                    if (foundBetterEnergy)
                    {
                        if(!doNotHeat)
                            temp *= temperatureDivider * temperatureDivider; // as long as better states are found, temperature can be kept high


                        for (int i = 0; i < NumParameters; i++)
                        {
                            currentParameters[i] = parameterOut.access<ParameterType>(i + foundId * numParametersItersPerWorkgroupWithUnused * workGroupThreads);
                        }

                        if (foundBestEnergy)
                        {
                            for (int i = 0; i < NumParameters; i++)
                            {
                                bestParameters[i] = parameterOut.access<ParameterType>(i + foundIdBest * numParametersItersPerWorkgroupWithUnused * workGroupThreads);
                            }
                        }

                        // new low-energy point becomes new guess for next iteration
                        for (int i = 0; i < NumParameters; i++)
                        {
                            parameterIn.access<ParameterType>(i) = currentParameters[i];
                        }

                        if (energyDebug && foundBestEnergy)
                            std::cout << "lower energy found: " << bestEnergy << std::endl;

                        if(foundBestEnergy)
                            callbackLowerEnergyFound(bestParameters.data());
                    }

                    temp /= temperatureDivider;
                    temperatureIn.access<ParameterType>(0) = temp;

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
                                std::cout << "reheating. num reheats left=" << reheat << std::endl;

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
            return bestParameters;
        }
    };
}