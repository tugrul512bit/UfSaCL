#include"UfSaCL.h"
int main()
{
    try
    {
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
        std::vector<int> test = { 1,2 };
        sim.addUserInput("test", test);
        sim.build();
        for (int i = 0; i < 10; i++)
        {
            std::vector<float> prm = sim.run(1, 0.001, 1.1, true,true);
            float s = 0.0f;
            for (auto& e : prm)
            {
                std::cout << e << std::endl;
            }
            std::cout << "---------------" << std::endl;

        }
    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }
    return 0;
}