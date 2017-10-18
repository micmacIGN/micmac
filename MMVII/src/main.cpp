#include "../include/all.h"

int MPDtest_main(int argc, char ** argv);

void  TestSharedPointer();
void  TestLE2();




void  ShowArgs(int argc, char ** argv)
{
    for (int aK=0 ; aK<argc ; aK++)
        std::cout  << "Arg[" << aK << "]=" << argv[aK] << "\n";
}

int main(int argc, char ** argv)
{
   TestSharedPointer();
   TestLE2();
/*
    std::shared_ptr<int> p0(new int(5));  
    std::shared_ptr<int> p1 = p0;
    std::cout << "Count = " << p0.use_count() << "\n";
    auto i = 3.0;
    std::cout << "Hello word \n";
    std::cout << typeid(p0).name() << "\n" ;
    std::cout << typeid(i).name() << "\n" ;
*/
    MPDtest_main(argc,argv);
}

