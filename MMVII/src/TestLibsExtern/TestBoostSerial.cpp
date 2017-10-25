#include "include/MMVII_all.h"

#include <algorithm>
#include <tuple>
#include <typeinfo>
#include <forward_list>
#include <unordered_map>
#include <functional>


#include <boost/config.hpp>
#include <boost/array.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <cstdio> // remove
#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{ 
    using ::remove;
}
#endif

#include <boost/archive/tmpdir.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>


//#include <Eigen/Dense>

namespace MMVII
{

class cEwelina 
{
	friend std::ostream & operator<<(std::ostream &os, const cEwelina &aE);

	public :
		cEwelina(int &at, std::string& bt) {a=at; b=bt;}
		int a; 
		std::string b;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version)
		{ ar & BOOST_SERIALIZATION_NVP (a)
             	& BOOST_SERIALIZATION_NVP (b); };

	private:
	    	friend class boost::serialization::access;
};

std::ostream & operator<<(std::ostream &os, const cEwelina &aE)
{
    os << aE.a << " " << aE.b ;
    return os;
}



/*************************************************************/
/*                                                           */
/*            cAppli_MMVII_TestBoostSerial                   */
/*                                                           */
/*************************************************************/
template<class Archive>
void serialize(Archive & ar, cPt2dr & aP, const unsigned int version)
{
    ar & aP.x();
    ar & aP.y();
}


void TestBoostSerial()
{

 // create class instance
    std::ofstream ofs("filename");
    ofs.precision(6);
    ofs << std::setprecision(6);
    // const 
    cPt2dr aP(1,2);

    // save data to archive
    {
        boost::archive::text_oarchive oa(ofs);
        // write class instance to archive
        oa << aP;
        oa << aP;
    	// archive and stream closed when destructors are called
    }

    // ... some time later restore the class instance to its orginal state
    cPt2dr aNewP(0,0);
    {
        // create and open an archive for input
        std::ifstream ifs("filename");
        boost::archive::text_iarchive ia(ifs);
        // read class state from archive
        ia >> aNewP;
        ia >> aNewP;
        // archive and stream closed when destructors are called
    }
    std::cout  << " AAATestBoostSerial " << aNewP.x() << " " <<  aNewP.y() << "\n";

    int i=3;
    std::string j="eeee";
    const cEwelina aER(i,j);
    std::ofstream xml_ofs("filename.xml");
    boost::archive::xml_oarchive oa(xml_ofs);
    oa << BOOST_SERIALIZATION_NVP(aER);
}

class cAppli_MMVII_TestBoostSerial : public cMMVII_Appli
{
     public :
        cAppli_MMVII_TestBoostSerial(int argc,char** argv) ;
        int Exe();
};

cAppli_MMVII_TestBoostSerial::cAppli_MMVII_TestBoostSerial (int argc,char **argv) :
    cMMVII_Appli
    (
        argc,
        argv,
        DirCur(),
        cArgMMVII_Appli
        (
        )
    )
{
}

int cAppli_MMVII_TestBoostSerial::Exe()
{
    TestBoostSerial();
}

tMMVII_UnikPApli Alloc_MMVII_TestBoostSerial(int argc,char ** argv)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_TestBoostSerial(argc,argv));
}


cSpecMMVII_Appli  TheSpec_TestBoostSerial
(
     "TBS",
      Alloc_MMVII_TestBoostSerial,
      "This command execute some experiments en boost serrialization",
      "Test",
      "None",
      "Console"
);



/*
*/




};
