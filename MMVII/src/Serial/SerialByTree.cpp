#include "MMVII_Stringifier.h"

/*
#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_MeasuresIm.h"
*/


/** \file SerialByTree.cpp
    \brief Implementation of serialisation using a tree represention, instead
    of a streaming.

    Streaming if more efficient for big data, for example when used for exchanging data via binary file
    or unstructurd text file, but is more complicated 

    In SerialByTree we firt create a universall tree representation and then of the data and then
    transform

*/


namespace MMVII
{
class cNodeSerial
{
      public :
          std::string             mTag;
	  std::string             mValue;
	  std::list<cNodeSerial>  mSons;
};

class  cXML_CreateSerial
{
};


void TestcNodeSerial()
{
	cNodeSerial aN;
	aN.mSons.push_back(aN);
}






};

