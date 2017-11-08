#include "include/MMVII_all.h"
#include <boost/regex.hpp> 
// #include <boost/cregex.hpp> 

namespace MMVII
{


/* ======================================== */
/*                                          */
/*      cNameSelector                       */
/*                                          */
/* ======================================== */

cNameSelector::~cNameSelector()
{
}

/* ======================================== */
/*                                          */
/*     cInterfRegex                         */
/*                                          */
/* ======================================== */

cInterfRegex::cInterfRegex(const std::string & aName) :
   mName (aName)
{
}

cInterfRegex::~cInterfRegex()
{
}

const std::string & cInterfRegex::Name() const
{
   return mName;
}
/* ======================================== */
/*                                          */
/*     cBoostRegex                          */
/*                                          */
/* ======================================== */

/// Boost implementation of Regex expression
class cBoostRegex : public  cInterfRegex
{
    public :
        cBoostRegex(const std::string &);
        bool Match(const std::string &) const override ;
    private :
        boost::regex mRegex;
};


cBoostRegex::cBoostRegex(const std::string & aName) :
   cInterfRegex (aName)
{
}

bool cBoostRegex::Match(const std::string & aStr) const 
{
    return regex_match(aStr,mRegex);
}

/*=================================*/

class cSetName
{
   public :

   private :

       std::vector<std::string> mV;
};

/* ==================================================== */
/*                                                      */
/*                                                      */
/*                                                      */
/* ==================================================== */
class cAppli_EditSet : public cMMVII_Appli
{
     public :
        cAppli_EditSet(int argc,char** argv);
        int Exe() override;
     private :
         std::string mXml;
         std::string mPat;
         std::string mOp;
         bool        mShow;
};


cAppli_EditSet::cAppli_EditSet(int argc,char** argv) :
  cMMVII_Appli (argc,argv),
  mShow        (false)
{
   InitParam
   (
      mArgObl 
        <<  Arg2007(mXml,"Full Name of Xml in/out",{eTA2007::ProjectDir})
        <<  Arg2007(mOp,"Operator (= += -= *= 0)")
        <<  Arg2007(mPat,"Pattern or Xml")
     ,
     mArgFac
        <<  AOpt2007(mShow,"Show","Full Name of Xml in/out",{})
  );
}

int cAppli_EditSet::Exe()
{
   return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_EditSet(int argc,char ** argv)
{
   return tMMVII_UnikPApli(new cAppli_EditSet(argc,argv));
}

cSpecMMVII_Appli  TheSpecEditSet
(
     "EditSet",
      Alloc_EditSet,
      "This command is used to edit set of file",
      {eApF::Project},
      {eApDT::Console,eApDT::Xml},
      {eApDT::Xml}

);



};

