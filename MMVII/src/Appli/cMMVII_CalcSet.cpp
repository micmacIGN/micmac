#include "include/MMVII_all.h"
#include <boost/regex.hpp> 
// #include <boost/cregex.hpp> 

/** \file cMMVII_CalcSet.cpp
    \brief Command for set calculation

    This file contain the command that compute a  set of file from File/Regex
  It's also the first "real" command of MMVII, so an occasion for tuning a 
  a lot of thing.

*/


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
   cInterfRegex (aName),
   mRegex       (aName)
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
       cSetName();
       cSetName(const std::string &,bool AllowPat);
       cSetName(const  cInterfSet<std::string> &);
       

       size_t size() const;
       void Add(const  std::string &);
       cInterfSet<std::string> * ToSet();
      
   private :
   // private :
       void InitFromString(const std::string &,bool AllowPat);
       void InitFromFile(const std::string &);
       void InitFromPat(const std::string & aFullPat);
       friend void  AddData(const cAuxAr2007 & anAux,cSetName & aSON);
       std::vector<std::string> mV;
};

/* ====================================== */
/*                                        */
/*             cSetName                   */
/*                                        */
/* ====================================== */

cSetName::cSetName()
{
}

cSetName::cSetName(const std::string & aName,bool AllowPat) :
    cSetName()
{
   InitFromString(aName,AllowPat);
}

cSetName::cSetName(const  cInterfSet<std::string> & aSet)
{
   std::vector<const std::string *> aVPtr;
   aSet.PutInSet(aVPtr);
   for (auto el : aVPtr)
       mV.push_back(*el);
}

cInterfSet<std::string> * cSetName::ToSet()
{
   cInterfSet<std::string> * aRes = AllocUS<std::string>();
   for (auto el:mV)
       aRes->Add(el);
   return aRes;
}


void cSetName::Add(const  std::string & aName)
{
   mV.push_back(aName);
}

size_t cSetName::size() const
{
     return mV.size();
}


void  AddData(const cAuxAr2007 & anAux,cSetName & aSON)
{
    AddData(cAuxAr2007(TagSetOfName,anAux) ,aSON.mV);
}

void cSetName::InitFromFile(const std::string & aNameFile)
{
   // MMVII_INTERNAL_ASSERT_user(false,"Non xml file"I);

    
    ReadFromFileWithDef(*this,aNameFile);
std::cout << "InitFromFile " << aNameFile  << " NB: " << mV.size() << "\n";
for (auto el:mV)
   std::cout << "III " << el << "\n";
}

void  cSetName::InitFromPat(const std::string & aFullPat)
{
     std::string aDir,aPat;
     SplitDirAndFile(aDir,aPat,aFullPat,false);

// std::cout << "D=" << aDir << " P=" << aPat << "\n";
     cBoostRegex aBE(aPat);
// std::cout << "HHHHHHH\n";
     GetFilesFromDir(mV,aDir,aBE);
}


void cSetName::InitFromString(const std::string & aName,bool AllowPat)
{
   if ((! AllowPat) || IsFile2007XmlOfGivenTag(aName,TagSetOfName))
      InitFromFile(aName);
   else 
      InitFromPat(aName);
}


/* ==================================================== */
/*                                                      */
/*                                                      */
/*                                                      */
/* ==================================================== */

/// An application for editing set of file
/**
    Given an XML memorizing a set of file, it is possible to :

      - add a new set (+=)
      - substract a new set (-=)
      - intersect a new set (*=)
      - overwrite with a new set (=)
*/
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
        <<  Arg2007(mXml,"Full Name of Xml in/out",{eTA2007::FileDirProj})
        <<  Arg2007(mOp,"Operator (= += -= *= 0)")
        <<  Arg2007(mPat,"Pattern or Xml")
     ,
     mArgFac
        <<  AOpt2007(mShow,"Show","Full Name of Xml in/out",{})
  );
}

int cAppli_EditSet::Exe()
{
   std::string Opers="= *= += -= =0";
   std::vector<std::string>  aVOps = SplitString(Opers," ");

   cSetName aInput(mXml,false);
   cSetName aNew(mPat,true);
    
   if (mOp==aVOps.at(0)) // =
   {
      aInput = aNew;
   }
   else
   {
       cInterfSet<std::string> *  aSIn = aInput.ToSet();
       cInterfSet<std::string> *  aSetNew = aNew.ToSet();

       if (mOp==aVOps.at(1)) // *=
       {
             *aSIn *=  *aSetNew;
       }
       else
       {
           MMVII_INTERNAL_ASSERT_user(false,"Unknown set operator :["+mOp+"] allowed: "+ Opers);
       }

       aInput  = cSetName(*aSetNew);
   }
   SaveInFile(aInput,mXml);

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

