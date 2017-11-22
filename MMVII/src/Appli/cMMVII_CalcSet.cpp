#include "include/MMVII_all.h"

/** \file cMMVII_CalcSet.cpp
    \brief Command for set calculation

    This file contain the command that compute a  set of file from File/Regex
  It's also the first "real" command of MMVII, so an occasion for tuning a 
  a lot of thing.

*/


namespace MMVII
{



/* ====================================== */
/*                                        */
/*             cSetName                   */
/*                                        */
/* ====================================== */

            //========== Constructors =============

cSetName::cSetName()
{
}

void cSetName::Sort()
{
    std::sort(mV.begin(),mV.end());
}

cSetName::cSetName(const std::string & aName,bool AllowPat) :
    cSetName()
{
   InitFromString(aName,AllowPat);
   Sort();
}

cSetName::cSetName(const  cInterfSet<std::string> & aSet)
{
   std::vector<const std::string *> aVPtr;
   aSet.PutInSet(aVPtr,true);
   for (const auto & el : aVPtr)
       mV.push_back(*el);
}


            //===== Constructor helper =====

void cSetName::InitFromFile(const std::string & aNameFile,int aNumV)
{
    cMMVII_Appli::SignalInputFormat(aNumV);
    if (aNumV==1)
    {
       MMV1InitSet(mV,aNameFile);
    }
    else
    {
       ReadFromFileWithDef(*this,aNameFile);
    }
}


void  cSetName::InitFromPat(const std::string & aFullPat)
{
     std::string aDir,aPat;
     SplitDirAndFile(aDir,aPat,aFullPat,false);

     GetFilesFromDir(mV,aDir,BoostAllocRegex(aPat));
}


void cSetName::InitFromString(const std::string & aName,bool AllowPat)
{
   if (IsFileXmlOfGivenTag(true,aName,TagSetOfName)) // MMVII
   {
      InitFromFile(aName,2);
   }
   else if (IsFileXmlOfGivenTag(false,aName,MMv1XmlTag_SetName))  // MMv1
   {
      InitFromFile(aName,1);
   }
   else if (AllowPat)
   {
      InitFromPat(aName);
   }
   else 
   {
      InitFromFile(aName,0);
   }
}

            //============== "Sophisticated" operation

cInterfSet<std::string> * cSetName::ToSet() const
{
   cInterfSet<std::string> * aRes = AllocUS<std::string>();
   for (const auto & el:mV)
       aRes->Add(el);
   return aRes;
}

void  cSetName::Filter(tNameSelector aSel)
{
   tCont aVF;
   for (const auto & el:mV)
   {
      if (aSel->Match(el))
         aVF.push_back(el);
   }

   mV = aVF;
}


        // ==== Basic accessor

size_t                   cSetName::size() const { return mV.size(); }
const  cSetName::tCont & cSetName::Cont() const { return mV;        }

     // ====== Global for serialization ===

void  AddData(const cAuxAr2007 & anAux,cSetName & aSON)
{
    AddData(cAuxAr2007(TagSetOfName,anAux) ,aSON.mV);
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
        cAppli_EditSet(int argc,char** argv,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
     private :
         std::string mXmlIn;
         std::string mXmlOut;
         std::string mPat;
         std::string mOp;
         bool        mShow;
         std::string mAllOp;

 

};

cCollecSpecArg2007 & cAppli_EditSet::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl 
         << Arg2007(mXmlIn,"Full Name of Xml in/out",{eTA2007::FileDirProj})
         << Arg2007(mOp,"Operator ("+mAllOp+")" )
         << Arg2007(mPat,"Pattern or Xml for modifying",{{eTA2007::MPatIm,"0"}});
}

cCollecSpecArg2007 & cAppli_EditSet::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
      anArgOpt
         << AOpt2007(mShow,"Show","Show detail of set before/after",{})
         << AOpt2007(mXmlOut,"Out","Destination, def=Input",{});
}

cAppli_EditSet::cAppli_EditSet(int argc,char** argv,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (argc,argv,aSpec),
  mShow        (false),
  mAllOp       ("= *= += -= =0")
{
}

int cAppli_EditSet::Exe()
{
   if (! IsInit(&mXmlOut)) mXmlOut = mXmlIn;
   std::vector<std::string>  aVOps = SplitString(mAllOp," ");

   cSetName aInput(mXmlIn,false);
   const cSetName & aNew =  MainSet0();

   
   // we make set of them to handle unicity
   std::unique_ptr<cInterfSet<std::string> >  aSInit ( aInput.ToSet());  
   std::unique_ptr<cInterfSet<std::string> >  aSetNew ( aNew.ToSet());
   std::unique_ptr<cInterfSet<std::string> >  aRes ; // AllocUS<std::string>());

   // do the modification
   if (mOp==aVOps.at(0)) // *=
   {
       aRes.reset(aSetNew->VDupl());
   }
   else if (mOp==aVOps.at(1)) // *=
   {
      aRes.reset(*aSInit * *aSetNew);
   }
   else if (mOp==aVOps.at(2)) // +=
   {
      aRes.reset(*aSInit + *aSetNew);
   }
   else if (mOp==aVOps.at(3)) // -=
   {
      aRes.reset(*aSInit - *aSetNew);
   }
   else
   {
      MMVII_INTERNAL_ASSERT_user(false,"Unknown set operator :["+mOp+"] allowed: "+ mAllOp);
   }

   if (mShow)
   {
       std::unique_ptr<cInterfSet<std::string> >  aTot(*aSInit+* aSetNew);

       std::vector<const std::string *> aV;
       aTot->PutInSet(aV,true);
       // 0 First time show unnmodifier, 1 show added, 2 show supressed
       for (int aK=0 ; aK<3 ; aK++)
       {
          for (const auto  & aPtrS : aV)
          {
              bool aInInit = aSInit->In(*aPtrS);
              bool aInRes  = aRes->In(*aPtrS);
              int aKPrint = (aInInit ? 0 : 2) + (aInRes ? 0 : 1);
              if (aKPrint== aK)
              {
                  std::cout <<  " " << (aInInit ? "+" : "-");
                  std::cout <<   (aInRes ? "+" : "-") << " ";
                  std::cout <<  *aPtrS << "\n";
              }
          }
       }
   }

   // Back to cSetName
   {
      cSetName aResSN(*aRes);
      SaveInFile(aResSN,mXmlOut);
   }

   return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_EditSet(int argc,char ** argv,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_EditSet(argc,argv,aSpec));
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

void BenchEditSet()
{
    cMMVII_Appli &  anAp = cMMVII_Appli::TheAppli();

    std::string aCom = anAp.StrCallMMVII
                       (
                          "EditSet",
                           anAp.StrObl() << "t.xml" << "+=" << ".*",
                           anAp.StrOpt() << t2S("Out","t2.xml")
                       );
    std::cout << "VVVVV=" << aCom << "\n";
}


};

