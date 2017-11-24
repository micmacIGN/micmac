#include "include/MMVII_all.h"

/** \file cMMVII_CalcSet.cpp
    \brief Command for set calculation

    This file contain the command that compute a  set of file from File/Regex
  It's also the first "real" command of MMVII, so an occasion for tuning a 
  a lot of thing.

*/


namespace MMVII
{

/*
template <class Type> cSetAsVect : public cMemCheck
{
   public :
       friend void  AddData(const cAuxAr2007 & anAux,cSetAsVect<Type> & aSON);  ///< For serialization
       typedef std::vector<Type> tCont;  ///< In case we change the container type

       cSetName();  ///< Do nothing for now
       cSetName(const  cInterfSet<Type> &);  ///<  Fill with set
       cSetName(const std::string &,bool AllowPat);  ///< From Pat Or File

       void InitFromString(const std::string &,bool AllowPat);  ///< Init from file if ok, from pattern else

       size_t size() const;           ///< Accessor
       const tCont & Cont() const;    ///< Accessor

       cInterfSet<Type> * ToSet() const; ///< generate set, usefull for boolean operation
       void Filter(cSelector<Type>);  ///< select name matching the selector
   private :
   // private :
       void Sort();
       void InitFromFile(const std::string &,int aNumV);  ///< Init from Xml file
       void InitFromPat(const std::string & aFullPat); ///< Init from pattern (regex)


       // Data part
       tCont mV;
};
*/



/* ====================================== */
/*                                        */
/*             cSetName                   */
/*                                        */
/* ====================================== */

            //========== Constructors =============

/*
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
      if (aSel.Match(el))
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
*/

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
   if (! IsInit(&mXmlOut)) 
       mXmlOut = mXmlIn;
    else
       mXmlOut = mDirProject + mXmlOut;

   std::vector<std::string>  aVOps = SplitString(mAllOp," ");

   tNameSet aInput = SetNameFromString(mXmlIn,false);
   const tNameSet & aNew =  MainSet0();
   tNameSet aRes(eTySC::NonInit);

   
   if (mOp==aVOps.at(0)) // *=
   {
       aRes = aNew;
   }
   else if (mOp==aVOps.at(1)) // *=
   {
      aRes = aInput * aNew;
   }
   else if (mOp==aVOps.at(2)) // +=
   {
      aRes = aInput + aNew;
   }
   else if (mOp==aVOps.at(3)) // -=
   {
      aRes = aInput - aNew;
   }
   else if (mOp==aVOps.at(4)) // -=
   {
      aRes.clear();
   }
   else
   {
      MMVII_INTERNAL_ASSERT_user(false,"Unknown set operator :["+mOp+"] allowed: "+ mAllOp);
   }

   if (mShow)
   {
       tNameSet   aTot(aInput+aNew);

       std::vector<const std::string *> aV;
       aTot.PutInVect(aV,true);
       // 0 First time show unnmodifier, 1 show added, 2 show supressed
       for (int aK=0 ; aK<3 ; aK++)
       {
          for (const auto  & aPtrS : aV)
          {
              bool aInInit = aInput.In(*aPtrS);
              bool aInRes  = aRes.In(*aPtrS);
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
   SaveInFile(aRes,mXmlOut);

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


/* ==================================================================== */
/*                                                                      */
/*                     BENCH PART                                       */
/*                                                                      */
/* ==================================================================== */

void OneBenchEditSet
    (
        const std::string & anOp,    // Operator
        bool InitInput,              // If true, Input is set to last output
        const std::string & aPat,    // Pattern of image
        int aNumAskedOut,            // Required num version
        int aRealNumOut,             // Real Num Version
        int ExpectCard,              // Number of element required, useless with ExpSet added
        const std::string & Interv,  // Interval,
        const std::string & ExpSet   // Expect set
    )
{
    cMMVII_Appli &  anAp = cMMVII_Appli::TheAppli();
    std::string aDirI = anAp.InputDirTestMMVII() + "Files/" ;
    std::string aDirT = anAp.TmpDirTestMMVII()  ;
    std::string Input = "Input.xml";
    std::string Ouput = "Ouput.xml";
    if (InitInput)
    {
       if (ExistFile(aDirT+Ouput))
          RenameFiles(aDirT+Ouput,aDirI+Input);
    }
    else
    {
       // First time, file may subsist from an old crash
       RemoveFile(aDirI+Input,true);
    }

    cColStrAOpt & anArgOpt = anAp.StrOpt() << t2S("Out",Ouput);

    if (aNumAskedOut!=0)
       anArgOpt <<  t2S("NumVOut",ToStr(aNumAskedOut));

    if (Interv!="")
       anArgOpt <<  t2S("FFI0",ToStr(Interv));


    anAp.ExeCallMMVII
    (
        "EditSet",
        anAp.StrObl() <<   aDirI+Input  << anOp << aPat,
        anArgOpt
    );

    RenameFiles(aDirI+Ouput,aDirT+Ouput);

    const std::string & aTag = (aRealNumOut==1) ?  MMv1XmlTag_SetName : TagSetOfName;
 // std::cout << "FFFfff " << aTag <<  " " << aDirT+Ouput << " " << aRealNumOut << "\n"; getchar();
    MMVII_INTERNAL_ASSERT_always
    (
        IsFileXmlOfGivenTag((aRealNumOut==2),aDirT+Ouput,aTag) ,
        "Tag in OneBenchEditSet"
    );

    tNameSet aSet = SetNameFromString(aDirT+Ouput,false);

    MMVII_INTERNAL_ASSERT_always
    (
         aSet.size()==ExpectCard,
        "Bad number in OneBenchEditSet exp: "+ToStr(ExpectCard) + " , got: " + ToStr(aSet.size())
    );

    if (InitInput)
       RemoveFile(aDirI+Input,false);
   
   for (int aK=0 ; aK<10 ; aK++)
   {
       std::string aNF = "F" + ToStr(aK) + ".txt";
       MMVII_INTERNAL_ASSERT_always(aSet.In(aNF)==(ExpSet.find('0'+aK)!=std::string::npos),"Exp Set in OneBenchEditSet");

   }
}


void BenchEditSet()
{                  
    std::string C09="0123456789";
  // Basic test, we create the file
    OneBenchEditSet("+=",false,".*txt"       ,0,2,10,"",C09); // 
    OneBenchEditSet("+=",false,".*txt"       ,1,1,10,"",C09);
    OneBenchEditSet("+=",false,".*txt"       ,2,2,10,"",C09);
    OneBenchEditSet("+=",false,"F[02468].txt",2,2,5,"","02468");
 // here we init from previous
    OneBenchEditSet("+=",true ,"F[3-5].txt" ,2,2,7,"","0234568"); // 0234568
    OneBenchEditSet("*=",true ,"F[0-5].txt" ,2,2,5,"","02345"); // 02345
    OneBenchEditSet("-=",true ,"F[0369].txt",2,2,3,"","245"); // 245

    OneBenchEditSet( "=",true ,"F[0369].txt",1,1,4,"","0369"); // 0369
    OneBenchEditSet("+=",true ,"F[02468].txt",0,1,7,"","0234689"); // 0234689
    // Specify V2, but entry is V1, so V1
    OneBenchEditSet( "=",true ,"F.*.txt",0,1,5,"],F4.txt]","01234"); // 01234
    OneBenchEditSet("+=",true ,"F.*.txt",0,1,6,"]F8.txt,]","012349"); // 012349

    OneBenchEditSet( "=",true ,"F.*.txt",0,1,4,"[F1.txt,F3.txt]]F6.txt,F8.txt[","1237"); // 
}


};

