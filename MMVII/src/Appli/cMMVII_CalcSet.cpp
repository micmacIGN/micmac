#include "include/MMVII_all.h"

/** \file cMMVII_CalcSet.cpp
    \brief Command for set calculation

    This file contain the command that compute a  set of file from File/Regex
  It's also the first "real" command of MMVII, so an occasion for tuning a 
  a lot of thing.

*/


namespace MMVII
{

/* ==================================================== */
/*                                                      */
/*          cAppli_EditSet                              */
/*                                                      */
/* ==================================================== */

/// An application for editing set of file
/**
    Given an XML memorizing a set of file, it is possible to :

      - add a new set (+=)
      - substract a new set (-=)
      - intersect a new set (*=)
      - overwrite with a new set (=)

    Most command take as input a set of file, single case can be pattern,
  but more complex require Xml file that can be edited.
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
         int         mShow;
};

cCollecSpecArg2007 & cAppli_EditSet::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl 
         << Arg2007(mXmlIn,"Full Name of Xml in/out",{eTA2007::FileDirProj})
         << Arg2007(mOp,"Operator in ("+StrAllVall<eOpAff>()+")" )
         << Arg2007(mPat,"Pattern or Xml for modifying",{{eTA2007::MPatIm,"0"}});
}

cCollecSpecArg2007 & cAppli_EditSet::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
      anArgOpt
         << AOpt2007(mShow,"Show","Show detail of set before/after , (def) 0->none, (1) modif, (2) all",{})
         << AOpt2007(mXmlOut,"Out","Destination, def=Input, no save for " + MMVII_NONE,{});
}

cAppli_EditSet::cAppli_EditSet(int argc,char** argv,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (argc,argv,aSpec),
  mShow        (0)
{
}

int cAppli_EditSet::Exe()
{
   InitOutFromIn(mXmlOut,mXmlIn);

   tNameSet aInput = SetNameFromString(mXmlIn,false);
   const tNameSet & aNew =  MainSet0();

   tNameSet aRes = aInput.Dupl();

   aRes.OpAff(Str2E<eOpAff>(mOp),aNew);

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
             bool ShowThis = (mShow>=2)  || (! aInInit) || (! aInRes);
             if (ShowThis)
             {
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
   }

   // Back to cSetName
   if (FileOfPath(mXmlOut,false) != MMVII_NONE)
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
      {eApDT::Xml},
      __FILE__
);



/* ==================================================== */
/*                                                      */
/*          cAppli_EditRel                              */
/*                                                      */
/* ==================================================== */

/// An application for editing set of cple of file
/**
    Given an XML memorizing a set of file, it is possible to :

      - add a new set (+=)
      - substract a new set (-=)
      - intersect a new set (*=)
      - overwrite with a new set (=)
*/
class cAppli_EditRel : public cMMVII_Appli
{
     public :
        cAppli_EditRel(int argc,char** argv,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
     private :
         bool        ValideCple(const std::string & aN1,const std::string &aN2) const;
         void        AddCple(const std::string & aN1,const std::string &aN2) ;

         std::string mXmlIn;
         std::string mXmlOut;
         std::string mPat;
         std::string mPat2;
         bool        m2Set;
         bool        mAllPair;
         std::string mOp;
         int         mShow;
         int         mLine;
         bool        mCirc;
         bool        mReflexif;
         tNameRel    mNewRel;
};

cAppli_EditRel::cAppli_EditRel(int argc,char** argv,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (argc,argv,aSpec),
  mAllPair     (false),
  mShow        (0),
  mLine        (0),
  mCirc        (true),
  mReflexif    (false)
{
}

cCollecSpecArg2007 & cAppli_EditRel::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl 
         << Arg2007(mXmlIn,"Full Name of Xml in/out",{eTA2007::FileDirProj})
         << Arg2007(mOp,"Operator in ("+StrAllVall<eOpAff>()+")" )
         << Arg2007(mPat,"Pattern or Xml for modifying",{{eTA2007::MPatIm,"0"}})
      ;
}

cCollecSpecArg2007 & cAppli_EditRel::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
      anArgOpt
         << AOpt2007(mPat2,"Pat2","Second Pattern or Xml for modifying, def=first",{{eTA2007::MPatIm,"1"}})
         << AOpt2007(mAllPair,"AllP","Put all pair, def false ",{})
         << AOpt2007(mLine,"Line","\"Linear graph\" : for k, add [k-l,k+l]")
         << AOpt2007(mCirc,"Circ","\"Circular\" in line mode")
         << AOpt2007(mReflexif,"Reflexif","Accept pair with identicall names, def=false")

      ;
/*
         // << AOpt2007(mShow,"Show","Show detail of set before/after , (def) 0->none, (1) modif, (2) all",{})
         << AOpt2007(mXmlOut,"Out","Destination, def=Input, no save for " + MMVII_NONE,{});
*/
}

bool cAppli_EditRel::ValideCple(const std::string & aN1,const std::string &aN2) const
{
   if ((aN1==aN2) && (!mReflexif))
      return false;

   return true;
}

void cAppli_EditRel::AddCple(const std::string & aN1,const std::string &aN2)
{
   if (ValideCple(aN1,aN2))
      mNewRel.Add(tNamePair(aN1,aN2));
}


int cAppli_EditRel::Exe() 
{
   InitOutFromIn(mXmlOut,mXmlIn);
   m2Set = IsInit(&mPat2);

   tNameRel aRelIn =  RelNameFromFile (mXmlIn);
   const tNameSet & aSet1 =  MainSet0();
   const tNameSet & aSet2 =  m2Set ?  MainSet1() : MainSet0() ;


   if (mAllPair)
   {
      std::vector<const std::string *> aV1;
      aSet1.PutInVect(aV1,true);
      std::vector<const std::string *> aV2;
      aSet2.PutInVect(aV2,true);

      for (const auto & aPtr1 : aV1)
      {
         for (const auto & aPtr2 : aV2)
         {
             AddCple(*aPtr1,*aPtr2);
         }
      }
   }

   if (mLine>0)
   {
       if (m2Set)
          Warning("Mode Line with 2 sets in EditRel",eTyW::eWLineAndCart,__LINE__,__FILE__);

      std::vector<const std::string *> aV1;
      int aNba = aV1.size();
      aSet1.PutInVect(aV1,true);
      for (int aKa=0 ; aKa < aNba ; aKa++)
      {
           int aKb0 = aKa-mLine;
           int aKb1 = aKa+mLine;
           if (!mCirc)
           {
                aKb0 = std::max(aKb0,0);
                aKb1 = std::min(aKb0,0);
           }
           for (int aKb = aKb0; aKb <= aKb1 ; aKb++)
           {
               AddCple(*aV1.at(aKa),*aV1.at(aKb));
           }
      }
   }


   tNameRel aRes = aRelIn.Dupl();
   aRes.OpAff(Str2E<eOpAff>(mOp),mNewRel);

   if (FileOfPath(mXmlOut,false) != MMVII_NONE)
      SaveInFile(aRes,mXmlOut);

   return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_EditRel(int argc,char ** argv,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_EditRel(argc,argv,aSpec));
}

cSpecMMVII_Appli  TheSpecEditRel
(
     "EditRel",
      Alloc_EditRel,
      "This command is used to edit set of pairs of files",
      {eApF::Project},
      {eApDT::Console,eApDT::Xml},
      {eApDT::Xml},
      __FILE__
);
/* ==================================================================== */
/*                                                                      */
/*                     BENCH PART                                       */
/*                                                                      */
/* ==================================================================== */

void OneBenchEditSet
    (
        int aNumTest,                // Change test condition
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

    bool UseDirP = (aNumTest==1);

    if (InitInput)
    {
       if (ExistFile(aDirT+Ouput))
       {
          RenameFiles(aDirT+Ouput,aDirI+Input);
       }
       else
       {
          MMVII_INTERNAL_ASSERT_always(false,"Incoherence in OneBenchEditSet");
       }
    }
    else
    {
       // First time, file may subsist from an old crash
       RemoveFile(aDirI+Input,true);
    }

    cColStrAOpt & anArgOpt = anAp.StrOpt() << t2S("Out",Ouput);

    if (aNumAskedOut!=0)
       anArgOpt <<  t2S(GOP_NumVO,ToStr(aNumAskedOut));

    if (Interv!="")
       anArgOpt <<  t2S(GOP_Int0,ToStr(Interv));

    if (UseDirP)
       anArgOpt <<  t2S(GOP_DirProj,aDirI);


    anAp.ExeCallMMVII
    (
        "EditSet",
        anAp.StrObl() <<   (UseDirP ? "" : aDirI)+Input  << anOp << aPat,
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
    for (int aK=0 ; aK<2 ; aK++)
    {
       std::string C09="0123456789";
     // Basic test, we create the file
       OneBenchEditSet(aK,"+=",false,".*txt"       ,0,2,10,"",C09); // 
       OneBenchEditSet(aK,"+=",false,".*txt"       ,1,1,10,"",C09);
       OneBenchEditSet(aK,"+=",false,".*txt"       ,2,2,10,"",C09);
       OneBenchEditSet(aK,"+=",false,"F[02468].txt",2,2,5,"","02468");
    // here we init from previous
       OneBenchEditSet(aK,"+=",true ,"F[3-5].txt" ,2,2,7,"","0234568"); // 0234568
       OneBenchEditSet(aK,"*=",true ,"F[0-5].txt" ,2,2,5,"","02345"); // 02345
       OneBenchEditSet(aK,"-=",true ,"F[0369].txt",2,2,3,"","245"); // 245

       OneBenchEditSet(aK,"=",true ,"F[0369].txt",1,1,4,"","0369"); // 0369
       OneBenchEditSet(aK,"+=",true ,"F[02468].txt",0,1,7,"","0234689"); // 0234689
     // Specify V2, but entry is V1, so V1
       OneBenchEditSet(aK,"=",true ,"F.*.txt",0,1,5,"],F4.txt]","01234"); // 01234
       OneBenchEditSet(aK,"+=",true ,"F.*.txt",0,1,6,"]F8.txt,]","012349"); // 012349

       OneBenchEditSet(aK,"=",true ,"F.*.txt",0,1,4,"[F1.txt,F3.txt]]F6.txt,F8.txt[","1237"); // 
       OneBenchEditSet(aK,"=0",true ,"F.*.txt",0,1,0,"",""); // 
    }
}

};

