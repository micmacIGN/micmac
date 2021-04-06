#include "include/MMVII_all.h"
#include "include/MMVII_Class4Bench.h"
#include "include/MMVII_2Include_Serial_Tpl.h"




/** \file BenchSerial.cpp
    \brief Bench 4 correctness of Serialization

*/


namespace MMVII
{

//  ==========  cTestSerial0 =================

bool cTestSerial0::operator ==   (const cTestSerial0 & aT0) const {return (mP1==aT0.mP1) && (mP2==aT0.mP2);}
cTestSerial0::cTestSerial0() : 
    mP1 (1,2) , 
    mP2(3,3) 
{
}

///  To serialize cTestSerial0, just indicate that it is made of mP1 and mP2
void AddData(const cAuxAr2007 & anAux, cTestSerial0 &    aTS0) 
{
    AddData(cAuxAr2007("P1",anAux),aTS0.mP1);
    AddData(cAuxAr2007("P2",anAux),aTS0.mP2);
}
//  ==========  cTestSerial1 =================

template <class Type> bool EqualCont(const Type &aV1,const Type & aV2)
{
    return  std::equal(aV1.begin(),aV1.end(),aV2.begin(),aV2.end());
}


cTestSerial1::cTestSerial1() : 
    mS("Hello"), 
    mP3(3.1,3.2) ,
    mLI{1,22,333},
    mVD {314,27,14},
    mO2 (cPt2dr(100,1000))
{
}

bool cTestSerial1::operator ==   (const cTestSerial1 & aT1) const 
{
   return   (mTS0==aT1.mTS0) 
         && (mS==aT1.mS) 
         && (mP3==aT1.mP3) 
         && (mO1==aT1.mO1) 
         && (mO2==aT1.mO2)
         && EqualCont(mLI,aT1.mLI)   
   ;
}


void AddData(const cAuxAr2007 & anAux, cTestSerial1 &    aTS1) 
{
    AddData(cAuxAr2007("TS0",anAux),aTS1.mTS0);
    AddData(cAuxAr2007("S",anAux),aTS1.mS);
    AddData(cAuxAr2007("P3",anAux),aTS1.mP3);
    AddData(cAuxAr2007("LI",anAux),aTS1.mLI);
    AddData(cAuxAr2007("VD",anAux),aTS1.mVD);
    AddOptData(anAux,"O1",aTS1.mO1);
    AddOptData(anAux,"O2",aTS1.mO2);
}


///  a class to illustrate flexibility in serialization
/**  This class illusrate that the serialization protocol
  is very flexible, in this class we save the mTS0.mP1 data
  field at the same xml-level 
*/

class cTestSerial2 : public cTestSerial1
{
};

void AddData(const cAuxAr2007 & anAux, cTestSerial2 &    aTS2) 
{
    AddData(cAuxAr2007("TS0:P1",anAux),aTS2.mTS0.mP1);
    AddData(cAuxAr2007("TS0:P2",anAux),aTS2.mTS0.mP2);
    AddData(cAuxAr2007("S",anAux),aTS2.mS);
    AddData(cAuxAr2007("P3",anAux),aTS2.mP3);
    AddData(cAuxAr2007("LI",anAux),aTS2.mLI);
    AddData(cAuxAr2007("VD",anAux),aTS2.mVD);
    AddOptData(anAux,"O1",aTS2.mO1);
    AddOptData(anAux,"O2",aTS2.mO2);
}

template <class Type> void BenchSerialIm2D(const std::string & aDirOut)
{
    for (int aK=0 ;aK<10 ; aK++)
    {
        bool isXml = ((aK%2)==0);
        cPt2di aSz(1+RandUnif_N(10),1+RandUnif_N(10));
        cIm2D<Type>  anIm1(aSz,nullptr,eModeInitImage::eMIA_RandCenter);

        std::string aNameFile = aDirOut + "Image." + StdPostF_ArMMVII(isXml);
        SaveInFile(anIm1.DIm(),aNameFile);

        cIm2D<Type>  anIm2(cPt2di(1,1));
        ReadFromFile(anIm2.DIm(),aNameFile);

        double aD = anIm1.DIm().L1Dist(anIm2.DIm());

        // StdOut() << "BENCHSssIm2Dddd " << aD << " "<<  tNumTrait<Type>::NameType() << " Sz=" << aSz <<"\n";
        MMVII_INTERNAL_ASSERT_bench(aD<1e-5,"BenchSerialIm2D");
    }
}

void BenchSerialization
    (
        cParamExeBench & aParam,
        const std::string & aDirOut,  ///< For write-read temp file
        const std::string & aDirIn  ///< For readin existing file (as Xml with comments)
    )
{
    if (! aParam.NewBench("Serial")) return;
    // std::string aDir= DirCur();
    {
        BenchSerialIm2D<tREAL4>(aDirOut);
        BenchSerialIm2D<tU_INT1>(aDirOut);
    }

    SaveInFile(cTestSerial1(),aDirOut+"F1."+PostF_XmlFiles);

    {
       cTestSerial1 aP12;
       aP12.mLI.clear();
       aP12.mVD.clear();
       ReadFromFile(aP12,aDirOut+"F1."+PostF_XmlFiles);
       // Check the value read is the same
       MMVII_INTERNAL_ASSERT_bench(aP12==cTestSerial1(),"cAppli_MMVII_TestSerial");

       cTestSerial1 aS1;

       // Same object, same key
       MMVII_INTERNAL_ASSERT_bench(HashValue(aS1,true)==HashValue(aP12,true),"Hash_1");
       MMVII_INTERNAL_ASSERT_bench(HashValue(aS1,false)==HashValue(aP12,false),"Hash_2");

       // Same object to a permutation, dif 4 ordered, equal 4 unordered
       aS1.mP3 = PSymXY(aS1.mP3);
       MMVII_INTERNAL_ASSERT_bench(HashValue(aS1,true)!=HashValue(aP12,true),"Hash_1");
       MMVII_INTERNAL_ASSERT_bench(HashValue(aS1,false)==HashValue(aP12,false),"Hash_2");

       // Different object, different key
       aS1.mP3.x()++;
       MMVII_INTERNAL_ASSERT_bench(HashValue(aS1,true)!=HashValue(aP12,true),"Hash_1");
       MMVII_INTERNAL_ASSERT_bench(HashValue(aS1,false)!=HashValue(aP12,false),"Hash_2");

       // aS1
       // Check that == return false if we change a few
       cTestSerial1 aPModif = aP12;
       aPModif.mO1 = cPt2dr(14,18);
       MMVII_INTERNAL_ASSERT_bench(!(aPModif==cTestSerial1()),"cAppli_MMVII_TestSerial");
       SaveInFile(aP12,aDirOut+"F2."+PostF_XmlFiles);
    }

    {
        cTestSerial1 aP23;
        ReadFromFile(aP23,aDirOut+"F2."+PostF_XmlFiles);
        SaveInFile(aP23,aDirOut+"F3."+PostF_DumpFiles);
    }


    // Check dump value are preserved
    {
        cTestSerial1 aP34;
        ReadFromFile(aP34,aDirOut+"F3."+PostF_DumpFiles);
        MMVII_INTERNAL_ASSERT_bench(aP34==cTestSerial1(),"cAppli_MMVII_TestSerial");
        SaveInFile(aP34,aDirOut+"F4."+PostF_XmlFiles);
    }


/*
    {
        SaveInFile(cTestSerial2(),aDirOut+"F_T2."+PostF_XmlFiles);
        cTestSerial2 aT2;
    
        ReadFromFile(aT2,aDirOut+"F3."+PostF_DumpFiles);   // OK also in binary, the format has no influence
SaveInFile(aT2,"DEBUG."+PostF_XmlFiles);
        // And the value is still the same as dump is compatible at binary level

        MMVII_INTERNAL_ASSERT_bench(aT2==cTestSerial2(),"cAppli_MMVII_TestSerial");
    }
*/
    {
        SaveInFile(cTestSerial2(),aDirOut+"F_T2."+PostF_XmlFiles);
        cTestSerial2 aT2;
        // Generate an error
        if (0)
          ReadFromFile(aT2,aDirOut+"F2."+PostF_XmlFiles);
        ReadFromFile(aT2,aDirOut+"F_T2."+PostF_XmlFiles); // OK , read what we wrote as usual
        // and the value is the same
        MMVII_INTERNAL_ASSERT_bench(aT2==cTestSerial1(),"cAppli_MMVII_TestSerial");
    
        ReadFromFile(aT2,aDirOut+"F3."+  PostF_DumpFiles);   // OK also in binary, the format has no influence
        // And the value is still the same as dump is compatible at binary level

        MMVII_INTERNAL_ASSERT_bench(aT2==cTestSerial2(),"cAppli_MMVII_TestSerial");
    }

    // Bench ReadFromFileWithDef
    {
        cTestSerial2 aT5;
        // ReadFromFile(aT5,aDirOut+"FILE-DO-NO-EXIT");   // => Problem file does not exist
        aT5.mP3.x() = 12345.6789;
        MMVII_INTERNAL_ASSERT_bench(!(aT5==cTestSerial1()),"cAppli_MMVII_TestSerial"); // cmp fails, P5 has been changed
        ReadFromFileWithDef(aT5,aDirOut+"FILE-DO-NO-EXIT");   // Initialized with def
        MMVII_INTERNAL_ASSERT_bench(aT5==cTestSerial1(),"cAppli_MMVII_TestSerial"); // cmp is now ok
    }


    // Bench IsFile2007XmlOfGivenTag 
    {
       MMVII_INTERNAL_ASSERT_bench( IsFileXmlOfGivenTag(true,aDirOut+"F2."+PostF_XmlFiles,"TS0"),"cAppli_MMVII_TestSerial");
       MMVII_INTERNAL_ASSERT_bench(!IsFileXmlOfGivenTag(true,aDirOut+"F2."+PostF_XmlFiles,"TS1"),"cAppli_MMVII_TestSerial");
       MMVII_INTERNAL_ASSERT_bench(!IsFileXmlOfGivenTag(true,aDirIn+"PBF2."+PostF_XmlFiles,"TS0"),"cAppli_MMVII_TestSerial");
    }

    aParam.EndBench();
    //StdOut() << "DONE SERIAL\n";

    // return EXIT_SUCCESS;
}


};

