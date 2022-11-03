#include "MMVII_Bench.h"
#include "MMVII_Class4Bench.h"
#include "MMVII_2Include_Serial_Tpl.h"

#include "MMVII_Geom2D.h"


/** \file BenchSerial.cpp
    \brief Bench 4 correctness of Serialization

*/


namespace MMVII
{

//  ==========  cTestSerial0 =================

bool cTestSerial0::operator ==   (const cTestSerial0 & aT0) const 
{
	// StdOut() << "cTestSerial0cTestSerial0\n";
	return     (mP1==aT0.mP1) 
		&& (mI1==aT0.mI1)
		&& (mR4==aT0.mR4)
		&& (mP2==aT0.mP2)
        ;
}

cTestSerial0::cTestSerial0() : 
    mP1 (1,2) , 
    mI1 (22),
    mR4 (4.5),
    mP2 (3,3) 
{
}

///  To serialize cTestSerial0, just indicate that it is made of mP1 and mP2
void AddData(const cAuxAr2007 & anAux, cTestSerial0 &    aTS0) 
{
    AddData(cAuxAr2007("P1",anAux),aTS0.mP1);
    AddData(cAuxAr2007("I1",anAux),aTS0.mI1);
    AddData(cAuxAr2007("R4",anAux),aTS0.mR4);
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
    AddData(cAuxAr2007("TS0:I1",anAux),aTS2.mTS0.mI1);
    AddData(cAuxAr2007("TS0:R4",anAux),aTS2.mTS0.mR4);
    AddData(cAuxAr2007("TS0:P2",anAux),aTS2.mTS0.mP2);
    AddData(cAuxAr2007("S",anAux),aTS2.mS);
    AddData(cAuxAr2007("P3",anAux),aTS2.mP3);
    AddData(cAuxAr2007("LI",anAux),aTS2.mLI);
    AddData(cAuxAr2007("VD",anAux),aTS2.mVD);
    AddOptData(anAux,"O1",aTS2.mO1);
    AddOptData(anAux,"O2",aTS2.mO2);
}

/* --------------- cTestSerial3 --------------------*/

template <class Type> bool EqualPtr(const Type *aPtr1,const Type * aPtr2)
{
   if ((aPtr1==nullptr) && (aPtr2==nullptr)) return true;
   if ((aPtr1!=nullptr) && (aPtr2!=nullptr)) return (*aPtr1==*aPtr2);
   return false;
}
class cTestSerial3  : public  cMemCheck
{
     public :
        cTestSerial3(int aVal)  :
           mPtrI  ( (aVal==-1) ? nullptr : (new int(aVal)))   ,
           mPtrS1 ( (aVal==-1) ? nullptr : (new cTestSerial1))
        {
            mS1.mLI.push_back(aVal);
            mS1B.mLI.push_back(aVal+1);
            if (mPtrS1) mPtrS1->mLI.push_back(aVal+12);
        }
        ~cTestSerial3() 
        {
            delete mPtrI;
            delete mPtrS1;
        }
        cTestSerial3(const cTestSerial3 & aS3) :
           mS1   ( aS3.mS1),
           mPtrI (  (aS3.mPtrI) ? new int(*aS3.mPtrI) : nullptr),
           mS1B  ( aS3.mS1B)
        {
        }

        cTestSerial1            mS1;
        int *                   mPtrI;
        cTestSerial1 *          mPtrS1;
        cTestSerial1            mS1B;
};

bool operator ==   (const cTestSerial3 & aT1,const cTestSerial3 & aT2)
{
    return      EqualPtr(aT1.mPtrI,aT2.mPtrI)
            &&  EqualPtr(aT1.mPtrS1,aT2.mPtrS1)
            &&  (aT1.mS1  == aT2.mS1)
            &&  (aT1.mS1B == aT2.mS1B)
    ;
}

void AddData(const cAuxAr2007 & anAux, cTestSerial3 &    aTS3) 
{
   AddData(anAux,aTS3.mS1);
   OnePtrAddData(anAux,aTS3.mPtrI);
   OnePtrAddData(anAux,aTS3.mPtrS1);
   AddData(anAux,aTS3.mS1B);
}



template <class Type> void BenchSerialIm2D(const std::string & aDirOut)
{
    // Check if vector of ptr are iniatialized to null
    {
        std::vector<cTestSerial1 *> aVPtr;
        for (int aK=0 ; aK< 3 ; aK++)
        {
            aVPtr = std::vector<cTestSerial1 *>(2);
            MMVII_INTERNAL_ASSERT_bench((aVPtr.at(0)==nullptr),"BenchSerial3-Ptr");
            MMVII_INTERNAL_ASSERT_bench((aVPtr.at(1)==nullptr),"BenchSerial3-Ptr");
            aVPtr.clear();
            cTestSerial1 aS1;
            for (int aK=0 ; aK< 3; aK++)
                aVPtr.push_back(&aS1);
        }
    }
    {
         cTestSerial3 aT1(-1);
         cTestSerial3 aT2(-1);
         cTestSerial3 aT3( 1);
         cTestSerial3 aT4( 1);
         cTestSerial3 aT5( 2);
         MMVII_INTERNAL_ASSERT_bench((aT1==aT2),"BenchSerial3-Ptr");
         MMVII_INTERNAL_ASSERT_bench((aT3==aT4),"BenchSerial3-Ptr");
         MMVII_INTERNAL_ASSERT_bench(!(aT3==aT5),"BenchSerial3-Ptr");
         MMVII_INTERNAL_ASSERT_bench(!(aT1==aT5),"BenchSerial3-Ptr");

         cTestSerial3 aT44(44);
         std::string aNameFile = aDirOut + "S3.dmp";

         SaveInFile(aT1,aNameFile);
         ReadFromFile(aT44,aNameFile);
         MMVII_INTERNAL_ASSERT_bench((aT1==aT44),"BenchSerial3-Ptr");

         SaveInFile(aT3,aNameFile);
         ReadFromFile(aT44,aNameFile);
         MMVII_INTERNAL_ASSERT_bench((aT3==aT44),"BenchSerial3-Ptr");
    }
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

/// Test both Cumul and its Read/Write mode
template <class TypeH,class TypeCumul> void BenchHistoAndSerial(const std::string & aDirOut)
{
    for (int aK=0 ;aK<10 ; aK++)
    {
        bool isXml = ((aK%2)==0);
        std::string aNameFile = aDirOut + "Histo." + StdPostF_ArMMVII(isXml);
        int  aSz =  1+RandUnif_N(10);
         
        cHistoCumul<TypeH,TypeCumul> aH(aSz);
        // We add 1 2 3  ... , in cumul we must have  N(N+1)/2
        for (int aX=0 ; aX<aSz ; aX++)
        {
             aH.AddV(aX,3.0);   // Add it in two time, just for test 
             aH.AddV(aX,aX-2.0);
        }
        aH.MakeCumul();

        SaveInFile(aH,aNameFile);
        cHistoCumul<TypeH,TypeCumul> aH2;
        ReadFromFileWithDef(aH2,aNameFile);
        
        
        MMVII_INTERNAL_ASSERT_bench(aH2.H().Sz()==aSz,"BenchHistoAndSerial");

        int aPTot = ((1+aSz) * aSz)/2;
        for (int aX=0 ; aX<aSz ; aX++)
        {
            double aProp = (((2+aX) * (1+aX))/2) / double(aPTot);
            double aDif = RelativeDifference(aProp,aH2.PropCumul(aX));
            MMVII_INTERNAL_ASSERT_bench(aDif<1e-7,"BenchHistoAndSerial");
        }
/*
StdOut() << "WwwwwwwwWWWWWWwwww " << aNameFile << " " << aH2.H().Sz() << "\n";
getchar();
*/
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
        BenchHistoAndSerial<tINT4,tREAL8>(aDirOut);
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
        cTestSerial1 aP34_0 = aP34;
        ReadFromFile(aP34,aDirOut+"F3."+PostF_DumpFiles);
        MMVII_INTERNAL_ASSERT_bench(aP34==aP34_0,"cAppli_MMVII_TestSerial");


	aP34.mTS0.mI1 =-123;
	aP34.mTS0.mR4 = 100.5;
        aP34_0 = aP34;
	//std::vector<string> aVPost ({PostF_DumpFiles,PostF_XmlFiles})

	for (const auto & aPost : {PostF_DumpFiles,PostF_XmlFiles})
        {
           SaveInFile(aP34,aDirOut+"F10."+aPost);
           cTestSerial1 aP34_Read;
           ReadFromFile(aP34_Read,aDirOut+"F10."+aPost);
           MMVII_INTERNAL_ASSERT_bench(aP34_Read==aP34_0,"cAppli_MMVII_TestSerial");
	}
    }


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

