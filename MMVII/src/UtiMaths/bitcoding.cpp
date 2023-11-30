#include "MMVII_SetITpl.h"
#include "cMMVII_Appli.h"

#include <fstream>

namespace MMVII
{

size_t GetNDigit_OfBase(size_t aNum,size_t aBase)
{
   size_t aNbD=1;
   size_t aVMax = aBase-1;

   while (aVMax < aNum)
   {
       aVMax = aVMax*aBase + aBase-1;
       aNbD++;
   }
   return aNbD;
};

std::string  NameOfNum_InBase(size_t aNum,size_t aBase,size_t aNbDigit)
{
    size_t aNbDigMin = GetNDigit_OfBase(aNum,aBase);
    UpdateMax(aNbDigit,aNbDigMin);

    std::string aRes(aNbDigit,'0');

    for (int aK = aRes.size() -1 ; aK>=int(aRes.size()-aNbDigMin) ; aK--)
    {
        size_t aCar = aNum % aBase;
        if (aCar < 10)
           aCar = '0' +  aCar;
        else
           aCar = 'A' + (aCar-10);
        aRes[aK] = aCar;
        aNum /= aBase;
    }

    return aRes;
}



/* ************************************************* */
/*                                                   */
/*       BIT MANIPULATION                            */
/*                                                   */
/* ************************************************* */

///  Number of bits to 1

inline size_t NbBitsGen(tU_INT4 aVal)
{
   size_t aCpt=0;
 
   while (aVal)
   {
       if (1&aVal)
          aCpt++;
       aVal >>=1;
   }
   return aCpt;
}

size_t NbBits(tU_INT4 aVal)
{
    static constexpr size_t TheMaxTab = (1<<20) +1;
    if (aVal < TheMaxTab)
    {
         static tU_INT1   TheTabul[TheMaxTab];
	 static bool First =true;
	 if (First)
	 {
             First = false;
	     for (size_t aK=0 ; aK< TheMaxTab ; aK++)
                 TheTabul[aK] = NbBitsGen(aK);
	 }
	 return TheTabul[aVal];
    }

    return NbBitsGen(aVal);
}

///  Hamming distance (number of bit different)

size_t HammingDist(tU_INT4 aV1,tU_INT4 aV2)
{
	return NbBits(aV1^aV2);
}


/// make a circular permutation of bits, assuming a size NbIt, with  aPow2= NbBit^2

size_t  LeftBitsCircPerm(size_t aSetFlag,size_t aPow2) 
{
     if (aSetFlag&1) aSetFlag |= aPow2;  // pre-transfer low bit, if exit, in high bit
     return aSetFlag >> 1;            // now left shit
}

size_t  N_LeftBitsCircPerm(size_t aSetFlag,size_t aPow2,size_t N)
{
    while (N!=0)
    {
        N--;
        aSetFlag= LeftBitsCircPerm(aSetFlag,aPow2);
    }

    return aSetFlag;
}


/// make a symetry bits, assuming a size NbIt, with  aPow2= NbBit^2

size_t  BitMirror(size_t aSetFlag,size_t aPow2) 
{
    size_t aRes =0;
    size_t aFlag = 1;
    aPow2 >>= 1;

    while (aPow2)
    {
       if (aPow2 & aSetFlag) 
          aRes |= aFlag;
       aPow2 >>= 1;
       aFlag <<= 1;
    }

    return aRes;
}

/// make a visualisation of bit flag as  (5,256) -> "10100000"

std::string  StrOfBitFlag(size_t aSetFlag,size_t aPow2) 
{
    std::string  aRes;
    for (size_t aFlag=1 ; aFlag<aPow2 ; aFlag<<=1)
    {
        aRes.push_back((aFlag&aSetFlag) ? '1' : '0');
    }
    return aRes;
}

/// Transformate a string-Visu in flag bits "10100000" -> 5
size_t  Str2BitFlag(const std::string & aStr)
{
  size_t aRes=0;
  for ( size_t aK=0 ; aStr[aK] ; aK++)
      if (aStr[aK]!='0')
          aRes |= (size_t(1)<<aK);
  return aRes;
}

/// Transormate a bit flage in vect of int, for easier manip
void  BitsToVect(std::vector<int> & aVBits,tU_INT4 aVal,size_t aPow2)
{
   aVBits.clear();
   for (size_t aFlag=1 ; aFlag<aPow2 ; aFlag<<=1)
   {
        aVBits.push_back((aVal&aFlag)!=0);
   }
}

///  return the maximal length of consecutive 0 & 1, interpreted circularly    (94="01111010", 256=2^8)  =>  (3,2)
///  fill vector will all interval
cPt2di MaxRunLength(tU_INT4 aVal,size_t aPow2,std::vector<cPt2di> & aVInterv0,std::vector<cPt2di> & aVInterv1)
{

   aVInterv0.clear();
   aVInterv1.clear();
   
   std::vector<int> aVBits;
   BitsToVect(aVBits,aVal,aPow2);

   int aNbB = aVBits.size();

   int aMaxR0=0;
   int aMaxR1=0;
   //  Parse all bit
   for (int aK1=0; aK1< aNbB ; aK1++)
   {
        // select bit that are different previous
        if (ValCirc(aVBits,aK1-1) !=ValCirc(aVBits,aK1))
	{
           int aK2 = aK1+1;
	   while ( ValCirc(aVBits,aK1)==ValCirc(aVBits,aK2) ) // reach next diff
                 aK2++;
           if (ValCirc(aVBits,aK1))  // update count for 0 or 1
           {
              aVInterv1.push_back(cPt2di(aK1,aK2));
              UpdateMax(aMaxR1,aK2-aK1);
           }
           else
           {
              aVInterv0.push_back(cPt2di(aK1,aK2));
              UpdateMax(aMaxR0,aK2-aK1);
           }
	}
   }

   // special, if there is only one run as no transition was detected
   if (aMaxR0==0)
   {
       if (aVBits[0])
          return cPt2di(0,aNbB);
       return cPt2di(aNbB,0);
   }

   return cPt2di(aMaxR0,aMaxR1);
}

cPt2di MaxRunLength(tU_INT4 aVal,size_t aPow2)
{
     std::vector<cPt2di> aVInterv0;
     std::vector<cPt2di> aVInterv1;
     return MaxRunLength(aVal,aPow2,aVInterv0,aVInterv1);
}

/// Max of both run (0 and 1)
size_t MaxRun2Length(tU_INT4 aVal,size_t aPow2)
{
   return NormInf(MaxRunLength(aVal,aPow2));
}

/* *************************** */
/*           cCelCC            */
/* *************************** */

cCelCC::cCelCC(size_t aLowestCode) :
    mLowCode       (aLowestCode)
{
}


size_t cCelCC::HammingDist(const cCelCC & aC2) const
{
    size_t aD= MMVII::HammingDist(mLowCode,aC2.mEquivCode[0]);

    for (size_t aK=1 ; aK< aC2.mEquivCode.size(); aK++)
        UpdateMin(aD,MMVII::HammingDist(mLowCode,aC2.mEquivCode[aK]));

    return aD;
}


/* ************************************************* */
/*                                                   */
/*             cCompEquiCodes                        */
/*                                                   */
/* ************************************************* */

/** Class for computing equivalent code, typicall code that are equal up to a circular permutation */


//std::map<std::string,cCompEquiCodes*> cCompEquiCodes::TheMapCodes;

cCompEquiCodes::cCompEquiCodes(size_t aNbBits,size_t aPer,bool WithMirror) :
     mNbBits       (aNbBits),
     mPeriod       (aPer),
    mNbCodeUC     (size_t(1)<<mNbBits),
     mVCodes2Cell  (mNbCodeUC,nullptr)
{
     MMVII_INTERNAL_ASSERT_strong((aNbBits%aPer)==0,"NbBit not multiple of period in cCompEquiCodes");
     for (size_t aCode=0 ; aCode < mNbCodeUC ; aCode++)
     {
          if (mVCodes2Cell[aCode] == nullptr)
	  {
              cCelCC * aNewCel = new cCelCC(aCode);
	      mVecOfCells.push_back(aNewCel);

	      AddCodeWithPermCirc(aCode,aNewCel);
	      if (WithMirror)
	          AddCodeWithPermCirc(BitMirror(aCode,mNbCodeUC),aNewCel);

	      AddAndResizeUp(mHistoNbBit,NbBits(aCode),1);
	  }
	  else
	  {
              // Nothing to do, code has been processed by equivalent lower codes
	  }
     }

     //for (const auto & AC : mVecOfCells)
         //StdOut()  << " AC " << AC->mEquivCode << std::endl;
}

cCompEquiCodes::~cCompEquiCodes()
{
     DeleteAllAndClear(mVecOfCells);
}

cCompEquiCodes * cCompEquiCodes::Alloc(size_t aNbBits,size_t aPer,bool WithMirror)
{
     return  new cCompEquiCodes(aNbBits,aPer,WithMirror);
     /*
     std::string aInd = ToStr(aNbBits)+"_"+ ToStr(aPer) + "_" +ToStr(WithMirror);
     cCompEquiCodes* & aRef = TheMapCodes[aInd];
     if (aRef==nullptr)
        aRef = new cCompEquiCodes(aNbBits,aPer,WithMirror);

     return aRef;
     */
}

const cCelCC &  cCompEquiCodes::CellOfCodeOK(size_t aCode) const 
{
    const cCelCC * aRes = CellOfCode(aCode);
    MMVII_INTERNAL_ASSERT_tiny(aRes!=nullptr,"cCompEquiCodes::CellOfCodeOK");

    return *aRes;
}


const cCelCC *  cCompEquiCodes::CellOfCode(size_t aCode) const
{
   if (aCode>=mVCodes2Cell.size()) return nullptr;

   return  mVCodes2Cell.at(aCode);
}



void cCompEquiCodes::AddCodeWithPermCirc(size_t aCode,cCelCC * aNewCel)
{
   for (size_t aBit=0 ; aBit<mNbBits ; aBit+=mPeriod)
   {
       // Test because code may have already been processed i
       // due to circular invariant code (ie as "0101" on 4 bits)
       if (mVCodes2Cell[aCode] == nullptr)
       {
            mVCodes2Cell[aCode] = aNewCel;
            aNewCel->mEquivCode.push_back(aCode);
       }
       aCode = N_LeftBitsCircPerm(aCode,mNbCodeUC,mPeriod);
   }
}

const std::vector<cCelCC*>  & cCompEquiCodes::VecOfCells() const {return mVecOfCells;}


std::vector<cCelCC*>  cCompEquiCodes::VecOfUsedCode(const std::vector<cPt2di> & aVXY,bool Used)
{
    for (auto aPCel : mVecOfCells)
       aPCel->mTmp = false;

    for (auto aXY : aVXY)
       mVCodes2Cell[aXY.y()]->mTmp = true;

    std::vector<cCelCC*> aRes;
    for (auto aPCel : mVecOfCells)
        if (aPCel->mTmp == Used)
           aRes.push_back(aPCel);
    return aRes;
}


std::list<cCompEquiCodes::tAmbigPair>  cCompEquiCodes::AmbiguousCode(const std::vector<cPt2di> & aVecXY)
{
    std::list<tAmbigPair>  aRes;

    std::map<cCelCC*,std::vector<cPt2di> >  aMapAmbig;

    for (const auto & aXY : aVecXY)
    {
         aMapAmbig[mVCodes2Cell.at(aXY.y())].push_back(aXY);
    }

    for (const auto & anIter : aMapAmbig)
    {
         if (anIter.second.size() >1)
         {
              aRes.push_back(tAmbigPair(anIter.first,anIter.second));
         }
    }

    return aRes;
}

std::string cCompEquiCodes::NameCERStuff(const std::string & aPrefix,size_t aNbBits)
{
	return     cMMVII_Appli::DirRessourcesMMVII() 
		+ "CodeCircTaget"  + StringDirSeparator()
		+ aPrefix + ToStr(aNbBits) + "bit_lookup.txt";
}
std::string cCompEquiCodes::NameCERNLookUpTable(size_t aNbBits) {return NameCERStuff("",aNbBits);}
std::string cCompEquiCodes::NameCERNPannel(size_t aNbBits) {return NameCERStuff("Positions-3D-",aNbBits);}

/// Low level function, read the pair Num->Code in a file
void  ReadCodesTarget(std::vector<cPt2di> & aVCode,const std::string & aNameFile)
{
     std::vector<std::vector<double>> aVV;
     ReadFilesNum(aNameFile,"FF",aVV,'#');
     aVCode.clear();

     for (const auto & aV : aVV)
         aVCode.push_back(cPt2di(round_ni(aV.at(0)),round_ni(aV.at(1))));
}

/** show some static of run lenght on certain codinf scheme */

void  TestComputeCoding(size_t aNBBCoding,int aParity,size_t)
{
   std::vector<std::list<cCelCC*>>  aVCodeByRun(aNBBCoding+1);

   std::unique_ptr<cCompEquiCodes> aCEC (cCompEquiCodes::Alloc(aNBBCoding));
   for (const auto & aPCel : aCEC->VecOfCells())
   {
	size_t aCode =  aPCel->mLowCode;
        int aNbB = NbBits(aCode);
	bool takeIt = (aNbB%2==0)  ? ((aParity & 2)!=0)  : ((aParity & 1) !=0);

	if (takeIt)
	{
         size_t aLenRun = MaxRun2Length(aCode,size_t(1)<<aNBBCoding);
	     aVCodeByRun.at(aLenRun).push_back(aPCel);
	}
   }

   int aCumul=0;
   for (size_t aL=0 ; aL<aVCodeByRun.size() ; aL++)
   {
        if (! aVCodeByRun[aL].empty())
	{
            int aSz = aVCodeByRun[aL].size() ;
	    aCumul += aSz;
            StdOut() << " For RunLength " << aL << " got " << aSz << " Cumul=" << aCumul << " codes" << std::endl;
	}
   }
   StdOut() << std::endl;
}


void cCompEquiCodes::Bench(size_t aNBB,size_t aPer,bool Miror)
{
     std::unique_ptr<cCompEquiCodes> aCEC (cCompEquiCodes::Alloc(aNBB,aPer,Miror));

     int aNBC = 0;
     for (const auto & aPC : aCEC->mVecOfCells)
     {
          aNBC += aPC->mEquivCode.size();
     }
     MMVII_INTERNAL_ASSERT_bench((aNBC==(1<<aNBB)),"Base representation");

     /*
     StdOut() << "Lllllllll " 
	      <<  aPer << " " 
	      << Miror << " " 
	      <<   aCEC->mVecOfCells.size() * (aNBB/aPer) * (1+Miror) / (double) aNBC << "\n";
	      */
}

void BenchCircCoding()
{
    for (auto aMir : {false,true})
    {
        for (auto aNbB : {10,11,12})
	{
             cCompEquiCodes::Bench(aNbB,aNbB,aMir);
	     cCompEquiCodes::Bench(aNbB,   1,aMir);
	}
        for (auto aPer : {1,2,3})
	{
             cCompEquiCodes::Bench(aPer*4,aPer,aMir);
             cCompEquiCodes::Bench(aPer*5,aPer,aMir);
	}
    }
    if (0)
    {
       TestComputeCoding(20,2,1);
       TestComputeCoding(14,2,1);
       TestComputeCoding(14,3,7);
    }

	/* 1.0  make test on low-level bits manipulations */

    MMVII_INTERNAL_ASSERT_bench(NameOfNum_InBase(256,16)=="100","Base representation");
    MMVII_INTERNAL_ASSERT_bench(NameOfNum_InBase(255,16)=="FF","Base representation");
    MMVII_INTERNAL_ASSERT_bench(NameOfNum_InBase(255,16,4)=="00FF","Base representation");
    MMVII_INTERNAL_ASSERT_bench(NameOfNum_InBase(71,36,3)=="01Z","Base representation");

	/* 1.1  make test on low-level bits manipulations */

    MMVII_INTERNAL_ASSERT_bench(NbBits(256)==1,"Nb Bits");
    MMVII_INTERNAL_ASSERT_bench(NbBits(255)==8,"Nb Bits");
    MMVII_INTERNAL_ASSERT_bench(Str2BitFlag("101100")==13,"Bits Visu");
    MMVII_INTERNAL_ASSERT_bench(LeftBitsCircPerm(Str2BitFlag("01011001"),256)==Str2BitFlag("10110010"),"LeftBitsCircPerm");

    for (size_t aVal : {0,123,512,7,8})
    {
        MMVII_INTERNAL_ASSERT_bench(Str2BitFlag(StrOfBitFlag(aVal,1024))==aVal,"Bits Visu");
	MMVII_INTERNAL_ASSERT_bench(NbBits(LeftBitsCircPerm(aVal,1024))==NbBits(aVal),"LeftBitsCircPerm/NbBits");
    }

    MMVII_INTERNAL_ASSERT_bench(MaxRunLength(Str2BitFlag("01011000"),256)==cPt2di(4,2),"MaxRunLength");
    MMVII_INTERNAL_ASSERT_bench(MaxRunLength(Str2BitFlag("01011000"),512)==cPt2di(5,2),"MaxRunLength");

    MMVII_INTERNAL_ASSERT_bench(BitMirror(Str2BitFlag("01011000"),256)==Str2BitFlag("00011010"),"BitMirror");


    //  Test coherence of existing coding 
    for (auto aNbB : {12,14,20})
    {
       std::vector<cPt2di>  aVCode;
       ReadCodesTarget(aVCode,cCompEquiCodes::NameCERNLookUpTable(aNbB));

       std::unique_ptr<cCompEquiCodes> aCEC (cCompEquiCodes::Alloc(aNbB));
       std::list<cCompEquiCodes::tAmbigPair>  aLamb = aCEC->AmbiguousCode(aVCode);

       std::vector<cPt2di> aListOddCode;
       for (const auto & aXY : aVCode)
       {
           if (NbBits(aXY.y())%2!=0)
              aListOddCode.push_back(aXY);
       }
       if (aNbB!=20)  // The 12 & 14 bits are ok
       {
           MMVII_INTERNAL_ASSERT_bench(aLamb.empty(),"Ambiguous coding");
           MMVII_INTERNAL_ASSERT_bench(aListOddCode.empty(),"Odd code");
       }
       else if (0)  // for now the 20 bits code are erroneous, so dont test ...
       {
           StdOut() << "------------AmbiguousCode: " << aNbB << " Sz=" << aLamb.size() << std::endl;
	   for (const auto & aPair : aLamb)
               StdOut() << "  * " << aPair.second << std::endl;
           StdOut() << "--------------  Parity check " << std::endl;
	   for (const auto &  aOdd : aListOddCode)
                StdOut() <<  "N= " << aOdd.x() <<  " NbB=" << NbBits(aOdd.y()) << " Code=" << aOdd.y() << std::endl;
	   StdOut() << "Nb odd codes : " << aListOddCode.size() << std::endl;
           getchar();
       }
    }
}



/* ******************************************************* */
/*                                                         */
/*          cHamingCoder                                   */
/*                                                         */
/* ******************************************************* */




int cHamingCoder::NbBitsOut() const { return mNbBitsOut; }
int cHamingCoder::NbBitsRed() const { return mNbBitsRed; }
int cHamingCoder::NbBitsIn () const { return mNbBitsIn ; }

/*
  x x   x
0 1 2 3 4 5 6 7


O2I: [-1,-1,-1,1,-1,2,3,4]
I2O: [-1,3,5,6,7]

*/  

int cHamingCoder::UnCodeWhenCorrect(tU_INT4 aVal)
{
   aVal *= 2;

    tU_INT4 aRes = 0;
    for(int aK=1 ; aK<=mNbBitsIn ; aK++)
    {
	  if (aVal & (1<<mNumI2O[aK]))
	     aRes |= (1<<(aK-1));		   
    }

    return (Coding(aRes) == aVal/2) ? aRes : -1;
}

tU_INT4 cHamingCoder::Coding(tU_INT4 aV) const
{
   cSetISingleFixed<tU_INT4> aSetV (aV);
   std::vector<int> aVecBits =aSetV.ToVect();

    int aRes = 0;
    for(const auto & aNumBit : aVecBits)
    {
          aRes |= (1<< mNumI2O[aNumBit+1]);
    }

    for (int aK=0 ; aK<mNbBitsRed ; aK++)
    {
         int aFlag = 1<< aK;
         int aCpt = 0;
         for  (const auto & aBit : aVecBits)
         {
             if ((mNumI2O[aBit+1])&aFlag)
                aCpt++;
         }
         if (aCpt%2)
            aRes |= (1<<aFlag);
    }

   return aRes/2;
}


// a basic implementation, but we don need to have it very efficient
cHamingCoder cHamingCoder::HCOfBitTot(int aNbBitsTot,bool WithParity)
{
   int  aNBI = 1;
   cHamingCoder aHC(aNBI);
   while ((aHC.NbBitsOut() <aNbBitsTot) || (WithParity &&(aHC.NbBitsOut()%2 != 0)) )
   {
         aNBI++;
         aHC = cHamingCoder(aNBI);

	 StdOut() << "HHHH " << aHC.NbBitsOut() << " " << aHC.NbBitsIn() << std::endl;
   }
   return aHC;
}

cHamingCoder::cHamingCoder(int aNbBitsIn) :
   mNbBitsIn  (aNbBitsIn),
   mNbBitsRed (1),
   mNbBitsOut (mNbBitsIn+mNbBitsRed)
{
    while (  (1<<mNbBitsRed) <= mNbBitsOut)
    {
        mNbBitsRed++;
        mNbBitsOut++;
    }
    //  StdOut() << "HHHC " << mNbBitsIn << " " << mNbBitsRed << " " <<  mNbBitsOut << std::endl;
    mIsBitRed = std::vector<bool>(mNbBitsOut+1,false);
    mNumI2O   = std::vector<int> (mNbBitsIn+1,-1);
    mNumO2I   = std::vector<int> (mNbBitsOut+1,-1);

    for (int aK=0 ; aK<mNbBitsRed ; aK++)
        mIsBitRed.at(size_t(1)<<aK) = true;

    int aKIn=1;
    for (int aKOut=1 ; aKOut<=mNbBitsOut ; aKOut++)
    {
         if (! mIsBitRed[aKOut])
         {
            mNumO2I[aKOut] = aKIn ;
            mNumI2O[aKIn ] = aKOut ;
            aKIn++;
         }
    }
    /*
StdOut()   << "O2I: " <<  mNumO2I << std::endl;
StdOut()   << "I2O: " <<  mNumI2O << std::endl;
getchar();
*/

}

void BenchHammingDist(int  aV1,int aV2)
{
   cSetISingleFixed<tU_INT4> aSetV (aV1^aV2);
   int aC1 = aSetV.Cardinality(); 
   int aC2 = HammingDist(aV1,aV2);

   MMVII_INTERNAL_ASSERT_bench(aC1==aC2,"Ham dist");
}

void BenchHammingCode(int aNbB)
{
   cHamingCoder aHC(aNbB);

   std::vector<int>  aVC;
   std::vector<bool>  aVIsCorrect(size_t(1)<<aHC.NbBitsOut(),false);
   for (int aK=0 ; aK<(1<<aNbB) ; aK++)
   {
      int aC = aHC.Coding(aK);
      aVC.push_back(aC);
      aVIsCorrect.at(aC) = true;
      MMVII_INTERNAL_ASSERT_bench(aK==aHC.UnCodeWhenCorrect(aC),"Ham decode");
      //  StdOut() << "HH " << aK << " "<< aC  << " " << aHC.UnCodeWhenCorrect(aC) << std::endl;
   }

   for (tU_INT4 aK=0 ; aK<aVIsCorrect.size() ; aK++)
   {
       if (!aVIsCorrect[aK])
       {
            MMVII_INTERNAL_ASSERT_bench(aHC.UnCodeWhenCorrect(aK)==-1,"Ham decode");
       }
   }
   for (int aK1=0 ; aK1<int(aVC.size()) ; aK1++)
   {
       cWhichMin<int,int> aWM(-1,100);
       for (int aK2=0 ; aK2<int(aVC.size()) ; aK2++)
       {
           if (aK1!=aK2)
           {
              aWM.Add(aK2,HammingDist(aVC[aK1],aVC[aK2]));
           }
       }
       // StdOut() << "DH " << aWM.ValExtre() << std::endl;
       MMVII_INTERNAL_ASSERT_bench(aWM.ValExtre()>=3 ,"Ham dist");
   }

}

void BenchHamming(cParamExeBench & aParam)
{
    Bench_Target_Encoding();
    BenchCircCoding();
    if (! aParam.NewBench("Hamming")) return;

    BenchHammingDist(0,2);
    for (int aK1=0 ; aK1<23; aK1++)
        for (int aK2=0 ; aK2<23; aK2++)
            BenchHammingDist(aK1,aK2);

    BenchHammingCode(4);
    BenchHammingCode(11);
    BenchHammingCode(13);
    aParam.EndBench();
}

};

