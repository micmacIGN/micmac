#include "MMVII_SetITpl.h"
#include "MMVII_Sys.h"
#include "cMMVII_Appli.h"

#include <fstream>

namespace MMVII
{

/// make a circular permutation of bits, assuming a size NbIt, with  aPow2= NbBit^2
size_t  BitsCircPerm(size_t aSetFlag,size_t aPow2) 
{
     if (aSetFlag&1) aSetFlag |= aPow2;  // pre-transfer low bit, if exit, if high bit
     return aSetFlag >> 1;            // now left shit
}

/// make a symetry bits, assuming a size NbIt, with  aPow2= NbBit^2
size_t  BitsSym(size_t aSetFlag,size_t aPow2) 
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


class cCelCC
{
     public :
        std::vector<size_t>  mEquivCode;
	size_t               mLowCode;
	int                  mNum;
	int                  mRepresentant;

	cCelCC(size_t aLowestCode);
     public :
	cCelCC(const cCelCC &) = delete;
};

cCelCC::cCelCC(size_t aLowestCode) :
    mLowCode       (aLowestCode),
    mNum           (-1),
    mRepresentant  (-1)
{
}






class cCircularCoding
{
   public :
       static std::string NameCERNLookUpTable(size_t aNbBits);

       cCircularCoding(size_t aNbBits);
   private :
       void AddCodeWithPermCirc(size_t aCode,cCelCC *);


       size_t                   mNbBits;      ///< Number of bits
       size_t                   mNbCodeUC;    ///<  Number of code uncircullar i.e. 2 ^NbBits
       size_t                   mNbDiffCode;  ///< Number of possible coding different taking account circularity
       std::vector<cCelCC*>     mVCodes2Cell; ///< Code->Cell  vector of all code for sharing equivalence
       std::vector<cCelCC*>     mVecOfCells;  ///< vector of all different cells
};


void cCircularCoding::AddCodeWithPermCirc(size_t aCode,cCelCC * aNewCel)
{
   for (size_t aBit=0 ; aBit<mNbBits ; aBit++)
   {
       if (mVCodes2Cell[aCode] == nullptr) // Code may have already been processed because of circular invariant code  0101 on 4 bits
       {
            mVCodes2Cell[aCode] = aNewCel;
            aNewCel->mEquivCode.push_back(aCode);
       }
       aCode = BitsCircPerm(aCode,mNbCodeUC);
   }
}

cCircularCoding::cCircularCoding(size_t aNbBits) :
     mNbBits       (aNbBits),
     mNbCodeUC     (1<<mNbBits),
     mNbDiffCode   (0),
     mVCodes2Cell  (mNbCodeUC,nullptr)
{
     for (size_t aCode=0 ; aCode < mNbCodeUC ; aCode++)
     {
          if (mVCodes2Cell[aCode] == nullptr)
	  {
              mNbDiffCode++;
              cCelCC * aNewCel = new cCelCC(aCode);
	      mVecOfCells.push_back(aNewCel);

	      AddCodeWithPermCirc(aCode,aNewCel);
	  }
	  else
	  {
              // Nothing to do, code has been processed by equivalent lower codes
	  }
     }

     /*
     for (const auto & aSpec : aVSpec)
     {
          cCelCC * aCel = mVCodes2Cell.at(aSpec.y());
	  if (aCel->mNum != -1)
	  {
              StdOut()  << "MULTIPLE CODE " 
		        <<  aSpec.x() << " -> " << aSpec.y()   << " :: "
			<< aCel->mNum << " -> " << aCel->mRepresentant 
			<< "\n";

	  }
	  else
	  {
             aCel->mNum = aSpec.x();
             aCel->mRepresentant  = aSpec.y();
	  }
     }
     */

     StdOut() << "NB DIFC=" << mNbDiffCode << "\n";
}

std::string cCircularCoding::NameCERNLookUpTable(size_t aNbBits)
{
	return     cMMVII_Appli::DirRessourcesMMVII() 
		+ "CodeCircTaget"  + StringDirSeparator()
		+ ToStr(aNbBits) + "bit_lookup.txt";
}
            
/// Low level function, read the pair Num->Code in a file
void  ReadTarget(std::vector<cPt2di> & aVCode,const std::string & aNameFile)
{
    std::ifstream infile(aNameFile);

    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        int a, b;
        if (!(iss >> a >> b)) 
	{ 
            MMVII_UnclasseUsEr(std::string("Bad target file for ") + aNameFile);
	}
	aVCode.push_back(cPt2di(a,b));
    }
}

void BenchCircCoding()
{
	/*
    for (auto aSetFlag : {3,128,254})
    {
        StdOut() <<  "BSSS " <<  aSetFlag << " -> " << BitsSym(aSetFlag,256)  << "\n";
        StdOut() <<  StrOfBitFlag(aSetFlag,256) << "\n";
        StdOut() <<  StrOfBitFlag(BitsSym(aSetFlag,256),256) << "\n";
    }
    size_t aFlag = 5;
    for (int aK=0 ; aK<12 ; aK++)
    {
        StdOut() << "FFF=" << aFlag << "\n";
	aFlag = BitsCircPerm(aFlag,256);
    }
    */

    for (auto aNbB : {12,14,20})
    {
       // std::vector<cPt2di>  aVCode;
       // ReadTarget(aVCode,cCircularCoding::NameCERNLookUpTable(aNbB));
       cCircularCoding aCC(aNbB);
FakeUseIt(aCC);
    }
       StdOut() << "WWWWWWWWW\n"; getchar();

}

/* ******************************************************* */
/*                                                         */
/*          cHamingCoder                                   */
/*                                                         */
/* ******************************************************* */

int HammingDist(tU_INT4 aV1,tU_INT4 aV2)
{
   int aCpt=0;
   tU_INT4 aDif = aV1^aV2;
 
   for (tU_INT4 aFlag=1; (aFlag<=aDif) ; aFlag <<= 1)
   {
       if (aFlag&aDif)
          aCpt++;
   }
   return aCpt;
}

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
    //  StdOut() << "HHHC " << mNbBitsIn << " " << mNbBitsRed << " " <<  mNbBitsOut << "\n";
    mIsBitRed = std::vector<bool>(mNbBitsOut+1,false);
    mNumI2O   = std::vector<int> (mNbBitsIn+1,-1);
    mNumO2I   = std::vector<int> (mNbBitsOut+1,-1);

    for (int aK=0 ; aK<mNbBitsRed ; aK++)
        mIsBitRed.at(1<<aK) = true;

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
StdOut()   << "O2I: " <<  mNumO2I << "\n";
StdOut()   << "I2O: " <<  mNumI2O << "\n";
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
   FakeUseIt(aHC);

   std::vector<int>  aVC;
   std::vector<bool>  aVIsCorrect(1<<aHC.NbBitsOut(),false);
   for (int aK=0 ; aK<(1<<aNbB) ; aK++)
   {
      int aC = aHC.Coding(aK);
      aVC.push_back(aC);
      aVIsCorrect.at(aC) = true;
      MMVII_INTERNAL_ASSERT_bench(aK==aHC.UnCodeWhenCorrect(aC),"Ham decode");
      //  StdOut() << "HH " << aK << " "<< aC  << " " << aHC.UnCodeWhenCorrect(aC) << "\n";
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
       // StdOut() << "DH " << aWM.ValExtre() << "\n";
       MMVII_INTERNAL_ASSERT_bench(aWM.ValExtre()>=3 ,"Ham dist");
   }

}

void BenchHamming(cParamExeBench & aParam)
{
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

