#include "MMVII_2Include_Tiling.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_MeasuresIm.h"
#include "MMVII_UtiSort.h"
#include "MMVII_Sensor.h"

#include "TieP.h"

/**
   \file  cImplemConvertHom

   \brief file contain classes for transforming pair of homologous point by pai 
          in a multiple points

*/


namespace MMVII
{

class  cGCP_PMGCP : public cInterfParsePMulGCP
{
     public :
	 void  Incr() override;
	 bool   End() const override;
	 cGCP_PMGCP(const cSetMesGndPt &);

     private :
        void SetCurPt();

       	const cSetMesGndPt &  mSet;
        const std::vector<cMultipleImPt> & mVMIP;
	size_t                mIndEnd;
	size_t                mCurInd;
};

cGCP_PMGCP::cGCP_PMGCP(const cSetMesGndPt & aSet) :
     cInterfParsePMulGCP (true),
     mSet    (aSet),
     mVMIP   (mSet.MesImOfPt()),
     mIndEnd (mVMIP.size()),
     mCurInd (0)
{
    SetCurPt();
}

void cGCP_PMGCP::SetCurPt()
{
    const cMultipleImPt & aMP = mVMIP.at(mCurInd);
    mResult.mVPIm = aMP.VMeasures();
    mResult.mVIm = aMP.VImages();

    const cMes1Gnd3D & aMG =   mSet.MesGCPOfMulIm(aMP);
    mResult.mPGround = aMG.mPt;
    mResult.mName = aMG.mNamePt;
}

bool   cGCP_PMGCP::End() const { return (mCurInd==mIndEnd); }

void  cGCP_PMGCP::Incr()
{
    mCurInd++;

    if (mCurInd!=mIndEnd)
       SetCurPt();
}

/******************************************************/
/*                                                    */
/*                    cMTP_PMGCP                      */
/*                                                    */
/******************************************************/

class cMTP_PMGCP : public cInterfParsePMulGCP
{
     public :
         const std::vector<std::string> & VNamesImage() const;
	 cMTP_PMGCP(const cComputeMergeMulTieP &,bool WithPGround);
	 void  Incr() override;
	 bool   End() const override;
     private :
	 typedef tMapTiePMult::const_iterator  tCIter;

         cMTP_PMGCP(const cMTP_PMGCP &) = delete;
	 void SetCurPt(bool WithConfigChg);

	 const cComputeMergeMulTieP &  mCMTP;
	 tCIter                        mCurIt; 
	 tCIter                        mEndIt; 

	 const std::vector<cPt2dr>*  mCurVP2;
	 size_t                      mCurNbPts;
	 size_t                      mCurMult;
	 size_t                      mIndCurPMul;
	 size_t                      mIndCurP2;
};

cMTP_PMGCP::cMTP_PMGCP(const cComputeMergeMulTieP & aCMTP,bool withPGround) :
    cInterfParsePMulGCP (withPGround),
    mCMTP       (aCMTP),
    mCurIt      (mCMTP.Pts().begin()),
    mEndIt      (mCMTP.Pts().end()),
    mCurVP2     (nullptr)
{
    SetCurPt(true);
}

inline const cVal1ConfTPM     & VALUE(const tPairTiePMult & aPair)    {return aPair.second;}


void cMTP_PMGCP::SetCurPt(bool WithConfigChg)
{
     if (mCurIt!=mEndIt)
     {
        if (WithConfigChg)
        {
           mIndCurPMul  = 0;
           mIndCurP2    = 0;
           mResult.mVIm = Config(*mCurIt);
           mCurNbPts = NbPtsMul(*mCurIt);
           mCurMult  = Multiplicity(*mCurIt);
           mResult.mVPIm.resize(mCurMult);
           mCurVP2 = &(mCurIt->second.mVPIm);

           if (0)  // JOE ???
           {
	      //  const std::vector<cPt2dr>*  mCurVP2;
	      //  mCurIt
              //  if instead of mCurIt->second we use the "equivalent" inline VALUE
              mCurVP2 = &(VALUE(*mCurIt).mVPIm);

              StdOut()  <<  "ddd1 " << mCurVP2 << " " << &(VALUE(*mCurIt).mVPIm) << std::endl; // The adress are the same
              StdOut()  <<  "ddd2 " << mCurVP2 - &(VALUE(*mCurIt).mVPIm) << std::endl;  // the adresses are realy the same
              StdOut()  <<  "ddd3 " << mCurVP2->data() - VALUE(*mCurIt).mVPIm.data() << std::endl; // the adresses are definitively the same
											    
              StdOut()  <<  "ccc1 " << (*mCurVP2)[0] << " " << VALUE(*mCurIt).mVPIm[0] << std::endl; // first element are different
              StdOut()  <<  "ccc2 " << mCurVP2->at(0) << " " << VALUE(*mCurIt).mVPIm.at(0) << std::endl; // first element are really different
              StdOut()  <<  "ccc2 " << mCurVP2->data()[0] << " " << VALUE(*mCurIt).mVPIm.data()[0] << std::endl; // first element are definitively different
              StdOut()  <<  "ccc4 " << (*mCurVP2)[1] << " " << VALUE(*mCurIt).mVPIm[1] << std::endl;  //  by the way, second elment are equal !!!!!!
              getchar();
	   }
        }

	for (size_t aK=0 ; aK<mCurMult ; aK++)
        {
            mResult.mVPIm.at(aK) = Val(*mCurIt).mVPIm.at(mIndCurP2+aK);
	}


	if (mWithPGround)
	{
           mResult.mPGround = BundleInter(*mCurIt,mIndCurPMul,mCMTP.VSensors());
	}
     }
}

void   cMTP_PMGCP::Incr()
{
    bool ChgConfig = false;
    if (mIndCurPMul+1<mCurNbPts)
    {
        mIndCurPMul++;
	mIndCurP2 += mCurMult;
    }
    else
    {
        mCurIt++;
        ChgConfig = true;
    }
    SetCurPt(ChgConfig);
}

bool   cMTP_PMGCP::End() const { return mCurIt==mEndIt; }



const std::vector<std::string> & cMTP_PMGCP::VNamesImage()  const
{
  return mCMTP.VNames();
}

/******************************************************/
/*                                                    */
/*               cInterfParsePMulGCP                  */
/*                                                    */
/******************************************************/

cInterfParsePMulGCP *  cInterfParsePMulGCP::Alloc_CMTP(const cComputeMergeMulTieP & aCMTP,bool WithPGround)
{
    return new  cMTP_PMGCP(aCMTP,WithPGround);
}

cInterfParsePMulGCP *  cInterfParsePMulGCP::Alloc_ImGCP(const cSetMesGndPt &aSet)
{
    return new cGCP_PMGCP (aSet);
}


cInterfParsePMulGCP::cInterfParsePMulGCP(bool withPGround) :
    mWithPGround (withPGround)
{
}

cInterfParsePMulGCP::~cInterfParsePMulGCP() {}

const cPMulGCPIm & cInterfParsePMulGCP::CurP() const {return mResult;}

}; // MMVII






