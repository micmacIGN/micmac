#include "include/MMVII_all.h"
#include "IndexBinaire.h"

#include "include/MMVII_Tpl_Images.h"

/** \file cCalcul_IndexBinaire.cpp
    \brief Command for computing the parameters of 

*/


namespace MMVII
{

/* ==================================================== */
/*                                                      */
/*           cMetaDataOneFileInvRad                     */
/*                                                      */
/* ==================================================== */


cMetaDataOneFileInvRad::cMetaDataOneFileInvRad
(
    cDataOneInvRad & aDOIR,
    const std::string & aName
)  :
   mDOIR (&aDOIR),
   mName (aName),
   mDFIm (cDataFileIm2D::Create(aDOIR.Dir() + DirSeparator() + mName))
{
}

void cMetaDataOneFileInvRad::SetNbPair()
{
   // Check szP divide sz glob, shouldbe always true due to computation by HCF
   MMVII_INTERNAL_ASSERT_strong(mDFIm.Sz().y()%mDOIR->SzP0().y()==0,"Size of Patch do not divide size image");

   mNbPair = mDFIm.Sz().y()/mDOIR->SzP0().y();
}


void cMetaDataOneFileInvRad::CheckCoherence(const cMetaDataOneFileInvRad& aMD2) const
{
   if ((mName != aMD2.mName) ||  (mNbPair!=aMD2.mNbPair))
   {
       StdOut() << "\n\n";
       StdOut() << "TYPES= " << E2Str(mDOIR->TIR()) << " " <<  E2Str(aMD2.mDOIR->TIR()) << "\n";
       StdOut() << "NAMES= " << mName << " " <<  aMD2.mName << "\n";
       StdOut() << "NbPatch= " << mNbPair  << " " << aMD2.mNbPair << "\n";
       MMVII_UsersErrror(eTyUEr::eUnClassedError,"Incohernt files of file in Invar");
   }
}

void  cMetaDataOneFileInvRad::AddPCar()
{
    cIm2D<tU_INT1> aImGlob(mDFIm.Sz());
    aImGlob.Read(mDFIm,cPt2di(0,0));
    cPt2di aSzP0 = mDOIR->SzP0();

    for (int aKPatch=0 ; aKPatch<2*mNbPair ; aKPatch++)
    {
        int aX0 = (aKPatch%2) *  aSzP0.x();
        int aY0 = (aKPatch/2) *  aSzP0.y();
        cPt2di aP0(aX0,aY0);

        //  Put in a small image, safer in we want to reduce ...
        cIm2D<tU_INT1> aImLoc(aSzP0);
        for (const auto & aPix : aImLoc.DIm())
            aImLoc.DIm().SetV(aPix,aImGlob.DIm().GetV(aPix+aP0));

        int aKVect =  mDOIR->PosInVect();
        cVecInvRad*  aIR = mDOIR->Appli().IR(mDOIR->KFill());
        for (const auto & aPix : aImLoc.DIm())
        {
            aIR->mVec.DIm().SetV(aKVect,aImLoc.DIm().GetV(aPix));
            aKVect++;
        }
        mDOIR->KFill() ++;
    }
}

/* ==================================================== */
/*                                                      */
/*           cDataOneInvRad                             */
/*                                                      */
/* ==================================================== */

cDataOneInvRad::cDataOneInvRad(cAppli_ComputeParamIndexBinaire & anAppli,cDataOneInvRad * aPrev,eTyInvRad aTIR) :
    mAppli (anAppli),
    mTIR   (aTIR),
    mDir   (anAppli.DirCurPC() + E2Str(aTIR) + DirSeparator()),
    mNbPixTot (0.0),
    mNbPatch  (0)
{
    std::vector<std::string>  aVS;
    {
       std::vector<std::string>  aVS0;
       GetFilesFromDir(aVS0,mAppli.DirCurPC() +  E2Str(mTIR)  ,BoostAllocRegex("Cple.*tif"));

       double aProp = mAppli.PropFile();
       for (int aK=0 ; aK<(int)aVS0.size() ; aK++)
           if (round_ni((aK-1)*aProp) != round_ni(aK*aProp))
              aVS.push_back(aVS0[aK]);
    }

    for (int aK=0 ; aK<int(aVS.size()) ; aK++)
    {
        mMDOFIR.push_back(cMetaDataOneFileInvRad(*this,aVS[aK]));
        // If first file, initialize to size of file
        if (aK==0)
        {
           mSzP0.x() = mMDOFIR.back().mDFIm.Sz().x();
           if (mSzP0.x()%2!=0)
           {
              MMVII_UsersErrror(eTyUEr::eUnClassedError,"exptected even witdh");
           }
           mSzP0.x() /= 2;
           mSzP0.y() = mMDOFIR.back().mDFIm.Sz().y();
        }
        else
        {
            // Sz.x of patch must be equal for all Image in one invariant
            if ((2*mSzP0.x())!= mMDOFIR.back().mDFIm.Sz().x())
            {
                 MMVII_UsersErrror(eTyUEr::eUnClassedError,"Variable size in Invariant rad");
            } 
            // compute Sz.y as HCF (PGCD) 
            mSzP0.y() = HCF(mSzP0.y(),mMDOFIR.back().mDFIm.Sz().y());
        }
    }
    for (auto &  aMD : mMDOFIR)
    {
       aMD.SetNbPair();
       mNbPixTot +=   aMD.mDFIm.NbElem();
       mNbPatch += 2 * aMD.mNbPair;
    }

    mPosInVect =  (aPrev==nullptr) ? 0 :  (aPrev->mPosInVect + aPrev->NbValByP()) ;
    // aPrev->mPosInVect + aPrev->NbValByP() ;
    
    StdOut()  <<  E2Str(mTIR)  
              << ", VS :" << aVS.size() 
              << " SZ=" << mSzP0 
              << " Pos=" << mPosInVect 
              // << " NbPatch " << mNbPatch 
              << " Pix=" << mNbPixTot 
              << "\n";
}



void cDataOneInvRad::CheckCoherence(const cDataOneInvRad& aD2) const
{
    if(mMDOFIR.size()!=aD2.mMDOFIR.size() || (mNbPatch!=aD2.mNbPatch))
    {
       StdOut() << "\n\n";
       StdOut() << "SIZES = " << mMDOFIR.size() << " " << aD2.mMDOFIR.size() << "\n";
       StdOut() << "TYPES = " << E2Str(mTIR) << " " << E2Str(aD2.mTIR) << "\n";
       MMVII_UsersErrror(eTyUEr::eUnClassedError,"Variable number of file in Invar");
    }
    for (int aK=0 ; aK<int(mMDOFIR.size()) ; aK++)
        mMDOFIR[aK].CheckCoherence(aD2.mMDOFIR[aK]);
}

int  cDataOneInvRad::NbValByP() const
{
    return mSzP0.x() * mSzP0.y();
}


void  cDataOneInvRad::AddPCar()
{
    mKFill = 0;
    for (auto &  aMD : mMDOFIR)
    {
         aMD.AddPCar();
    }
    MMVII_INTERNAL_ASSERT_strong(mKFill==mNbPatch,"Internal check in AddPCar");
}


cAppli_ComputeParamIndexBinaire& cDataOneInvRad::Appli()  {return mAppli;}
const std::string & cDataOneInvRad::Dir() const {return mDir;}
const cPt2di &  cDataOneInvRad::SzP0() const {return mSzP0;}
eTyInvRad  cDataOneInvRad::TIR() const {return mTIR;}
tREAL8  cDataOneInvRad::NbPixTot() const {return mNbPixTot;}
int  cDataOneInvRad::NbPatch() const {return mNbPatch;}
int  cDataOneInvRad::PosInVect() const {return mPosInVect;}

int& cDataOneInvRad::KFill() {return mKFill;}

/* ==================================================== */
/*                                                      */
/*           cVecInvRad                                 */
/*                                                      */
/* ==================================================== */

cVecInvRad::cVecInvRad(int aNbVal) :
   mVec       (aNbVal),
   mSelected  (true)
{
}


/*
void cVecInvRad::Add2Stat(cDenseVect<tREAL8>& aTmp,cDenseVect<tREAL8>& aMoy,cDenseMatrix<tREAL8> & aCov) const
{
   CopyIn(aTmp.DIm(),mVec.DIm());
   AddIn(aMoy.DIm(),aTmp.DIm());
   aCov.Add_tAA(aTmp);
}
void cVecInvRad::PostStat(cDenseVect<tREAL8>& aMoy,cDenseMatrix<tREAL8> & aCov,double aPdsTot)
{
    DivCsteIn(aMoy.DIm(),aPdsTot);
    DivCsteIn(aCov.DIm(),aPdsTot);
    aCov.Sub_tAA(aMoy);
    aCov.SelfSymetrizeBottom();
}
*/


/*
std::vector<std::string> & cDataOneInvRad::VS()
       std::vector<std::string> & VS();
*/


};

