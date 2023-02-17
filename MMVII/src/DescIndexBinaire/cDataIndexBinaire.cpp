
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
   mDFIm (cDataFileIm2D::Create(aDOIR.Dir() + StringDirSeparator() + mName,true))
{
}

void cMetaDataOneFileInvRad::SetNbPair()
{
   // Check szP divide sz glob, shouldbe always true due to computation by HCF
   MMVII_INTERNAL_ASSERT_strong(mDFIm.Sz().y()%mDOIR->SzP0Init().y()==0,"Size of Patch do not divide size image");

   mNbPair = mDFIm.Sz().y()/mDOIR->SzP0Init().y();
}


void cMetaDataOneFileInvRad::CheckCoherence(const cMetaDataOneFileInvRad& aMD2) const
{
   //  As folder must have the same structure, and selection must be identic,
   // at the end we must have same name.  Also number of pair must be identic
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
    cIm2D<tU_INT1> aImGlob(mDFIm.Sz());  //! Contain global image
    aImGlob.Read(mDFIm,cPt2di(0,0));     // read file
    cPt2di aSzP0Init = mDOIR->SzP0Init();  //! Size of a Patch before reduce

    for (int aKPatch=0 ; aKPatch<2*mNbPair ; aKPatch++)
    {
        int aX0 = (aKPatch%2) *  aSzP0Init.x();  //! contains begin.x of current pair in global image
        int aY0 = (aKPatch/2) *  aSzP0Init.y();  //! contains begin.y of current pair  in global image
        cPt2di aP0(aX0,aY0); //! begin of current pair  in global image

        //  Put in a small image, safer in we want to reduce ...
        cIm2D<tU_INT1> aImLoc(aSzP0Init);
        for (const auto & aPix : aImLoc.DIm()) // for each pixel of aImLoc 
            aImLoc.DIm().SetV(aPix,aImGlob.DIm().GetV(aPix+aP0)); //  copy Big in Small using offset

        aImLoc = mDOIR->Appli().MakeImSz(aImLoc);
        

        // Now put the small image at the end of current vect
        int aKVect =  mDOIR->PosInVect();
        cVecInvRad*  aIR = mDOIR->Appli().IR(mDOIR->KFill());
        for (const auto & aPix : aImLoc.DIm())
        {
            aIR->mVec.DIm().SetV(aKVect,aImLoc.DIm().GetV(aPix));
            aKVect++; // increment position in current vector
        }
        mDOIR->KFill() ++; // increment position of current vector
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
    mDir   (anAppli.DirCurPC() + E2Str(aTIR) + StringDirSeparator()),
    mSzP0Init(-1,-1),
    mSzP0Final(-1,-1),
    mNbPixTot (0.0),
    mNbPatch  (0)
{
    std::vector<std::string>  aVS; //! list of files in the folder that will be selected
    //  In tuning mode, we dont want to process all folder, but only a proportion "mAppli.PropFile()"
    // This code make the selection
    {
       std::vector<std::string>  aVS0;  //! Get all file corresponding to regular expression
       GetFilesFromDir(aVS0,mAppli.DirCurPC() +  E2Str(mTIR)  ,AllocRegex("Cple.*tif"));

       //  Select a subset with a given proportion (parametre PropF of command)
       double aProp = mAppli.PropFile();
       for (int aK=0 ; aK<(int)aVS0.size() ; aK++)
           if (round_ni((aK-1)*aProp) != round_ni(aK*aProp))  //! Mathematicall formula to select a proportion of aProp
              aVS.push_back(aVS0[aK]);
    }

    // read the information on each file, compute an check coherence of size
    // Sz.x must be equal for all file, Sz.y is computed by Highest Common Factor

    for (int aK=0 ; aK<int(aVS.size()) ; aK++)
    {
        // Memorize for each file meta data information 
        mMDOFIR.push_back(cMetaDataOneFileInvRad(*this,aVS[aK]));
        // If first file, initialize to size of file
        if (aK==0)
        {
           mSzP0Init.x() = mMDOFIR.back().mDFIm.Sz().x();
           if (mSzP0Init.x()%2!=0) // Sz in x must be even, as it contain a pair of patch
           {
              MMVII_UsersErrror(eTyUEr::eUnClassedError,"exptected even witdh");
           }
           mSzP0Init.x() /= 2;
           mSzP0Init.y() = mMDOFIR.back().mDFIm.Sz().y(); // Initialize to full size
        }
        else
        {
            // Sz.x of patch must be equal for all Image in one invariant
            if ((2*mSzP0Init.x())!= mMDOFIR.back().mDFIm.Sz().x())
            {
                 MMVII_UsersErrror(eTyUEr::eUnClassedError,"Variable size in Invariant rad");
            } 
            // compute Sz.y as HCF (PGCD) 
            mSzP0Init.y() = HCF(mSzP0Init.y(),mMDOFIR.back().mDFIm.Sz().y());
        }
    }
    // For each  meta data
    for (auto &  aMD : mMDOFIR)
    {
       aMD.SetNbPair();  //   fix nb of pair by division (and check division works)
       mNbPixTot +=   aMD.mDFIm.NbElem();  // for statistic on RAM occupation maybe, not usefull really
       mNbPatch += 2 * aMD.mNbPair;  // Number of sub images total
    }

    // Compute size of reduced image, a bit "heavy" but this way we are sure that we will get the
    // same value that when we really compute them 
    {
       cIm2D<tU_INT1> aImgTmp(mSzP0Init);
       aImgTmp = mAppli.MakeImSz(aImgTmp);
       SetSzP0Final(aImgTmp.DIm().Sz());
    }

    // Compute position where to concatenate value of this invariant
    mPosInVect =  (aPrev==nullptr) ? 0 :  (aPrev->mPosInVect + aPrev->NbValByP()) ;
    // aPrev->mPosInVect + aPrev->NbValByP() ;
    
    StdOut()  <<  E2Str(mTIR)  
              << ", VS :" << aVS.size() 
              << " SZ=" << mSzP0Init
              << " Pos=" << mPosInVect 
              // << " NbPatch " << mNbPatch 
              << " Pix=" << mNbPixTot 
              << "\n";
}



void cDataOneInvRad::CheckCoherence(const cDataOneInvRad& aD2) const
{
    // Numbet of file must be equal in all folder
    if(mMDOFIR.size()!=aD2.mMDOFIR.size() || (mNbPatch!=aD2.mNbPatch))
    {
       StdOut() << "\n\n";
       StdOut() << "SIZES = " << mMDOFIR.size() << " " << aD2.mMDOFIR.size() << "\n";
       StdOut() << "TYPES = " << E2Str(mTIR) << " " << E2Str(aD2.mTIR) << "\n";
       MMVII_UsersErrror(eTyUEr::eUnClassedError,"Variable number of file in Invar");
    }
    // Check coherence between corresponding files
    for (int aK=0 ; aK<int(mMDOFIR.size()) ; aK++)
        mMDOFIR[aK].CheckCoherence(aD2.mMDOFIR[aK]);
}

int  cDataOneInvRad::NbValByP() const
{
// StdOut() << "mSzP0Final.xmSzP0Final.x " << mSzP0Final << "\n";
    return SzP0Final().x() * SzP0Final().y();
}


void  cDataOneInvRad::AddPCar()
{
    mKFill = 0;
    // Parse all files
    for (auto &  aMD : mMDOFIR)
    {
         aMD.AddPCar();
    }
    MMVII_INTERNAL_ASSERT_strong(mKFill==mNbPatch,"Internal check in AddPCar");
}


cAppli_ComputeParamIndexBinaire& cDataOneInvRad::Appli()  {return mAppli;}
const std::string & cDataOneInvRad::Dir() const {return mDir;}
const cPt2di &  cDataOneInvRad::SzP0Init() const {return mSzP0Init;}
eTyInvRad  cDataOneInvRad::TIR() const {return mTIR;}
tREAL8  cDataOneInvRad::NbPixTot() const {return mNbPixTot;}
int  cDataOneInvRad::NbPatch() const {return mNbPatch;}
int  cDataOneInvRad::PosInVect() const {return mPosInVect;}

int& cDataOneInvRad::KFill() {return mKFill;}

void  cDataOneInvRad::SetSzP0Final(const cPt2di & aP0F)
{
   MMVII_INTERNAL_ASSERT_strong(mSzP0Final.x()<0,"Multiple Init in SetSzP0Final");
   mSzP0Final = aP0F;
}
const cPt2di &  cDataOneInvRad::SzP0Final() const 
{
   MMVII_INTERNAL_ASSERT_strong(mSzP0Final.x()>0,"SetSzP0Final non initialized");
   return mSzP0Final;
}


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

