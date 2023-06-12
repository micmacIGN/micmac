#include "MMVII_2Include_Tiling.h"
#include "MMVII_MMV1Compat.h"

/**
   \file  cImplemConvertHom

   \brief file contain classes for transforming pair of homologous point by pai 
          in a multiple points

*/


namespace MMVII
{

/**  Methof fir sorting simultaneously 2 vector
      => maybe in future moe=re efficient implementation using a permutation ?
*/

template <class T1,class T2>  void Sort2Vect(std::vector<T1> &aV1,std::vector<T2> & aV2)
{
     MMVII_INTERNAL_ASSERT_tiny(aV1.size()==aV2.size(),"Diff size in Sort2V");

     std::vector<std::pair<T1,T2> > aV12;

     for (size_t aK=0 ; aK<aV1.size() ; aK++)
         aV12.push_back(std::pair<T1,T2>(aV1[aK],aV2[aK]));

     std::sort(aV12.begin(),aV12.end());

     for (size_t aK=0 ; aK<aV1.size() ; aK++)
     {
          aV1[aK] = aV12[aK].first;
          aV2[aK] = aV12[aK].second;
     }
}

/**  Assure that P0,P1 are non empty box, using a minimum changes
*/

void  MakeBoxNonEmptyWithMargin(cPt2dr & aP0 ,cPt2dr & aP1,tREAL8 aStdMargin,tREAL8 aMarginSemiEmpty,tREAL8 aMarginEmpty)
{
     cPt2dr aSz = aP1-aP0;

     // Different Precaution for box empty
     if (aSz==cPt2dr(0,0))
        aSz = cPt2dr(aMarginEmpty,aMarginEmpty);
     else if (aSz.x()==0)
        aSz =  cPt2dr(aSz.y()*aMarginSemiEmpty,0.0);
     else if (aSz.y()==0)
        aSz =  cPt2dr(0.0,aSz.x()*aMarginSemiEmpty);
     else
        aSz = aSz * aStdMargin;

     aP0 += -aSz;
     aP1 +=  aSz;
}

/*********************************************************************/
/*********************************************************************/
/*********************************************************************/

/**  Class for representing a "topological" tie-poin, i.e no geometry is stored */
class cTopoMMP
{
      public :

        cTopoMMP(const cPt2di&,const cPt2di&);
        std::vector<cPt2di>  mVIP; // Vector Image(x) Point-Indexe(y)
        bool  mOk;  // Not Ok, as soon as for an image we get 2 different point
};

/*
cTopoMMP::cTopoMMP() :
    mOk (true)
{
}
*/

/**   Class for presenting a merged/compactified version of multiple homologous point
 * of one image.  The same point being present in several set of homol, at the
 * end :
 *      * the point is represented only once in tab mVPts
 *      * the multiple reference to this file are integer (pointing to mVPts)
 *         store in mIndPts[aKIm]
 *
 */
class cOneImMEff2MP
{
     public :
          cOneImMEff2MP();
	  void AddCnx(int aNumIm,bool IsFirt);
	  void SetNameIm(const std::string & aNameIm);
	  void SetNumIm(int aNumIm);
	  const std::string &  NameIm() const;

	  void ComputeIndexPts(cInterfImportHom &,const  std::vector<cOneImMEff2MP> &  mVIms);

	  void CreatMultiplePoint(std::vector<cOneImMEff2MP> &  mVIms);

	  const cPt2dr & Pt(int aNum) {return mVPts.at(aNum);}


     private :
	  typedef std::vector<int> tIndCoord;
	  cOneImMEff2MP(const cOneImMEff2MP &) = delete;
	  void CreatMultiplePoint(int aKIm,cOneImMEff2MP &);

          ///  if  NumIm is an image connected to this, return K such that mImCnx[K] = NumIm
	  int  FindNumIm(int aNumIm) const;

          std::string            mNameIm;
	  int                    mNumIm;
	  std::vector<int>       mImCnx;    // image connected 
	  std::vector<bool>      mIsFirst;  // is it the first
	  std::vector<tIndCoord> mIndPts;   // store the index of coords
	  std::vector<cTopoMMP*> mMerge;   // store the index of coords
	  std::vector<cPt2dr>    mVPts;
	  size_t                 mNbIm;
};

class cToMulH_SpInd
{
      public :
	static constexpr int     Dim = 2;
        typedef cPt2dr           tPrimGeom;
        typedef cOneImMEff2MP *  tArgPG;  /// unused here

        const tPrimGeom & GetPrimGeom(tArgPG anIm) const {return anIm->Pt(mNum);}

        cToMulH_SpInd(int aNum) : mNum (aNum) { }
	int Num() const {return mNum;}

    private :
        int mNum;
};



const std::string &  cOneImMEff2MP::NameIm() const
{
	return mNameIm;
}
void cOneImMEff2MP::SetNameIm(const std::string & aNameIm)
{
    mNameIm = aNameIm;
}

void cOneImMEff2MP::SetNumIm(int aNumIm)
{
    mNumIm = aNumIm;
}

void cOneImMEff2MP::AddCnx(int aNumIm,bool IsFirst)
{
	mImCnx.push_back(aNumIm);
	mIsFirst.push_back(IsFirst);
}

int  cOneImMEff2MP::FindNumIm(int aKIm) const
{
    // Use binary search for fast recover
    auto anIt = std::lower_bound(mImCnx.begin(),mImCnx.end(),aKIm);
    MMVII_INTERNAL_ASSERT_tiny(anIt!=mImCnx.end(),"Can find in FindNumIm");

    // convert iterator to int
    int aRes = anIt-mImCnx.begin();

    // 4 now, called only one way, so check coherence, maybe to relax later
    MMVII_INTERNAL_ASSERT_tiny(!mIsFirst.at(aRes),"Incoherence in mIsFirst");

    return aRes;
}


void cOneImMEff2MP::ComputeIndexPts(cInterfImportHom & anImport,const  std::vector<cOneImMEff2MP> &  aVIms)
{
    // sort the  num of connected , for fast retrieval using binary search
     Sort2Vect(mImCnx,mIsFirst);
     mNbIm = mImCnx.size();
     mIndPts.resize(mNbIm);


     std::vector<cSetHomogCpleIm > aVHom(mNbIm);

     // Load points & compute box + number of points
     size_t  aNbPts = 0;
     cTplBoxOfPts<tREAL8,2>  aBox;
     for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
     {
         bool IsFirst = mIsFirst[aKIm];
         std::string aN1 = mNameIm;
         std::string aN2 = aVIms[mImCnx[aKIm]].mNameIm;
	 if (! IsFirst) 
            std::swap(aN1,aN2);
         anImport.GetHom (aVHom[aKIm],aN1,aN2);

	 for (const auto & aPair : aVHom[aKIm].SetH())
	 {
             aBox.Add(aPair.Pt(IsFirst));
	 }
	 aNbPts += aVHom[aKIm].SetH().size();
     }

     // Precaution if no points, not sure we can handle it, warn 4 now
     if (aNbPts==0) 
     {
        MMVII_DEV_WARNING("To Multiple hom, found an imag w/o point");
        return;
     }

     //  Precaution if box is empty
     cPt2dr aP0 = aBox.P0();
     cPt2dr aP1 = aBox.P1();
     MakeBoxNonEmptyWithMargin(aP0,aP1,1e-3,0.25,1.0);

     cTiling<cToMulH_SpInd> aTile(cBox2dr(aP0,aP1),false,aNbPts,this);

     for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
     {
         bool IsFirst = mIsFirst[aKIm];
	 size_t aNbPair = aVHom[aKIm].SetH().size();
	 mIndPts.at(aKIm).resize(aNbPair);

	 for (size_t aKPair = 0 ; aKPair<aNbPair ; aKPair++)
	 {
             const auto & aPair = aVHom[aKIm].SetH().at(aKPair);
             int aInd=-1;
             // try to find ind at this position
             const cPt2dr & aPt = aPair.Pt(IsFirst);
	     cToMulH_SpInd * aSpInd = aTile.GetObjAtPos(aPt);
	     // if nothing at position, create a new point 
	     if (aSpInd==nullptr)
	     {
                aInd =  mVPts.size();
		mVPts.push_back(aPt);
                aTile.Add(cToMulH_SpInd(aInd));
	     }
	     else
	     {
                aInd = aSpInd->Num();
	     }
	     mIndPts.at(aKIm).at(aKPair) = aInd;
	 }
     }
     // adjust exactly size of points
     mVPts.shrink_to_fit();
     mMerge.resize(mVPts.size());
}

void cOneImMEff2MP::CreatMultiplePoint(std::vector<cOneImMEff2MP> &  mVIms)
{
     for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
     {
         if (mIsFirst[aKIm])  // need to do it only one way
	 {
             CreatMultiplePoint(aKIm,mVIms[aKIm]);
	 }
     }
}

void cOneImMEff2MP::CreatMultiplePoint(int aKIm1,cOneImMEff2MP &  aIm2)
{
    int aKIm2  = aIm2.FindNumIm(mNumIm);

    const std::vector<int>  & aVInd1 = mIndPts[aKIm1];
    const std::vector<int>  & aVInd2 = aIm2.mIndPts[aKIm2];

    MMVII_INTERNAL_ASSERT_tiny(aVInd1.size()==aVInd2.size(),"Diff size in Sort2V");

    for (size_t aK=0 ; aK<aVInd1.size() ; aK++)
    {
        int aIndP1 = aVInd1.at(aK);
	int aIndP2 = aVInd2.at(aK);
        cTopoMMP * & aT1 =  mMerge.at(aIndP1);
        cTopoMMP * & aT2 =  aIm2.mMerge.at(aIndP2);

        cPt2di  aIP1 (mNumIm      , aIndP1);
        cPt2di  aIP2 (aIm2.mNumIm , aIndP2);

	if ((aT1==nullptr) && (aT2==nullptr))
	{
              cTopoMMP * aNew = new cTopoMMP(aIP1,aIP2);

              aT1 = aNew;
              aT2 = aNew;
	}
    }

}


/**
 *     Do the conversion.
 *     This class is targeted to be memory efficient, but not specially fast (for example it read
 *     several times the file).  It is adapted for conversion of "big" data, but not for
 *     conversion on the fly if effecienciey is required.
 *
 *
 */


class cMemoryEffToMultiplePoint
{
      public :
           cMemoryEffToMultiplePoint(cInterfImportHom &,const std::vector<std::string>& aVNames);
      private :
           cInterfImportHom &           mInterImport;
	   size_t                       mNbIm;
	   std::vector<cOneImMEff2MP>   mVIms;
};


cMemoryEffToMultiplePoint::cMemoryEffToMultiplePoint(cInterfImportHom & anInterf,const std::vector<std::string>& aVNames) :
    mInterImport  (anInterf),
    mNbIm         (aVNames.size()),
    mVIms         (mNbIm)
{

    // transfert the name in structure
    for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
    {
        mVIms.at(aKIm).SetNameIm(aVNames.at(aKIm));
        mVIms.at(aKIm).SetNumIm(aKIm);
    }

    // memorize the connexions : list of image conected and sens of connexion
    for (size_t aKIm1=0 ; aKIm1<mNbIm ; aKIm1++)
    {
         for (size_t aKIm2=0 ; aKIm2<mNbIm ; aKIm2++)
	 {
             if (mInterImport.HasHom(mVIms.at(aKIm1).NameIm() ,mVIms.at(aKIm2).NameIm() ))
             {
                 mVIms.at(aKIm1).AddCnx(aKIm2,true);
                 mVIms.at(aKIm2).AddCnx(aKIm1,true);
             }
	 }
    }
    
    // Load point and make a merge-compact representation
    for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
    {
        mVIms.at(aKIm).ComputeIndexPts(mInterImport,mVIms);
    }

    // Create  the multiple points
    for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
    {
        mVIms.at(aKIm).CreatMultiplePoint(mVIms);
    }
}





}; // MMVII

