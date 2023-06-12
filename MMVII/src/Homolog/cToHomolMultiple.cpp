#include "MMVII_2Include_Tiling.h"
#include "MMVII_MMV1Compat.h"

/**
   \file  cImplemConvertHom

   \brief file contain classes for transforming pair of homologous point by pai 
          in a multiple points

*/


namespace MMVII
{

/*  ************************************************* */
/*                                                    */
/*      CLASS/METHOD of general ineterest that will   */
/*   put in headers (later)                           */
/*                                                    */
/*  ************************************************* */

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

typedef std::pair<int,int>  tIP;  // Type Image point
				
int Im(const tIP & aPair) {return aPair.first;}
int Pt(const tIP & aPair) {return aPair.second;}

/**  Class for representing a "topological" tie-poin, i.e no geometry is stored */
class cTopoMMP : public cMemCheck
{
      public :

        cTopoMMP(const tIP&,const tIP&);

	/// Compute if, for a given image, the point is unique
	void ComputeOk();

	void Add(const tIP&,const tIP&);
	void Add(const cTopoMMP & aT2);
        std::vector<tIP>  mVIP; // Vector Image(x) Point-Indexe(y)
        bool  mOk;    // used to store the result of correctness (is each point represent by only one image)
        bool  mDone;  // marker used to parse points only once

        // bool  mOk;  // Not Ok, as soon as for an image we get 2 different point
};

cTopoMMP::cTopoMMP(const tIP& aP1,const tIP& aP2)  :
     mOk    (false),
     mDone  (false)
{
	Add(aP1,aP2);
}

void cTopoMMP::Add(const tIP& aP1,const tIP& aP2) 
{
	mVIP.push_back(aP1);
	mVIP.push_back(aP2);
}

void cTopoMMP::Add(const cTopoMMP & aT2)
{
    AppendIn(mVIP,aT2.mVIP);
}


void cTopoMMP::ComputeOk()
{
    if (mDone) 
       return;

    mOk = true;
    mDone = true;

    // Lexicographic sort, Im then Pt, that's OK
    std::sort(mVIP.begin(),mVIP.end());

    for (size_t aK=1 ; (aK<mVIP.size()) && mOk ; aK++)
         if (   (  Im(mVIP[aK-1]) == Im(mVIP[aK])  )  && (  Pt(mVIP[aK-1]) == Pt(mVIP[aK])  )   )
            mOk = false;

}

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
	  void MarkeMergeUndone();

	  void ComputeMergedOk();


     private :
	  typedef std::vector<int> tIndCoord;
	  cOneImMEff2MP(const cOneImMEff2MP &) = delete;
	  void CreatMultiplePoint(int aKIm,cOneImMEff2MP &,std::vector<cOneImMEff2MP> &  mVIms);

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
     mMerge.resize(mVPts.size());  // we know now the size of merged points
}

void cOneImMEff2MP::CreatMultiplePoint(std::vector<cOneImMEff2MP> &  aVIms)
{
     for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
     {
         if (mIsFirst[aKIm])  // need to do it only one way
	 {
             CreatMultiplePoint(aKIm,aVIms[aKIm],aVIms);
	 }
     }
}

void cOneImMEff2MP::CreatMultiplePoint(int aKIm1,cOneImMEff2MP &  aIm2,std::vector<cOneImMEff2MP> &  aVIms)
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

        tIP  aIP1 (mNumIm      , aIndP1);
        tIP  aIP2 (aIm2.mNumIm , aIndP2);

	// case no point exist, 
	if ((aT1==nullptr) && (aT2==nullptr))
	{
	      // we create a new one with, initially, only two points
              cTopoMMP * aNew = new cTopoMMP(aIP1,aIP2); 
              aT1 = aNew;
              aT2 = aNew;
	}
	else if ((aT1!=nullptr) && (aT2==nullptr))
	{
           // If T2 dont exist, T1 is the common merged point
           aT1->Add(aIP1,aIP2);
           aT2 = aT1;
	}
	else if ((aT1==nullptr) && (aT2!=nullptr))
	{
		aT2->Add(aIP1,aIP2);
		aT1 = aT2;
	}
	else 
	{
           aT1->Add(aIP1,aIP2);
           if (aT1==aT2) // nothing to do, we have  added IP1,IP2 in new point
	   {
	   }
	   else
	   {
               // More complicated case
               aT1->Add(*aT2); // fisrt put information of T2 in T1
               for (const auto & aIP : aT2->mVIP)
	       {
                   // Now replace T2 by T1 in all point that where refering to T2
                   aVIms.at(Im(aIP)).mMerge.at(MMVII::Pt(aIP)) = aT1;
	       }
	       delete aT2;
	   }
	}
    }
}

void cOneImMEff2MP::ComputeMergedOk()
{
    for (auto & aMerged : mMerge)
        aMerged->ComputeOk();
}

void cOneImMEff2MP::MarkeMergeUndone()
{
    for (auto & aMerged : mMerge)
        aMerged->mDone = false;
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
	   void MarkeMergeUndone();
           cInterfImportHom &           mInterImport;
	   size_t                       mNbIm;
	   std::vector<cOneImMEff2MP>   mVIms;
};

void cMemoryEffToMultiplePoint::MarkeMergeUndone()
{
    for (auto & aIm : mVIms)
        aIm.MarkeMergeUndone();
}

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

    // Make all point undone (to avoid multiple computation)
    MarkeMergeUndone();


    // Compute for all merged point if it OK
    for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
    {
        mVIms.at(aKIm).ComputeMergedOk();
    }

}

/*********************************************************/
/*                                                       */
/*                                                       */
/*                                                       */
/*********************************************************/

namespace NB_BenchMergeHomol
{

class cImage
{
    public :
       cImage(int aNum);

       cPt2dr  mSz;
       cGeneratePointDiff<2>  mGenPts;
};

class  cMultipleTieP
{
      public :
         std::vector<cPt2dr>  mVPts;
         std::vector<int>     mNumIm;
};


class cSimulHom
{
     public :
         cSimulHom(int aNbImage,int aNbPts,int MaxCard);
         ~cSimulHom();

         cMultipleTieP GenMulTieP();
         void  GenEdges(const cMultipleTieP & aMTP,bool WithError);
     private :
	 std::vector<cImage *>    mVIm;
	 std::vector<std::string> mVNames;

	 std::map<cPt2di,cSetHomogCpleIm>   mMapHom;

         int               mNbImage;
         int               mNbPts;
         int               mMaxCard;
};


cSimulHom::cSimulHom(int aNbImage,int aNbPts,int aMaxCard) :
    mNbImage  (aNbImage),
    mNbPts    (aNbPts),
    mMaxCard  (aMaxCard)
{
    for (int aK=0 ; aK<aNbImage ; aK++)
    {
         mVIm.push_back(new cImage(aK));
	 mVNames.push_back(ToStr(aK));
    }
}



cMultipleTieP cSimulHom::GenMulTieP()
{
    cMultipleTieP aRes;

    // 0- Compute multiplicity
    int aMult = 1 + round_up((mMaxCard-1)*std::pow(RandUnif_0_1(),2));
    aMult = std::max(2,std::min(aMult,mMaxCard));

    // 1- Compute  set of images
    aRes.mNumIm = RandSet(aMult,mNbImage);

    // 2- Compute  set of images
    for (const auto & aNI :  aRes.mNumIm )
        aRes.mVPts.push_back(mVIm[aNI]->mGenPts.GetNewPoint());

    return aRes;
}

typedef  std::tuple<int,int,bool> tEdge;
int  S1(const tEdge & anE)       {return std::get<0>(anE);}
int  S2(const tEdge & anE)       {return std::get<1>(anE);}
bool IsRedund(const tEdge & anE) {return std::get<2>(anE);}

void  cSimulHom::GenEdges(const cMultipleTieP & aMTP,bool WithError)
{
    int aMult = aMTP.mNumIm.size();

    std::vector<tEdge > aVEdgesInit;
    
    // 1  Compute  random tree
    std::vector<int>  anOrder = RandPerm(aMult);

    aVEdgesInit.push_back(tEdge({anOrder[0],anOrder[1],false}));
    for (int aK=2; aK<aMult ; aK++)
    {
        int aS1 = anOrder[aK];  // next som un reached
	int aS2 = anOrder[RandUnif_N(aK)];  //
	// randomize the orde
	if (HeadOrTail())
           std::swap(aS1,aS2);
        aVEdgesInit.push_back(tEdge({aS1,aS2,false}));
    }
    
    // 2  Add random edges to make it more multiple

    int aNbAdd = round_ni( ((aMult-1) * (aMult) - (aMult-1)) * std::pow(RandUnif_0_1(),2)) ;

    for (int aK= 0 ; aK<aNbAdd ; aK++)
    {
         int aS1 = RandUnif_N(aMult);
         int aS2 = RandUnif_N(aMult);
	 while (aS1==aS2)
             aS2 = RandUnif_N(aMult);
        aVEdgesInit.push_back(tEdge({aS1,aS2,true}));
    }

    // 3  filter duplicatas

    std::sort(aVEdgesInit.begin(),aVEdgesInit.end());

    std::vector<tEdge > aVEdges;
    aVEdges.push_back(aVEdgesInit[0]);

    for (size_t aK=1 ; aK<aVEdgesInit.size() ; aK++)
    {
        const tEdge & aPrec = aVEdgesInit.at(aK-1);
        const tEdge & aCur  = aVEdgesInit.at(aK);

	if ( (!IsRedund(aCur)) || (S1(aPrec)!=S1(aCur)) || (S2(aPrec)!=S2(aCur)) )
	{
            aVEdges.push_back(aCur);
	}
    }
    
    // 4 generate the tie point itself

}

};



}; // MMVII






