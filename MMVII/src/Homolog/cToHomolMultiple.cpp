#include "MMVII_2Include_Tiling.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_2Include_Serial_Tpl.h"

/**
   \file  cImplemConvertHom

   \brief file contain classes for transforming pair of homologous point by pai 
          in a multiple points

*/


namespace MMVII
{
class cOneImMEff2MP;
class cMemoryEffToMultiplePoint;

/*  ************************************************* */
/*                                                    */
/*      CLASS/METHOD of general ineterest that will   */
/*   put in headers (later)                           */
/*                                                    */
/*  ************************************************* */

template <class TypeKey,class TypeVal> void AddData(const cAuxAr2007 & anAux,std::map<TypeKey,TypeVal> & aMap)
{
    size_t aNb=aMap.size();
    // put or read the number
    AddData(cAuxAr2007("Nb",anAux),aNb);
    // In input, nb is now intialized, we must set the size of list
    //
    if (anAux.Input())
    {
       for (size_t aK=0 ; aK<aNb ; aK++)
       {
          {
            cAuxAr2007 anAuxPair("Pair",anAux);
            TypeKey aKey;
            AddData(anAuxPair,aKey);
	    AddData(anAuxPair,aMap[aKey]);
          }
       }
    }
    else
    {
        for (auto & aPair : aMap)
        {
            cAuxAr2007 anAuxPair("Pair",anAux);
            AddData(anAuxPair,const_cast<TypeKey&>(aPair.first));
            AddData(anAuxPair,const_cast<TypeVal&>(aPair.second));
	    //AddData(anAuxPair,aPair->second);
        }
    }
}

void TMAP()
{
    std::map<std::string,std::vector<cPt2dr>> aMap;
    aMap["1"] = std::vector<cPt2dr>{{1,1}};
    aMap["2"] = std::vector<cPt2dr>{{1,1},{2,2}};
    aMap["0"] = std::vector<cPt2dr>{};

    SaveInFile(aMap,"toto.xml");

    std::map<std::string,std::vector<cPt2dr>> aMap2;
    ReadFromFile(aMap2,"toto.xml");
    aMap2["3"] = std::vector<cPt2dr>{{1,1},{2,2},{3,3}};
    SaveInFile(aMap2,"tata.xml");
}


/**  Methof for sorting simultaneously 2 vector uing lexicographic order => the 2 type must be comparable
      => maybe in future moe=re efficient implementation using a permutation ?
*/

template <class T1,class T2>  void Sort2VectLexico(std::vector<T1> &aV1,std::vector<T2> & aV2)
{
     MMVII_INTERNAL_ASSERT_tiny(aV1.size()==aV2.size(),"Diff size in Sort2V");

     std::vector<std::pair<T1,T2> > aV12;

     for (size_t aK=0 ; aK<aV1.size() ; aK++)
         aV12.push_back(std::pair<T1,T2>(aV1.at(aK),aV2.at(aK)));

     std::sort(aV12.begin(),aV12.end());
     // std::sort(aV12.begin(),aV12.end(),[](const auto &aP1,const auto & aP2){return aP1.first < aP2.first;});

     for (size_t aK=0 ; aK<aV1.size() ; aK++)
     {
          aV1.at(aK) = aV12.at(aK).first;
          aV2.at(aK) = aV12.at(aK).second;
     }
}

/**  Methof for sorting simultaneously 2 vector using only first vector; only first type must be comparible,
       and order among equivalent first value is undefined
      => maybe in future moe=re efficient implementation using a permutation ?
*/

template <class T1,class T2>  void Sort2VectFirstOne(std::vector<T1> &aV1,std::vector<T2> & aV2)
{
     MMVII_INTERNAL_ASSERT_tiny(aV1.size()==aV2.size(),"Diff size in Sort2V");

     std::vector<std::pair<T1,T2> > aV12;

     for (size_t aK=0 ; aK<aV1.size() ; aK++)
         aV12.push_back(std::pair<T1,T2>(aV1.at(aK),aV2.at(aK)));

     std::sort(aV12.begin(),aV12.end(),[](const auto &aP1,const auto & aP2){return aP1.first < aP2.first;});

     for (size_t aK=0 ; aK<aV1.size() ; aK++)
     {
          aV1.at(aK) = aV12.at(aK).first;
          aV2.at(aK) = aV12.at(aK).second;
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

template <class Type,const int Dim> bool  PtLexCompare(const cPtxd<Type,Dim> & aP1,const cPtxd<Type,Dim> & aP2)
{
	return std::lexicographical_compare(aP1.PtRawData(),aP1.PtRawData()+Dim,aP2.PtRawData(),aP2.PtRawData()+Dim);
}

template <class Type,const int Dim>  
         bool VPtLexCompare (const std::vector<cPtxd<Type,Dim>> & aV1,const std::vector<cPtxd<Type,Dim>> & aV2)
{
	return  std::lexicographical_compare(aV1.begin(),aV1.end(),aV2.begin(),aV2.end(),PtLexCompare<Type,Dim>);
}
#if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny )
template <class Type>  void AssertSorted(const std::vector<Type> & aV)
{
    for (size_t aK=1 ; aK<aV.size() ; aK++)
    {
	    MMVII_INTERNAL_ASSERT_tiny(aV[aK-1]<=aV[aK],"AssertSorted");
    }
}
#define ASSERT_SORTED(V) {AssertSorted(V);}
template <class Type> void AssertInRange(const std::vector<Type> & aVect,const Type & aSz)
{
     for (const auto &  aV : aVect)
         MMVII_INTERNAL_ASSERT_tiny(aV<aSz,"AssertInRange");
}
#define ASSERT_IN_RANGE(VEC,SZ) {AssertInRange(VEC,SZ);}


#else
#define ASSERT_SORTED(V) {}
#define ASSERT_IN_RANGE(VEC,SZ) {}
#endif

/*  ************************************************* */
/*                                                    */
/*           class That Will be exported              */
/*   specifically for multiple tie points             */
/*                                                    */
/*  ************************************************* */



class cSetMultipleTiePoints
{
     public :
        typedef std::vector<int>     tConfigIm;
        typedef std::vector<cPt2dr>  tPtsOfConfig;
        cSetMultipleTiePoints(const  std::vector<std::string> & aVNames);

	void AddPMul(const tConfigIm&,const std::vector<cPt2dr> &);

	void  AddData(const cAuxAr2007 & anAux);

	void TestEq(cSetMultipleTiePoints &) const;
     private  :
        std::vector<std::string>          mVNames;
	std::map<tConfigIm,tPtsOfConfig>  mPts;
};

cSetMultipleTiePoints::cSetMultipleTiePoints(const std::vector<std::string> & aVNames) :
    mVNames (aVNames)
{
}
	
void cSetMultipleTiePoints::AddPMul(const tConfigIm& aConfig,const std::vector<cPt2dr> & aVPts)
{
     ASSERT_SORTED(aConfig);
     MMVII_INTERNAL_ASSERT_tiny((aConfig.size()==aVPts.size()) ,"Diff size in Add PMul");
     ASSERT_IN_RANGE(aConfig,(int)mVNames.size())

     AppendIn(mPts[aConfig],aVPts);
}

void cSetMultipleTiePoints::TestEq(cSetMultipleTiePoints &aS2) const
{
    const std::map<tConfigIm,tPtsOfConfig> & aMapPts1 = mPts ;
    const std::map<tConfigIm,tPtsOfConfig> & aMapPts2 = aS2.mPts ;

    MMVII_INTERNAL_ASSERT_tiny( aMapPts1.size()== aMapPts2.size(),"SetMultipleTiePoints::TestEq");

    for (const auto  &  aP1 : aMapPts1)
    {
        const auto & aItP2 =  aMapPts2.find(aP1.first);
        MMVII_INTERNAL_ASSERT_tiny( aItP2!=aMapPts2.end() ,"SetMultipleTiePoints::TestEq");

        tPtsOfConfig aVPts1 = aP1.second;
        tPtsOfConfig aVPts2 = aItP2->second;

        StdOut()  << VPtLexCompare(aVPts1,aVPts2) << VPtLexCompare(aVPts2,aVPts1) << aVPts1 << " " << aVPts2 << "\n";

        MMVII_INTERNAL_ASSERT_tiny(VPtLexCompare(aVPts1,aVPts2)==0 ,"SetMultipleTiePoints::TestEq");
        MMVII_INTERNAL_ASSERT_tiny(VPtLexCompare(aVPts2,aVPts1)==0 ,"SetMultipleTiePoints::TestEq");

    }
}




/*********************************************************************/
/*********************************************************************/
/*********************************************************************/

typedef std::pair<int,int>  tIP;  // Type Image point
				
int Im(const tIP & aPair) {return aPair.first;}
int Pt(const tIP & aPair) {return aPair.second;}
/*
template <class T1,class T2> std::ostream & operator << (std::ostream & OS,const std::pair<T1,T2> &aPair)
{
    OS  << "{" << aPair.first  << "," << aPair.second << "}";
    return OS;
}
*/


/**  Class for representing a "topological" tie-poin, i.e no geometry is stored */
class cTopoMMP : public cMemCheck
{
      public :

        cTopoMMP(const tIP&,const tIP&);

	/// Compute if, for a given image, the point is unique
	void ComputeOk(cSetMultipleTiePoints & aRes,const std::vector<cOneImMEff2MP> &  aVIms);

	void Add(const tIP&,const tIP&);
	void Add(const cTopoMMP & aT2);

        std::vector<tIP>  mVIP; // Vector Image(x) Point-Indexe(y)
        bool  mOk;    // used to store the result of correctness (is each point represent by only one image)
        bool  mDone;  // marker used to parse points only once

	int   mCpt;
	bool   mKilled;

        // bool  mOk;  // Not Ok, as soon as for an image we get 2 different point
	
};

/**   Class for presenting a merged/compactified version of multiple homologous point
 * of one image.  The same point being present in several set of homol, at the
 * end :
 *      * the point is represented only once in tab mVPts
 *      * the multiple reference to this file are integer (pointing to mVPts)
 *         store in mIndPts[aKIm]
 *
 */
class cMemoryEffToMultiplePoint;
class cOneImMEff2MP
{
     public :
          // friend class cMemoryEffToMultiplePoint;
          cOneImMEff2MP() {}
	  void AddCnx(int aNumIm,bool IsFirt);
	  void SetNameIm(const std::string & aNameIm);
	  void SetNumIm(int aNumIm);
	  const std::string &  NameIm() const;

	  void ComputeIndexPts(cInterfImportHom &,const  cMemoryEffToMultiplePoint &);
	  void CreatMultiplePoint(cMemoryEffToMultiplePoint &);
	  const cPt2dr & Pt(int aNum) const {return mVPts.at(aNum);}

	  /// Mark Done False for all points
	  void MarkeMergeUndone();

	  void ComputeMergedOk(cSetMultipleTiePoints & aRes,const std::vector<cOneImMEff2MP> &  aVIms);

          void DeleteMTP(std::vector<cOneImMEff2MP> &  aVIms);

          void ShowMerged(std::vector<cOneImMEff2MP> &  aVIms);
	  void ShowInit() const;
	  void ShowCur() const;
     private :

	  void ShowTestMerge(cTopoMMP*);

	  typedef std::vector<int> tIndCoord;
	  cOneImMEff2MP(const cOneImMEff2MP &) = delete;
	  void CreatMultiplePoint(int aKIm,cOneImMEff2MP &,cMemoryEffToMultiplePoint &);

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







cTopoMMP::cTopoMMP(const tIP& aP1,const tIP& aP2)  :
     mOk    (false),
     mDone  (false)
{
static int aCpt = 0 ;  mCpt  = aCpt++; mKilled = false;

     Add(aP1,aP2);
}

void cTopoMMP::Add(const tIP& aP1,const tIP& aP2) 
{
// StdOut ()  << "cTopoMMP::Add " <<this << "\n";
     mVIP.push_back(aP1);
     mVIP.push_back(aP2);
// StdOut ()  << "----------cTopoMMP::Add " <<this << "\n";
}

void cTopoMMP::Add(const cTopoMMP & aT2)
{
    AppendIn(mVIP,aT2.mVIP);
}


void cTopoMMP::ComputeOk(cSetMultipleTiePoints & aRes,const std::vector<cOneImMEff2MP> &  aVIms)
{
    if (mDone) 
       return;

    mOk = true;
    mDone = true;

    // Lexicographic sort, Im then Pt, that's OK
    std::sort(mVIP.begin(),mVIP.end());

    std::vector<int>    aVIm;
    std::vector<cPt2dr> aVPts;
    for (size_t aK=0 ; (aK<mVIP.size()) && mOk ; aK++)
    {
         int aNumIm  = Im(mVIP.at(aK));
         int aNumP  =  Pt(mVIP.at(aK));
         if  ((aK>0) && ( Im(mVIP.at(aK-1))==aNumIm) ) 
	 {
	    if (  Pt(mVIP.at(aK-1))!=aNumP )
                mOk = false;
	 }
	 else
	 {
             aVIm.push_back(aNumIm);
             aVPts.push_back(aVIms.at(aNumIm).Pt(aNumP));
	 }
    }

    aRes.AddPMul(aVIm,aVPts);

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
           cMemoryEffToMultiplePoint(cInterfImportHom &,const std::vector<std::string>& aVNames,cSetMultipleTiePoints & aRes);
           ~cMemoryEffToMultiplePoint();
	   std::vector<cOneImMEff2MP> &  VIms() ;  // Accessor
	   const std::vector<cOneImMEff2MP> &  VIms() const;  // Accessor

	   void ShowCur() const;

      private :
	  /// Mark Done False for all points
	   void MarkeMergeUndone();
           cInterfImportHom &           mInterImport;
	   size_t                       mNbIm;
	   std::vector<cOneImMEff2MP>   mVIms;
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
    //  StdOut() << "+++ FNii " << mImCnx << aKIm << "\n";
    // Use binary search for fast recover
    auto anIt = std::lower_bound(mImCnx.begin(),mImCnx.end(),aKIm);
    MMVII_INTERNAL_ASSERT_tiny(anIt!=mImCnx.end(),"At end in FindNumIm");

    // convert iterator to int
    int aRes = anIt-mImCnx.begin();
    MMVII_INTERNAL_ASSERT_tiny(mImCnx.at(aRes)==aKIm,"Can find in FindNumIm");

    // StdOut() << "FNii " << aRes << " " << aKIm << " " << mImCnx[aRes] << "\n";

    // 4 now, called only one way, so check coherence, maybe to relax later
    MMVII_INTERNAL_ASSERT_tiny(!mIsFirst.at(aRes),"Incoherence in mIsFirst");

    return aRes;
}

void cOneImMEff2MP::ShowTestMerge(cTopoMMP* aT2)
{
     if (! mMerge.empty())
     {
         // StdOut() << "STM: " << mIndPts << "\n";
	 for (auto & aMerge : mMerge)
         {
              MMVII_INTERNAL_ASSERT_tiny(aMerge!=aT2,"Incoherence ShowTestMerge");
	 }
     }
}


void cOneImMEff2MP::ComputeIndexPts(cInterfImportHom & anImport,const  cMemoryEffToMultiplePoint &  aMEff2MP)
{
    const  std::vector<cOneImMEff2MP> &  aVIms = aMEff2MP.VIms();
    // sort the  num of connected , for fast retrieval using binary search
     Sort2VectLexico(mImCnx,mIsFirst);
     mNbIm = mImCnx.size();
     mIndPts.resize(mNbIm);


     std::vector<cSetHomogCpleIm > aVHom(mNbIm);

     // Load points & compute box + number of points
     size_t  aNbPts = 0;
     cTplBoxOfPts<tREAL8,2>  aBox;
     for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
     {
         bool IsFirst = mIsFirst.at(aKIm);
         std::string aN1 = mNameIm;
         std::string aN2 = aVIms.at(mImCnx.at(aKIm)).mNameIm;
	 if (! IsFirst) 
            std::swap(aN1,aN2);

         anImport.GetHom (aVHom.at(aKIm),aN1,aN2);

	 for (const auto & aPair : aVHom.at(aKIm).SetH())
	 {
             aBox.Add(aPair.Pt(IsFirst));
	 }
	 aNbPts += aVHom.at(aKIm).SetH().size();
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
         bool IsFirst = mIsFirst.at(aKIm);
	 size_t aNbPair = aVHom.at(aKIm).SetH().size();
	 mIndPts.at(aKIm).resize(aNbPair);

	 for (size_t aKPair = 0 ; aKPair<aNbPair ; aKPair++)
	 {
             const auto & aPair = aVHom.at(aKIm).SetH().at(aKPair);
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
     MMVII_INTERNAL_ASSERT_tiny(mMerge.empty(),"Merge not empty");
     mMerge.resize(mVPts.size(),nullptr);  // we know now the size of merged points
}

void cOneImMEff2MP::CreatMultiplePoint(cMemoryEffToMultiplePoint & aMEff2MP)
{
     std::vector<cOneImMEff2MP> &  aVIms = aMEff2MP.VIms();
     for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
     {
         if (mIsFirst.at(aKIm))  // need to do it only one way
	 {
             CreatMultiplePoint(aKIm,aVIms.at(mImCnx.at(aKIm)),aMEff2MP);
	 }
     }
}

void cOneImMEff2MP::CreatMultiplePoint(int aKIm1,cOneImMEff2MP &  aIm2,cMemoryEffToMultiplePoint & aMEff2MP)
{
     std::vector<cOneImMEff2MP> &  aVIms = aMEff2MP.VIms();
    // find what is the index of I2 in I1
    int aKIm2  = aIm2.FindNumIm(mNumIm);

    const std::vector<int>  & aVInd1 = mIndPts.at(aKIm1);
    const std::vector<int>  & aVInd2 = aIm2.mIndPts.at(aKIm2);

    MMVII_INTERNAL_ASSERT_tiny(aVInd1.size()==aVInd2.size(),"Diff size in CreatMultiplePoint");

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
//StdOut() << "AAAAAAAAAAA "<< __LINE__  << aIP1 << aIP2 << "\n";
//aMEff2MP.ShowCur();
//MMVII_INTERNAL_ASSERT_tiny(mMerge.at(aIndP1)==aNew,"T2T1 assertion");
//MMVII_INTERNAL_ASSERT_tiny(aIm2.mMerge.at(aIndP2)==aNew,"T2T1 assertion");
	}
	else if ((aT1!=nullptr) && (aT2==nullptr))
	{
           // If T2 dont exist, T1 is the common merged point
           aT1->Add(aIP1,aIP2);
           aT2 = aT1;

//StdOut() << "BBBBBBB "<< __LINE__  << aIP1 << aIP2 << "\n";
//aMEff2MP.ShowCur();
//MMVII_INTERNAL_ASSERT_tiny(aIm2.mMerge.at(aIndP2)==aT1,"T2T1 assertion");
	}
	else if ((aT1==nullptr) && (aT2!=nullptr))
	{
		aT2->Add(aIP1,aIP2);
		aT1 = aT2;
//StdOut() << "CCCCCCC "<< __LINE__  << aIP1 << aIP2 << "\n";
//aMEff2MP.ShowCur();
//MMVII_INTERNAL_ASSERT_tiny(mMerge.at(aIndP1)==aT2,"T2T1 assertion");
	}
	else 
	{
           aT1->Add(aIP1,aIP2);
           if (aT1==aT2) // nothing to do, we have  added IP1,IP2 in new point
	   {
	   }
	   else
	   {
// StdOut() << "DDDDDDD T1:"   << aT1  << " " << aT1->mCpt << " ::: "  << " T2:" << aT2   << " " << aT2->mCpt << "\n"; 
//MMVII_INTERNAL_ASSERT_tiny(aT1!=aT2,"0:T1==T2)");
//aMEff2MP.ShowCur();
               // More complicated case
               aT1->Add(*aT2); // fisrt put information of T2 in T1
	       cTopoMMP * aAdrT2 =  aT2;  // Because at end aT2 value T1 for delete !!!!
	       aT2->mKilled = true;
               for (const auto & aIP : aT2->mVIP)
	       {
//MMVII_INTERNAL_ASSERT_tiny(aT1!=aT2,"1:T1==T2)");
                   // Now replace T2 by T1 in all point that where refering to T2
//MMVII_INTERNAL_ASSERT_tiny(aVIms.at(Im(aIP)).mMerge.at(MMVII::Pt(aIP))==aT2,"T2T1 assertion");
                   aVIms.at(Im(aIP)).mMerge.at(MMVII::Pt(aIP)) = aT1;
//MMVII_INTERNAL_ASSERT_tiny(aVIms.at(Im(aIP)).mMerge.at(MMVII::Pt(aIP))==aT1,"T2T1 assertion");
//MMVII_INTERNAL_ASSERT_tiny(aT1!=aT2,"2:T1==T2)");
	       }
	       delete aAdrT2;
// MMVII_INTERNAL_ASSERT_tiny(aIm2.mMerge.at(aIndP2)==aT1,"XXXXX T2T1 assertion");
//MMVII_INTERNAL_ASSERT_tiny(aT1!=aT2,"5:T1==T2)");
//StdOut() << "Iiiiiiiiiiiiiiiiiiiii T1:" <<   aT1 << " T2=" << aT2 << "\n"; 
//StdOut() << "EEEEEE "<< __LINE__  << aIP1 << aIP2 << "\n";
//aMEff2MP.ShowCur();
// for (auto & aIm:aVIms) { aIm.ShowTestMerge(aT2); }
	   }
	}
    }
// aMEff2MP.ShowCur();
}

void cOneImMEff2MP::ComputeMergedOk(cSetMultipleTiePoints & aRes,const std::vector<cOneImMEff2MP> &  aVIms)
{
    for (auto & aMerged : mMerge)
        aMerged->ComputeOk(aRes,aVIms);
}

void cOneImMEff2MP::MarkeMergeUndone()
{
    for (auto & aMerged : mMerge)
        aMerged->mDone = false;
}

void cOneImMEff2MP::DeleteMTP(std::vector<cOneImMEff2MP> &  aVIms)
{
     for (auto aMTP : mMerge)
     {
         // if is has not been erased yet
         if (aMTP != nullptr)
	 {
             // erase all its copy
             for (const auto & aIP :  aMTP->mVIP)
	     {
		   aVIms.at(Im(aIP)).mMerge.at(MMVII::Pt(aIP)) = nullptr;
	     }
	     // and finally delete
             delete aMTP;
	 }
     }
}


void cOneImMEff2MP::ShowMerged(std::vector<cOneImMEff2MP> &  aVIms)
{
    for (auto & aMerged : mMerge)
    {
        StdOut() << "ADRMERGED" << aMerged << "\n";
        if (!  aMerged->mDone)
	{
           aMerged->mDone = true;
	   for (size_t aK=0 ; aK<aMerged->mVIP.size() ; aK++)
	   {
		   int aKI = Im(aMerged->mVIP.at(aK));
		   int aKP = MMVII::Pt(aMerged->mVIP.at(aK));
		   StdOut() << aKI  << aVIms.at(aKI).mVPts.at(aKP) << " ";
	   }
           StdOut() <<  "\n";
           // StdOut() << " VIP " << aMerged->mVIP << "\n";
	}
    }
}

void cOneImMEff2MP::ShowInit() const
{
   StdOut() << "Im=" << mNameIm    << mVPts << "\n";
   for (size_t aK= 0 ;  aK< mNbIm; aK++)
   {
       StdOut()   << " ** "; 
       StdOut()  << (mIsFirst.at(aK)  ? "+" : "-") << " ";
       StdOut()  << mImCnx.at(aK) ;
       StdOut()  << " " << mIndPts.at(aK) ;
       StdOut()   << "\n"; 
   }
}
void cOneImMEff2MP::ShowCur() const
{
   StdOut() << "Im=" << mNameIm   ;
   
   for (size_t aK=0 ; aK< mMerge.size()  ; aK++)
   {
        if (mMerge[aK] ==nullptr)  StdOut() << "[/]" ;
	else if (mMerge[aK]->mKilled )  StdOut() << "[?]" ;
	else  StdOut() << "[" << mMerge[aK]->mCpt << "]";
   }
   StdOut() << "\n";
}


/* ****************************************************************** */
/*                                                                    */
/*             cMemoryEffToMultiplePoint                              */
/*                                                                    */
/* ****************************************************************** */


cMemoryEffToMultiplePoint::cMemoryEffToMultiplePoint
(
      cInterfImportHom & anInterf,
      const std::vector<std::string>& aVNames,
      cSetMultipleTiePoints & aRes
):
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
                 mVIms.at(aKIm2).AddCnx(aKIm1,false);
             }
	 }
    }

    // Load point and make a merge-compact representation
    for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
    {
        mVIms.at(aKIm).ComputeIndexPts(mInterImport,*this);
    }


    /*
    for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
    {
        mVIms.at(aKIm).ShowInit();
    }
    */

    // ShowCur();

    // Create  the multiple points
    for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
    {
        mVIms.at(aKIm).CreatMultiplePoint(*this);
    }

    // Make all point undone (to avoid multiple computation)
    MarkeMergeUndone();

    // Compute for all merged point if it OK
    for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
    {
        mVIms.at(aKIm).ComputeMergedOk(aRes,mVIms);
    }
   
    // Make all point undone (to avoid multiple computation)
    /*
    MarkeMergeUndone();
    for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
    {
        mVIms.at(aKIm).ShowMerged(mVIms);
    }
    */
}

cMemoryEffToMultiplePoint::~cMemoryEffToMultiplePoint()
{
    for (auto & aIm : mVIms)
        aIm.DeleteMTP(mVIms);
}


void cMemoryEffToMultiplePoint::MarkeMergeUndone()
{
    for (auto & aIm : mVIms)
        aIm.MarkeMergeUndone();
}

void cMemoryEffToMultiplePoint::ShowCur() const
{
    StdOut() << "------------------------------------------\n";
    for (auto & aIm : mVIms)
        aIm.ShowCur();
}

      std::vector<cOneImMEff2MP> &  cMemoryEffToMultiplePoint::VIms()       {return mVIms;}
const std::vector<cOneImMEff2MP> &  cMemoryEffToMultiplePoint::VIms() const {return mVIms;}




namespace NS_BenchMergeHomol
{

/*********************************************************/
/*                                                       */
/*                         cMultiplePt                   */
/*                                                       */
/*********************************************************/


/** Minimal class for representing multiple-points */
class  cMultiplePt
{
      public :
         cMultiplePt();

	 void Show();
	 void Sort();
         bool mGotEr;
         std::vector<cPt2dr>  mVPts;
         std::vector<int>     mNumIm;
};


cMultiplePt::cMultiplePt() :
   mGotEr (false)
{
}

void cMultiplePt::Show()
{
    StdOut() << "SHOWMPT ";
    for (size_t aK=0 ; aK<mVPts.size() ; aK++)
       StdOut() << mNumIm.at(aK) << mVPts.at(aK) << " ";
    StdOut() << "\n";
}

bool operator < (cMultiplePt &aPM1,cMultiplePt &aPM2)
{
    if (aPM1.mNumIm < aPM2.mNumIm) return true;
    if (aPM1.mNumIm > aPM2.mNumIm) return false;

    if (VPtLexCompare(aPM1.mVPts,aPM2.mVPts)) return true;
    if (VPtLexCompare(aPM2.mVPts,aPM1.mVPts)) return false;

    return false;
}

void cMultiplePt::Sort()
{
      Sort2VectFirstOne(mNumIm,mVPts);

}

/*********************************************************/
/*                                                       */
/*               NS_BenchMergeHomol                      */
/*                                                       */
/*********************************************************/


class cImage
{
    public :
       cImage(int aNum);

       int     mNum;
       cPt2dr  mSz;
       cGeneratePointDiff<2>  mGenPts;
};

cImage::cImage(int aNum) :
   mNum     (aNum),
   mSz      (3000,2000),
   mGenPts  (cBox2dr(cPt2dr(0,0),ToR(mSz)),0.1)
{
}



typedef std::pair<std::string,std::string>  tSS;  // Type Image point

class cSimulHom : public  cInterfImportHom
{
     public :
         cSimulHom(int aNbImage,int aNbPts,int MaxCard,bool Debug);
         ~cSimulHom();

         cMultiplePt GenMulTieP();
         void  GenEdges(cMultiplePt & aMTP,bool WithError);
	 const std::vector<std::string> & VNames() const;
     private :
	 void GetHom(cSetHomogCpleIm &,const std::string & aNameIm1,const std::string & aNameIm2) const override;
         bool HasHom(const std::string & aNameIm1,const std::string & aNameIm2) const override;

	 std::vector<cImage *>    mVIm;
	 std::vector<std::string> mVNames;

	 std::map<tSS,cSetHomogCpleIm>   mMapHom;

         int               mNbImage;
         int               mNbPts;
         int               mMaxCard;
	 bool              mDebug;
};


bool cSimulHom::HasHom(const std::string & aNameIm1,const std::string & aNameIm2) const
{
    return mMapHom.find(tSS(aNameIm1,aNameIm2)) != mMapHom.end();
}

void cSimulHom::GetHom(cSetHomogCpleIm & aSet,const std::string & aNameIm1,const std::string & aNameIm2) const 
{
   MMVII_INTERNAL_ASSERT_tiny(HasHom(aNameIm1,aNameIm2),"Cannot get hom in cSimulHom");
   aSet = mMapHom.find(tSS(aNameIm1,aNameIm2))->second;
}

cSimulHom::~cSimulHom()
{
    DeleteAllAndClear(mVIm);
}

cSimulHom::cSimulHom(int aNbImage,int aNbPts,int aMaxCard,bool Debug) :
    mNbImage  (aNbImage),
    mNbPts    (aNbPts),
    mMaxCard  (aMaxCard),
    mDebug    (Debug)
{
    for (int aK=0 ; aK<aNbImage ; aK++)
    {
         mVIm.push_back(new cImage(aK));
	 mVNames.push_back(ToStr(aK));
    }
}

const std::vector<std::string> & cSimulHom::VNames() const {return mVNames;}


cMultiplePt cSimulHom::GenMulTieP()
{
    cMultiplePt aRes;

    // 0- Compute multiplicity
    int aMult = 1 + round_up((mMaxCard-1)*std::pow(RandUnif_0_1(),2));
    aMult = std::max(2,std::min(aMult,mMaxCard));

    // 1- Compute  set of images
    aRes.mNumIm = RandSet(aMult,mNbImage);

    // 2- Compute  set of images
    for (const auto & aNI :  aRes.mNumIm )
        aRes.mVPts.push_back(mVIm.at(aNI)->mGenPts.GetNewPoint());

    if (mDebug) 
    {
       StdOut() << "NumIms=" << aRes.mNumIm  << "\n";
    }
    return aRes;
}

typedef  std::tuple<int,int,bool> tEdge;
int  S1(const tEdge & anE)       {return std::get<0>(anE);}
int  S2(const tEdge & anE)       {return std::get<1>(anE);}
bool IsRedund(const tEdge & anE) {return std::get<2>(anE);}

void  cSimulHom::GenEdges(cMultiplePt & aMTP,bool WithError)
{
    int aMult = aMTP.mNumIm.size();

    std::vector<tEdge > aVEdgesInit;
    
    // 1  Compute  random tree
    std::vector<int>  anOrder = RandPerm(aMult);

    aVEdgesInit.push_back(tEdge({anOrder.at(0),anOrder.at(1),false}));
    for (int aK=2; aK<aMult ; aK++)
    {
        int aS1 = anOrder.at(aK);  // next som un reached
	int aS2 = anOrder.at(RandUnif_N(aK));  //
	// randomize the order
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
    aVEdges.push_back(aVEdgesInit.at(0));

    for (size_t aK=1 ; aK<aVEdgesInit.size() ; aK++)
    {
        const tEdge & aPrec = aVEdgesInit.at(aK-1);
        const tEdge & aCur  = aVEdgesInit.at(aK);

	if ( (!IsRedund(aCur)) || (S1(aPrec)!=S1(aCur)) || (S2(aPrec)!=S2(aCur)) )
	{
            aVEdges.push_back(aCur);
	}
    }

    // StdOut() << "M=" << aMult << " E=" << aVEdges.size() << "\n";
    
    // 4 generate the tie points itself

    for (const auto & anE : aVEdges)
    {
         cPt2dr aP1 =  aMTP.mVPts.at(S1(anE));
	 int aI1    =  aMTP.mNumIm.at(S1(anE));

         cPt2dr aP2 = aMTP.mVPts.at(S2(anE));
	 int aI2    =  aMTP.mNumIm.at(S2(anE));

	 if (IsRedund(anE) && WithError && (HeadOrTail()))
	 {
             aMTP.mGotEr = true;
             if (HeadOrTail())
                aP1 = mVIm.at(aI1)->mGenPts.GetNewPoint();
	     else
                aP2 = mVIm.at(aI2)->mGenPts.GetNewPoint();
	 }

	 mMapHom[tSS(ToStr(aI1),ToStr(aI2))].Add(cHomogCpleIm(aP1,aP2));

	 if (mDebug)
            StdOut() <<  "-EeeE=" << aI1 << " " << aI2 << "\n";
    }
}

void OneBench(int aNbImage,int aNbPts,int aMaxCard,bool DoIt)
{
    // StdOut() << "NbImage= " << aNbImage << "\n";
    cSimulHom aSimH(aNbImage,aNbPts,aMaxCard,false);
    cSetMultipleTiePoints aSetMTP1(aSimH.VNames());
    cSetMultipleTiePoints aSetMTP2(aSimH.VNames());

    for (int aKPts=0 ; aKPts<aNbPts ; aKPts++)
    {
        cMultiplePt aMTP = aSimH.GenMulTieP();
	aMTP.Sort();

	//if (DoIt)
	    //aMTP.Show();

	aSetMTP1.AddPMul(aMTP.mNumIm,aMTP.mVPts);
	aSimH.GenEdges(aMTP,false);
    }

    if (DoIt)
    {
        cMemoryEffToMultiplePoint aToMP(aSimH,aSimH.VNames(),aSetMTP2);

	aSetMTP1.TestEq(aSetMTP2);
	// StdOut() << "DONNEEE \n";
	// getchar();
    }

    // getchar();
}

void Bench()
{

	/*
     for (int aK=0 ; aK<10000 ; aK++)
     {
          OneBench(4,2,4,true); //(aK==16));
     }
     */
    // OneBench(10,8,4);
	/*
     for (int aK=0 ; aK<100 ; aK++)
     {
	  //StdOut() << "k=" << aK << "\n";
          OneBench(10,RandInInterval(3,50),4,true); //(aK==16));
     }
    // OneBench(10,8,4);
     //OneBench(10,40,4);

    for (int aK=0 ; aK<100 ; aK++)
    {
        // StdOut() << "k=" << aK << "\n";
        int aNbIm = RandInInterval(3,50);
        OneBench(aNbIm,40,std::min(aNbIm,6),true);
    }
    */
}

};

void Fonc1()
{
       int i0=0; int i1=1;
       std::vector<int *> aVPtrA{&i0,&i1};
       int * & aPI0 = aVPtrA.at(0);
       int * & aPI1 = aVPtrA.at(1);

       aVPtrA.at(1) = aPI0;

       std::cout  << "Zero=" << *aPI0  << " One=" << *aPI1 << "\n";
}



void Bench_ToHomMult(cParamExeBench & aParam)
{
   if (! aParam.NewBench("HomMult")) return;

   // TMAP(); getchar();
   // Fonc1(); Fonc2(); getchar();


   NS_BenchMergeHomol::Bench();

   aParam.EndBench();
}



}; // MMVII






