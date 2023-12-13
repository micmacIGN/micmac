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



    //================================  tPairTiePMult  ===================
    //================================  tPairTiePMult  ===================

size_t Multiplicity(const tPairTiePMult & aPair) { return  Config(aPair).size(); }
size_t NbPtsMul(const tPairTiePMult & aPair)
{
	return Val(aPair).mVPIm.size()  / Multiplicity(aPair);
}

cPt3dr BundleInter(const tPairTiePMult & aPair,size_t aKPts,const std::vector<cSensorImage *>&  aVSI)
{
    const auto &  aConfig = Config(aPair);
    const cVal1ConfTPM & aVal =  Val(aPair);
    size_t aMult = aConfig.size();

    if (aMult<2) 
       return cPt3dr(0,0,0);

    size_t aKP0 = aKPts*aMult;
    std::vector<tSeg3dr>  aVSeg;
    for (size_t aK= 0 ; aK<aMult ; aK++)
    {
        const cPt2dr & aPIm = aVal.mVPIm.at(aKP0+aK);
	cSensorImage * aSI  = aVSI.at(aConfig.at(aK));

	aVSeg.push_back(aSI->Image2Bundle(aPIm));
    }

    return BundleInters(aVSeg);
}


void MakePGround(tPairTiePMult & aPair,const std::vector<cSensorImage *> & aVSI)
{
    std::vector<cPt3dr> & aVPts = Val(aPair).mVPGround;
    aVPts.clear();
    size_t aNbPts = NbPtsMul(aPair);

    for (size_t aKP=0 ; aKP<aNbPts; aKP++)
    {
        aVPts.push_back(BundleInter(aPair,aKP,aVSI));
    }
}




///========================================================




class cOneImMEff2MP;
class cMemoryEffToMultiplePoint;
class cCstrMulP ;

typedef std::pair<int,int>  tIP;  // Type Image point
static int Im(const tIP & aPair) {return aPair.first;}
static int Pt(const tIP & aPair) {return aPair.second;}


/*********************************************************************/
/*********************************************************************/
/*********************************************************************/



/**  Class for representing a multiple tie point in "construction".  The point
 *    store only "topology/relations" od tie-poin, i.e no geometry is stored .
 *    The point is made of list (KIm,KPt) = "mVIP"  wher KIm is the number of the image
 *    and KPt the number of the point in image
 * */
class cCstrMulP : public cMemCheck
{
      public :

        cCstrMulP(const tIP&,const tIP&);

	/// Compute if, for a given image, the point is unique
	void ComputeCoherence(cComputeMergeMulTieP & aRes,const std::vector<cOneImMEff2MP> &  aVIms);

	void Add(const tIP&,const tIP&);
	void Add(const cCstrMulP & aT2);

        std::vector<tIP>  mVIP; ///< Vector Image(x) Point-Indexe(y)
        bool  mOk;    ///< used to store the result of correctness (is each point represent by only one image)
        bool  mDone;  ///< marker used to parse points only once

	int   mCpt;    ///<  Counter for debug
	bool   mKilled;  ///< For debuging, instead of deleting, marked killed
};

/**   Class for presenting a merged/compactified version of multiple homologous point
 * of one image.  At input same point willbe present in several set of pair of homol, at the
 * end :
 *      * the point is represented only once in the tab "mVPts"
 *      * the multiple reference to this point are integer (pointing to mVPts)
 *        store in "mIndPts"[aKIm]
 *
 *    Let I1 be I2 two image :
 *
 *       *  let K1 be such that I1.mImCnx[K1]  = I2.mNumIm , and id for K2, (K1 can be fast extracted with "FindNumIm")
 *       *  at the , for any k  I1.mIndPt[K1][k] and I2.mIndPt[K2][k] are each referencing a pair of homologous point
 *
 */
class cOneImMEff2MP
{
     public :
          friend class cMemoryEffToMultiplePoint;
          friend class cCstrMulP ;
          /// default constructor
          cOneImMEff2MP(); 

     private :
	  void SetNameIm(const std::string & aNameIm); ///< Modifier
	  void SetNumIm(int aNumIm);  ///< Modifier
	  const std::string &  NameIm() const;   ///< Accessor

	  /// Accessor to Pts
	  const cPt2dr & Pt(int aNum) const {return mVPts.at(aNum);}

	  ///  Show merged point 
          void ShowMerged(std::vector<cOneImMEff2MP> &  aVIms);
	  void ShowInit() const;    ///  Show cnx after initialisation
	  void ShowCur() const;     /// Show curent state of conexion

	  void AddCnx(int aNumIm,bool IsFirt);  /// Add a connexion with aNumIm, IsFirst indicate if was firt of the pair
	  /// Read the point&store a unique geomtry & Store the Homolog as vector of int in each image ref to geom
	  void ComputeIndexPts(cInterfImportHom &,const  cMemoryEffToMultiplePoint &);
	  ///  Create the multiple point structure by adding the edges in 
	  void CreatMultiplePoint(cMemoryEffToMultiplePoint &);

	  /// Mark Done False for all points, to avoid redundancy in parsing
	  void MarkeMergeUndone();

	  ///  Compute for each point if there is no incoherence, and if yes add it to results "aRes"
	  void ComputeMergedOk(cComputeMergeMulTieP & aRes,const std::vector<cOneImMEff2MP> &  aVIms);

	  ///  Delete the Multie point referenced
          void DeleteMTP(std::vector<cOneImMEff2MP> &  aVIms);

	  void ShowTestMerge(cCstrMulP*);

	  typedef std::vector<int> tIndCoord;
	  cOneImMEff2MP(const cOneImMEff2MP &) = delete;
	  void CreatMultiplePoint(int aKIm,cMemoryEffToMultiplePoint &);

          ///  if  NumIm is an image connected to this, return K such that mImCnx[K] = NumIm
	  int  FindNumIm(int aNumIm) const;

          std::string            mNameIm;   ///< Name of image
	  int                    mNumIm;    ///< number of image
	  std::vector<int>       mImCnx;    ///< image connected  : size "NbIm"
	  std::vector<bool>      mIsFirst;  ///< is it the first  : size "mNbIm"
	  std::vector<tIndCoord> mIndPts;   ///< store the index of coords : size "mNbIm"
	  std::vector<cCstrMulP*> mMerge;    ///< store the merged points : size "mNbPts"
	  std::vector<cPt2dr>    mVPts;     ///<  store the geometry of points
	  size_t                 mNbIm;     ///<  number of image
};

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
           cMemoryEffToMultiplePoint(cInterfImportHom &,const std::vector<std::string>& aVNames,cComputeMergeMulTieP & aRes);
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

/* ***************************************************** */
/*                                                       */
/*                  cComputeMergeMulTieP                 */
/*                                                       */
/* ***************************************************** */

bool operator == (const cVal1ConfTPM & aV1,const cVal1ConfTPM & aV2)
{
	return     (aV1.mVPIm== aV2.mVPIm)
	       &&  (aV1.mVPGround== aV2.mVPGround)
	       &&  (aV1.mVIdPts== aV2.mVIdPts)
        ;
}

cComputeMergeMulTieP::cComputeMergeMulTieP
(
       const std::vector<std::string> & aVNames,
       cInterfImportHom * anIIH,
       cPhotogrammetricProject*  aPhP ,
       bool                      WithImageIndexe
) :
    mVNames (aVNames)
{
   ASSERT_SORTED(aVNames);
   if (anIIH)
   {
      cMemoryEffToMultiplePoint aToMP(*anIIH,mVNames,*this);
   }

   if (aPhP)
   {
      for (const auto & aName : mVNames)
          mVSensors.push_back(aPhP->LoadSensor(aName,false));
   }

   if (WithImageIndexe)
      SetImageIndexe();
}
const std::vector<std::list<std::pair<size_t,tPairTiePMult*>>> & cComputeMergeMulTieP::IndexeOfImages()  const
{
	return mImageIndexes;
}

void cComputeMergeMulTieP::SetImageIndexe()
{
     mImageIndexes.resize(mVNames.size());

     for (auto & aPair : mPts)
     {
         auto & aConfig = aPair.first;
	 tPairTiePMult * anAdr = & aPair;
	 for (size_t aKI=0 ; aKI<aConfig.size() ; aKI++)
         {
              mImageIndexes.at(aConfig.at(aKI)).push_back(std::pair(aKI,anAdr));
	 }
     }
}

const std::vector<std::string> & cComputeMergeMulTieP::VNames() const { return mVNames;}

const std::vector<cSensorImage *> &  cComputeMergeMulTieP::VSensors() const
{
   MMVII_INTERNAL_ASSERT_tiny(!mVSensors.empty(),"Sensor non initialized in cComputeMergeMulTieP");

   return mVSensors;
}

const std::map<std::vector<int>,cVal1ConfTPM> &  cComputeMergeMulTieP::Pts() const {return mPts;}
std::map<std::vector<int>,cVal1ConfTPM> &  cComputeMergeMulTieP::Pts() {return mPts;}

	
void cComputeMergeMulTieP::AddPMul(const tConfigIm& aConfig,const std::vector<cPt2dr> & aVPts)
{
     // Different check of coherence
     ASSERT_SORTED(aConfig);
     MMVII_INTERNAL_ASSERT_tiny((aConfig.size()==aVPts.size()) ,"Diff size in Add PMul");
     ASSERT_IN_RANGE(aConfig,(int)mVNames.size())

     //  finally add new pts to config
     AppendIn(mPts[aConfig].mVPIm,aVPts);
}

void cComputeMergeMulTieP::TestEq(cComputeMergeMulTieP &aS2) const
{
    const std::map<tConfigIm,cVal1ConfTPM> & aMapPts1 = mPts ;
    const std::map<tConfigIm,cVal1ConfTPM> & aMapPts2 = aS2.mPts ;

    if  (0)
    {
	    /*
        StdOut() << "SZZZZ " << aMapPts1.size() << " " << aMapPts2.size() <<  std::endl ;
        for (const auto  &  aP1 : aMapPts1)
            StdOut() <<  " * C1=" << aP1.first  << " Nb=" << aP1.second.size() << std::endl ;
        StdOut() << "==================" << std::endl ;
        for (const auto  &  aP2 : aMapPts2)
            StdOut() <<  " * C2=" << aP2.first  << " Nb=" << aP2.second.size() << std::endl;
        StdOut() << "==================" << std::endl ;
	*/
    }

    MMVII_INTERNAL_ASSERT_bench( aMapPts1.size()== aMapPts2.size(),"SetMultipleTiePoints::TestEq");

    for (const auto  &  aP1 : aMapPts1)
    {
        const auto & aConfig = aP1.first;
        const auto & aItP2 =  aMapPts2.find(aConfig);
	Fake4ReleaseUseIt(aItP2);
        MMVII_INTERNAL_ASSERT_bench( aItP2!=aMapPts2.end() ,"SetMultipleTiePoints::TestEq");

	auto aVPMul1 = PUnMixed(aConfig,true);  // true = sort
	auto aVPMul2 = aS2.PUnMixed(aConfig,true); // because order would not be preserved
        MMVII_INTERNAL_ASSERT_bench(aVPMul1.size() ==aVPMul2.size(),"SetMultipleTiePoints::TestEq");
        for (size_t aK=0 ;  aK<aVPMul1.size() ; aK++)
        {
            MMVII_INTERNAL_ASSERT_bench(aVPMul1[aK].mVPIm==aVPMul2[aK].mVPIm,"SetMultipleTiePoints::TestEq");
            MMVII_INTERNAL_ASSERT_bench(aVPMul1[aK].mVIm==aVPMul2[aK].mVIm,"SetMultipleTiePoints::TestEq");
        }
    }
}

/*   If , for example, NbMul=3 :
 *
 *   [  a b d a b c e ...]  =>  [[a b d] [a b c] [e ...] ...]  ,  case unsorted
 *   [  a b d a b c e ...]  =>  [[a b c] [a b d] [e ...] ...]  ,  case sorted (on lexicographic order)
 *
 *   Sorted is usefull essentially for checking equality 
 */
std::vector<cPMulGCPIm>
    cComputeMergeMulTieP::PUnMixed(const tConfigIm & aConfigIm,bool Sorted) const
{
    const auto & anIt  = mPts.find(aConfigIm);
    MMVII_INTERNAL_ASSERT_tiny(anIt!=mPts.end(),"cComputeMergeMulTieP::SortedPtsOf");
    const auto & aValue = anIt->second;

    size_t aMult = aConfigIm.size();
    size_t aNbPMul = aValue.mVPIm.size() / aMult;

    //  unmix
    std::vector<cPMulGCPIm> aRes(aNbPMul);
    for (size_t aK=0 ; aK<aNbPMul ; aK++)
    {
        aRes.at(aK).mVPIm = std::vector<cPt2dr>(aValue.mVPIm.begin()+aK*aMult,aValue.mVPIm.begin()+(aK+1)*aMult);
	if (! aValue.mVIdPts.empty())
            aRes.at(aK).mName = ToStr(aValue.mVIdPts.at(aK));
	if (! aValue.mVPGround.empty())
            aRes.at(aK).mPGround = aValue.mVPGround.at(aK);
         aRes.at(aK).mVIm = aConfigIm;
    }

    // sort
    if (Sorted)
    {
         std::sort
         (
	     aRes.begin(),aRes.end(),
	     [](const auto & aVal1,const auto& aVal2) {return aVal1.mVPIm < aVal2.mVPIm;}
         );
    }
     /*
    */

    return aRes;
}

void cComputeMergeMulTieP::Shrink()
{
     for (auto &  aPair : mPts)
     {
         // aPair.first.shrink_to_fit();  : generate constness  issue
         aPair.second.mVPIm.shrink_to_fit();
         aPair.second.mVIdPts.shrink_to_fit();
         aPair.second.mVPGround.shrink_to_fit();
     }
}

void cComputeMergeMulTieP::SetPGround()
{
    for (auto & aPair : mPts)
        MakePGround(aPair,mVSensors);
}




/* ************************************************* */
/*                                                   */
/*                 cCstrMulP                          */
/*                                                   */
/* ************************************************* */

cCstrMulP::cCstrMulP(const tIP& aP1,const tIP& aP2)  :
     mOk    (false),
     mDone  (false)
{
     //  debug info
     {
	     static int aCpt = 0 ;  mCpt  = aCpt++; 
	     mKilled = false;
     }

     Add(aP1,aP2);  // initialize with one pair
}

void cCstrMulP::Add(const tIP& aP1,const tIP& aP2)   // just memorize the pair  Im-Pt
{
     mVIP.push_back(aP1);
     mVIP.push_back(aP2);
}

void cCstrMulP::Add(const cCstrMulP & aT2)
{
    AppendIn(mVIP,aT2.mVIP);  // concat pair Im-Pt  of T2 to T1
}


void cCstrMulP::ComputeCoherence(cComputeMergeMulTieP & aRes,const std::vector<cOneImMEff2MP> &  aVIms)
{
    // Marker "mDone" is used to avoid multiple computation on shared point
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
	 //  if image equal previous one
         if  ((aK>0) && ( Im(mVIP.at(aK-1))==aNumIm) ) 
	 {
	    // if we find a merge-point with two different  2D point in the same image : bad
	    if (  Pt(mVIP.at(aK-1))!=aNumP )
                mOk = false;
	 }
	 else
	 {
             // if first or != previous one, then add it to merge
             aVIm.push_back(aNumIm);
             aVPts.push_back(aVIms.at(aNumIm).Pt(aNumP));
	 }
    }

    // if coherent, put it in result
    if (mOk)
       aRes.AddPMul(aVIm,aVPts);
}

/* ************************************************* */
/*                                                   */
/*                 cOneImMEff2MP                     */
/*                                                   */
/* ************************************************* */

     // ========== construction an +- destruction ===========

cOneImMEff2MP:: cOneImMEff2MP()
{
}

void cOneImMEff2MP::DeleteMTP(std::vector<cOneImMEff2MP> &  aVIms)
{
     for (auto aMTP : mMerge)  // parse merge point of image
     {
         // if is has not been erased yet
         if (aMTP != nullptr)
	 {
             // erase all its copy to avoid multiple delete
             for (const auto & aIP :  aMTP->mVIP)
	     {
		   aVIms.at(Im(aIP)).mMerge.at(MMVII::Pt(aIP)) = nullptr;
	     }
	     // and finally delete
             delete aMTP;
	 }
     }
}

     // ========== Modifier / Accessor ===========

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

void cOneImMEff2MP::MarkeMergeUndone()
{
    for (auto & aMerged : mMerge)
        aMerged->mDone = false;
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
    MMVII_INTERNAL_ASSERT_tiny(anIt!=mImCnx.end(),"At end in FindNumIm");

    // convert iterator to int
    int aRes = anIt-mImCnx.begin();
    MMVII_INTERNAL_ASSERT_tiny(mImCnx.at(aRes)==aKIm,"Can find in FindNumIm");

    // StdOut() << "FNii " << aRes << " " << aKIm << " " << mImCnx[aRes] << std::endl;

    // 4 now, called only one way, so check coherence, maybe to relax later
    MMVII_INTERNAL_ASSERT_tiny(!mIsFirst.at(aRes),"Incoherence in mIsFirst");

    return aRes;
}

/*   Create the topology from the geometry,  ie from pair of homologous point, create
 *   a structure +- equivalent to a graph.  
 *
 */

void cOneImMEff2MP::ComputeIndexPts(cInterfImportHom & anImport,const  cMemoryEffToMultiplePoint &  aMEff2MP)
{
     //  Spatial index to recover multiple point,  it contains only on int which is
     //  referencing the "mPts" of the imafe
     class cToMulH_SpInd
     {
           public :
	     //  static constexpr int     Dim = 2;  => dont work with local class, hapilly enum works
	     enum {Dim=2};
             typedef cPt2dr           tPrimGeom;  // geometric primitives indexed are points
	     // type of arg that we will used in call back "GetPrimGeom", we need to refer to images
             typedef cOneImMEff2MP *  tArgPG;     

	     /// from image & index compute the point refered by index
             const tPrimGeom & GetPrimGeom(tArgPG anIm) const {return anIm->Pt(mNum);}

             cToMulH_SpInd(int aNum) : mNum (aNum) { }  ///< constructor
	     int Num() const {return mNum;}   ///< accessor

         private :
             int mNum;
    };

     const  std::vector<cOneImMEff2MP> &  aVIms = aMEff2MP.VIms();
     // sort the  num of connected , for fast retrieval using binary search, for coherence sort in // "mIsFirst"
     Sort2VectLexico(mImCnx,mIsFirst);
     mNbIm = mImCnx.size();  // utilitary now we know the size
     mIndPts.resize(mNbIm);


     std::vector<cSetHomogCpleIm > aVHom(mNbIm);

     // 1- Load points & compute box + number of points
     size_t  aNbPts = 0;
     cTplBoxOfPts<tREAL8,2>  aBox;
     for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
     {
         // 1.1 load the homologous point
         bool IsFirst = mIsFirst.at(aKIm);  // we need to know if point of this image are 1 or 2
         std::string aN1 = mNameIm;
         std::string aN2 = aVIms.at(mImCnx.at(aKIm)).mNameIm;
	 if (! IsFirst)  // put name in right order
            std::swap(aN1,aN2);
         anImport.GetHom (aVHom.at(aKIm),aN1,aN2);  // now store the point

	 //  1.2  add all point to box
	 for (const auto & aPair : aVHom.at(aKIm).SetH())
	 {
             aBox.Add(aPair.Pt(IsFirst));
	 }
	 aNbPts += aVHom.at(aKIm).SetH().size();
     }

     // Precaution if no points, not sure we can handle it, warn 4 now
     if (aNbPts==0) 
     {
        //  MMVII_DEV_WARNING("To Multiple hom, found an imag w/o point");  apparently, donnt create problem
        return;
     }

     //  Precaution if box is empty
     cPt2dr aP0 = aBox.P0();
     cPt2dr aP1 = aBox.P1();
     MakeBoxNonEmptyWithMargin(aP0,aP1,1e-3,0.25,1.0);

     // 2- Compute the index & the graph-like structure
     cTiling<cToMulH_SpInd> aTile(cBox2dr(aP0,aP1),false,aNbPts,this);  // tiling for geometric indexing

     for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
     {
         bool IsFirst = mIsFirst.at(aKIm);
	 size_t aNbPair = aVHom.at(aKIm).SetH().size();
	 mIndPts.at(aKIm).resize(aNbPair,-1);  // fix the adjacency struct

	 for (size_t aKPair = 0 ; aKPair<aNbPair ; aKPair++)
	 {
             const auto & aPair = aVHom.at(aKIm).SetH().at(aKPair);
             int aInd=-1;
             const cPt2dr & aPt = aPair.Pt(IsFirst); // extract point 1 or 2
             // try to find an index already create  at this position
	     cToMulH_SpInd * aSpInd = aTile.GetObjAtPos(aPt);
	     if (aSpInd==nullptr) // if nothing at position, must create a new point 
	     {
                aInd =  mVPts.size(); // ie 0 if first point ...
		mVPts.push_back(aPt); // memorize point that will be refered
                aTile.Add(cToMulH_SpInd(aInd));  // memorize index
	     }
	     else
	     {
                aInd = aSpInd->Num();
	     }
	     mIndPts.at(aKIm).at(aKPair) = aInd;  // create "edge" on the graph like struct
	 }
     }
     mVPts.shrink_to_fit(); // adjust exactly size of points
     MMVII_INTERNAL_ASSERT_tiny(mMerge.empty(),"Merge not empty");
     mMerge.resize(mVPts.size(),nullptr);  // we know now the size of merged points
}

void cOneImMEff2MP::CreatMultiplePoint(cMemoryEffToMultiplePoint & aMEff2MP)
{
     for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
     {
         if (mIsFirst.at(aKIm))  // need to do it only one way
	 {
             CreatMultiplePoint(aKIm,aMEff2MP);
	 }
     }
}

void cOneImMEff2MP::CreatMultiplePoint(int aKIm1,cMemoryEffToMultiplePoint & aMEff2MP)
{
    std::vector<cOneImMEff2MP> &  aVIms = aMEff2MP.VIms();
    cOneImMEff2MP &  aIm2 = aVIms.at(mImCnx.at(aKIm1));
    // find what is the index of I2 in I1
    int aKIm2  = aIm2.FindNumIm(mNumIm);

    const std::vector<int>  & aVInd1 = mIndPts.at(aKIm1);
    const std::vector<int>  & aVInd2 = aIm2.mIndPts.at(aKIm2);

    MMVII_INTERNAL_ASSERT_tiny(aVInd1.size()==aVInd2.size(),"Diff size in CreatMultiplePoint");

    for (size_t aK=0 ; aK<aVInd1.size() ; aK++)
    {
        int aIndP1 = aVInd1.at(aK);
	int aIndP2 = aVInd2.at(aK);
        cCstrMulP * & aT1 =  mMerge.at(aIndP1);
        cCstrMulP * & aT2 =  aIm2.mMerge.at(aIndP2);

        tIP  aIP1 (mNumIm      , aIndP1);
        tIP  aIP2 (aIm2.mNumIm , aIndP2);

	// case no point exist, 
	if ((aT1==nullptr) && (aT2==nullptr))
	{
	      // we create a new one with, initially, only two points
              cCstrMulP * aNew = new cCstrMulP(aIP1,aIP2); 
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
               // More complicated case we must merge, arbitrarily merge T1 in T2
               aT1->Add(*aT2); // fisrt put information of T2 in T1
	       cCstrMulP * aAdrT2 =  aT2;  // !!!!!!!!!    Because at end aT2 value aT1 for delete !!!!
	       aT2->mKilled = true;
               
	       // all the point that where refering to T2 must now refer to T1
               for (const auto & aIP : aT2->mVIP)
	       {
                   aVIms.at(Im(aIP)).mMerge.at(MMVII::Pt(aIP)) = aT1;
	       }
	       delete aAdrT2;  // delete aT2 => CRACK BOUM SCRATCH
	   }
	}
    }
}

void cOneImMEff2MP::ComputeMergedOk(cComputeMergeMulTieP & aRes,const std::vector<cOneImMEff2MP> &  aVIms)
{
    for (auto & aMerged : mMerge)
        aMerged->ComputeCoherence(aRes,aVIms);
}

    // =======   Different printing for debugging =================

void cOneImMEff2MP::ShowMerged(std::vector<cOneImMEff2MP> &  aVIms)
{
    for (auto & aMerged : mMerge)
    {
        StdOut() << "ADRMERGED" << aMerged << std::endl;
        if (!  aMerged->mDone)
	{
           aMerged->mDone = true;
	   for (size_t aK=0 ; aK<aMerged->mVIP.size() ; aK++)
	   {
		   int aKI = Im(aMerged->mVIP.at(aK));
		   int aKP = MMVII::Pt(aMerged->mVIP.at(aK));
		   StdOut() << aKI  << aVIms.at(aKI).mVPts.at(aKP) << " ";
	   }
           StdOut() <<  std::endl;
           // StdOut() << " VIP " << aMerged->mVIP << std::endl;
	}
    }
}

void cOneImMEff2MP::ShowInit() const
{
   StdOut() << "Im=" << mNameIm    << mVPts << std::endl;
   for (size_t aK= 0 ;  aK< mNbIm; aK++)
   {
       StdOut()   << " ** "; 
       StdOut()  << (mIsFirst.at(aK)  ? "+" : "-") << " ";
       StdOut()  << mImCnx.at(aK) ;
       StdOut()  << " " << mIndPts.at(aK) ;
       StdOut()   << std::endl; 
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
   StdOut() << std::endl;
}

void cOneImMEff2MP::ShowTestMerge(cCstrMulP* aT2)
{
     if (! mMerge.empty())
     {
         // StdOut() << "STM: " << mIndPts << std::endl;
	 for (auto & aMerge : mMerge)
         {
              Fake4ReleaseUseIt(aMerge);
              MMVII_INTERNAL_ASSERT_tiny(aMerge!=aT2,"Incoherence ShowTestMerge");
	 }
     }
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
      cComputeMergeMulTieP & aRes
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
   
}

cMemoryEffToMultiplePoint::~cMemoryEffToMultiplePoint()
{
    // delete the temporary multiple points
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
    StdOut() << "------------------------------------------" << std::endl;
    for (auto & aIm : mVIms)
        aIm.ShowCur();
}

      std::vector<cOneImMEff2MP> &  cMemoryEffToMultiplePoint::VIms()       {return mVIms;}
const std::vector<cOneImMEff2MP> &  cMemoryEffToMultiplePoint::VIms() const {return mVIms;}


  /**********************************************************************************************/
  /**********************************************************************************************/
  /**********************************************************************************************/
  /**********                                                                         ***********/
  /**********                          BENCH PART                                     ***********/
  /**********                                                                         ***********/
  /**********************************************************************************************/
  /**********************************************************************************************/
  /**********************************************************************************************/


namespace NS_BenchMergeHomol
{

/**  For generating data-check for multiple points :
 *
 *     - generate random sets "S" of multiple point i.e  random set of point (Card=N) 
 *        + random set of int (images they belong to, same Card)
 *
 *     - for the homologous points :
 *
 *            - a random tree on "S" that is the minimal two recover the
 *            - a random subset of random edges
 *            - optionnaly add some error to generate wrong points
 */

/** class for representing simulated multiple-points */
class  cMultiplePt
{
      public :
         cMultiplePt();  ///< constructor

	 void Show();                  ///< Show details
         bool mGotEr;                  ///<  Did we generate an error in 
         std::vector<cPt2dr>  mVPts;   ///<  Simulated point
         std::vector<int>     mNumIm;  ///< Simlated num of images
};

/** class for representing one image in simul */

class cImage
{
    public :
       cImage(int aNum);

       int     mNum;   ///<  Num of image
       cPt2dr  mSz;    ///<  Size of image
       cGeneratePointDiff<2>  mGenPts;  ///< Generate point all different
};

typedef std::pair<std::string,std::string>  tSS;  // pair of name for storing Name x Name => Cple homol

/** class for generating Multiple random points + correspond random
 * homologous for generating them
 */

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

     int NbPts() const { return mNbPts;}    //CM: avoid mNbPts unused
     int KPts() const { return mKPts;}      //CM: avoid mKPts unused

	 std::vector<cImage *>    mVIm;
	 std::vector<std::string> mVNames;

	 std::map<tSS,cSetHomogCpleIm>   mMapHom;

         int               mNbImage;
         int               mNbPts;
         int               mMaxCard;
	 bool              mDebug;
	 int               mKPts;
};



    /*=======================================================*/
    /*                         cMultiplePt                   */
    /*=======================================================*/

cMultiplePt::cMultiplePt() :
   mGotEr (false)
{
}

void cMultiplePt::Show()
{
    StdOut() << "SHOWMPT ";
    for (size_t aK=0 ; aK<mVPts.size() ; aK++)
       StdOut() << mNumIm.at(aK) << mVPts.at(aK) << " ";
    StdOut() << std::endl;
}


    /*=======================================================*/
    /*                         cImage                        */
    /*=======================================================*/

cImage::cImage(int aNum) :
   mNum     (aNum),
   mSz      (3000,2000),
   mGenPts  (cBox2dr(cPt2dr(0,0),ToR(mSz)),0.1)
{
}


    /*=======================================================*/
    /*                      cSimulHom                        */
    /*=======================================================*/


bool cSimulHom::HasHom(const std::string & aNameIm1,const std::string & aNameIm2) const
{
    return mMapHom.find(tSS(aNameIm1,aNameIm2)) != mMapHom.end();
}

void cSimulHom::GetHom(cSetHomogCpleIm & aSet,const std::string & aNameIm1,const std::string & aNameIm2) const 
{
   MMVII_INTERNAL_ASSERT_bench(HasHom(aNameIm1,aNameIm2),"Cannot get hom in cSimulHom");
   aSet = mMapHom.find(tSS(aNameIm1,aNameIm2))->second;
}

cSimulHom::~cSimulHom()
{
    DeleteAllAndClear(mVIm);
}

static constexpr int NbDigit = 4; // 4 digit to be sorted lexicographically

cSimulHom::cSimulHom(int aNbImage,int aNbPts,int aMaxCard,bool Debug) :
    mNbImage  (aNbImage),
    mNbPts    (aNbPts),
    mMaxCard  (aMaxCard),
    mDebug    (Debug),
    mKPts     (0)
{
    if (mDebug) StdOut() << "Ddddddddddddddddddddddddddddddeeeeeeeeeeeeebugggggggggggggggg cSimulHom::cSimulHom" << std::endl;
    for (int aK=0 ; aK<aNbImage ; aK++)
    {
         mVIm.push_back(new cImage(aK));
	 mVNames.push_back(ToStr(aK,NbDigit));  
    }
}


const std::vector<std::string> & cSimulHom::VNames() const {return mVNames;}


cMultiplePt cSimulHom::GenMulTieP()
{
    cMultiplePt aRes;

    // 0- Generate multiplicity
    int aMult = 1 + round_up((mMaxCard-1)*std::pow(RandUnif_0_1(),2));
    aMult = std::max(2,std::min(aMult,mMaxCard));

    // 1- Compute  set of images
    aRes.mNumIm = RandSet(aMult,mNbImage);
    std::sort(aRes.mNumIm.begin(),aRes.mNumIm.end());

    // 2- Compute  pts
    for (const auto & aNI :  aRes.mNumIm )
    {
        aRes.mVPts.push_back(mVIm.at(aNI)->mGenPts.GetNewPoint());
    }

    if (mDebug) 
    {
       StdOut() << "NumIms=" << aRes.mNumIm  << std::endl;
    }

    return aRes;
}

//  data for storing an edge "S1,S2,IsRedundant"
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

    // begin by an edge on the two first sum "randommly" selected
    aVEdgesInit.push_back(tEdge({anOrder.at(0),anOrder.at(1),false}));
    for (int aK=2; aK<aMult ; aK++)
    {
        int aS1 = anOrder.at(aK);  // next som un reached
	int aS2 = anOrder.at(RandUnif_N(aK));  //random som among one already reached
	// randomize the order inside the nez pair
	if (HeadOrTail())
           std::swap(aS1,aS2);
        aVEdgesInit.push_back(tEdge({aS1,aS2,false}));
    }
    
    // 2  Add random edges to make it more multiple, 
    // in a first time dont bother add same time same edge

    int aNbAdd = round_ni( ((aMult-1) * (aMult) - (aMult-1)) * std::pow(RandUnif_0_1(),2)) ;

    for (int aK= 0 ; aK<aNbAdd ; aK++)
    {
         int aS1 = RandUnif_N(aMult);
         int aS2 = RandUnif_N(aMult);
	 while (aS1==aS2)  // avoid add  same sum
             aS2 = RandUnif_N(aMult);
        aVEdgesInit.push_back(tEdge({aS1,aS2,true}));
    }

    // 3  now filter duplicatas, just to be more realistic,
    // in fact if we dont eliminate duplicatas, the bench still work

    std::sort(aVEdgesInit.begin(),aVEdgesInit.end());

    std::vector<tEdge > aVEdges;
    aVEdges.push_back(aVEdgesInit.at(0));

    for (size_t aK=1 ; aK<aVEdgesInit.size() ; aK++)
    {
        const tEdge & aPrec = aVEdgesInit.at(aK-1);
        const tEdge & aCur  = aVEdgesInit.at(aK);

	// we maintain non redundant edge to be sure to have initial tree
	if ( (!IsRedund(aCur)) || (S1(aPrec)!=S1(aCur)) || (S2(aPrec)!=S2(aCur)) )
	{
            aVEdges.push_back(aCur);
	}
    }

    
    // 4 generate the tie points itself

    for (const auto & anE : aVEdges)
    {
         cPt2dr aP1 =  aMTP.mVPts.at(S1(anE));
	 int aI1    =  aMTP.mNumIm.at(S1(anE));

         cPt2dr aP2 = aMTP.mVPts.at(S2(anE));
	 int aI2    =  aMTP.mNumIm.at(S2(anE));

	 //  Iw we are in error mode, generate random some incoherence
	 if (IsRedund(anE) && WithError && (HeadOrTail()))
	 {
             aMTP.mGotEr = true;
             if (HeadOrTail())
                aP1 = mVIm.at(aI1)->mGenPts.GetNewPoint();
	     else
                aP2 = mVIm.at(aI2)->mGenPts.GetNewPoint();
	 }

	 mMapHom[tSS(ToStr(aI1,NbDigit),ToStr(aI2,NbDigit))].Add(cHomogCpleIm(aP1,aP2));

	 if (mDebug)
            StdOut() <<  "-EeeE=" << aI1 << " " << aI2 << std::endl;
    }
}

void OneBench(int aNbImage,int aNbPts,int aMaxCard,bool DoIt)
{
    // StdOut() << "NbImage= " << aNbImage << std::endl;
    cSimulHom aSimH(aNbImage,aNbPts,aMaxCard,false);
    cComputeMergeMulTieP aSetMTP1(aSimH.VNames());

    int aCptErr = 0;
    for (int aKPts=0 ; aKPts<aNbPts ; aKPts++)
    {
        cMultiplePt aMTP = aSimH.GenMulTieP();

	aSimH.GenEdges(aMTP,true);
	if (! aMTP.mGotEr)
	{
	   aSetMTP1.AddPMul(aMTP.mNumIm,aMTP.mVPts);
	}
	else
	{
           aCptErr++;
	}
    }

    aSetMTP1.Shrink();

    if (DoIt)
    {
        // cMemoryEffToMultiplePoint aToMP(aSimH,aSimH.VNames(),aSetMTP2);
        //cMemoryEffToMultiplePoint aToMP(aSimH,aSetMTP2.VNames(),aSetMTP2);
	//StdOut() << "cMemoryEffToMultiplePointcMemoryEffToMultiplePoint " << std::endl;

        cComputeMergeMulTieP aSetMTP2(aSimH.VNames(),&aSimH);
        aSetMTP2.Shrink();
	aSetMTP1.TestEq(aSetMTP2);

        cInterfParsePMulGCP *  aIter =  cInterfParsePMulGCP::Alloc_CMTP(aSetMTP2,false);
	for (const auto & aPair :  aSetMTP2.Pts())
	{
             const auto & aVV = aSetMTP2.PUnMixed(Config(aPair),false);

	     for (const auto & aVal : aVV)
	     {
                 MMVII_INTERNAL_ASSERT_bench(!aIter->End(),"Error on iter") ;
                 MMVII_INTERNAL_ASSERT_bench(aVal.mVPIm==aIter->CurP().mVPIm,"Error on iter") ;
                 aIter->Incr();
	     }
	}
        MMVII_INTERNAL_ASSERT_bench(aIter->End(),"Error on iter") ;
	delete aIter;
    }

    // getchar();
}

void Bench()
{

     for (int aK=0 ; aK<100 ; aK++)
     {
          OneBench(4,2,4,true);
     }
     for (int aK=0 ; aK<100 ; aK++)
     {
          OneBench(10,RandInInterval(3,50),4,true); //(aK==16));
     }

    for (int aK=0 ; aK<100 ; aK++)
    {
        int aNbIm = RandInInterval(3,50);
        OneBench(aNbIm,40,std::min(aNbIm,6),true);
    }

    for (int aK=0 ; aK<100 ; aK++)
    {
        int aNbIm = RandInInterval(3,50);
        int aMult = std::min(aNbIm,round_ni(RandInInterval(3,20)));
        OneBench(aNbIm,RandInInterval(3,200),aMult,true);
    }
}

};

void PiegeACon()
{
       int i0=0; int i1=1;
       std::vector<int *> aVPtrA{&i0,&i1};
       int * & aPI0 = aVPtrA.at(0);
       int * & aPI1 = aVPtrA.at(1);

       aVPtrA.at(1) = aPI0;

       // Guess what is printed  ...
       std::cout  << "Zero=" << *aPI0  << " One=" << *aPI1 << std::endl;
}



void Bench_ToHomMult(cParamExeBench & aParam)
{
   if (! aParam.NewBench("HomMult")) return;

   NS_BenchMergeHomol::Bench();

   aParam.EndBench();
}



}; // MMVII






