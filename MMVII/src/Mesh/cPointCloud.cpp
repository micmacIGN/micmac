#include "MMVII_PointCloud.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Geom2D.h"
#include "MMVII_util_tpl.h"

#include "MMVII_2Include_Tiling.h"

namespace MMVII
{

cPointCloud::cPointCloud(bool isM8) :
   mOffset     (0,0,0),
   mMode8      (isM8),
   mMulDegVis  (-1),
   mDensity    (-1),
   mBox3dOfPts (),
   mLeavesUnit (-1)
{
}

tREAL8   cPointCloud::CurBasicDensity() const 
{
	  return NbPts() / Box2d().NbElem() ;
}
tREAL8   cPointCloud::CurStdDensity() const 
{
    return (mDensity>0) ? mDensity : CurBasicDensity();
}

//====================================================================================

class cTil2D_PC
{
    public :
        typedef cTiling<cTil2D_PC> tTiling;      
        static constexpr int TheDim = 2;          // Pre-requite for instantite cTilingIndex
        typedef cPt2dr             tPrimGeom;     // Pre-requite for instantite cTilingIndex
        typedef const  cPointCloud *  tArgPG; // Pre-requite for instantite cTilingIndex

        /**  Pre-requite for instantite cTilingIndex : indicate how we extract geometric primitive from one object */

        tPrimGeom  GetPrimGeom(tArgPG aPtrPC) const {return Proj(ToR(aPtrPC->KthPt(mInd)));}

        cTil2D_PC(size_t anInd) : mInd(anInd) {}
        size_t  Ind() const {return mInd;}

        static tTiling *  ComputeTiling(const cPointCloud & aPC,int aNbByCase=20);

    private :
        size_t  mInd;
};



cTiling<cTil2D_PC> *  cTil2D_PC::ComputeTiling(const cPointCloud & aPC,int aNbByCase)
{
     cBox2dr aBox = aPC.Box2d();
     int aNbCase = aPC.NbPts() / aNbByCase;

     tTiling *  aTil = new tTiling (aBox,true,aNbCase,&aPC);
     for (size_t aKPt=0 ; aKPt<aPC.NbPts() ; aKPt++)
         aTil->Add(cTil2D_PC(aKPt));

     return aTil;
}

tREAL8  ComputeDensity
     (
         const cPointCloud & aPC,
         cTiling<cTil2D_PC> *  aTilInit=nullptr
     )
{
    cTiling<cTil2D_PC> * aTil = (aTilInit == nullptr) ? cTil2D_PC::ComputeTiling(aPC,10) : aTilInit;

    tREAL8 aDist = std::sqrt(5 / (M_PI * aPC.CurStdDensity()));
    cWeightAv<tREAL8>  aWSz;

    for (size_t aKPt=0 ; aKPt< aPC.NbPts() ; aKPt++)
    {
        int aNb = aTil->GetObjAtDist(Proj(aPC.KthPt(aKPt)),aDist).size();
	aWSz.Add(1.0,aNb-1);
    }
    tREAL8 aNbAv =  aWSz.Average();
    // Surf * Density = NB
    tREAL8 aDensity =  aNbAv / (M_PI * Square(aDist));


    if (aTilInit == nullptr) 
       delete aTil;

    return aDensity;
}

tREAL8   cPointCloud::ComputeCurFineDensity() const
{
    return ComputeDensity(*this);
}



// --------------------- Colours access ------------------------------------------
void cPointCloud::SetNbColours(int aNbC)
{
    mColors.resize(aNbC);
    for (auto & aCol :mColors)
        aCol.resize(NbPts());
}
int  cPointCloud::GetNbColours() const {return mColors.size();}
std::vector<tU_INT1> & cPointCloud::GrayColors()
{
    MMVII_INTERNAL_ASSERT_always(GetNbColours()==1,"Bad Nb channel in cPointCloud::GrayColors");
    return mColors.at(0);
}
// --------------------- Colours access ------------------------------------------

void cPointCloud::SetMulDegVis(tREAL8 aMulDegVis) 
{
    if (mDegVis.empty())
    {
        mDegVis.resize(NbPts(),0);
        mMulDegVis = aMulDegVis;
    }
    else
    {
        MMVII_INTERNAL_ASSERT_always(aMulDegVis==mMulDegVis,"cPointCloud::SetMulDegVis chg MulDegVis");
    }
}
void cPointCloud::SetDegVis(int aK,tREAL8 aDeg) { mDegVis.at(aK) = round_ni(aDeg*mMulDegVis); }
tREAL8 cPointCloud::GetDegVis(int aK) const {return mDegVis.at(aK) / mMulDegVis;}

bool   cPointCloud::DegVisIsInit() const {return ! mDegVis.empty();}



// --------------------- Leaves  ------------------------------------------

void cPointCloud::SetLeavesUnit(tREAL8 aPropAvgD,bool SVP)
{
    if (! mSzLeaves.empty())
    {
        MMVII_INTERNAL_ASSERT_always(SVP,"cPointCloud::SetLeavesUnit");
        return;
    }
    mLeavesUnit = aPropAvgD / std::sqrt(mDensity);
    mSzLeaves.resize(NbPts(),0);
}
void cPointCloud::SetSzLeaves(int aK,tREAL8 aSz) 
{ 
     mSzLeaves.at(aK) = std::min(255,round_ni(aSz/mLeavesUnit));
}


tREAL8 cPointCloud::GetSzLeave(int aK) const {return mSzLeaves.at(aK) * mLeavesUnit;}

tU_INT1 cPointCloud::GetIntSzLeave(int aK) const {return mSzLeaves.at(aK);}
tREAL8  cPointCloud::ConvertInt2SzLeave(int aInd) const 
{
  
   MMVII_INTERNAL_ASSERT_User_UndefE(mLeavesUnit>=0,"mLeavesUnit not itialized");
   return mLeavesUnit * aInd;
}

bool  cPointCloud::LeavesIsInit() const 
{
    return ! mSzLeaves.empty();
}





cBox3dr  cPointCloud::Box3d()   const  {return mBox3dOfPts.CurBox();}
cBox2dr  cPointCloud::Box2d()   const 
{    
    cBox3dr aB3 = Box3d();
    return cBox2dr(Proj(aB3.P0()),Proj(aB3.P1()));
}

cPt3dr   cPointCloud::Centroid() const
{
   return mSumPt / tREAL8(NbPts());
}


void cPointCloud::ToPly(const std::string & aName,bool WithOffset) const
{
    cMMVII_Ofs anOfs(aName,eFileModeOut::CreateText);

    size_t aNbP = NbPts();
    size_t aNbC = mColors.size(); 
    bool  WithVis = (mMulDegVis>0);
    if (WithVis)
    {
        MMVII_INTERNAL_ASSERT_always(aNbC==0,"Colors & DegVis ...");
        aNbC=1;
    }
    // with use 8 byte if initially 8 byte, or if we use the offset that creat big coord
    bool  aMode8 =  mMode8 || WithOffset;

    std::string aSpecCoord = aMode8 ? "float64" : "float32";
    anOfs.Ofs() <<  "ply\n";
    anOfs.Ofs() <<  "format ascii 1.0\n";
    anOfs.Ofs() <<  "comment Generated by MMVVI\n";
    anOfs.Ofs() <<  "element vertex " << aNbP << "\n";
    anOfs.Ofs() <<  "property " <<  aSpecCoord  <<" x\n";
    anOfs.Ofs() <<  "property " <<  aSpecCoord  <<" y\n";
    anOfs.Ofs() <<  "property " <<  aSpecCoord  <<" z\n";
    if (aNbC) 
    {
        anOfs.Ofs() <<  "property uchar red\n"; 
        anOfs.Ofs() <<  "property uchar green\n"; 
        anOfs.Ofs() <<  "property uchar blue\n"; 
    }
    anOfs.Ofs() <<  "end_header\n";


    for (size_t aKPt=0 ; aKPt<aNbP ; aKPt++)
    {
        if (aMode8)
        {
            cPt3dr aPt = WithOffset ? KthPt(aKPt) :  KthPtWoOffs(aKPt);
            anOfs.Ofs() <<  aPt.x() << " " << aPt.y() << " " << aPt.z();
        }
        else
        {
            const cPt3df&  aPt = mPtsF.at(aKPt);
            anOfs.Ofs() <<  aPt.x() << " " << aPt.y() << " " << aPt.z();
        }
        if (aNbC)
        {
           if (aNbC==1)
           {
              size_t aC =  WithVis ? round_ni(GetDegVis(aKPt) *255)  : mColors.at(0).at(aKPt);
              anOfs.Ofs() << " " << aC << " " << aC << " " << aC;
           }
           else if (aNbC==3)
           {
               for (size_t aKC=0 ; aKC<aNbC ; aKC++)
                  anOfs.Ofs() << " " << (size_t)  mColors.at(aKC).at(aKPt);
           }
           else 
           {
               MMVII_INTERNAL_ERROR("Bad number of channel in ply generate : " + ToStr(aNbC));
           }
        }
        anOfs.Ofs() << "\n";
    }
}

void cPointCloud::AddPt(const cPt3dr& aPt0)
{

  mBox3dOfPts.Add(aPt0);
  mSumPt += aPt0;

  cPt3dr aPt = aPt0 - mOffset;
  
  if (mMode8)
     mPtsR.push_back(aPt);
  else
     mPtsF.push_back(cPt3df::FromPtR(aPt));
}

void cPointCloud::Clip(cPointCloud& aPC,const cBox2dr & aBox) const
{
    // MMVII_INTERNAL_ERROR("cPointCloud::Clip  2 Finalize");
    aPC = cPointCloud(mMode8);
    aPC.mDensity = mDensity;
    
    // aPC.mBox2d   = aBox;
    // aPC.mPtsR.clear();
    // aPC.mPtsF.clear();
    size_t aNbCol = mColors.size();
    aPC.mColors = std::vector<std::vector<tU_INT1>>(aNbCol);

    aPC.SetOffset(mOffset);
    // aPC.mOffset = mOffset;
    // aPC.mMode8 = mMode8;
    aPC.mMulDegVis = mMulDegVis;
    aPC.mLeavesUnit = mLeavesUnit;

    for (size_t aKPt=0 ; aKPt<NbPts() ; aKPt++)
    {
        cPt3dr aPt = KthPt(aKPt);
        if (aBox.Inside(Proj(aPt)))
        {
           aPC.AddPt(aPt);
           for (size_t aKC=0 ; aKC<aNbCol ; aKC++)
           {
               aPC.mColors.at(aKC).push_back(mColors.at(aKC).at(aKPt));
           }
           if (mMulDegVis>0)
              aPC.mDegVis.push_back(mDegVis.at(aKPt));
           if (mLeavesUnit>0)
              aPC.mSzLeaves.push_back(mSzLeaves.at(aKPt));
        }
    }
}

// template <class Type,const int Dim>  void  AddData(const  cAuxAr2007 & anAux,cTplBox<Type,Dim> & aBox) { aBox.AddData(anAux); }

void cPointCloud::SetOffset(const cPt3dr & anOffset)
{
    MMVII_INTERNAL_ASSERT_always(NbPts()==0,"cPointCloud::SetOffset not empty");
    mOffset = anOffset;
}

void cPointCloud::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("M8",anAux),mMode8);
    MMVII::AddData(cAuxAr2007("Params",anAux),mParams);
    MMVII::AddData(cAuxAr2007("Offset",anAux),mOffset);
    MMVII::AddData(cAuxAr2007("PtsR",anAux),mPtsR);
    MMVII::AddData(cAuxAr2007("PtsF",anAux),mPtsF);
    MMVII::AddData(cAuxAr2007("Colors",anAux),mColors);

    MMVII::AddData(cAuxAr2007("MulDegVis",anAux),mMulDegVis);

    MMVII::AddData(cAuxAr2007("DegVis",anAux),mDegVis);

    MMVII::AddData(cAuxAr2007("Density",anAux),mDensity);

    MMVII::AddData(cAuxAr2007("Box3d",anAux),mBox3dOfPts);
    MMVII::AddData(cAuxAr2007("SumPt",anAux),mSumPt);

    MMVII::AddData(cAuxAr2007("LeavesUnit",anAux),mLeavesUnit);
    MMVII::AddData(cAuxAr2007("LeavesSize",anAux),mSzLeaves);
}

void AddData(const  cAuxAr2007 & anAux,cPointCloud & aPC)
{
   aPC.AddData(anAux);
}

/*
cBox3dr  cPointCloud::Box() const
{
   cTplBoxOfPts<tREAL8,3> aTplB;
   for (size_t aK=0 ; aK<NbPts() ; aK++)
       aTplB.Add(KthPt(aK));
   return aTplB.CurBox();
}
*/

#if (0)

#endif
};

