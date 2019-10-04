#include "include/V1VII.h"


#include "src/uti_image/NewRechPH/cParamNewRechPH.h"
#include "../CalcDescriptPCar/AimeTieP.h"

#include "include/im_tpl/cPtOfCorrel.h"
#include "include/algo_geom/qdt.h"
#include "include/algo_geom/qdt_implem.h"
#include "include/ext_stl/heap.h"


namespace MMVII
{




//=============  tNameRel ====================

void TestTimeV1V2()
{
    for (int aK= 1 ; aK<1000000 ; aK++)
    {
         int aSom=0;
         ElTimer aChronoV1;
         double aT1 = cMMVII_Appli::CurrentAppli().SecFromT0();
         for (int aI=0 ; aI<10000; aI++)
         {
              for (int aJ=0 ; aJ<10000; aJ++)
              {
                  aSom += 1/(1+aI+aJ);
              }
         }
         if (aSom==-1)
            return;
         double aT2 = cMMVII_Appli::CurrentAppli().SecFromT0();

         double aDV1 = aChronoV1.uval();
         double aDV2 = aT2-aT1;

         StdOut()  << "Ratio " << aDV1 / aDV2  << " TimeV1: " << aDV1 << "\n";
    }
}

//=============  tNameRel ====================

tNameRel  MMV1InitRel(const std::string & aName)
{
   tNameRel aRes;
   cSauvegardeNamedRel aSNR = StdGetFromPCP(aName,SauvegardeNamedRel);
   for (const auto & aCpl : aSNR.Cple())
   {
       aRes.Add(tNamePair(aCpl.N1(),aCpl.N2()));
   }

   return aRes;
}

/// Write a rel in MMV1 format
template<> void  MMv1_SaveInFile(const tNameRel & aSet,const std::string & aName)
{
   std::vector<const tNamePair *> aV;
   aSet.PutInVect(aV,true);

   cSauvegardeNamedRel aSNR;
   for (const auto & aPair : aV)
   {
      aSNR.Cple().push_back(cCpleString(aPair->V1(),aPair->V2()));
   }
   MakeFileXML(aSNR,aName);
}

//=============  tNameSet ====================

/// Read MMV1 Set
tNameSet  MMV1InitSet(const std::string & aName)
{
   tNameSet aRes ;
   cListOfName aLON = StdGetFromPCP(aName,ListOfName);
   for (const auto & el : aLON.Name())
       aRes.Add(el);
   return aRes;
}

/// Write a set in MMV1 format
template<> void  MMv1_SaveInFile(const tNameSet & aSet,const std::string & aName)
{
    std::vector<const std::string *> aV;
    aSet.PutInVect(aV,true);

    cListOfName aLON;
    for (const auto & el : aV)
        aLON.Name().push_back(*el);
    MakeFileXML(aLON,aName);
}

/********************************************************/



   // This 3 thresholds are used for size of neighbooring window for selecting points
   // they take into account the local scale (scale in the octave)

#define  VAR_RHO  6.0      // Distance used for computing variance
#define AC_RHO  5.0        // Distance used for circle of auto correl
#define AC_SZW  3          // Size of auto correl

   // This 3 thresholds are used to compute initial scores

// #define  POW_CORREL 2.0  // privilegiate low self correlation
// #define  POW_VAR    1.0  // privilegiate high variance
// #define  POW_SCALE  1.0  // privilegiate high scale

    //  Thresholds for spatial filtering itself

// #define  STD_DIST_PC  35.0 //   Standard targeted distance between points after spatial filtering
// #define  MUL_RAY_PC   3.0  //   Multipler to compute zone of influence of a selected points
// #define  PROP_NO_SF   0.2  //   Proportion that dont use spatial filtering


/// Class implementing services promized by cInterf_ExportAimeTiep

/**  This class use MMV1 libray to implement service described in cInterf_ExportAimeTiep
*/
template <class Type> class cImplem_ExportAimeTiep : public cInterf_ExportAimeTiep<Type>
{
     public :
         cImplem_ExportAimeTiep
         (
               const cPt2di& aSzIm0,
               bool IsMin,int ATypePt,const std::string & aName,bool ForInspect,
               const cGP_Params & aParam
         );
         virtual ~cImplem_ExportAimeTiep();

         void AddAimeTieP(const cProtoAimeTieP & aPATP ) override;
         void Export(const std::string &) override;
         void SetCurImages(cIm2D<Type>,cIm2D<Type>,double aScaleInO) override;

         void FiltrageSpatialPts();

     private :
          typedef typename tElemNumTrait<Type>::tBase tBase;
          typedef Im2D<Type,tBase>           tImV1;
          typedef TIm2D<Type,tBase>          tTImV1;
          typedef cCutAutoCorrelDir<tTImV1>  tCACD;
          typedef std::unique_ptr<cFastCriterCompute> tFCC;
 
          cPt2di          mSzIm0; ///< Sz of Image at full resolution
          cXml2007SetPtOneType  mPtsXml; ///< Result
          cIm2D<Type>     mIm0V2;  ///< Image "init",  MMVII version
          tImV1           mIm0V1;  ///<  Image "init", MMV1 Version
          tTImV1          mTIm0V1; ///< T Image Init V1
          cIm2D<Type>     mImStdV2;  ///<  Imagge Std eq Lapl, Corner ...
          tImV1           mImStdV1;  ///<  Image Std V2
          tTImV1          mTImStdV1;  /// T Image Std V2
          std::shared_ptr<tCACD> mCACD;  ///< For computing auto correl
          tFCC                   mFCC;  ///< For computing fast
          bool                   mForInspect;  ///< When inspect, save all points to inspect rejection
          cTplBoxOfPts<tREAL8,2> mBox;
          cGP_Params             mParam;  ///< Parameters 
          cFilterPCar &          mFP;  ///< Parameters 
};
/* ================================= */
/*    cInterf_ExportAimeTiep         */
/* ================================= */

template <class Type> cInterf_ExportAimeTiep<Type>::~cInterf_ExportAimeTiep()
{
}

template <class Type> cInterf_ExportAimeTiep<Type> * cInterf_ExportAimeTiep<Type>::Alloc(const cPt2di& aSzIm0,bool IsMin,int ATypePt,const std::string & aName,bool ForInspect,const cGP_Params & aParam )
{
    return new cImplem_ExportAimeTiep<Type>(aSzIm0,IsMin,ATypePt,aName,ForInspect,aParam);
}



/* ================================= */
/*    cImplem_ExportAimeTiep         */
/* ================================= */

template <class Type> 
   cImplem_ExportAimeTiep<Type>::cImplem_ExportAimeTiep
   (
        const cPt2di & aSzIm0,
        bool IsMin,
        int ATypePt,
        const std::string & aNameType,
        bool ForInspect,
        const cGP_Params & aParam
    ) :
    mSzIm0    (aSzIm0),
    mIm0V2    (cPt2di(1,1)),
    mIm0V1    (1,1),
    mTIm0V1   (mIm0V1),
    mImStdV2  (cPt2di(1,1)),
    mImStdV1  (1,1),
    mTImStdV1 (mImStdV1),
    mCACD  (nullptr),
    mFCC   (cFastCriterCompute::Circle(3.0)),
    mForInspect (ForInspect),
    mParam      (aParam),
    mFP         (mParam.mFPC)
{
    mPtsXml.IsMin() = IsMin;
    mPtsXml.TypePt() = IsMin;
    mPtsXml.NameTypePt() = aNameType;
    
}
template <class Type> cImplem_ExportAimeTiep<Type>::~cImplem_ExportAimeTiep()
{
}

template <class Type> void cImplem_ExportAimeTiep<Type>::AddAimeTieP(const cProtoAimeTieP & aPATP)
{
    static int anIdent=0;  // for debugging and computing identifier
    anIdent++;

    Pt2di  aV1PIm = round_ni(ToMMV1(aPATP.Pt()) / double(1<<aPATP.NumOct()));
    cPt2di aV2PIm = ToMMVII(aV1PIm);

    bool  aAutoCor = mCACD->AutoCorrel(aV1PIm,mFP.AC_CutInt(),mFP.AC_CutReal(),mFP.AC_Threshold());
    //     bool  aAutoCor = mCACD->AutoCorrel(aV1PIm,AC_CutInt,AC_CutReal,AC_Threshold);
    
    cXml2007Pt aPXml;

    aPXml.Pt() = ToMMV1(aPATP.Pt());
    aPXml.Id() = anIdent;
    aPXml.NumOct() = aPATP.NumOct();
    aPXml.NumIm() = aPATP.NumIm();
    aPXml.ScaleInO() = aPATP.ScaleInO();
    aPXml.ScaleAbs() = aPATP.ScaleAbs();
    aPXml.OKAc() = ! aAutoCor;
    aPXml.AutoCor() = mCACD->mCorOut;
    aPXml.NumChAC() = mCACD->mNumOut;

    if (aPXml.OKAc() || mForInspect)
    {
        Pt2dr aFQ =  FastQuality(mTImStdV1,aV1PIm,*mFCC,! mPtsXml.IsMin() ,Pt2dr(0.75,0.85));
        aPXml.FastStd() = aFQ.x;
        aPXml.FastConx() = aFQ.y;
        // Compute variance weighted by a pseudo Gauss
        aPXml.Var() = CubGaussWeightStandardDev(mIm0V2.DIm(),aV2PIm,aPATP.ScaleInO()*VAR_RHO);
        if (aPXml.Var()<=0)  // Case degenerate
            aPXml.OKAc() = false;
        else
        {
            aPXml.Score()  =      pow(1-aPXml.AutoCor()  , mFP.PowAC())
                               *  pow(aPXml.Var()        , mFP.PowVar())
                               *  pow(aPATP.ScaleAbs()   , mFP.PowScale())
                             ;
            aPXml.ScoreRel()  = aPXml.Score(); // Initially no point selected, relative=absolute
        }

        if (aPXml.OKAc() || mForInspect)
        {
            aPXml.SFSelected() = false;
            mBox.Add(aPATP.Pt());
            mPtsXml.Pts().push_back(aPXml);
        }
    }
}

template <class Type> void cImplem_ExportAimeTiep<Type>::Export(const std::string & aName)
{
     MakeFileXML(mPtsXml,aName);
}

template <class Type> void cImplem_ExportAimeTiep<Type>::SetCurImages(cIm2D<Type> anIm0,cIm2D<Type> anImStd,double aScaleInO) 
{
   mIm0V2 = anIm0;
   mIm0V1 = cMMV1_Conv<Type>::ImToMMV1(mIm0V2.DIm());
   mTIm0V1 =  tTImV1(mIm0V1);

   int aSzW = round_ni(AC_SZW*aScaleInO);
   double aFact  = aSzW / double(AC_SZW);
   double aRho = AC_RHO * aFact;


   mCACD = std::shared_ptr<tCACD>(new tCACD(mTIm0V1,Pt2di(0,0),aRho,aSzW));

   mImStdV2 = anImStd;
   mImStdV1 = cMMV1_Conv<Type>::ImToMMV1(mImStdV2.DIm());
   mTImStdV1 =  tTImV1(mImStdV1);
}

     // ================= Fitlrage spatial , point "bons" et bien repartis ===============

         //  ------ Qt stuff ---------------
typedef  cXml2007Pt * tP2007Ptr;
class cFuncPtOfXml2007
{   // argument du qauad tri
    // comment à partir un objet, je recuper sa pt2D
      public :
         Pt2dr operator () (tP2007Ptr  aXP) {return Pt2dr(aXP->Pt());}
};
typedef ElQT<tP2007Ptr,Pt2dr,cFuncPtOfXml2007> tQtXml2007;

         //  ------ Heap stuff ---------------
class cAimeFS_HeapIndex
{
     public :
        static void SetIndex(tP2007Ptr   aXP,int i)
        {
                aXP->HeapIndexe() = i;
        }
        static int  Index(tP2007Ptr aXP)
        {
             return aXP->HeapIndexe();
        }
};

class cAimeFS_HeapCmp
{
    public :

        bool operator () (tP2007Ptr    aXP1,tP2007Ptr  aXP2)
        {
              return aXP1->ScoreRel() > aXP2->ScoreRel();   // compare score correl global
        }
        // est ce que objet 1 est meuilleur que 2
};
typedef ElHeap<tP2007Ptr,cAimeFS_HeapCmp,cAimeFS_HeapIndex> tHeapXml2007;


template <class Type> void cImplem_ExportAimeTiep<Type>::FiltrageSpatialPts()
{
     if (mPtsXml.Pts().empty())  // may be degenerated (bounding box ...)
     {
        return;
     }
     cBox2dr aB = mBox.CurBox().Dilate(10);

     // Quod tree for spacial indexation
     cFuncPtOfXml2007 aFctrPt;
     tQtXml2007  aQt(aFctrPt,ToMMV1(aB),10,20.0);

     // Heap for handling priority
     cAimeFS_HeapCmp aFctrCmp;
     tHeapXml2007  aHeap(aFctrCmp);

     // Put everyting in heap and quod tree, put only validated (they may remain when inspect mode)
     for (auto & aP : mPtsXml.Pts())
     {
         if (aP.OKAc() )
         {
            aQt.insert(&aP);
            aHeap.push(&aP);
         }
     }
     
     //  do the spatial filtering
     // double aMulRay = mFPC.mPSF.at(1);
     // double aPropNoSF = mFPC.mPSF.at(2);
     double aDsf = mFP.DistSF();
     int NbToGet= (mSzIm0.x()*mSzIm0.y()) / Square(aDsf);
     while (NbToGet>0)
     {
        double aDistInfl = aDsf * mFP.MulDistSF();
        double aProp =  mFP.PropNoSF();
        tP2007Ptr aNewP = nullptr;
        if (aHeap.pop(aNewP)) // if get next best pts 
        {
            aQt.remove(aNewP);  // supress from Qt
            aNewP->SFSelected() = true;  // memorize it is selected
            std::set<tP2007Ptr> aSet;
            aQt.RVoisins(aSet,aNewP->Pt(),aDistInfl);  // Extract neighboors

            for (const auto & aVois : aSet)
            {
                if (! aVois->SFSelected())
                {
                    double aD = euclid(aNewP->Pt(),aVois->Pt()); // Distance to new selected
                    double aRatio = aD/aDistInfl;  // put as ratio to max
                    aRatio = aProp + aRatio *(1-aProp); // take into account non spatial part
                    double aNewSc = aVois->Score() * aRatio;  // comput new score 
                    if (aNewSc < aVois->ScoreRel())  // if new score is lower
                    {
                        aVois->ScoreRel() = aNewSc;  // update score
                        aHeap.MAJ(aVois);            // udpate position in heap
                    }
                }
            }
            NbToGet--; // One less to get
        }
        else // If heap empty then end
        {
           NbToGet =0;
        }
     }

     if (! mForInspect)
     {
         erase_if
         (
             mPtsXml.Pts(),
	     [] (const cXml2007Pt & aXP) { return  (!aXP.OKAc() ) || (! aXP.SFSelected());}
         );
     }
}


// ============  INSTANTIATION ======================

template class cInterf_ExportAimeTiep<tREAL4>;
template class cInterf_ExportAimeTiep<tINT2>;
template class cImplem_ExportAimeTiep<tREAL4>;
template class cImplem_ExportAimeTiep<tINT2>;


};
