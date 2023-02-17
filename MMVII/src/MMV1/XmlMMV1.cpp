#include "include/V1VII.h"

#include "MMVII_Mappings.h"
#include "MMVII_AimeTieP.h"


#include "src/uti_image/NewRechPH/cParamNewRechPH.h"
// #include "../CalcDescriptPCar/AimeTieP.h"

#include "im_tpl/cPtOfCorrel.h"
#include "algo_geom/qdt.h"
#include "algo_geom/qdt_implem.h"
#include "ext_stl/heap.h"


namespace MMVII
{

/* *********************************** */
/*                                     */	
/*       cMasq_MMV1asBoundeSet         */
/*                                     */	
/* *********************************** */


class cMasq_MMV1asBoundeSet : public cDataBoundedSet<tREAL8,3> 
{
    public :
	cMasq_MMV1asBoundeSet(const cBox3dr &,const std::string & aNameFile);
	 tREAL8 Insideness(const tPt &) const override;
	 ~cMasq_MMV1asBoundeSet();
     private :
	 cMasqBin3D * mV1Masq3D;

};

cMasq_MMV1asBoundeSet::cMasq_MMV1asBoundeSet(const cBox3dr & aBox,const std::string & aNameFile) :
    cDataBoundedSet<tREAL8,3>(aBox),
    mV1Masq3D    (cMasqBin3D::FromSaisieMasq3d(aNameFile))
{
}

cMasq_MMV1asBoundeSet::~cMasq_MMV1asBoundeSet()
{
    delete mV1Masq3D;
}

tREAL8 cMasq_MMV1asBoundeSet::Insideness(const tPt & aPt) const
{
	return mV1Masq3D->IsInMasq(ToMMV1(aPt)) ? 1 : -1.0;
}

cDataBoundedSet<tREAL8,3> *  MMV1_Masq(const cBox3dr &aBox,const std::string & aNameFile)
{
    return new cMasq_MMV1asBoundeSet(aBox,aNameFile);
}



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

#define AFEXTR_RADIUS  2.0  // Ray of extremum refinement 
#define AFEXTR_DIST_INST  1.5  // Distance of instability
#define AFEXTR_DIST_CONV  0.05  // Ray of extremum refinement 
#define AFEXTR_NB_ITER    2  // Ray of extremum refinement 

   // This 3 thresholds are used to compute initial scores


    //  Thresholds for spatial filtering itself



/// Class implementing services promized by cInterf_ExportAimeTiep

/**  This class use MMV1 libray to implement service described in cInterf_ExportAimeTiep
*/
template <class Type> class cImplem_ExportAimeTiep : public cInterf_ExportAimeTiep<Type>
{
     public :
         cImplem_ExportAimeTiep
         (
               const cPt2di& aSzIm0,
               bool IsMax,eTyPyrTieP ATypePt,const std::string & aName,bool ForInspect,
               const cGP_Params & aParam
         );
         virtual ~cImplem_ExportAimeTiep();

         void AddAimeTieP(cProtoAimeTieP<Type>  aPATP ) override;
         void Export(const std::string &,bool SaveV1) override;
         void FiltrageSpatialPts() override;

          std::string NameExport(const std::string & aName,eModeOutPCar aMode);
     private :
          typedef typename tElemNumTrait<Type>::tBase tBase;
          typedef Im2D<Type,tBase>           tImV1;
          typedef TIm2D<Type,tBase>          tTImV1;
          typedef cCutAutoCorrelDir<tTImV1>  tCACD;
          typedef std::unique_ptr<cFastCriterCompute> tFCC;
 
          cPt2di          mSzIm0; ///< Sz of Image at full resolution
          cXml2007SetPtOneType  mMMV1_XmlPts; ///< Result
          std::vector<cProtoAimeTieP<Type> > mVecProtoPts;
          // std::vmector<cAimePCar >            mVecAPC;
          cSetAimePCAR       mSetAPC;
          // cIm2D<Type>     mImStdV2;  ///<  Imagge Std eq Lapl, Corner ...
          // tImV1           mImStdV1;  ///<  Image Std V2
          // tTImV1          mTImStdV1;  /// T Image Std V2
          // tFCC                   mFCC;  ///< For computing fast
          bool                   mForInspect;  ///< When inspect, save all points to inspect rejection
          cTplBoxOfPts<tREAL8,2> mBox;
          cGP_Params             mParam;  ///< Parameters 
          cFilterPCar &          mFPC;  ///< Parameters 
};
/* ================================= */
/*    cInterf_ExportAimeTiep         */
/* ================================= */

template <class Type> cInterf_ExportAimeTiep<Type>::~cInterf_ExportAimeTiep()
{
}

template <class Type> cInterf_ExportAimeTiep<Type> * cInterf_ExportAimeTiep<Type>::Alloc(const cPt2di& aSzIm0,bool IsMax,eTyPyrTieP ATypePt,const std::string & aName,bool ForInspect,const cGP_Params & aParam )
{
    return new cImplem_ExportAimeTiep<Type>(aSzIm0,IsMax,ATypePt,aName,ForInspect,aParam);
}



/* ================================= */
/*    cImplem_ExportAimeTiep         */
/* ================================= */

template <class Type> 
   cImplem_ExportAimeTiep<Type>::cImplem_ExportAimeTiep
   (
        const cPt2di & aSzIm0,
        bool IsMax,
        eTyPyrTieP ATypePt,
        const std::string & aNameType,
        bool ForInspect,
        const cGP_Params & aParam
    ) :
    mSzIm0    (aSzIm0),
    // mImStdV2  (cPt2di(1,1)),
    // mImStdV1  (1,1),
    // mTImStdV1 (mImStdV1),
    // mFCC   (cFastCriterCompute::Circle(3.0)),
    mSetAPC     (ATypePt,IsMax),
    mForInspect (ForInspect),
    mParam      (aParam),
    mFPC        (mParam.mFPC)
{
    mMMV1_XmlPts.IsMax() =  IsMax;
    mMMV1_XmlPts.TypePt() = int(ATypePt);
    mMMV1_XmlPts.NameTypePt() = aNameType;
    
    if (mFPC.LPS_CensusMode())
    {
         mSetAPC.Census() = true;
         mSetAPC.Ampl2N()   =  cSetAimePCAR::TheCensusMult;
    }
    else
    {
         mSetAPC.Census() = false;
         mSetAPC.Ampl2N()   = mFPC.LPS_Mult();
    }
}
template <class Type> cImplem_ExportAimeTiep<Type>::~cImplem_ExportAimeTiep()
{
}

template <class Type> cXml2007Pt ToXmlMMV1(const cProtoAimeTieP<Type>&  aPATP,cAimePCar * aAPCPtr)
{
    cXml2007Pt aPXml;

    aPXml.PtInit() = ToMMV1(aPATP.mPFileInit);
    aPXml.PtAff() = ToMMV1(aPATP.mPFileRefined);
    // PtAff
    aPXml.Id() = aPATP.mId;
    aPXml.NumOct() = aPATP.NumOct();
    aPXml.NumIm() = aPATP.NumIm();
    aPXml.ScaleInO() = aPATP.ScaleInO();
    aPXml.ScaleAbs() = aPATP.ScaleAbs();
    aPXml.OKAc() = aPATP.mOkAutoCor;
    aPXml.SFSelected() = aPATP.mSFSelected;
    aPXml.Stable() = aPATP.mStable;
    aPXml.AutoCor() = aPATP.mAutoCor;
    aPXml.NumChAC() = aPATP.mNumOutAutoC;
    // aPXml.FastStd() =  aPATP.mCritFastStd;
    // aPXml.FastConx() =  aPATP.mCritFastCnx;
    aPXml.Score()  =  aPATP.mScoreInit;
    aPXml.ScoreRel()  = aPATP.mScoreRel;
    aPXml.Var()  = aPATP.mStdDev;
    aPXml.ChgMaj()  = aPATP.mChgMaj;
    aPXml.OKLP()  = aPATP.mOKLP;

    if (aAPCPtr!=nullptr)
    {
        aPXml.ImLP() = cMMV1_Conv<tU_INT1>::ImToMMV1(aAPCPtr->Desc().ILP().DIm());
    }
    return aPXml;
}


template <class Type> void cImplem_ExportAimeTiep<Type>::AddAimeTieP(cProtoAimeTieP<Type>  aPATP)
{
    static int anIdent=0;  // for debugging and computing identifier
    anIdent++;

    cPt2di aV2PIm = aPATP.mPImInit;
    Pt2di  aV1PIm = ToMMV1(aV2PIm);


    int aSzW = round_ni(AC_SZW*aPATP.ScaleInO());
    double aFact  = aSzW / double(AC_SZW);
    double aRho = AC_RHO * aFact;
    static double aLastFact = aFact;
    
    cIm2D<Type>     aIm0V2 = aPATP.mGPI->ImOriHom()-> ImG();
    tImV1  aIm0V1 = cMMV1_Conv<Type>::ImToMMV1(aIm0V2.DIm());  ///<  Image "init", MMV1 Version
    tTImV1 aTIm0V1(aIm0V1);
    // Heuristic way to avoid creating multiple 
    static tCACD  aCACD(aTIm0V1,Pt2di(0,0),aRho,aSzW);
    if (aFact != aLastFact)
    {
       aCACD = tCACD(aTIm0V1,Pt2di(0,0),aRho,aSzW);
       aLastFact = aFact;
    }

    bool  aAutoCor;
    {
        cAutoTimerSegm aATS("AutoCorrel");
        aAutoCor  = aCACD.AutoCorrel(aV1PIm,mFPC.AC_CutInt(),mFPC.AC_CutReal(),mFPC.AC_Threshold());
    }


    
    // cXml2007Pt aPXml;

    // aPXml.Pt() = ToMMV1(aPATP.mPFileInit);
    aPATP.mId = anIdent;
    // aPXml.NumOct() = aPATP.NumOct();
    // aPXml.NumIm() = aPATP.NumIm();
    // aPXml.ScaleInO() = aPATP.ScaleInO();
    // aPXml.ScaleAbs() = aPATP.ScaleAbs();
    aPATP.mOkAutoCor = ! aAutoCor;
    aPATP.mAutoCor = aCACD.mCorOut;
    aPATP.mNumOutAutoC = aCACD.mNumOut;

    if (aPATP.mOkAutoCor || mForInspect)
    {
/*
        Pt2dr aFQ =  FastQuality(mTImStdV1,aV1PIm,*mFCC,! mMMV1_XmlPts.IsMin() ,Pt2dr(0.75,0.85));
        aPATP.mCritFastStd = aFQ.x;
        aPATP.mCritFastCnx = aFQ.y;
*/
        // Compute variance weighted by a pseudo Gauss
        {
           cAutoTimerSegm aATS("VarGauss");
           aPATP.mStdDev = CubGaussWeightStandardDev(aIm0V2.DIm(),aV2PIm,aPATP.ScaleInO()*VAR_RHO);
        }
        if (aPATP.mStdDev<=0)  // Case degenerate
            aPATP.mOkAutoCor = false;
        else
        {
            aPATP.mScoreInit  =      pow(1-aPATP.mAutoCor  , mFPC.PowAC())
                                  *  pow(aPATP.mStdDev     , mFPC.PowVar())
                                  *  pow(aPATP.ScaleAbs()  , mFPC.PowScale())
                             ;
            aPATP.mScoreRel  = aPATP.mScoreInit; // Initially no point selected, relative=absolute
        }

        if (aPATP.mOkAutoCor || mForInspect)
        {
            aPATP.mSFSelected = false;
            mBox.Add(aPATP.mPFileInit);
            mVecProtoPts.push_back(aPATP);
        }
    }
}

template <class Type> std::string cImplem_ExportAimeTiep<Type>::NameExport(const std::string & aNameIm,eModeOutPCar aMode)
{
     return  mParam.mAppli->NamePCar
             (
                aNameIm,
                aMode,
                mSetAPC.Type(),
                false,
                mSetAPC.IsMax(),
                mParam.mNumTile
             );
}

template <class Type> void cImplem_ExportAimeTiep<Type>::Export(const std::string & aNameIm,bool SaveV1)
{
     // Eventualy save to MMV1 format for visual inspection
     if (SaveV1)
     {
        std::string aNameV1 = NameExport(aNameIm, eModeOutPCar::eMNO_PCarV1);
        for (auto const & aPMMV2 : mVecProtoPts)
        {
            int aNum = aPMMV2.mNumAPC;
            cAimePCar * aAPCPtr = (aNum>=0) ? &(mSetAPC.VPC().at(aNum)) : nullptr; 
            mMMV1_XmlPts.Pts().push_back(ToXmlMMV1(aPMMV2,aAPCPtr));
        }
        MakeFileXML(mMMV1_XmlPts,aNameV1);
        mMMV1_XmlPts.Pts().clear() ;
     }
     // Now save in V2 format , what we really need
     std::string aNameV2 = NameExport(aNameIm, eModeOutPCar::eMNO_BinPCarV2);
     mSetAPC.SaveInFile(aNameV2);
}


     // ================= Fitlrage spatial , point "bons" et bien repartis ===============

         //  ------ Qt stuff ---------------
template <class Type> class cFuncPtOfXml2007
{   // argument du qauad tri
    // comment à partir un objet, je recuper sa pt2D
      public :
         Pt2dr operator () (cProtoAimeTieP<Type> *  aXP) {return ToMMV1(aXP->mPFileInit);}
};
// typedef  cXml2007Pt * tP2007Ptr;
// typedef ElQT<tP2007Ptr,Pt2dr,cFuncPtOfXml2007> tQtXml2007;

         //  ------ Heap stuff ---------------
template <class Type>  class cAimeFS_HeapIndex
{
     public :
        static void SetIndex(cProtoAimeTieP<Type> *   aPP,int i)
        {
                aPP->mHeapIndexe = i;
        }
        static int  Index(cProtoAimeTieP<Type> *  aPP)
        {
             return aPP->mHeapIndexe;
        }
};

template <class Type> class cAimeFS_HeapCmp
{
    public :

        bool operator () (cProtoAimeTieP<Type> *  aPP1,cProtoAimeTieP<Type> *  aPP2)
        {
              return aPP1->mScoreRel > aPP2->mScoreRel;   // compare score correl global
        }
        // est ce que objet 1 est meuilleur que 2
};
// typedef ElHeap<tP2007Ptr,cAimeFS_HeapCmp,cAimeFS_HeapIndex> tHeapXml2007;


template <class Type> void cImplem_ExportAimeTiep<Type>::FiltrageSpatialPts()
{
     for (auto & aPATP : mVecProtoPts)
     {
         if (aPATP.mOkAutoCor)
         {
            cAutoTimerSegm aATS("Refined");
            double aScale = aPATP.ScaleInO();
            cAffineExtremum<Type> anAE(aPATP.mGPI->ImG().DIm(),AFEXTR_RADIUS*aScale);

            cPt2dr aP0 = ToR(aPATP.mPImInit);
            cPt2dr aPLast = aP0;
            bool   IsInstable = false;
            bool   GoOn = true;
            int    aKIter = 0;
         
            while (GoOn)
            {
                cPt2dr aPNext = anAE.OneIter(aPLast);
                if (Norm2(aPNext-aP0)>AFEXTR_DIST_INST*aScale)
                {
                     IsInstable = true;
                     GoOn = false;
                }
                else if (Norm2(aPNext-aPLast) < AFEXTR_DIST_CONV *aScale)
                {
                     GoOn = false;
                }

                aKIter ++;
                if (aKIter >=  AFEXTR_NB_ITER)
                {
                     GoOn = false;
                }
                aPLast = aPNext;
            }
          
            aPATP.mPRImRefined     = aPLast;
            aPATP.mPFileRefined =   aPATP.mGPI->Im2File(aPLast);
            aPATP.mStable = ! IsInstable;
         }
         else
         {
            aPATP.mPFileRefined =   aPATP.mPFileInit;
            aPATP.mStable = true;
         }
     }
     {
        cAutoTimerSegm aATS("TestOKLogPol");
        for (auto & aPATP : mVecProtoPts)
        {
            if (aPATP.mStable  && aPATP.mOkAutoCor)
            {
                aPATP.mOKLP = aPATP.TestFillAPC(mFPC);
            }
            else
            {
                aPATP.mOKLP = true;
            }
        }
     }
     cAutoTimerSegm aATS("SpatialFilter");
     if (mVecProtoPts.empty())  // may be degenerated (bounding box ...)
     {
        return;
     }
     cBox2dr aB = mBox.CurBox().Dilate(10);

     // Quod tree for spacial indexation
     cFuncPtOfXml2007<Type> aFctrPt;
     ElQT<cProtoAimeTieP<Type>*,Pt2dr,cFuncPtOfXml2007<Type> >  aQt(aFctrPt,ToMMV1(aB),10,20.0);

     // tQtXml2007  aQt(aFctrPt,ToMMV1(aB),10,20.0);

     // Heap for handling priority
     cAimeFS_HeapCmp<Type> aFctrCmp;
     ElHeap<cProtoAimeTieP<Type>*,cAimeFS_HeapCmp<Type>,cAimeFS_HeapIndex<Type>> aHeap(aFctrCmp);

     // Put everyting in heap and quod tree, put only validated (they may remain when inspect mode)
     for (auto & aP : mVecProtoPts)
     {
         if (aP.mOkAutoCor && aP.mStable && aP.mOKLP)
         {
            aQt.insert(&aP);
            aHeap.push(&aP);
         }
     }
     
     //  do the spatial filtering
     // double aMulRay = mFPC.mPSF.at(1);
     // double aPropNoSF = mFPC.mPSF.at(2);
     double aDsf = mFPC.DistSF();
     int NbToGet= (mSzIm0.x()*mSzIm0.y()) / Square(aDsf);
     while (NbToGet>0)
     {
        double aDistInfl = aDsf * mFPC.MulDistSF();
        double aProp =  mFPC.PropNoSF();
        cProtoAimeTieP<Type>* aNewP = nullptr;
        if (aHeap.pop(aNewP)) // if get next best pts 
        {
            aQt.remove(aNewP);  // supress from Qt
            aNewP->mSFSelected = true;  // memorize it is selected
            std::set<cProtoAimeTieP<Type>* > aSet;
            aQt.RVoisins(aSet,ToMMV1(aNewP->mPFileInit),aDistInfl);  // Extract neighboors

            for (const auto & aVois : aSet)
            {
                if (! aVois->mSFSelected)
                {
                    double aD = Norm2(aNewP->mPFileInit-aVois->mPFileInit); // Distance to new selected
                    double aRatio = aD/aDistInfl;  // put as ratio to max
                    aRatio = aProp + aRatio *(1-aProp); // take into account non spatial part
                    double aNewSc = aVois->mScoreInit * aRatio;  // comput new score 
                    if (aNewSc < aVois->mScoreRel)  // if new score is lower
                    {
                        aVois->mScoreRel = aNewSc;  // update score
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
             mVecProtoPts,
	     [] (const cProtoAimeTieP<Type> & aPP) 
                { return  (!aPP.mOkAutoCor) || (! aPP.mSFSelected) || (!aPP.mStable) || (!aPP.mOKLP);}
         );
     }
     {
        cAutoTimerSegm aATS("ImLogPol");
        for (auto & aPATP : mVecProtoPts)
        {
            // Test required because has not been erase in inspect mode
            if (aPATP.mOkAutoCor &&  aPATP.mSFSelected && aPATP.mStable && aPATP.mOKLP)
            {
                 cAimePCar aAPC;
                 // bool Ok = aPATP.FillAPC(mFPC,aAPC,false);
                 MMVII_INTERNAL_ASSERT_tiny(aPATP.FillAPC(mFPC,aAPC,false),"Incoherence in FillAPC");
                 aPATP.mNumAPC = mSetAPC.VPC().size();
                 mSetAPC.VPC().push_back(aAPC);
            }
        }
     }
}


// ============  INSTANTIATION ======================

template class cInterf_ExportAimeTiep<tREAL4>;
template class cInterf_ExportAimeTiep<tINT2>;
template class cImplem_ExportAimeTiep<tREAL4>;
template class cImplem_ExportAimeTiep<tINT2>;


};
