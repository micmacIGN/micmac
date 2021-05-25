/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "Apero.h"


/*******************************************/
/*                                         */
/*    cSolBasculeRig                        */
/*    cBasculementRigide                   */
/*                                         */
/*******************************************/



     //---------------------------------------------
     //     UTILITAIRES
     //---------------------------------------------

Pt3dr cAppliApero::CpleIm2PTer(const cAperoPointeStereo & anAPS)
{
    const CamStenope * aCS1 = PoseFromName(anAPS.Im1())->CurCam();
    const CamStenope * aCS2 = PoseFromName(anAPS.Im2())->CurCam();

    return   aCS1->PseudoInter(anAPS.P1(),*aCS2,anAPS.P2());
}

Pt3dr cAppliApero::PImetZ2PTer(const cAperoPointeMono & anAPM,double aZ)
{

    const CamStenope * aCS = PoseFromName(anAPM.Im())->CurCam();
    return aCS->F2AndZtoR3(anAPM.Pt(),aZ);
}




     //---------------------------------------------
     //     cArgGetPtsTerrain
     //---------------------------------------------

cArgGetPtsTerrain::cArgGetPtsTerrain(double aResolMAsq,double aLimBsH) :
     mResol      (aResolMAsq),
     mMasq       (0),
     mMode       (eModeAGPIm),
     mLimBsH     (aLimBsH),
     mDoByIm     (false),
     mSymDoByIm  (false)
{
}

void cArgGetPtsTerrain::SetByIm(bool DoByIm,bool Sym)
{
   mDoByIm = DoByIm;
   mSymDoByIm = Sym;
}



double cArgGetPtsTerrain::LimBsH() const {return mLimBsH;}

void cArgGetPtsTerrain::InitColiorageDirec(Pt3dr aDir,double aPer)
{
   mMode = eModeAGPHypso;
   double aFreq = 1/aPer;
   mDirCol  =   vunit(aDir) * aFreq;
}

void cArgGetPtsTerrain::InitModeNormale()
{
   mMode = eModeAGPNormale;
}

void cArgGetPtsTerrain::InitModeNoAttr()
{
   mMode = eModeAGPNoAttr;
}

void cArgGetPtsTerrain::SetMasq(Im2D_Bits<1> * aMasq)
{
   mMasq = aMasq;
}

const  double mAGPFactN = 255;

void cArgGetPtsTerrain::AddAGP
     (
           Pt2dr aPIm,
           const Pt3dr & aPts,
           double aPds,
           bool aReduc,
           const std::vector<double> * aVPds ,
           const std::vector<cGenPoseCam *> * aVPose
     )
{
   if (mMasq)
   {
       Pt2di aPInd = round_ni(aPIm/mResol);
       if (mMasq->get_def(aPInd.x,aPInd.y,0)==0)
          return;
   }
   if (aPds > 0)
   {
      mPts.push_back(aPts);
      mPds.push_back(aPds);
      if (mDoByIm && aVPose)
      {
          int aNb = mSymDoByIm ? (int)aVPose->size() : ElMin(1,int(aVPose->size()));
          for (int aK=0 ; aK<aNb ; aK++)
              (*aVPose)[aK]->AddPtsVu(aPts);
      }

      if (mMode == eModeAGPIm)
      {
         if (mVIms.size())
         {
            Pt2dr aP = aReduc ? (aPIm/mStepImRed) : aPIm;
            std::vector<Im2DGen *> & aVI = aReduc ? mVImRed : mVIms;
            mCouls.push_back
            (
                 Pt3di
                 (
                     aVI[mKR]->GetI(Pt2di::RP2ToThisT(aP),0),
                     aVI[mKG]->GetI(Pt2di::RP2ToThisT(aP),0),
                     aVI[mKB]->GetI(Pt2di::RP2ToThisT(aP),0)
                 )
            );
         }
      }
      else if (mMode==eModeAGPHypso)
      {
           double  aTeinte = scal(aPts,mDirCol) ;
           Elise_colour aCol = Elise_colour::its(0.5,aTeinte,0.5);
            mCouls.push_back
            (
                 Pt3di
                 (
                     round_ni(aCol.r()*255),
                     round_ni(aCol.g()*255),
                     round_ni(aCol.b()*255)
                 )
            );

      }
      else if (mMode==eModeAGPNormale)
      {
           ELISE_ASSERT(aVPds!=0,"eModeAGPNormale no Pds/Cam");
           Pt3dr aNorm(0,0,0);
           double aSomP = 0;
           for (int aK=0 ; aK<int(aVPds->size()) ; aK++)
           {
               double aPds = (*aVPds)[aK];
               if (aPds > 0)
               {
                   const cBasicGeomCap3D  * aCS = (*aVPose)[aK]->GenCurCam();
                   // Pt3dr aC = aCS->PseudoOpticalCenter();
                   Pt3dr aC = aCS->OpticalCenterOfPixel(aPIm);
                   aSomP +=  aPds;
                   aNorm  = aNorm + vunit(aC-aPts) * aPds;
               }
           }
           aNorm = vunit(aNorm/aSomP);
           mCouls.push_back
           (
                 Pt3di
                 (
                     round_ni(aNorm.x*mAGPFactN),
                     round_ni(aNorm.y*mAGPFactN),
                     round_ni(aNorm.z*mAGPFactN)
                 )
           );


      }
      else if (mMode==eModeAGPNoAttr)
      {
      }
   }
}

void cArgGetPtsTerrain::AddPts(Pt3dr aP1,Pt3di aCoul)
{
   mCouls.push_back(aCoul);
   mPts.push_back(aP1);
}

void cArgGetPtsTerrain::AddSeg(Pt3dr aP1,Pt3dr aP2,double aStep,Pt3di aCoul)
{
   if (aCoul.x<0) return;
   double aD = euclid(aP1-aP2);
   int aNb = round_up(aD/aStep);
   for (int aKP=0 ; aKP<=aNb ; aKP++)
   {
        double aPds = aKP / double(aNb);
        Pt3dr aP = aP1*aPds + aP2*(1.0-aPds);
        mCouls.push_back(aCoul);
        mPts.push_back(aP);
   }
}

void cArgGetPtsTerrain::InitFileColor(const std::string & aName,double aStepIm,const std::string & aImRef,int NbChan)
{
   mMode = eModeAGPIm;
   DeleteAndClear(mVIms);
   DeleteAndClear(mVImRed);
   mStepImRed=aStepIm;

   std::cout << aName << " " << mVIms.size() << " " << mVImRed.size() << "\n";

   double aMul=1;
   if (aImRef!="")
   {
       const cMetaDataPhoto & aMDP =     cMetaDataPhoto::CreateExiv2(aName);
       const cMetaDataPhoto & aMDPRef =  cMetaDataPhoto::CreateExiv2(aImRef);
       aMul = aMDP.MultiplierEqual(aMDPRef,0);
   }
   Tiff_Im aTin = Tiff_Im::StdConvGen(aName,NbChan,false);
   mVIms = aTin.VecOfIm(aTin.sz());
   ELISE_COPY(aTin.all_pts(),Min(255,aMul*aTin.in()),StdOut(mVIms));
   if (mVIms.size()==1)
   {
        mKR = 0;
        mKG = 0;
        mKB = 0;
   }
   else if (mVIms.size()==3)
   {
        mKR = 0;
        mKG = 1;
        mKB = 2;
   }
   else if (mVIms.size()==4)
   {
		std::cout << "Warning: 4 channel image considered as RGB" << "\n";
        mKR = 0;
        mKG = 1;
        mKB = 2;
   }
   else
   {
       ELISE_ASSERT(false,"Bas size in cArgGetPtsTerrain::InitFileColor");
   }

   if (mStepImRed > 0)
   {
       for (int aKIM=0 ; aKIM<int(mVIms.size()) ; aKIM++)
       {
            Im2DGen * aI1 = mVIms[aKIM];
            mVImRed.push_back(aI1->ImOfSameType(round_up(Pt2dr(aI1->sz())/mStepImRed)));
            Im2DGen * aIR = mVImRed[aKIM];

            ELISE_COPY
            (
                aIR->all_pts(),
                StdFoncChScale
                (
                     aI1->in_proj(),
                     Pt2dr(0,0),
                     Pt2dr(mStepImRed,mStepImRed)
                ),
                aIR->out()
            );
       }
   }
}


const std::vector<Pt3dr>  &  cArgGetPtsTerrain::Pts() const  {return mPts;}
const std::vector<double> &  cArgGetPtsTerrain::Pds() const  {return mPds;}

const std::vector<Pt3di>  &  cArgGetPtsTerrain::Cols() const  {return mCouls;}


     //---------------------------------------------
     //     BasculeCentre
     //---------------------------------------------

void TestBasc
     (
           const std::vector<std::string> & aVName,
           std::vector<Pt3dr>               aV1,
           std::vector<Pt3dr>               aV2
     )
{
   Pt3dr aC1(0,0,0);
   Pt3dr aC2(0,0,0);
   int aNb = (int)aV1.size();
   for (int aK=0 ; aK<aNb ; aK++)
   {
       aC1 = aC1 + aV1[aK];
       aC2 = aC2 + aV2[aK];
   }
   aC1 = aC1/aNb;
   aC2 = aC2/aNb;
   std::cout << aC1 << aC2 << "\n";
   for (int aK=0 ; aK<aNb ; aK++)
   {
       aV1[aK]  = aV1[aK] -aC1;
       aV2[aK]  = aV2[aK] -aC2;
       std::cout << "==  " << aVName[aK] << " " <<  aV1[aK] << " " << aV2[aK] << "\n";
   }
}

/******************************************************/
/*                                                    */
/*            cCompBascNonLin                         */
/*                                                    */
/******************************************************/

// cAnalyseZoneLiaison
// seg_mean_square
class cCompBascNonLin;
class cPtBNL;



class cPtBNL
{
   public :
      bool operator < (const cPtBNL & aP2) const {return mPLine.x < aP2.mPLine.x;}
      cPtBNL(const cCompBascNonLin & aCBNL,int aK);
      cPtBNL();
      const Pt3dr & PLine() const {return mPLine;}
      const Pt3dr & ErLine() const {return mErLine;}
      inline Pt3dr PAvant () const;
      inline Pt3dr PApres () const;
      inline Pt3dr PGot () const;
      inline const std::string  & Name() const;
      bool  UseForEstim() const;

   private :
      const cCompBascNonLin * mCBNL;
      int  mK;
      Pt3dr  mPLine;
      Pt3dr  mErLine;  // Erreur en coord line

};

class cModelQuadXY
{
    public :

       cModelQuadXY(bool DoShow,const std::string & aName,int aFlagMasq);
       void AddObs(const Pt3dr & aP,double aV);
       void Solve();
       double  ValOfPt(const Pt3dr & aP) const;
       void Show();

    private :
       bool Masqued(const int & aK) {return 0== (mMasq & (1<<aK));}

       std::string     mName;
       L2SysSurResol   mSys;
       Im1D_REAL8      mSol;
       double *        mDS;
       int             mMasq; // Masque des variables anhilees

};



class cCompBascNonLin : public cTransfo3D
{
    friend class cPtBNL;
    public :
       cCompBascNonLin
       (
             const cRansacBasculementRigide & aBasc,
             const cSolBasculeRig   & aSBR,
             const cAerialDeformNonLin &
       );
       virtual ~cCompBascNonLin(){}
       Pt3dr ToLineC(const Pt3dr &) const;
       Pt3dr FromLineC(const Pt3dr &) const;

       std::vector<Pt3dr> Src2Cibl(const std::vector<Pt3dr> &) const;
       Pt3dr Src2Cibl(const Pt3dr &) const;
       Pt3dr ModeleOfCorr(const Pt3dr & ) const; // Line-Coord to Line-Coord
    private :
       cCompBascNonLin (const cCompBascNonLin &); // N.I.
       const cRansacBasculementRigide & mBasc;
       const cSolBasculeRig   &         mSBR;
       const cAerialDeformNonLin &      mADLN;
       bool                             mShow;
       SegComp                          mSegLine;
       int                              mNb;
       std::vector<cPtBNL>              mVPt;

       cModelQuadXY                     mMQX;
       cModelQuadXY                     mMQY;
       cModelQuadXY                     mMQZ;
       mutable cElRegex                 mSelEstim;
};



    // ============= cModelQuadXY  =======================

cModelQuadXY::cModelQuadXY(bool DoShow,const std::string & aName,int aFlagMasq) :
   mName (aName),
   mSys  (6),
   mSol  (6),
   mDS   (0),
   mMasq (aFlagMasq)
{
   if ( DoShow) Show();
}

const char * NamePolQuadXY[6] = {"1","X","Y","X2","XY","Y2"};
void cModelQuadXY::Show()
{

    std::cout << " MQ:" << mName << " [";
    for (int aK=0 ; aK<6 ; aK++)
        if (! Masqued(aK))
           std::cout << NamePolQuadXY[aK] << " ";
    std::cout << "]\n";
}

double  cModelQuadXY::ValOfPt(const Pt3dr & aP) const
{
   return   mDS[0]
         +  mDS[1] * aP.x
         +  mDS[2] * aP.y
         +  mDS[3] * aP.x * aP.x
         +  mDS[4] * aP.x * aP.y
         +  mDS[5] * aP.y * aP.y ;
}


void cModelQuadXY::AddObs(const Pt3dr & aP,double aV)
{
    double aCoeff[6];

    aCoeff[0] = 1;
    aCoeff[1] = aP.x;
    aCoeff[2] = aP.y;
    aCoeff[3] = aP.x * aP.x;
    aCoeff[4] = aP.x * aP.y;
    aCoeff[5] = aP.y * aP.y;

    for (int aFlag=0 ; aFlag<6 ; aFlag++)
    {
       if (Masqued(aFlag))
       {
         aCoeff[aFlag] = 0;
       }
    }

    mSys.AddEquation(1.0,aCoeff,aV);
}

void cModelQuadXY::Solve()
{
    for (int aFlag=0 ; aFlag<6 ; aFlag++)
    {
       if (Masqued(aFlag))
       {
          double aCoeff[6];
          for (int aK=0 ; aK<6 ; aK++)
          {
               aCoeff[aK] = (aK==aFlag);
          }
          mSys.AddEquation(1.0,aCoeff,0.0);
       }
    }

    mSol = mSys.Solve((bool *) 0);
    mDS = mSol.data();
}

    // ============= cPtBNL =======================

const std::string  & cPtBNL::Name() const { return mCBNL->mBasc.Names()[mK]; }
Pt3dr   cPtBNL::PAvant() const { return mCBNL->mBasc.PAvant()[mK]; }
Pt3dr   cPtBNL::PApres() const { return mCBNL->mBasc.PApres()[mK]; }
Pt3dr   cPtBNL::PGot() const { return mCBNL->mSBR(mCBNL->mBasc.PAvant()[mK]); }

cPtBNL::cPtBNL(const cCompBascNonLin & aCBNL,int aK) :
   mCBNL   (&aCBNL),
   mK      (aK),
   mPLine  (mCBNL->ToLineC(PGot())),
   mErLine (mCBNL->ToLineC(PApres())-mPLine)
{
}

bool   cPtBNL::UseForEstim() const
{
   return mCBNL->mSelEstim.Match(Name());
}
    // ============= cCompBascNonLin =======================

cCompBascNonLin::cCompBascNonLin
(
             const cRansacBasculementRigide & aBasc,
             const cSolBasculeRig   & aSBR,
             const cAerialDeformNonLin & anADLN
)  :
   mBasc (aBasc),
   mSBR  (aSBR),
   mADLN (anADLN),
   mShow (mADLN.Show().Val()),
   mNb   ((int)mBasc.PAvant().size()),
   mMQX  (mShow,"X",anADLN.FlagX().Val()),
   mMQY  (mShow,"Y",anADLN.FlagY().Val()),
   mMQZ  (mShow,"Z",anADLN.FlagZ().Val()),
   mSelEstim (anADLN.PattEstim().Val(),10)
{
   // Calcul du segment
   RMat_Inertie aMat;
   for (int aK=0 ; aK<mNb ; aK++)
   {
       Pt3dr aTarg =  mBasc.PApres()[aK];
       aMat.add_pt_en_place(aTarg.x,aTarg.y,1.0);
   }
   mSegLine = seg_mean_square(aMat,1.0);

   // Calcul du vecteur de points
   for (int aK=0 ; aK<mNb ; aK++)
   {
       mVPt.push_back(cPtBNL(*this,aK));
   }
   std::sort(mVPt.begin(),mVPt.end());

   // Calcul de l'estimateur
   for (int aK=0 ; aK<mNb ; aK++)
   {
       const cPtBNL & aPt = mVPt[aK];
       if (aPt.UseForEstim())
       {
          Pt3dr aPLine = aPt.PLine();
          Pt3dr aPEr = aPt.ErLine();

          mMQX.AddObs(aPLine,aPEr.x);
          mMQY.AddObs(aPLine,aPEr.y);
          mMQZ.AddObs(aPLine,aPEr.z);
       }
   }

   mMQX.Solve();
   mMQY.Solve();
   mMQZ.Solve();

   // Calcul du vecteur de points
   if (mShow)
   {
       for (int aK=0 ; aK<mNb ; aK++)
       {
          const cPtBNL & aPt = mVPt[aK];
          Pt3dr aPLine = aPt.PLine();
          Pt3dr aPEr = aPt.ErLine();

          Pt3dr aModEr = ModeleOfCorr(aPLine);

          Pt3dr aPG = aPt.PGot();
          Pt3dr aPCor = Src2Cibl(aPG);
          Pt3dr aCibl = aPt.PApres();

          // Pt3dr aVerif = (aCibl-aPCor) - (aModEr-aPEr);
          double aVerif = ElAbs(euclid(aCibl-aPCor) - euclid(aModEr-aPEr)) / (euclid(aCibl));
          ELISE_ASSERT(aVerif<1e-10,"cCompBascNonLin::cCompBascNonLin");

          std::cout << (aPt.UseForEstim() ? "* " : "  ")
                    << aPt.Name() << " ErInit : " << euclid(aPEr)
                    // << " Ver " << aVerif
                    << " => ErCor : " << euclid(aModEr-aPEr)  << " DZ=" << (aModEr.z-aPEr.z) << "\n";
       }
   }

}

Pt3dr cCompBascNonLin::ToLineC(const Pt3dr & aP) const
{
    Pt2dr aRes2 = mSegLine.to_rep_loc(Pt2dr(aP.x,aP.y));
    return Pt3dr(aRes2.x,aRes2.y,aP.z);
}

Pt3dr cCompBascNonLin::FromLineC(const Pt3dr & aP) const
{
    Pt2dr aRes2 = mSegLine.from_rep_loc(Pt2dr(aP.x,aP.y));
    return Pt3dr(aRes2.x,aRes2.y,aP.z);
}

std::vector<Pt3dr> cCompBascNonLin::Src2Cibl(const std::vector<Pt3dr> & aInput) const
{
    std::vector<Pt3dr> aRes;
    for (int aK=0 ; aK<int(aInput.size()) ; aK++)
       aRes.push_back(Src2Cibl(aInput[aK]));

   return aRes;
}

Pt3dr  cCompBascNonLin::ModeleOfCorr(const Pt3dr &  aP) const // Line-Coord to Line-Coord
{
    return Pt3dr
           (
               mMQX.ValOfPt(aP),
               mMQY.ValOfPt(aP),
               mMQZ.ValOfPt(aP)
           );
}
Pt3dr cCompBascNonLin::Src2Cibl(const Pt3dr & aPBasc) const
{
   Pt3dr aPLine =  ToLineC(aPBasc);
   Pt3dr aCorrec = ModeleOfCorr(aPLine);

   return FromLineC(aPLine+aCorrec);
}

/*************************************************************/


cSolBasculeRig   Xml2EL(const cXml_ParamBascRigide & aXml)
{
   return   cSolBasculeRig::SBRFromElems
            ( 
                aXml.Trans(),
                ImportMat(aXml.ParamRotation()),
                aXml.Scale()
            );
}

cXml_ParamBascRigide   EL2Xml(const cSolBasculeRig & aSBR)
{
    cXml_ParamBascRigide aRes;
    aRes.ParamRotation() = ExportMatr(aSBR.Rot());
    aRes.Trans() = aSBR.Tr();
    aRes.Scale() = aSBR.Lambda();

    return aRes;
}




cSolBasculeRig cAppliApero::BasculePoints
     (
           const cBasculeOnPoints & aBOP,
           cSetName  &            aSelectorEstim,
           // cElRegex &            aSelectorEstim,
           cElRegex &            aSelectorApply
     )
{
   bool aForceSol = aBOP.ForceSol().IsInit();

   bool aBonC = aBOP.BascOnCentre().IsInit();
   bool CalcV = aBonC &&  aBOP.BascOnCentre().Val().EstimateSpeed().Val();


   cRansacBasculementRigide aBasc(CalcV);
   int aKC=-1;
   std::vector<std::string> aVName;
   bool Test = false;

   if (aBonC)
   {
      const cBascOnCentre aBC = aBOP.BascOnCentre().Val();
      for (int aKPose=0 ; aKPose<int(mVecPose.size()) ; aKPose++)
      {

          cPoseCam * aPC = mVecPose[aKPose];
          if (
                   aSelectorEstim.IsSetIn(aPC->Name())
                && (aPC->RotIsInit())
                && (aPC->HasObsOnCentre())
                && ((! CalcV) || aPC->HasObsOnVitesse())
             )
          {
              // std::cout << "BASCULE CENTRE DO " << aPC->Name() << "\n";
              Pt3dr aC0 = aPC->CurRot().ImAff(Pt3dr(0,0,0));

              Pt3dr aCObs = aPC->ObsCentre();

              // const cObserv1Im<cTypeEnglob_Centre> & anOC = ObsCentre(aBC.IdBDC(),aPC->Name());
              //
                if (Test) aVName.push_back(aPC->Name());

               if (CalcV)
               {
                   Pt3dr aV = aPC->Vitesse();
                   aBasc.AddExemple(aC0,aCObs,&aV,aPC->Name());
               }
               else
               {
                   aBasc.AddExemple(aC0,aCObs,0,aPC->Name());
               }

               if (   aBOP.PoseCentrale().IsInit()
                   && (! CalcV)
                   && (aBOP.PoseCentrale().Val()==aPC->Name())
                  )
               {
                  aKC = aBasc.CurK();
               }
          }
      }
   }



   cBdAppuisFlottant * aBAF = 0;

   if (aBOP.BascOnAppuis().IsInit())
   {
       cBascOnAppuis aBOA = aBOP.BascOnAppuis().Val();

       cPackGlobAppuis  * aPtrPGA = PtrPackGlobApp(aBOA.NameRef(),true);

       // Compat historique avec points appuis purs
       if (aPtrPGA)
       {
           cPackGlobAppuis  & aPGA = *aPtrPGA;
           std::map<int,cOneAppuiMul *> * aMAP = aPGA.Apps();
           for
           (
               std::map<int,cOneAppuiMul *>::iterator itO = aMAP->begin();
               itO != aMAP->end();
               itO++
           )
           {
                cOneAppuiMul * anOAM = itO->second;
                if (anOAM->NbInter() >=2)
                {
                    aBasc.AddExemple(anOAM->PInter(),anOAM->PTer(),0,ToString(itO->first));
                }
           }
       }
       else
       {
           aBAF = BAF_FromName(aBOA.NameRef(),0);
           const std::map<std::string,cOneAppuisFlottant *> &  aMAP = aBAF->Apps();
           for
           (
               std::map<std::string,cOneAppuisFlottant *>::const_iterator itF= aMAP.begin();
               itF!= aMAP.end();
               itF++
           )
           {
                cOneAppuisFlottant * anOAF = itF->second;
                if ((anOAF->NbMesures() >=2) && (anOAF->HasGround()))
                {
                    Pt3dr aPInc = anOAF->PInc();
                    ///std::cout << "BASCULEFFF " <<  anOAF->Name() << " " << aPInc << "\n";
                    if ((aPInc.x>0) && (aPInc.y>0) && (aPInc.z>0))
                    {
                        aBasc.AddExemple(anOAF->PInter(),anOAF->PtRes(),0,anOAF->Name());
                    }
                }
           }
           // std::cout << "BAF= " << aBAF << "\n";
           // ELISE_ASSERT(false,"jjjjjjjjjjjjjjjjjghgjghfffj");
       }
   }



   bool OkBasc=false;
   if (aKC!=-1)
   {
      OkBasc = aBasc.CloseWithTrOnK(aKC,true);
   }
   else
   {
      OkBasc = aBasc.CloseWithTrGlob(true);
   }

   if (!OkBasc)
   {
       if (aBAF)
       {
          aBAF->ShowError();
       }
       ELISE_ASSERT
       (
           false,
           "Not enough samples (Min 3) in cRansacBasculementRigide"
       );
   }

   aBasc.ExploreAllRansac();

   cSolBasculeRig  aSBR = aBasc.BestSol();


   if (Test)
   {
        //TestBasc(aVName,aBasc. PAvant(),aBasc.PApres());
   }

   if (aBOP.ModeL2().Val())
   {
       cSetEqFormelles aSetEq (cNameSpaceEqF::eSysPlein);
       cL2EqObsBascult  * aL2Basc = aSetEq.NewEqObsBascult(aSBR,false);
       aSetEq.SetClosed();

       // std::cout << "PARAM INIT " << aSBR.Tr() << " " <<  aSBR.Lambda()<< "\n";
       for (int aKEt=0 ; aKEt< 5 ; aKEt++)
       {
           aSetEq.SetPhaseEquation();
           const std::vector<Pt3dr>  & aV1 = aBasc. PAvant();
           const  std::vector<Pt3dr> & aV2 = aBasc.PApres() ;
           for (int aKP=0 ; aKP<int(aV1.size()) ; aKP++)
           {
               // std::cout << "   " << aSBR(aV1[aKP]) - aV2[aKP] << "\n";
               aL2Basc->AddObservation(aV1[aKP],aV2[aKP]);
           }
           aSetEq.SolveResetUpdate();
           aSBR=aL2Basc->CurSol();
           // std::cout << "PARAM K  " << aSBR.Tr() << " " <<  aSBR.Lambda()<< "\n";
       }

   }

   if (aForceSol)
   {
        cXml_ParamBascRigide anXPBR =  StdGetFromPCP(mDC+aBOP.ForceSol().Val(),Xml_ParamBascRigide);
        aSBR = Xml2EL(anXPBR);
   }

   cCompBascNonLin * aPtrBNL=0;
   const cAerialDeformNonLin * anADNL = aBOP.AerialDeformNonLin().PtrVal();
   if (anADNL!=0)
   {
       aPtrBNL = new cCompBascNonLin(aBasc,aSBR,*anADNL);
   }

   //for (int aKPose=0 ; aKPose<int(mVecPose.size()) ; aKPose++)
   // Pour conserver l'ordre alphabetique, + utile pour l'affichage on passe
   // par le dico
   for
   (
       tDiPo::const_iterator itD=mDicoPose.begin();
       itD!=mDicoPose.end();
       itD++
   )
   {
       //cPoseCam * aPC = mVecPose[aKPose];
       cPoseCam * aPC = itD->second;
       if (
                aSelectorApply.Match(aPC->Name())
             && (aPC->RotIsInit())
          )
       {
            aPC->SetBascRig(aSBR);

            if (aPtrBNL)
            {
               std::vector<ElCamera *> aVC;
               CamStenope * aCS = aPC->DupCurCam();
               aCS->UnNormalize();
               aCS->SetAltiSol(aPC->AltiSol());
               aCS->SetProfondeur(aPC->Profondeur());
               aVC.push_back(aCS);
               bool FTR = anADNL->ForceTrueRot().Val();
               ElCamera::ChangeSys(aVC,*aPtrBNL,FTR,!aBonC);
               if (FTR)
               {
                  aPC->PCSetCurRot(aCS->Orient().inv());
               }
               else
               {
                  // Si mode non ortho, on ne pourra plus faire de compensation
                  SetSqueezeDOCOAC();
                  aPC->SetCamNonOrtho(aCS);
               }
/*
*/
            }

            if (aPC->HasObsOnCentre() && ((!CalcV) || (aPC->HasObsOnVitesse())))
            {
               Pt3dr aCObs = aPC->ObsCentre();
               Pt3dr aC = aPC->CurRot().tr();
               if (CalcV)
                  aCObs = aCObs +  aPC-> Vitesse() * aBasc.Delay();
               std::cout << "Basc-Residual " <<  aPC->Name() << " " << aCObs-aC << "\n";
            }
       }
   }


   if (aBOP.NameExport().IsInit())
   {
      cXml_SolBascRigide aXSBR; //  cSolBasculeRig

      aXSBR.Param() = EL2Xml(aSBR);
      const std::vector<Pt3dr>  & aV1 = aBasc. PAvant();
      const  std::vector<Pt3dr> & aV2 = aBasc.PApres() ;
      const  std::vector<std::string> & aVN = aBasc.Names();
      ELISE_ASSERT(aV1.size()==aV2.size(),"Bad size in cRansacBasculementRigide");
      ELISE_ASSERT(aV1.size()==aVN.size(),"Bad size in cRansacBasculementRigide");

      aXSBR.MoyenneDist()      = 0.0;
      aXSBR.MoyenneDistAlti()  = 0.0;
      aXSBR.MoyenneDistPlani() = 0.0;

      aXSBR.Worst().Dist() = 0.0;
      aXSBR.Worst().Offset() = Pt3dr(0,0,0);
      aXSBR.Worst().Name() = "NONE";

 
      for (int aK=0 ; aK <int(aV1.size()) ; aK++)
      {
          cXml_ResiduBascule aRes;
          aRes.Name() = aVN[aK];
          Pt3dr Offset = aV2[aK]- aSBR(aV1[aK]);
          aRes.Offset() = Offset;
          aRes.Dist() = euclid(aRes.Offset());

          if (aRes.Dist() > aXSBR.Worst().Dist())
          {
              aXSBR.Worst() = aRes;
          }
          aXSBR.Residus().push_back(aRes);
          aXSBR.MoyenneDist()      += euclid(Offset) ;
          aXSBR.MoyenneDistAlti()  +=  ElAbs(Offset.z);
          aXSBR.MoyenneDistPlani() +=  euclid(Pt2dr(Offset.x,Offset.y));
      }
      aXSBR.MoyenneDist() /= aV1.size();
      aXSBR.MoyenneDistAlti() /= aV1.size();
      aXSBR.MoyenneDistPlani() /= aV1.size();


      ELISE_fp::MkDirSvp(DirOfFile(aBOP.NameExport().Val()));
      MakeFileXML(aXSBR,aBOP.NameExport().Val());
   }



/*
   if (anADNL && anADNL->Show().Val())
   {
      if (aBAF)
      {
         const std::map<std::string,cOneAppuisFlottant *> &  aMAP = aBAF->Apps();
         for
         (
               std::map<std::string,cOneAppuisFlottant *>::const_iterator itF= aMAP.begin();
               itF!= aMAP.end();
               itF++
         )
         {
                cOneAppuisFlottant * anOAF = itF->second;
                if ((anOAF->NbMesures() >=2) && (anOAF->HasGround()))
                {
                    Pt3dr aPInc = anOAF->PInc();
                    ///std::cout << "BASCULEFFF " <<  anOAF->Name() << " " << aPInc << "\n";
                    if ((aPInc.x>0) && (aPInc.y>0) && (aPInc.z>0))
                    {
                        Pt3dr aPG = anOAF->PtInit();
                        Pt3dr aPI = anOAF->PInter();
                        Pt3dr aDif = aPI-aPG;
                        std::cout << "Non Linear Basc : " <<  anOAF->Name() << " " << aPG << " " << euclid(aDif) << " " << aDif  << "\n";
                    }
                }
           }
           getchar();
           // std::cout << "BAF= " << aBAF << "\n";
      }
   }
*/


   delete aPtrBNL;

   return aSBR;
}


     //---------------------------------------------
     //     BasculeLiaison
     //---------------------------------------------
     //

cElPlan3D cAppliApero::EstimPlan
          (
                const cParamEstimPlan &       aPEP,
                cSetName &                    aSelectorEstim,
                const char *                  anAttr
          )
{
   cPackObsLiaison * aPOL = PackOfInd(aPEP.IdBdl());
   cArgGetPtsTerrain aAGPt(1.0,aPEP.LimBSurH().Val());

   aPOL->GetPtsTerrain (aPEP, aSelectorEstim, aAGPt,anAttr);


   // const std::vector<Pt3dr>  &  aVPts = aAGPt.Pts();
   // const std::vector<double> &  aVPds = aAGPt.Pds();
   std::vector<Pt3dr>   aVPts = aAGPt.Pts();
   std::vector<double>  aVPds = aAGPt.Pds();

   std::cout << "NbPts 4 plan " << aVPts.size() << "\n";

   if ((aVPts.size() == 0) && (aPEP.AcceptDefPlanIfNoPoint().Val()))
   {
       aVPts.push_back(Pt3dr(0,0,0));
       aVPts.push_back(Pt3dr(1,0,0));
       aVPts.push_back(Pt3dr(0,1,0));
       aVPds.push_back(1.0);
       aVPds.push_back(1.0);
       aVPds.push_back(1.0);
   }

/*
{
   Pt3dr aPMin(1e9,1e9,1e9);
   Pt3dr aPMax(-1e9,-1e9,-1e9);
   for (int aK=0 ; aK <int(aVPts.size()) ; aK++)
   {
       aPMax= Sup(aPMax,aVPts[aK]);
       aPMin= Inf(aPMin,aVPts[aK]);
   }
    std::cout << "EEEEEppl "  << aPMin << aPMax << "\n";
}
*/

   // cElPlan3D aPlan(aVPts,&aVPds);
   cElPlan3D aPlan =  RobustePlan3D(aVPts,&aVPds,1e6);


    // std::cout << "EEEEEppl "  << aPlan.Norm() << "\n";


   ElRotation3D aRE2Pl = aPlan.CoordPlan2Euclid().inv();
   double aSomZ = 0.0;
   for (int aKP=0 ; aKP<int(mVecPose.size()) ; aKP++)
   {
       if (aSelectorEstim.IsSetIn(mVecPose[aKP]->Name()))
       {

           const CamStenope * aCS =  mVecPose[aKP]->CurCam();
           Pt3dr aP = aRE2Pl.ImAff(aCS->PseudoOpticalCenter());
           aSomZ += aP.z;
       }

   }
   if (aSomZ<0)
      aPlan.Revert();



   if (0)
   {
      ElRotation3D  aRP2E = aPlan.CoordPlan2Euclid();
      std::vector<double> aVZ;
      for (int aK=0 ; aK<int(aVPts.size()) ; aK++)
      {
         Pt3dr aPP = aRP2E.ImRecAff(aVPts[aK]);
         aVZ.push_back(aPP.z);
         //std::cout << aPP.z << "\n";
      }

      std::sort(aVZ.begin(),aVZ.end());
      int aNbPerc = 20;
      for (int aK=0 ; aK<= aNbPerc ; aK++)
      {
          double aPerc =  (100.0*aK)/aNbPerc ;
          std::cout << aPerc << " " << ValPercentile(aVZ,aPerc) << "\n";
      }
   }
   return aPlan;
}

ElSeg3D   cAppliApero::PointeMono2Seg(const cAperoPointeMono & aPM)
{
    return PoseFromName(aPM.Im())->CurCam()->F2toRayonR3(aPM.Pt());
}

Pt3dr    cAppliApero::PointeMonoAndPlan2Pt(const cAperoPointeMono & aPM,const cElPlan3D & aPlan)
{
   return aPlan.Inter(PointeMono2Seg(aPM));
}

/*
Pt3dr  cAppliApero::PointeStereo2Pt(const cAperoPointeStereo & aPS)
{
    const CamStenope * aCS1 = PoseFromName(aPS.Im1())->CurCam();
    const CamStenope * aCS2 = PoseFromName(aPS.Im2())->CurCam();
    return aCS1->PseudoInter(aPS.P1(),*aCS2,aPS.P2());
}
*/

void AjustNormalSortante(bool Sortante,Pt3dr & aNorm, const ElCamera * aCS1,const Pt2dr &aPIm)
{
// std::cout << "AjustNormalSortante " << Sortante << " " << aNorm << " " << aCS1->F2toDirRayonR3(aPIm) << " " << scal(aNorm,aCS1->F2toDirRayonR3(aPIm)) << "\n";
    // Si Sortante, le produit scal Norm/DirVisee  doit etre <0, donc ((>0) == Sortante) => chg sign
    if (   (scal(aNorm,aCS1->F2toDirRayonR3(aPIm)) >0 ) == Sortante)
         aNorm = - aNorm;
// std::cout << "AjustNormalSortante "  << aNorm  << "\n";
}

void cAppliApero::BasculePlan
     (
        const cBasculeLiaisonOnPlan & aBL,
        cSetName &            aSelectorEstim,
        cElRegex &                    aSelectorApply
     )
{
   const cOrientInPlane * anOIP =0;
   cSetOfMesureAppuisFlottants aSMAF;
   if (aBL.OrientInPlane().IsInit())
   {
        anOIP = &(aBL.OrientInPlane().Val());
        aSMAF = StdGetMAF(anOIP->FileMesures());
   }


   cElPlan3D * aPlanFromNamedPt=0;


   // On regarde si le plan peut etre defini par les points 3D de nom "Plan*"
   if (1) // ( AVANT MPD MM() mais utile a tout le monde ?)
   {
       std::map<std::string,int> aCptName;
       std::list<std::string>    aLName;
       static cElRegex  aRegPlan("Plan[0-9]*",10);
       for (   std::list<cMesureAppuiFlottant1Im>::const_iterator itMAF=aSMAF.MesureAppuiFlottant1Im().begin();
               itMAF!=aSMAF.MesureAppuiFlottant1Im().end() ;
               itMAF++
           )
       {
           for 
           (
                std::list<cOneMesureAF1I>::const_iterator itMes=itMAF->OneMesureAF1I().begin();
                itMes !=itMAF->OneMesureAF1I().end();
                itMes++
           )
           {
                std::string aName = itMes->NamePt();
                if (aRegPlan.Match(itMes->NamePt()))
                {
                    aCptName[aName]++;
                    if (aCptName[aName]==2)
                       aLName.push_back(aName);
                }
           }
       }
       if (aLName.size() >=3)
       {
           std::vector<Pt3dr> aVPt;
           for (std::list<std::string>::const_iterator itN=aLName.begin() ; itN!=aLName.end() ; itN++)
           {
               Pt3dr  aPt    = CreatePtFromPointeMonoOrStereo(aSMAF,*itN,(cElPlan3D*)0);
               aVPt.push_back(aPt);
           }
           aPlanFromNamedPt = new cElPlan3D(aVPt,0);
       }
   }


   cElPlan3D aPlan=  aPlanFromNamedPt ? (*aPlanFromNamedPt) :EstimPlan(aBL.EstimPl(), aSelectorEstim,(const char *)0);

   //std::cout << "ZZZ::cAppliApero::BasculePlan  \n";
   ElRotation3D  aRP2E = aPlan.CoordPlan2Euclid();
   ElMatrix<REAL>  aMP2E = aRP2E.Mat();

   double aRatio = 1;
   Pt3dr aPOrig =  aRP2E.ImAff(Pt3dr(0,0,0));


   if (aBL.OrientInPlane().IsInit())
   {
        // const cOrientInPlane & anOIP = aBL.OrientInPlane().Val();
        // cSetOfMesureAppuisFlottants aSMAF = StdGetMAF(anOIP.FileMesures());

         Pt3dr aP1 = aRP2E.ImAff(Pt3dr(1,0,0));
         aP1 = CreatePtFromPointeMonoOrStereo(aSMAF,"Line1",&aPlan,"USEDEF",&aP1);

         Pt3dr aP2 = aRP2E.ImAff(Pt3dr(2,0,0));
         aP2 = CreatePtFromPointeMonoOrStereo(aSMAF,"Line2",&aPlan,"USEDEF",&aP2);

         aPOrig    = CreatePtFromPointeMonoOrStereo(aSMAF,"Origine",&aPlan,"USEDEF",&aP1);

/*
        cAperoPointeMono aPt1 =  CreatePointeMono(aSMAF,"Line1");
        Pt3dr aP1 = PointeMonoAndPlan2Pt(aPt1,aPlan);
        Pt3dr aP2 = PointeMonoAndPlan2Pt(CreatePointeMono(aSMAF,"Line2"),aPlan);
        aPOrig = PointeMonoAndPlan2Pt(CreatePointeMono(aSMAF,"Origine",&aPt1),aPlan);
*/


        double aD = anOIP->DistFixEch().ValWithDef(0);

        if (aD >0)
        {
            // Pt3dr aPE1 =  CpleIm2PTer(CreatePointeStereo(aSMAF,"Ech1"));
            // Pt3dr aPE2 =  CpleIm2PTer(CreatePointeStereo(aSMAF,"Ech2"));
             Pt3dr aPE1 =  CreatePtFromPointeMonoOrStereo(aSMAF,"Ech1",0);
             Pt3dr aPE2 =  CreatePtFromPointeMonoOrStereo(aSMAF,"Ech2",0);
             aRatio = aD/ euclid(aPE1-aPE2) ;
        }

        Pt3dr aNorm = aRP2E.ImVect(Pt3dr(0,0,1));

        std::vector<cOneMesureAF1I>  aVM = GetMesureOfPts(aSMAF,"Line1");
        if (aVM.size())
        {
           cPoseCam * aPose1 = PoseFromName  (aVM[0].NamePt());
           AjustNormalSortante(true,aNorm,aPose1->CurCam(),aVM[0].PtIm());
        }

// void AjustNormalSortante(Pt3dr & aNorm, const ElCamera * aCS1,const Pt2dr &aPIm)


        Pt3dr aDirX = vunit(aP2-aP1);
        // Pt3dr aDirY = aNorm ^ aDirX;

        Pt3dr aV[3];
        ElMatrix<double>::PermRot(anOIP->AlignOn().Val(),aV);


        aMP2E = ComplemRotation(aV[0],aV[1],aNorm,aDirX);
        // CpleIm2PTer
        // aHF.VecFOH().push_back(CreatePointeMono(aSMAF,aA2P.NameP1()));

   }

   cSolBasculeRig  aSBR
                   (
                       aPOrig,
                       Pt3dr(0,0,0),
                       gaussj(aMP2E),
                       aRatio
                   );

   for
   (
       tDiPo::const_iterator itD=mDicoPose.begin();
       itD!=mDicoPose.end();
       itD++
   )
   {
       cPoseCam * aPC = itD->second;
       if (
                aSelectorApply.Match(aPC->Name())
             && (aPC->RotIsInit())
          )
       {
//  -- SENS DE LA COMBINAISON :
//  Rc (Cam) = Monde
//  aRP2E (Plan) = Monde
//
//       (-1)
//  {RP2E     * Rc}(Cam)  = Plan
//
// std::cout << euclid(aPC->CurRot().tr()) <<  " " <<  aPC->CurRot().tr()  << "\n";
//            std::cout << "BASCULE PLAN DONE FOR " << aPC->Name() << "\n";
// std::cout << euclid(aPC->CurRot().tr()) <<  " " << aPC->CurRot().tr() << "\n\n";
            aPC->SetBascRig(aSBR);
       }
   }

   delete   aPlanFromNamedPt;
}



/*
   if (aBL.FixEchelle())
   {
   }

   if (aBL.FixOrientation())
   {
   }

*/



     //---------------------------------------------
     //    cAppliApero::Bascule
     //---------------------------------------------


  

void cAppliApero::Bascule(const cBasculeOrientation & aBO,bool CalledAfter)
{

   if (CalledAfter != aBO.AfterCompens().Val()) return;
   // cElRegex aSelectorEstim(aBO.PatternNameEstim().Val(),10);
   cSetName *  aSelectorEstim = mICNM->KeyOrPatSelector(aBO.PatternNameEstim().Val());
// std::cout << "PAT= " << aBO.PatternNameEstim().Val() << "\n";
   cElRegex aSelectorApply(aBO.PatternNameApply().Val(),10);

   cSolBasculeRig aSol = cSolBasculeRig::Id();

   if (aBO.BasculeOnPoints().IsInit())
   {
     aSol = BasculePoints(aBO.BasculeOnPoints().Val(),*aSelectorEstim,aSelectorApply);
   }
   else if (aBO.BasculeLiaisonOnPlan().IsInit())
   {
      BasculePlan(aBO.BasculeLiaisonOnPlan().Val(),*aSelectorEstim,aSelectorApply);
   }

   if (aBO.FileExportDir().IsInit())
      MakeFileXML(EL2Xml(aSol),mDC+aBO.FileExportDir().Val());
   if (aBO.FileExportInv().IsInit())
      MakeFileXML(EL2Xml(aSol.Inv()),mDC+aBO.FileExportInv().Val());
}

#if (0)
void cAppliApero::Bascule(const cBasculeOrientation & aBO)
{
}
#endif

/****************************************************************/
/*                                                              */
/*                    ECHELLE ECHELLE ECHELLE ECHELLE           */
/*                                                              */
/****************************************************************/

double  cAppliApero::StereoGetDistFE(const cStereoFE & aSFE)
{
  ELISE_ASSERT(aSFE.HomFE().size()==2,"cAppliApero::StereoGetDistFE");
  Pt3dr aP1 = CpleIm2PTer(aSFE.HomFE()[0]);
  Pt3dr aP2 = CpleIm2PTer(aSFE.HomFE()[1]);

  return euclid(aP1-aP2);
}

void cAppliApero::FixeEchelle(const cFixeEchelle & aFE)
{
   double aDTer = 0;

   if (aFE.StereoFE().IsInit())
   {
     aDTer = StereoGetDistFE(aFE.StereoFE().Val());
   }
   else if (aFE.FEFromFile().IsInit())
   {
         const cApero2PointeFromFile&  aA2P  = aFE.FEFromFile().Val();
         cSetOfMesureAppuisFlottants aSMAF = StdGetMAF(aA2P.File());

         cStereoFE  aSE;
         aSE.HomFE().push_back(CreatePointeStereo(aSMAF,aA2P.NameP1()));
         aSE.HomFE().push_back(CreatePointeStereo(aSMAF,aA2P.NameP2()));

         aDTer = StereoGetDistFE(aSE);
   }
   else
   {
       ELISE_ASSERT(false,"Incoherence in cAppliApero::FixeEchelle");
   }

   double aMult = aFE.DistVraie()  / aDTer;

   for
   (
       tDiPo::const_iterator itD=mDicoPose.begin();
       itD!=mDicoPose.end();
       itD++
   )
   {
       cPoseCam * aPC = itD->second;
       if ( aPC->RotIsInit())
       {
            ElRotation3D  aR = aPC->CurRot();
            aPC->PCSetCurRot(ElRotation3D(aR.tr()*aMult,aR.Mat(),true));
       }
   }

}

/****************************************************************/
/*                                                              */
/*                    ROTATION PLANE                            */
/*                                                              */
/****************************************************************/

Pt2dr  cAppliApero::GetVecHor(const cHorFOP &  aH)
{
  ELISE_ASSERT(aH.VecFOH().size()==2,"cAppliApero::StereoGetDistFE");
  Pt3dr aP1 =  PImetZ2PTer(aH.VecFOH()[0],aH.Z().Val());
  Pt3dr aP2 =  PImetZ2PTer(aH.VecFOH()[1],aH.Z().Val());


  return Pt2dr(aP2.x-aP1.x,aP2.y-aP1.y);
}

std::string AddDirIfRequired(const std::string & aDir,const std::string & aFile);



cDicoAppuisFlottant cAppliApero::StdGetDAF(const std::string & aName)
{
   return StdGetObjFromFile<cDicoAppuisFlottant>
          (
              AddDirIfRequired(mDC,aName),
              //mDC+aName,
              StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
              "DicoAppuisFlottant",
              "DicoAppuisFlottant"
          );

}


cSetOfMesureAppuisFlottants cAppliApero::StdGetMAF(const std::string & aName)
{
   return StdGetObjFromFile<cSetOfMesureAppuisFlottants>
          (
              AddDirIfRequired(mDC,aName),
              //mDC+aName,
              StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
              "SetOfMesureAppuisFlottants",
              "SetOfMesureAppuisFlottants"
          );

}

cMesureAppuiFlottant1Im cAppliApero::StdGetOneMAF(const std::string & aName)
{
   return StdGetObjFromFile<cMesureAppuiFlottant1Im>
          (
              // mDC+aName,
              AddDirIfRequired(mDC,aName),
              StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
              "MesureAppuiFlottant1Im",
              "MesureAppuiFlottant1Im"
          );

}

Pt3dr cAppliApero::CreatePtFromPointeMonoOrStereo
      (
            const cSetOfMesureAppuisFlottants & aMAF,
            const std::string & aNamePt,
            const cElPlan3D  * aPlan,
            const std::string & aNameSec,
            const Pt3dr * aPDef
      )
{
   Pt3dr aRes(0,0,0);
   std::vector<cOneMesureAF1I>  aV = GetMesureOfPts(aMAF,aNamePt);  // !!! LES IMAGES SONT DANS NamePt

   if (aV.size()==0)
   {
       aV = GetMesureOfPts(aMAF,aNameSec);   
   }

   if (int(aV.size()) <1)
   {
      if (aPDef!=0)
          return *aPDef; 
      std::cout << "For name point = " << aNamePt << "\n";
      ELISE_ASSERT(false,"cAppliApero::CreatePtFromPointe No Pointe");
   }

   std::vector<ElSeg3D> aVS;
   for (int aK=0 ; aK<int(aV.size()) ; aK++)
   {
       cOneMesureAF1I aPM = aV[aK];
       aVS.push_back(PoseFromName(aPM.NamePt())->CurCam()->F2toRayonR3(aPM.PtIm()));
   }

   if (aV.size()==1)
   {
      if (aPlan==0)
      {
         std::cout << "For name point = " << aNamePt << "\n";
         ELISE_ASSERT(false,"cAppliApero::CreatePtFromPointe 1 Pointe ");
      }
      return aPlan->Inter(aVS[0]);
   }

   return  ElSeg3D::L2InterFaisceaux(0,aVS);
}



cAperoPointeMono cAppliApero::CreatePointeMono
                 (
                     const cSetOfMesureAppuisFlottants & aMAF,
                     const std::string & aNamePt,
                     const cAperoPointeMono * aDef
                 )
{
   std::vector<cOneMesureAF1I>  aV = GetMesureOfPts(aMAF,aNamePt);

   if (int(aV.size()) <1)
   {
       if (aDef)
       {
          return * aDef;
       }
       std::cout << "For Pt = " << aNamePt << "\n";
       ELISE_ASSERT(false,"Point is not measured  (CreatePointeMono)");
   }

   if (int(aV.size())>1)
   {
       cElWarning::GeomPointTooManyMeasured.AddWarn
       (
           "Nb Mes =" + ToString(int(aV.size())) + " For point "  +aNamePt + " wil use image " + aV[0].NamePt(),
           __LINE__,
           __FILE__
       );
   }

   cAperoPointeMono aRes;
   aRes.Pt() = aV[0].PtIm();
   aRes.Im() = aV[0].NamePt();
   return aRes;
}

cAperoPointeStereo cAppliApero::CreatePointeStereo(const cSetOfMesureAppuisFlottants & aMAF,const std::string & aNamePt)
{
   std::vector<cOneMesureAF1I>  aV = GetMesureOfPts(aMAF,aNamePt);

   if (int(aV.size()) <2)
   {
       std::cout << "For Pt = " << aNamePt << "\n";
       ELISE_ASSERT(false,"Point is not measured  twice (CreatePointeStereo)");
   }

   if (int(aV.size())>2)
   {
       cElWarning::GeomPointTooManyMeasured.AddWarn
       (
           "Nb Mes =" + ToString(int(aV.size())) + " For point "  +aNamePt + " wil use images "
          + aV[0].NamePt() +  " and " +  aV[1].NamePt() ,
           __LINE__,
           __FILE__
       );
   }

   cAperoPointeStereo aRes;
   aRes.P1() = aV[0].PtIm();
   aRes.Im1() = aV[0].NamePt();
   aRes.P2() = aV[1].PtIm();
   aRes.Im2() = aV[1].NamePt();
   return aRes;
}



void cAppliApero::FixeOrientPlane(const cFixeOrientPlane & aFOP)
{
   Pt2dr aVTer0 (0,0);

   if ( aFOP.HorFOP().IsInit())
   {
      aVTer0 = GetVecHor(aFOP.HorFOP().Val());
   }
   else if (aFOP.HorFromFile().IsInit())
   {
         const cApero2PointeFromFile&  aA2P  = aFOP.HorFromFile().Val();
         cSetOfMesureAppuisFlottants aSMAF = StdGetMAF(aA2P.File());

         cHorFOP  aHF;
         aHF.Z().SetVal(0);
         aHF.VecFOH().push_back(CreatePointeMono(aSMAF,aA2P.NameP1()));
         aHF.VecFOH().push_back(CreatePointeMono(aSMAF,aA2P.NameP2()));

         aVTer0 = GetVecHor(aHF);
   }
   else
   {
       ELISE_ASSERT(false,"Incoherence in cAppliApero::FixeEchelle");
   }

   Pt2dr aCFinal0 = aFOP.Vecteur();



   ElMatrix<REAL> aMat = ComplemRotation
                         (
                              PZ0(aVTer0),
                              PZ0(aVTer0 * Pt2dr(0,1)),
                              PZ0(aCFinal0),
                              PZ0(aCFinal0 * Pt2dr(0,1))
                         );
    ElRotation3D aR(Pt3dr(0,0,0),aMat,true);



   for
   (
       tDiPo::const_iterator itD=mDicoPose.begin();
       itD!=mDicoPose.end();
       itD++
   )
   {
       cPoseCam * aPC = itD->second;
       if ( aPC->RotIsInit())
       {
// std::cout << aR.Mat() * (Pt3dr(0,0,1)) << "\n";
// std::cout << aPC->CurRot().Mat() * (Pt3dr(0,0,1)) << "\n";
            aPC->PCSetCurRot(aR*aPC->CurRot());
       }
   }
}

cAperoPointeMono  Pointe(const Pt2dr & aP,const std::string &aIm)
{
   cAperoPointeMono anAPM;

   anAPM.Pt() = aP;
   anAPM.Im() = aIm;

   return anAPM;
}

void cAppliApero::BasicFixeOrientPlane(const std::string & aName)
{
   Pt2dr aDir(1,0);
   Pt2dr aP0 =  PoseFromName(aName)->Calib()->SzIm() / 2.0;
   cFixeOrientPlane  aFOP;
   aFOP.Vecteur()=aDir;

   cHorFOP  aHF;
   aHF.Z().SetVal(0.0);
   aHF.VecFOH().push_back(Pointe(aP0,aName));
   aHF.VecFOH().push_back(Pointe(aP0+aDir,aName));
   aFOP.HorFOP().SetVal(aHF);

   FixeOrientPlane(aFOP);
}

/****************************************************************/
/*                                                              */
/*                    cArgVerifAero                             */
/*                                                              */
/****************************************************************/

    //-----------------
    //   cAVA_PtHS
    //-----------------

cAVA_PtHS::cAVA_PtHS
(
     Pt2dr aPIM,
     double aDZ,
     const std::vector<double> &  aPds,
     const cOnePtsMult & aPM
) :
    mPIM  (aPIM),
    mDZ   (aDZ),
    mPDS  (aPds),
    mPM   (&aPM)
{
}

bool operator < (const cAVA_PtHS & aPH1,const cAVA_PtHS & aPH2)
{
   return ElAbs(aPH1.mDZ) >  ElAbs(aPH2.mDZ);
}

    //-----------------
    //   cAVA_Residu
    //-----------------

cAVA_Residu::cAVA_Residu(const Pt2dr & aPt,const Pt2dr & aResidu) :
   mP   (aPt),
   mRes (aResidu)
{
}

    //-----------------
    //   cArgVerifAero
    //-----------------

void cArgVerifAero::AddResidu (const Pt2dr & aP1,const  Pt2dr & aRes)
{
   mRes.push_back(cAVA_Residu(aP1,aRes));
}

void cArgVerifAero::AddPImDZ
     (
         Pt2dr aPIM,
         double aDZ,
         const std::vector<double> & aVPds,
         const cOnePtsMult & aPM
     )
{
   Pt2dr aPR = Pt2dr(aPIM) / mResol;
   mW.draw_circle_loc
   (
       aPR,
       2.0,
       mW.pdisc()((aDZ>0) ? P8COL::red  : P8COL::blue)
   );

   Pt2di aPI = round_ni(aPR);
   if (mImZ.Inside(aPI))
   {
       mImZ.SetR(aPI,aDZ+mImZ.GetR(aPI));
       mImPds.SetR(aPI,1.0+mImPds.GetR(aPI));
   }

   if (ElAbs(aDZ) > mVA.SeuilTxt())
   {
      mPHS.push_back(cAVA_PtHS(aPIM,aDZ,aVPds,aPM));
   }
   // std::cout << aDZ << "\n";
}


cArgVerifAero::cArgVerifAero
(
     cAppliApero &           anAppli,
     Pt2di                   aSz,
     const cVerifAero  &     aVA,
     const std::string &     aPref,
     const std::string &     aPost
) :
  mAppli (anAppli),
  mVA    (aVA),
  mName  (aPost),
  mNameS (aPref + "-Sign-" + aPost),
  mNameR (aPref + "-Regul-" + aPost),
  mNameB (aPref + "-Brut-" + aPost),
  mNameT (aPref + "-Txt-" + aPost),
  mResol (aVA.Resol()),
  mSz    (round_ni(Pt2dr(aSz)/mResol)),
  mW     ("toto.tif", GlobPal(),mSz),
  mImZ   (mSz.x,mSz.y,0.0),
  mImPds (mSz.x,mSz.y,0.0),
  mPasR   (aVA.PasR()),
  mPasB   (aVA.PasB())
{
   ELISE_COPY(mW.all_pts(),0,mW.ogray());
}


Fonc_Num Filter(Im2D_REAL4 aI,int aNb,double aF)
{
    Fonc_Num aRes = aI.in(0);
    for (int aK=0 ; aK<aNb; aK++)
       aRes = canny_exp_filt(aRes,aF,aF);

   return aRes;
}

const cVerifAero & cArgVerifAero::VA() const
{
   return mVA;
}

cArgVerifAero::~cArgVerifAero()
{


   if (mVA.TypeVerif() == eVerifDZ)
   {
       mW.make_tif(mNameS.c_str());
       int aNbIter= 3;
       double aFact = 0.90;

       Tiff_Im::Create8BFromFonc
       (
            mNameR,
            mSz,
            Max(0,Min(255,128 + (Filter(mImZ,aNbIter,aFact) / Max(1e-6, Filter(mImPds,aNbIter,aFact))) / mPasR))
       );

       Tiff_Im::Create8BFromFonc
       (
            mNameB,
            mSz,
            Max(0,Min(255,128 + mImZ.in()/mPasB))
       );

       FILE * aFP = ElFopen(mNameT.c_str(),"w");
       std::sort(mPHS.begin(),mPHS.end());
       for (int aKPM=0 ; aKPM<int(mPHS.size()) ; aKPM++)
       {
           const cAVA_PtHS & aPHS = mPHS[aKPM];
           fprintf(aFP,"============ %lf ========\n",aPHS.mDZ);
           for (int aKP=0 ; aKP<int(aPHS.mPDS.size()) ; aKP++)
           {
                if (aPHS.mPDS[aKP] >0)
                {
                    fprintf(aFP,"  -%s\n",aPHS.mPM->GenPoseK(aKP)->Name().c_str());
                }
           }
       }
       ElFclose(aFP);
  }
  else if (mVA.TypeVerif()==eVerifResPerIm)
  {
       Tiff_Im aTF = Tiff_Im::StdConvGen(mAppli.DC()+mName,-1,false);
       ELISE_COPY
       (
          mW.all_pts(),
          StdFoncChScale(aTF.in_proj(),Pt2dr(0,0),Pt2dr(mResol,mResol)),
          mW.ogray()
       );
       for (int aKR=0 ; aKR<int(mRes.size()) ; aKR++)
       {
           if (euclid(mRes[aKR].mRes)<mVA.PasB())
           {
              mW.draw_seg
              (
                  mRes[aKR].mP/mResol,
                  mRes[aKR].mP/mResol+mRes[aKR].mRes*mVA.PasR(),
                  mW.pdisc()(P8COL::red)
              );
              mW.draw_circle_abs(mRes[aKR].mP/mResol,1.0,mW.pdisc()(P8COL::green));
           }
       }
       mW.make_tif(mNameS.c_str());
  }

}


  // Fait toute les verif

void cAppliApero::VerifAero(const cVerifAero & aVA)
{
   cElRegex  anAutom(aVA.PatternApply(),10);

   for (int aKP=0; aKP<int(mVecPose.size()) ; aKP++)
   {
       const std::string & aNameP = mVecPose[aKP]->Name();
       if (anAutom.Match(aNameP))
       {
            cObsLiaisonMultiple * anOLM = PackMulOfIndAndNale(aVA.IdBdLiaison(),aNameP);
            VerifAero(aVA,mVecPose[aKP],*anOLM);
       }
   }

}

  // Fait la verif pour une image

void cAppliApero::VerifAero
     (
          const cVerifAero & aVA,
          cPoseCam *            aPC,
          cObsLiaisonMultiple & anOLM
     )
{
    cPonderationPackMesure aPPM = aVA.Pond();
    aPPM.Add2Compens().SetVal(false);
    cCalibCam * aCC = aPC->Calib();

    cArgVerifAero anArgVA
                  (
                      *this,
                      aCC->SzIm(),
                      aVA,
                      mDC + aVA.Prefixe(),
                     aPC->Name()
                  );

    cStatObs aSO(false);
    anOLM.AddObsLM(aPPM,0,0,&anArgVA,aSO,0);

}


/*****************************************************/
/*                                                   */
/*           Bascule de block                        */
/*                                                   */
/*****************************************************/

cParamBascBloc::cParamBascBloc()  :
   mSomInvNb (0.0)
{
}
   // cRansacBasculementRigide aBasc;

void cAppliApero::ResetNumTmp(const std::vector<cPoseCam *> & aVP,int aNumInit,int aNumNonInit)
{
    for (int aKP=0 ; aKP<int(aVP.size()) ; aKP++)
    {
       cPoseCam & aPC =  *(aVP[aKP]);
       aPC.NumTmp() =  (aPC.RotIsInit() ? aNumInit : aNumNonInit);
    }
}
void cAppliApero::ResetNumTmp(int aNumI,int aNumNonI)
{
   ResetNumTmp(mVecPose,aNumI,aNumNonI);
}

double  BSurH(const std::vector<ElSeg3D> & aVS,const std::vector<double>  * aVPds=0)
{
   Pt3dr aSomT(0,0,0);
   Pt3dr aSomT2(0,0,0);
   double aSP = 0;
   double aSP2 = 0;
   for (int aK=0 ; aK<int(aVS.size()) ; aK++)
   {
          double aPds = aVPds ? (*aVPds)[aK] : 1.0 ;
          Pt3dr aT = aVS[aK].TgNormee();
          aSomT = aSomT + aT*aPds;
          aSomT2 = aSomT2 + Pcoord2(aT) *aPds;
          aSP += aPds;
          aSP2 += ElSquare(aPds);
   }
   aSomT = aSomT / aSP;
   aSomT2 = aSomT2 / aSP;
   aSomT2 = aSomT2 - Pcoord2(aSomT);
      // Ce debiaisement est necessaire, par exemple si tous les poids sauf 1 sont
      // presque nuls
   double aDebias = 1 - aSP2/ElSquare(aSP);
   ELISE_ASSERT(aDebias>0,"Singularity in cManipPt3TerInc::CalcPTerInterFaisceauCams ");
   aSomT2 =  aSomT2/ aDebias;

   double aEc2 = aSomT2.x+aSomT2.y+aSomT2.z;
   if (aEc2 <= -1e-7)
   {
           std::cout << "EC2 =" << aEc2 << "\n";
          ELISE_ASSERT(aEc2>-1e-7,"Singularity in cManipPt3TerInc::CalcPTerInterFaisceauCams ");
   }
   aEc2 = sqrt(ElMax(0.0,aEc2) );
      // Adaptation purement heuristique
   return 1.35 * aEc2;

}



void  cAppliApero::BlocBasculeOneWay
                   (
                        cParamBascBloc &     aBlc,
                        const std::vector<cPoseCam *> & aVPose1,
                        std:: vector<Pt3dr> &           aVPts1,
                        int                             aNum1,
                        const std::vector<cPoseCam *> & aVPos2,
                        std:: vector<Pt3dr> &           aVPts2,
                        int                             aNum2,
                        const std::string & anIndPts
                   )
{
   double aSeuilBSurH = 1e-2;
   for (int aKPos1 = 0 ; aKPos1<int(aVPose1.size() ) ; aKPos1++)
   {
        cObsLiaisonMultiple * anOLM = PackMulOfIndAndNale(anIndPts,aVPose1[aKPos1]->Name());
        const std::vector<cOnePtsMult *> &  aVMul = anOLM->VPMul();


        for (int aKPm=0 ; aKPm<int(aVMul.size()) ; aKPm++)
        {
            cOnePtsMult * aPM = aVMul[aKPm];
            cOneCombinMult * aCOM = aPM->OCM();
            const std::vector<cGenPoseCam *> & aVP =  aCOM->GenVP();

            int aNb1=0;
            int aNb2=0;
  //  Recherche rapide des mesures potentiellement valides
            for (int aKP=0 ; aKP<int (aVP.size()) ; aKP++)
            {
                  cGenPoseCam & aPC = *(aVP[aKP]);
                  if (aPC.NumTmp() == aNum1) aNb1++;
                  if (aPC.NumTmp() == aNum2) aNb2++;
            }

// if (aNb1 && aNb2) std::cout << " NB " << aNb1 << " " << aNb2 << "\n";

            if ((aNb1 >=2) && (aNb2 >=2))
            {
                std::vector<ElSeg3D> aV1;
                std::vector<ElSeg3D> aV2;
                const cNupletPtsHomologues & aNP = aPM->NPts();
                for (int aKP=0 ; aKP<int (aVP.size()) ; aKP++)
                {
                      cGenPoseCam & aPC = *(aVP[aKP]);
                      const cBasicGeomCap3D * aCS =   aPC.GenCurCam ();
                      if (aPC.NumTmp() == aNum1) aV1.push_back(aCS->Capteur2RayTer(aNP.PK(aKP)));
                      if (aPC.NumTmp() == aNum2) aV2.push_back(aCS->Capteur2RayTer(aNP.PK(aKP)));

                      double aBH1 = BSurH(aV1);
                      if (aBH1 > aSeuilBSurH)
                      {
                           double aBH2 = BSurH(aV2);
                           if (aBH2 > aSeuilBSurH)
                           {
                                aVPts1.push_back(ElSeg3D::L2InterFaisceaux(0,aV1));
                                aVPts2.push_back(ElSeg3D::L2InterFaisceaux(0,aV2));
                                aBlc.mBsH.push_back(ElMin(aBH1,aBH2));
                                aBlc.mSomInvNb += 1.0 / (aNb1+aNb2);
                           }
                      }
                }
            }
        }

   }

}




void    cAppliApero::PrepareBlocBascule
        (
              cParamBascBloc &                 aRes,
              const std::vector<cPoseCam *> & aVP1,
              const std::vector<cPoseCam *> & aVP2,
              const std::string &             anInd
        )
{

    ResetNumTmp(-1,-1);
    ResetNumTmp(aVP1,0,-1);
    ResetNumTmp(aVP2,1,-1);

    BlocBasculeOneWay(aRes,aVP1,aRes.mP1,0,aVP2,aRes.mP2,1,anInd);
    BlocBasculeOneWay(aRes,aVP2,aRes.mP2,1,aVP1,aRes.mP1,0,anInd);

}



void cAppliApero::BasculeBloc
     (
              const std::vector<cPoseCam *> & aVP1,
              const std::vector<cPoseCam *> & aVP2,
              const std::string &             anInd
     )
{
    cParamBascBloc  aPBB;
    PrepareBlocBascule(aPBB,aVP1,aVP2,anInd);

    std::cout << "BBBBloc " << aPBB.mSomInvNb << " " <<  aPBB.mP1.size() << "\n";
}



std::vector<cPoseCam *> cAppliApero::PoseOfPattern( const std::string & aKeyPat)
{
   std::vector<cPoseCam *>  aRes;
   cSetName *  aSelectorEstim = mICNM->KeyOrPatSelector(aKeyPat);

   for (int aKP=0 ; aKP<int(mVecPose.size()) ; aKP++)
   {
      if (aSelectorEstim->IsSetIn(mVecPose[aKP]->Name()))
      {
         aRes.push_back(mVecPose[aKP]);
      }
   }

   return aRes;
}


void  cAppliApero::BasculeBloc(const cBlocBascule & aBB)
{
   std::vector<cPoseCam *> aVP1 = PoseOfPattern(aBB.Pattern1());
   std::vector<cPoseCam *> aVP2 = PoseOfPattern(aBB.Pattern2());

   BasculeBloc(aVP1,aVP2,aBB.IdBdl());
}







/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
