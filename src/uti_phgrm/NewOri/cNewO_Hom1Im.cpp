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

#include "NewOri.h"

#define HOM_NbMinPTs 5


class cOrHom_AttrSom;
class cOrHom_AttrASym;
class cOrHom_AttrArc;
class cAppli_Hom1Im;

typedef  ElSom<cOrHom_AttrSom,cOrHom_AttrArc>         tSomGT;
typedef  ElArc<cOrHom_AttrSom,cOrHom_AttrArc>         tArcGT;
typedef  ElSomIterator<cOrHom_AttrSom,cOrHom_AttrArc> tItSGT;
typedef  ElArcIterator<cOrHom_AttrSom,cOrHom_AttrArc> tItAGT;
typedef  ElGraphe<cOrHom_AttrSom,cOrHom_AttrArc>      tGrGT;
typedef  ElSubGraphe<cOrHom_AttrSom,cOrHom_AttrArc>   tSubGrGT;


double PropPond(std::vector<Pt2df> &  aV,double aProp,int * aKMed=0);


/************************************************/
/*                                              */
/*         cOrHom_AttrSom                       */
/*                                              */
/************************************************/

class cOneSolSim
{
     public :
         cOneSolSim(tSomGT * aSom,const ElRotation3D & aRot,int aNbPts,double aDist) :
            mRot (aRot),
            mNbPts (aNbPts),
            mDist  (aDist),
            mSom   (aSom)
         {
         }

         ElRotation3D  mRot;
         int           mNbPts;
         double        mDist;
         tSomGT        *mSom;
};

class cOrHom_AttrSom
{
     public :
        void OneSensEqHomRig(double aPds,const std::vector<int> & aVInd,const Pt2dr & aP1, const Pt2dr & aP2,const std::vector<cElHomographie> & aVH1To2,const cElHomographie &aH1To2Init,double aEps);


        cOrHom_AttrSom(int aNum,const std::string & aName,cAppli_Hom1Im &);
        cOrHom_AttrSom();
        const std::string & Name() const  {return mName;}
        cAppli_Hom1Im & Appli() {return *mAppli;}
        void AddSol(const cOneSolSim & aSol);
        void ExportGps();
        int    NbPtsMax() const {return mNbPtsMax;}
        double DistPMax() const {return mDistPMax;}
        

        // Zone Homographie

        void AddObsLin2Var(int aKV1,double aCoef1,int aKV2,double aCoef2,double aVal,double aPds,bool CoordLoc);
        void AddObsGround(double aPds,bool CoordLoc);
        void AddObsFixAffinite(bool CoordLoc);
        void AddObsFixSimil(bool CoordLoc);

        void AddEqH12(tArcGT & anArc, double aPds,bool ModeAff,bool CoordLoc);
        L2SysSurResol &  Sys();
        Pt2dr   SetSol(double * aSolGlob,bool CoordLoc);
        Pt3dr & GPS() {return    mGPS;}
        const cElHomographie  & CurHom() {return mCurHom;}

        // Homographie courante avec une variation Epsilon du parametre K
        cElHomographie HomogPerturb(int aKParam,double aEpsil);

        Pt3dr TestSol(tArcGT & anArc);
        void SaveSolHom(int aKEtape);
     private :
        void AddOnePEq12(bool IsX,const Pt2dr & aP1,cOrHom_AttrSom & aS2,const Pt2dr & aP2,double aPds,bool ModeAff);
        void   AddObsFix(int aKVal,double aVal,double aPds,bool CoordLoc);
        void Save(const std::string & aNameOri,const ElRotation3D &);
        std::string mName;
        cAppli_Hom1Im * mAppli;
        cNewO_OneIm * mI;
        Pt3dr    mGPS;

        int                      mNbPtsMax;
        double                   mDistPMax;
        std::vector<cOneSolSim>  mVSol;
        //  Zone de param pour homographie
        // L'homographie inconnue est t.q.  Hom(I,I) = (X,Y) I,J "pixel", X,Y Terrain
        int                       mNum;
        int                       mN0Hom;
        int                       mN1Hom;
        double                    mCurParH[8];
        cElHomographie            mCurHom;
};


class cOrHom_AttrArcSym
{
     public :
        cOrHom_AttrArcSym(const cXml_Ori2Im &);
        const cXml_Ori2Im & Xml() const {return mXmlO;}
        cElHomographie Hom(bool Dir);

        const double & PdsFin() const {return mPdsFin;}  // Module pour que Somme = NbSom
        const double & PdsRes() const {return mPdsRes;}  // Module par les residus
        const double & PdsNb() const {return mPdsNb;}    // Poids qui tient compte du nombre
        const double & ResH() const {return mResH;}

        void SetStdPds(double aRStd);
        void SetMulFin(double aRStd);

        const std::vector<Pt2dr> & VP1() {return mVP1;}
        const std::vector<Pt2dr> & VP2() {return mVP2;}
     private :
        cXml_Ori2Im     mXmlO;
        double          mPdsNb;
        double          mPdsRes;
        double          mPdsFin;
        double          mResH;
        cElHomographie  mHom12;
        std::vector<Pt2dr> mVP1;
        std::vector<Pt2dr> mVP2;
};

class cOrHom_AttrArc
{
    public :
        cOrHom_AttrArc(cOrHom_AttrArcSym * ,bool Direct);
        void GetDistribGaus(std::vector<Pt2dr> & aVPts,int aN);
        bool Direct() const {return mDirect;}
        cOrHom_AttrArcSym & ASym() {return *mASym;}
    private :
         cOrHom_AttrArcSym * mASym;
         bool                mDirect;
         cElHomographie      mHom;
};


class cSubGraphe_NO_Hom  : public tSubGrGT
{
    public :
        bool   inA(tArcGT & anArc)  {return anArc.attr().Direct();}
};


class cAppli_Hom1Im : public cCommonMartiniAppli
{
    public :
        cAppli_Hom1Im(int argc,char ** argv,bool ModePrelim,bool aModeGps);
        tGrGT & Gr() {return mGr;}
        bool ModeGps() const {return mModeGps;}
        cNewO_NameManager * NM() {return mNM;}
        std::string GpsOut() {return mGpsOut;}
        L2SysSurResol &  Sys();
        Pt2dr  ToW(const Pt2dr &aPIm) const
        {
              return  (aPIm-mGpsMin) * mScaleW;
        }
        void DrawSeg(const Pt2dr & aP1,const Pt2dr & aP2, int aCoul);
        void WClear();
        void WClik();

    private:
        tSomGT * AddAnIm(int aNum,const std::string & aName);
        tSomGT * GetAnIm(const std::string & aName,bool SVP);
        tArcGT *   AddArc(tSomGT * aS1,tSomGT * aS2);

        bool        mModePrelim;
        bool        mModeGps;
        std::string mPat;
        std::string mNameC;
        cElemAppliSetFile mEASF;
        std::map<std::string,tSomGT *> mMapS;
        std::vector<tSomGT *>          mVecS;
        tSomGT *                       mSomC;
        tGrGT                          mGr;
        cSubGraphe_NO_Hom              mSubAS;
        // tSubGrGT                       mSubAS;
        cNewO_NameManager *            mNM;
        std::string                    mGpsOut;
        int                            mNbInc;
        L2SysSurResol *                mSys;
        double                         mResiduStd;
        Pt2dr                          mGpsMin;
        Pt2dr                          mGpsMax;
        double                         mScaleW;
        Pt2dr                          mSzW;
        double                         mMulRab;  // Pour mode visuel uniquement
        int                            mNbEtape;
        bool                           mModeVert;
#if (ELISE_X11)
        Video_Win *                    mW;
#endif
        Pt3dr                          mCdgGPS;
        double                         mMulGps;
};


/************************************************/
/*                                              */
/*         cOrHom_AttrSom                       */
/*                                              */
/************************************************/



cOrHom_AttrSom::cOrHom_AttrSom(int aNum,const std::string & aName,cAppli_Hom1Im & anAppli) :
   mName  (aName),
   mAppli (&anAppli),
   mI     (new cNewO_OneIm  (*anAppli.NM(),aName)),
   mNbPtsMax  (-1),
   mNum       (aNum),
   mN0Hom     (mNum * 8),
   mN1Hom     (mN0Hom + 8),
   mCurHom    (cElHomographie::Id())
{
   if (mAppli->ModeGps())
   {
       mGPS =    mAppli->GpsVal(mI);
   }
}

cElHomographie HomogrFromCoords(double * aParam)
{
    cElComposHomographie aHX(aParam[0],aParam[1],aParam[2]);
    cElComposHomographie aHY(aParam[3],aParam[4],aParam[5]);
    cElComposHomographie aHZ(aParam[6],aParam[7],1.0);

    return   cElHomographie(aHX,aHY,aHZ);
}


cElHomographie  cOrHom_AttrSom::HomogPerturb(int aKParam,double aEpsil)
{
    double aParam[8] ;
    for (int aK=0 ; aK<8 ; aK++)
        aParam[aK]= mCurParH[aK];

    aParam[ aKParam] += aEpsil;
    return HomogrFromCoords(aParam);
}



Pt2dr   cOrHom_AttrSom::SetSol(double * aSolGlob,bool CoordLoc)
{
    for (int aN= mN0Hom ; aN<mN1Hom ; aN++)
    {
        if (CoordLoc)
        {
            // std::cout << "CccCc: " <<  mCurParH[aN-mN0Hom] << " "<<  aSolGlob[aN] << "\n";
            // mCurParH[aN-mN0Hom] += aSolGlob[aN] * 0.1;
            mCurParH[aN-mN0Hom] += aSolGlob[aN] ;
        }
        else 
        
            mCurParH[aN-mN0Hom] = aSolGlob[aN];
    }

/*
    cElComposHomographie aHX(mCurParH[0],mCurParH[1],mCurParH[2]);
    cElComposHomographie aHY(mCurParH[3],mCurParH[4],mCurParH[5]);
    cElComposHomographie aHZ(mCurParH[6],mCurParH[7],1.0);

    mCurHom =  cElHomographie(aHX,aHY,aHZ);
*/
    mCurHom = HomogrFromCoords(mCurParH);

    Pt2dr aG2(mGPS.x,mGPS.y);
    Pt2dr aRes = aG2-mCurHom(Pt2dr(0,0));
    return aRes;
}

cOrHom_AttrSom::cOrHom_AttrSom() :
   mAppli (0),
   mCurHom    (cElHomographie::Id())
{
}

void cOrHom_AttrSom::AddSol(const cOneSolSim & aSol)
{
   if (aSol.mNbPts > mNbPtsMax)
   {
       mNbPtsMax  = aSol.mNbPts;
       mDistPMax  = aSol.mDist;
   }
   mVSol.push_back(aSol);
}

void cOrHom_AttrSom::ExportGps()
{
    double aScoreMax= -1;
    cOneSolSim * aBestSol0 =nullptr;

    for (auto & aSol : mVSol)
    {
        double aScore = aSol.mNbPts ;//  * aSol.mDist;
        if (aScore> aScoreMax)
        {
             aScoreMax = aScore;
             aBestSol0 = & aSol;
        }
    }
    ELISE_ASSERT(aBestSol0!=nullptr,"cOrHom_AttrSom::ExportGps");

    std::cout << "Sol for : " << mName << " => " << aBestSol0->mSom->attr().Name() << "\n";

    Save(mAppli->GpsOut(),aBestSol0->mRot);
}

void  cOrHom_AttrSom::Save(const std::string & aPrefixOri,const ElRotation3D & aRot)
{
    CamStenope * aCal =  mAppli->NM()->CalibrationCamera(mName);
    std::string aNameCal = mAppli->NM()->ICNM()->StdNameCalib(aPrefixOri,mName);
    if (! ELISE_fp::exist_file(aNameCal))
    {
         ELISE_fp::MkDirSvp(DirOfFile(aNameCal));
         cCalibrationInternConique aCIC = aCal->ExportCalibInterne2XmlStruct(aCal->Sz());

         MakeFileXML(aCIC,aNameCal);
    }


    aCal->SetOrientation(aRot.inv());
    cOrientationConique anOC =  aCal->StdExportCalibGlob();
    anOC.Interne().SetNoInit();
    anOC.FileInterne().SetVal(aNameCal);

    std::string aNameOri =  mAppli->NM()->ICNM()->NameOriStenope(aPrefixOri,mName);

    MakeFileXML(anOC,aNameOri);

    /*
        anOC.Interne().SetNoInit();
   */

    // std::cout << aCal << " " << aCal->Focale() << aNameOri << "\n";

}

void cOrHom_AttrSom::SaveSolHom(int aKEtape)
{
    CamStenope * aCal =  mAppli->NM()->CalibrationCamera(mName);
    Pt2dr aSz = Pt2dr(aCal->Sz());
    double aStep = 0.3;

    std::vector<double> aPAF;
    CamStenopeIdeale aCamI(true,1.0,Pt2dr(0,0),aPAF);
    std::list<Pt2dr>  aLIm;
    std::list<Pt3dr>  aLTer;
    for (double aPx=0.1 ; aPx<= 0.901 ; aPx += aStep)
    {
         for (double aPy=0.1 ; aPy<= 0.901 ; aPy += aStep)
         {
             Pt2dr aPIm  =  aSz.mcbyc(Pt2dr(aPx,aPy));
             Pt2dr aPDir = aCal->F2toPtDirRayonL3(aPIm);
             aLIm.push_back(aPDir);
             Pt2dr aPTer = mCurHom(aPDir);
             aLTer.push_back(Pt3dr(aPTer.x,aPTer.y,0.0));
         }
    }

    double aEcart;
    ElRotation3D aR= aCamI.CombinatoireOFPA(true,1000,aLTer,aLIm,&aEcart);

    CamStenopeIdeale aCamII(true,1.0,Pt2dr(0,0),aPAF);
    aCamI.SetOrientation(aR);
    aCamII.SetOrientation(aR.inv());

    {
        Box2dr aBox(Pt2dr(0,0),aSz);
        Pt2dr  aCorn[4];
        aBox.Corners(aCorn);
        for (int aK=0 ; aK< 4 ; aK++)
        {
             Pt2dr aPDir = aCal->F2toPtDirRayonL3(aCorn[aK]);
             Pt2dr aP2Ter = mCurHom(aPDir);
             aCorn[aK] = aP2Ter;
        }
        for (int aK=0 ; aK< 4 ; aK++)
        {
            int aCoul = (aEcart<100) ? P8COL::blue : P8COL::red;
            mAppli->DrawSeg(aCorn[aK],aCorn[(aK+1)%4], aCoul);
        }
    }

    {
       double aD = 0;
       double aDI = 0;
       int aNb = 0;
       for (double aPx=0.1 ; aPx<= 0.901 ; aPx += aStep)
       {
         for (double aPy=0.1 ; aPy<= 0.901 ; aPy += aStep)
         {
             Pt2dr aPIm  =  aSz.mcbyc(Pt2dr(aPx,aPy));
             Pt2dr aPDir = aCal->F2toPtDirRayonL3(aPIm);
             Pt2dr aP2Ter = mCurHom(aPDir);
             Pt3dr aPTer(aP2Ter.x,aP2Ter.y,0);
             aD += euclid(aPDir-aCamI.R3toF2(aPTer));
             aDI += euclid(aPDir-aCamII.R3toF2(aPTer));
             aNb++;
         }
       }
    }
    
}
   


L2SysSurResol & cOrHom_AttrSom::Sys()
{
    return mAppli->Sys();
}

/*    Homographie

          a0 I + a1 J + a2         a3 I + a4 J + a5
      X = ----------------     Y = ----------------
          a6 I + a7 J +1           a6 I + a7 J +1
*/

void   cOrHom_AttrSom::AddObsFix(int aKVal,double aVal,double aPds,bool CoordLoc)
{
    std::vector<int> aVInd;
    aVInd.push_back(mN0Hom+aKVal);
    double aCoeff = 1.0;

    if (CoordLoc)
      aVal -= mCurParH[aKVal];

    Sys().GSSR_AddNewEquation_Indexe(0,0,0,aVInd,aPds,&aCoeff,aVal,NullPCVU);
}

void cOrHom_AttrSom::AddObsLin2Var(int aKV1,double aCoef1,int aKV2,double aCoef2,double aVal,double aPds,bool CoordLoc)
{
    std::vector<int> aVInd;
    std::vector<double> aVCoeff;

    aVInd.push_back(mN0Hom+aKV1);
    aVCoeff.push_back(aCoef1);

    aVCoeff.push_back(aCoef2);
    aVInd.push_back(mN0Hom+aKV2);

    if (CoordLoc)
    {
       aVal -= mCurParH[aKV1] *aCoef1 + mCurParH[aKV2] *aCoef2 ;
    }
    Sys().GSSR_AddNewEquation_Indexe(0,0,0,aVInd,aPds,&(aVCoeff[0]),aVal,NullPCVU);
}

void cOrHom_AttrSom::AddObsGround(double aPds,bool CoordLoc)
{
     /*  
         On fait l'appoximition que le faisceau partant de (0,0) est vertical

          mGPS.x = a2/1    mGPS.y = a5/1
     */
     AddObsFix(2,mGPS.x,aPds,CoordLoc);
     AddObsFix(5,mGPS.y,aPds,CoordLoc);
}

void cOrHom_AttrSom::AddObsFixAffinite(bool CoordLoc)
{
     AddObsFix(6,0,1.0,CoordLoc);
     AddObsFix(7,0,1.0,CoordLoc);
}


void cOrHom_AttrSom::AddObsFixSimil(bool CoordLoc)
{
   //   X =    a0 I + a1 J + a2       Y=  a3 I + a4 J + a5 
   //  ==>    a0-a4=0      a1+ a3 = 0
   AddObsLin2Var(0,1.0,4,-1.0,0.0,1e4,CoordLoc);
   AddObsLin2Var(1,1.0,3,+1.0,0.0,1e4,CoordLoc);
}





Pt3dr cOrHom_AttrSom::TestSol(tArcGT & anArc)
{
     cOrHom_AttrSom & aS2 = anArc.s2().attr();
     cOrHom_AttrArcSym & aA12 = anArc.attr().ASym();

     const std::vector<Pt2dr> & aVP1 =  aA12.VP1();
     const std::vector<Pt2dr> & aVP2 =  aA12.VP2();
     int aNbPts = aVP1.size();

     cElHomographie    aH1  = CurHom();
     cElHomographie    aH1I = aH1.Inverse();
     // const cElHomographie  &  aH2 = aS2.CurHom().Inverse();
     cElHomographie    aH2 = aS2.CurHom();
     cElHomographie    aH2I = aH2.Inverse();

     cElHomographie aH1To2 = aH2I * aH1;
     cElHomographie aH2To1 = aH1I * aH2;


     double aS=0;
     double aSI1=0;
     double aSI2=0;
     for (int aKp=0 ; aKp<aNbPts ; aKp++)
     {
         aS += euclid(aH1(aVP1[aKp]) - aH2(aVP2[aKp]));

         aSI1 += euclid(aH1To2(aVP1[aKp]) - aVP2[aKp]);
         aSI2 += euclid(aH2To1(aVP2[aKp]) - aVP1[aKp]);
     }
     return Pt3dr(aS,aSI1,aSI2) / aNbPts;
}

// PB


void cOrHom_AttrSom::AddEqH12(tArcGT & anArc, double aPdsGlob,bool ModeAff,bool CoordLoc)
{
     cOrHom_AttrSom & aS1 = *this;
     cOrHom_AttrSom & aS2 = anArc.s2().attr();
     cOrHom_AttrArcSym & aA12 = anArc.attr().ASym();

     const std::vector<Pt2dr> & aVP1 =  aA12.VP1();
     const std::vector<Pt2dr> & aVP2 =  aA12.VP2();
     int aNbPts = aVP1.size();

     double  aPds = (aPdsGlob * aA12.PdsFin()) / aNbPts;

     if (ModeAff)
     {
         ELISE_ASSERT(!CoordLoc,"Coord loc in affine mod");
         for (int aKp=0 ; aKp<aNbPts ; aKp++)
         {
             for (int aKx=0 ; aKx<2 ; aKx++)
                 AddOnePEq12((aKx==0),aVP1[aKp],aS2,aVP2[aKp],aPds,ModeAff);
         }
         return;
     }

     // Si ce n'est pas un mode lineaire, on aborde une linearisation par 
     // differences finies

     ELISE_ASSERT(CoordLoc,"No Coord loc in homogr mod");
     
     double aEpsil = 1e-4;
     
     cElHomographie aH1To2Init =  aS2.CurHom().Inverse() *  aS1.CurHom();
     cElHomographie aH2To1Init =  aS1.CurHom().Inverse() *  aS2.CurHom();
     std::vector<int> aVInd;
     std::vector<cElHomographie> aVH1To2;  // H1 to 2 en fonction des 16 perturbation possibles de parametre
     std::vector<cElHomographie> aVH2To1;
     // 16 parametres : 8 pour chaque homographie
     for (int aKPTot=0 ; aKPTot<16 ; aKPTot++)
     {
         //   H1_0 H1_1 ... H1_7   H2_0 H2_1 ... H2_7
         int aKPLoc = aKPTot % 8;
         cElHomographie aH1 = aS1.CurHom();
         cElHomographie aH2 = aS2.CurHom();
         if (aKPTot<8)
         {
              aH1 = aS1.HomogPerturb(aKPLoc,aEpsil);
              aVInd.push_back(aS1.mN0Hom+aKPLoc);
         }
         else
         {
            aH2 = aS2.HomogPerturb(aKPLoc,aEpsil);
            aVInd.push_back(aS2.mN0Hom+aKPLoc);
         }
         aVH1To2.push_back(aH2.Inverse() * aH1);
         aVH2To1.push_back(aH1.Inverse() * aH2);
     }


     for (int aKPts=0 ; aKPts<aNbPts ; aKPts++)
     {
         Pt2dr aP1 = aVP1[aKPts];
         Pt2dr aP2 = aVP2[aKPts];
         OneSensEqHomRig(aPds,aVInd,aP1,aP2,aVH1To2,aH1To2Init,aEpsil);
         OneSensEqHomRig(aPds,aVInd,aP2,aP1,aVH2To1,aH2To1Init,aEpsil);
     }
}

void cOrHom_AttrSom::OneSensEqHomRig(double aPds,const std::vector<int> & aVInd,const Pt2dr & aP1, const Pt2dr & aP2,const std::vector<cElHomographie> & aVH1To2,const cElHomographie &aH1To2Init,double aEps)
{
    // aH1To2Init[aP1] = aP2;
    double aCoeff_X[16];
    double aCoeff_Y[16];

    Pt2dr aPTh2 = aH1To2Init(aP1);
    Pt2dr aVal = aP2 - aPTh2 ;
    for (int aK=0 ; aK< 16 ; aK++)
    {
        Pt2dr aDif2 = (aVH1To2[aK](aP1) -aPTh2) / aEps;
        aCoeff_X[aK] = aDif2.x;
        aCoeff_Y[aK] = aDif2.y;
    }

    Sys().GSSR_AddNewEquation_Indexe(0,0,0,aVInd,aPds,aCoeff_X,aVal.x,NullPCVU);
    Sys().GSSR_AddNewEquation_Indexe(0,0,0,aVInd,aPds,aCoeff_Y,aVal.y,NullPCVU);
}


void cOrHom_AttrSom::AddOnePEq12(bool IsX,const Pt2dr & aP1,cOrHom_AttrSom & aS2,const Pt2dr & aP2,double aPds,bool ModeAff)
{
    // a0 I + a1 J + a2         a3 I + a4 J + a5
    int Ind1 = mN0Hom + (IsX ? 0 : 3);
    int Ind2 = aS2.mN0Hom + (IsX ? 0 : 3);

    if (!ModeAff)
    {
        ELISE_ASSERT(false,"cOrHom_AttrSom::AddOnePEq12");
    }

    double aCoeff[6];
    std::vector<int> aVInd;
    double aVal=0;
    aCoeff[0] = aP1.x;  aVInd.push_back(Ind1+0);
    aCoeff[1] = aP1.y;  aVInd.push_back(Ind1+1);
    aCoeff[2] =   1.0;  aVInd.push_back(Ind1+2);

    aCoeff[3] = -aP2.x;  aVInd.push_back(Ind2+0);
    aCoeff[4] = -aP2.y;  aVInd.push_back(Ind2+1);
    aCoeff[5] = -  1.0;  aVInd.push_back(Ind2+2);
    
    Sys().GSSR_AddNewEquation_Indexe(0,0,0,aVInd,aPds,aCoeff,aVal,NullPCVU);
}


    

/************************************************/
/*                                              */
/*         cOrHom_AttrArc                       */
/*                                              */
/************************************************/

void cOrHom_AttrArcSym::SetStdPds(double aRStd)
{
     mPdsRes = mPdsNb / (1 + ElSquare(mResH/aRStd));
}
void cOrHom_AttrArcSym::SetMulFin(double aMul)
{
    mPdsFin = mPdsRes * aMul;
}

cOrHom_AttrArcSym::cOrHom_AttrArcSym(const cXml_Ori2Im & aXml) :
   mXmlO   (aXml),
   mHom12  (mXmlO.Geom().Val().HomWithR().Hom())
{


    int aNbP = mXmlO.NbPts();
    double aRedund = ElMax(0.0,(aNbP-5) / double(aNbP));

    mPdsNb = aNbP * pow(aRedund,3.0);  // Total arbitraire l'exposant
    mResH = aXml.Geom().Val().HomWithR().ResiduHom();

    cGenGaus2D aGC(aXml.Geom().Val().Elips2().Val());

    aGC.GetDistribGaus(mVP1,2,2);
    for (const auto & aP1 : mVP1)
        mVP2.push_back(mHom12(aP1));


}

cElHomographie cOrHom_AttrArcSym::Hom(bool aDir)
{
    cElHomographie aRes = mHom12;
    if (! aDir)
       aRes = aRes.Inverse();

    return aRes;
}

        // cGenGaus2D(const cXml_Elips2D & anEl );

         // =================================



cOrHom_AttrArc::cOrHom_AttrArc(cOrHom_AttrArcSym * anAsym ,bool Direct) :
   mASym(anAsym),
   mDirect (Direct),
   mHom    (anAsym->Hom(Direct))
{
}

void cOrHom_AttrArc::GetDistribGaus(std::vector<Pt2dr> & aVPts,int aN)
{
}

/*
*/

/************************************************/
/*                                              */
/*         cAppli_Hom1Im                        */
/*                                              */
/************************************************/

L2SysSurResol &  cAppli_Hom1Im::Sys()
{
   ELISE_ASSERT(mSys!=0,"No Sys in cAppli_Hom1Im");
   return *mSys;
}


void cAppli_Hom1Im::DrawSeg(const Pt2dr & aP1,const Pt2dr & aP2, int aCoul)
{
#if (ELISE_X11)
    if (!mW) return;
    mW->draw_seg(ToW(aP1),ToW(aP2),mW->pdisc()(aCoul));
#endif
}

void cAppli_Hom1Im::WClear()
{
#if (ELISE_X11)
    if (!mW) return;
    mW->clear();
#endif
}

void cAppli_Hom1Im::WClik()
{
#if (ELISE_X11)
    if (!mW) return;
    mW->clik_in();
#endif
}



tArcGT *   cAppli_Hom1Im::AddArc(tSomGT * aS1,tSomGT * aS2)
{
        ELISE_ASSERT(aS1->attr().Name()<aS2->attr().Name(),"Order assertion in cAppli_Hom1Im::AddArc");
/*
     if (mShow) 
        std::cout << "  ARC " <<  aS1->attr().Name() << " " <<  aS2->attr().Name() << "\n";
*/

    cXml_Ori2Im  aXml = mNM->GetOri2Im(aS1->attr().Name(),aS2->attr().Name());
    int aNbMin = HOM_NbMinPTs;

    if ((!aXml.Geom().IsInit())  || (aXml.NbPts() < aNbMin))
        return nullptr;

    const cXml_O2IComputed & aG = aXml.Geom().Val();

    if ((!aG.Elips2().IsInit()))
       return nullptr;


/*
    const cXml_O2IComputed & aXG = aXml.Geom().Val();
    if (mModeGps)
    {
       if (!  aXG.OriCpleGps().IsInit())  
          return;
       const cXml_OriCple  & aCpl = aXG.OriCpleGps().Val();
       
       int aNbPts = aXml.NbPts();
       double aDist = euclid(aCpl.Ori1().Centre()-aCpl.Ori2().Centre());

       aS1->attr().AddSol(cOneSolSim(aS2,Xml2El(aCpl.Ori1()),aNbPts,aDist));
       aS2->attr().AddSol(cOneSolSim(aS1,Xml2El(aCpl.Ori2()),aNbPts,aDist));
    }
*/


    cOrHom_AttrArcSym * anAttr = new cOrHom_AttrArcSym(aXml);

    tArcGT & aRes = aS1->attr().Appli().Gr().add_arc(*aS1,*aS2,cOrHom_AttrArc(anAttr,true),cOrHom_AttrArc(anAttr,false));
    // add_arc

    return & aRes;
}




tSomGT * cAppli_Hom1Im::AddAnIm(int aNum,const std::string & aName)
{
   if (mMapS[aName] == 0)
   {
      if (aNum<0)
      {
         if (aNum==-1)
         {
            std::cout << "For name=" << aName << "\n";
            ELISE_ASSERT(false,"cAppli_Hom1Im::AddAnIm cannot get Image");
         }
         return 0;
      }
      mMapS[aName]  = &(mGr.new_som(cOrHom_AttrSom(aNum,aName,*this)));
      mVecS.push_back(mMapS[aName]);

/*
     if (mShow) 
         std::cout <<" Add " << aName << "\n";
*/
   }

   return mMapS[aName];
}
tSomGT * cAppli_Hom1Im::GetAnIm(const std::string & aName,bool SVP) {return AddAnIm(SVP? -2 : -1,aName);}


cAppli_Hom1Im::cAppli_Hom1Im(int argc,char ** argv,bool aModePrelim,bool aModeGps) :
   mModePrelim (aModePrelim),
   mModeGps    (aModeGps),
   mSomC       (0),
   mNM         (0),
   mNbInc      (0),
   mSys        (nullptr),
   mMulRab     (0.2),
   mNbEtape    (50),
   mCdgGPS     (0,0,0)
{
#if (ELISE_X11)
    mW = 0;
#endif
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mPat,"Central image"),
        LArgMain() << ArgCMA()
                   << EAM(mMulRab,"MulRab",true," Rab multiplier in visusal mode, def= 0.2")
                   << EAM(mNbEtape,"NbIter",true," Number of steps")
                   << EAM(mModeVert,"Vert",true,"Compute vertical orientation")
   );

   if (!EAMIsInit(&mNbEtape) && (mModeVert))
   {
      mNbEtape = 1;
   }

   if (aModeGps)
   {
      ELISE_ASSERT(EAMIsInit(&mOriGPS),"No GPS ORI in GPS mode");
   }

   mGpsOut  = "Out-"+mOriGPS;

   mEASF.Init(mPat);
   mNM =   cCommonMartiniAppli::NM(mEASF.mDir);
   const cInterfChantierNameManipulateur::tSet * aVN = mEASF.SetIm();

   for(int aK=0 ; aK<int(aVN->size()) ; aK++)
   {
       AddAnIm(mVecS.size(),(*aVN)[aK]);
   }

   // Cas preliminaire, on rajoute tous les sommets connexe
   if (mModePrelim)
   {
       ELISE_ASSERT(aVN->size()==1,"Expect just one image in preliminary mode");
       mNameC = (*aVN)[0];
       mSomC =  GetAnIm(mNameC,false);
       std::list<std::string> aLC = mNM->Liste2SensImOrientedWith(mNameC);
       for (std::list<std::string>::const_iterator itL=aLC.begin() ; itL!=aLC.end() ; itL++)
       {
           AddAnIm(mVecS.size(),*itL);
       }
   }
    

   // Ajout des arcs

   int aNbATest=0 ;
   std::vector<Pt2df>  aVP;
   for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
   {
       tSomGT * aS1 = mVecS[aKS];
       std::list<std::string> aLC = mNM->ListeImOrientedWith(aS1->attr().Name());
       for (std::list<std::string>::const_iterator itL=aLC.begin() ; itL!=aLC.end() ; itL++)
       {
std::cout << "Aaaa " << *itL << "\n";
           tSomGT  * aS2 =  GetAnIm(*itL,true);
std::cout << "BBbbb\n";
           if (aS2)
           {
              tArcGT * anA = AddArc(aS1,aS2);
              if (anA)
              {
                 const cOrHom_AttrArcSym & anAS = anA->attr().ASym();
                 aVP.push_back(Pt2df(anAS.ResH(),anAS.PdsNb()));
                 aNbATest++;
              }
           }
       }
   }
   mResiduStd = PropPond(aVP,0.66);

    
   double aSomPdsRes = 0.0;
   for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
   {
       tSomGT * aS1 = mVecS[aKS];
       for (tItAGT  ait=aS1->begin(mSubAS); ait.go_on()       ; ait++)
       {
           cOrHom_AttrArcSym & aAS = (*ait).attr().ASym();
           aAS.SetStdPds(mResiduStd);
           aSomPdsRes += aAS.PdsRes();
           aNbATest--;
       }

   }
   double aMul = (mVecS.size()) / aSomPdsRes;
   aSomPdsRes = 0; // For check
   for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
   {
       tSomGT * aS1 = mVecS[aKS];
       for (tItAGT  ait=aS1->begin(mSubAS); ait.go_on()       ; ait++)
       {
           cOrHom_AttrArcSym & aAS = (*ait).attr().ASym();
           aAS.SetMulFin(aMul);
           aSomPdsRes += aAS.PdsFin();
       }

   }

   if (mModeGps)
   {
       mNbInc = mVecS.size() * 8;
       mSys = new L2SysSurResol(mNbInc);

       Pt3dr aSomGPS(0,0,0);
       for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
       {
           aSomGPS = aSomGPS + mVecS[aKS]->attr().GPS();
       }
       mCdgGPS = aSomGPS / mVecS.size();

       aSomGPS = Pt3dr(0,0,0);
       double aSomD2 = 0.0;
       for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
       {
            Pt3dr & aGPSK = mVecS[aKS]->attr().GPS();
            aGPSK  = aGPSK  - mCdgGPS;
            aSomD2 += euclid(aGPSK);
            aSomGPS = aSomGPS + aGPSK;
       }
       double aMoyD =  (aSomD2/mVecS.size()) ;
       mMulGps = (1/aMoyD) * sqrt(mVecS.size());

       aSomD2 = 0.0;
       mGpsMin = Pt2dr( 1e5, 1e5);
       mGpsMax = Pt2dr(-1e5,-1e5);
       for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
       {
            Pt3dr & aGPSK = mVecS[aKS]->attr().GPS();
            aGPSK  = aGPSK * mMulGps;
            aSomD2 += euclid(aGPSK);
            mGpsMin = Inf(mGpsMin,Pt2dr(aGPSK.x,aGPSK.y));
            mGpsMax = Sup(mGpsMax,Pt2dr(aGPSK.x,aGPSK.y));
       }
       double aRab = dist4(mGpsMax-mGpsMin) * mMulRab;
       Pt2dr aPRab(aRab,aRab);
       mGpsMin  = mGpsMin - aPRab;
       mGpsMax  = mGpsMax + aPRab;
       std::cout << "GGg " << aSomGPS<< " D2 " << aSomD2 / mVecS.size() << mGpsMin << mGpsMax << "\n";
#if (ELISE_X11)
       if (1)
       {   
           Pt2dr aDG = mGpsMax-mGpsMin;
           Pt2dr aSzWMax(1000,800);
           mScaleW = ElMin(aSzWMax.x/aDG.x,aSzWMax.y/aDG.y);
           mW = Video_Win::PtrWStd(round_ni(aDG*mScaleW));
       }
#endif



       //  Calcul d'une solution initiale Affine
       for (int aKEtape=0 ; aKEtape<mNbEtape ; aKEtape++)
       {
           double aPdsGround = 1e-2;
           if (aKEtape>=3)
              aPdsGround = aPdsGround * pow(1e-1,(aKEtape-3)/3.0);


           double aPdsAffin = 1.0;
           bool   ModeAffine = (aKEtape==0) || mModeVert;
           bool   CoordLoc = (aKEtape!=0);


           mSys->SetPhaseEquation(nullptr);
           // Poids sur les gps et contrainte affine
           for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
           {
               tSomGT & aSom = *(mVecS[aKS]);
               aSom.attr().AddObsGround(aPdsGround,CoordLoc);
               if (ModeAffine)
               {
                   ELISE_ASSERT(!CoordLoc,"No affine cst in loc coord");
                   aSom.attr().AddObsFixAffinite(CoordLoc);
               }
               if (mModeVert)
               {
                   aSom.attr().AddObsFixSimil(CoordLoc);
               }
           }
           for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
           {
               if (aKS%10==0)
                  std::cout << "Fill Eq " << (aKS *100.0) / mVecS.size() << " % \n";
               tSomGT * aS1 = mVecS[aKS];
               for (tItAGT  ait=aS1->begin(mSubAS); ait.go_on()       ; ait++)
               {
                   aS1->attr().AddEqH12(*ait,aPdsAffin,ModeAffine,CoordLoc);
               }
           }

           std::cout << "sssSolving \n";
           bool aOk; 
           Im1D_REAL8 aSol = mSys->GSSR_Solve(&aOk);
           ELISE_ASSERT(aOk,"GSSR_Solve");

           Pt2dr aResT(0,0);
           double aDResT=0;
           for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
           {
               tSomGT & aSom = *(mVecS[aKS]);
               Pt2dr aRes =aSom.attr().SetSol(aSol.data(),CoordLoc);
               aResT = aResT + aRes;
               aDResT += euclid(aRes);
           }
           std::cout << " PdsG= " << aPdsGround << "\n";
           std::cout << " Dist " << aDResT /mVecS.size() << " Bias " << aResT / double(mVecS.size())  << "\n";

           Pt3dr aSomD(0,0,0);
           double aSomP=0;
           for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
           {
               tSomGT * aS1 = mVecS[aKS];
               for (tItAGT  ait=aS1->begin(mSubAS); ait.go_on()       ; ait++)
               {
                   Pt3dr aD = aS1->attr().TestSol(*ait);
                   double aP = (*ait).attr().ASym().PdsFin();
                   aSomD = aSomD + aD * aP;
                   aSomP += aP;
               }
           }
           std::cout << "MoyD " << aSomD / aSomP << "\n";

           mSys->GSSR_Reset(true);
           if (aKEtape>0)
              WClik();
           WClear();

           if (aKEtape > -1)
           {
               for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
               {
                   tSomGT * aS1 = mVecS[aKS];
                   aS1->attr().SaveSolHom(aKEtape);
               }
           }
           // getchar();
       }

/*
       // 
       std::vector<int>     aVNbMax;
       std::vector<double>  aVDistMax;
       for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
       {
            tSomGT * aS1 = mVecS[aKS];
            int aNbMax = aS1->attr().NbPtsMax();
            if (aNbMax<0)
            {
               ELISE_ASSERT(aNbMax>=0,"No sol by gps");
            }
            aVNbMax.push_back(aS1->attr().NbPtsMax());
            aVDistMax.push_back(aS1->attr().DistPMax());
       }
       
       for (int aKS=0 ; aKS<int(mVecS.size()) ; aKS++)
       {
            tSomGT * aS1 = mVecS[aKS];
            aS1->attr().ExportGps();
       }
*/
   }
}


int  TestNewOriHom1Im_main(int argc,char ** argv)
{
    cAppli_Hom1Im anAppli(argc,argv,true,false);

    return EXIT_SUCCESS;
}

int  TestNewOriGpsSim_main(int argc,char ** argv)
{
    cAppli_Hom1Im anAppli(argc,argv,false,true);

    return EXIT_SUCCESS;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les mêmes raisons,
seule une responsabilite restreinte pese sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement,    l'utilisation,    la modification et/ou au
developpement et  la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe 
manipuler et qui le reserve donc   des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a  charger  et  tester  l'adequation  du
logiciel a  leurs besoins dans des conditions permettant d'assurer la
securite de leurs systemes et ou de leurs donnees et, plus generalement,
a  l'utiliser et l'exploiter dans les mêmes conditions de securite.

Le fait que vous puissiez acceder a  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007*/


