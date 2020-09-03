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


  // Test de Greg


#include "StdAfx.h"

// #include "DEBUG_numeric.h"

// CPP_CmpDenseMap
// CPP_FermDenseMap
// CPP_DenseMapToHom      DMatch2Hom
// CPP_CalcMapAnalitik
// CPP_ReechImMap


// PIERROT DESEILLIGNY
// PIERROT DESEILLIGNY

/*
void XXXXX(FILE * aF)
{
  int x;
  int TOTO;
  TOTO = fscanf(aF,"%d",&x);
}
*/

extern bool ERupnik_MM();

/**********************************************************/
/*                                                        */
/*         cStdNamePx2D                                   */
/*                                                        */
/**********************************************************/

class cStdNamePx2D
{
    public :
        cStdNamePx2D(const std::string & aPref,const std::string & aNIm1,const std::string & aNIm2,int aNum=-1);
        std::string mNX;
        std::string mNY;
    private :
        std::string mDir;
};


cStdNamePx2D::cStdNamePx2D(const std::string & aPref,const std::string & aNIm1,const std::string & aNIm2,int aNum) :
   mDir ("MEC-"+aPref+"-"+aNIm1+"-"+aNIm2 +"/")
{
    std::string aNameXml = mDir+"MMLastNuage.xml";
    cXML_ParamNuage3DMaille aXN = StdGetFromSI(aNameXml,XML_ParamNuage3DMaille);
    std::string aN1 = aXN.Image_Profondeur().Val().Image();
    if (aNum>=0)
    {
        aN1= std::string("Px1_Num")+ToString(aNum)+"_DeZoom1_LeChantier.tif";
    }
    std::string aN2 = aN1;
    aN2[2] = '2';
    mNX = mDir + aN1;
    mNY = mDir + aN2;
}


    // ===============================================================
/*******************************************************************/
/*                                                                 */
/*                cParamMap2DRobustInit                            */
/*                                                                 */
/*******************************************************************/

class cParamMap2DRobustInit
{
      public :
         cParamMap2DRobustInit(eTypeMap2D aType,int aNbTirRans,const std::vector<std::string> * aVAux);

         eTypeMap2D          mType;
         int                 mNbTirRans;
         int                 mNbMaxPtsRansac;
         int                 mNbTestFor1P;
         double              mPropRan;  // Percentile (entre 0 et 1) pour estimer l'erreur lors du Ransac
         int                 mNbIterL2;
         cElMap2D*           mRes;
         std::vector<std::string>  mVAux;
         std::vector<double>       mVPdsSol;
};


cParamMap2DRobustInit::cParamMap2DRobustInit(eTypeMap2D aType,int aNbTirRans,const std::vector<std::string> * aVAux) :
    mType           (aType),
    mNbTirRans      (aNbTirRans),
    mNbMaxPtsRansac (2e9),
    mNbTestFor1P    (4),
    mPropRan        (0.8),
    mNbIterL2       (4)
{
   if (aVAux)
      mVAux  = *aVAux;
}


void  Map2DRobustInit(const ElPackHomologue & aPackFull,cParamMap2DRobustInit & aParam);
template <class Type> Type TplMap2DRobustInit(const ElPackHomologue & aPackFull,double aPropRan,int aNbTir,eTypeMap2D anIdType,std::vector<double> * aVPds)
{
    cParamMap2DRobustInit aParam(anIdType,aNbTir,nullptr);
    aParam.mPropRan = aPropRan;
    Map2DRobustInit(aPackFull,aParam);
    if (! aParam.mRes)
    {
        std::cout << "NBPOINT= " << aPackFull.size() << "\n";
        ELISE_ASSERT( false,"TplMap2DRobustInit no result");
    }
    ELISE_ASSERT(aParam.mRes->Type()==anIdType,"TplMap2DRobustInit");

    if (aVPds)
       *aVPds = aParam.mVPdsSol;
    Type * aPtrRes = static_cast<Type *>(aParam.mRes);

    Type aRes = * aPtrRes;
    delete aPtrRes;
    return aRes;

}


cElMap2D *  L2EstimMapHom(cElMap2D * aRes,const ElPackHomologue & aPack);
cElMap2D * L2EstimMapHom(eTypeMap2D aType,const ElPackHomologue & aPack,const std::vector<std::string> * aVAux=0);


cElHomographie HomogrRobustInit(const ElPackHomologue & aPackFull,double aPropRan,int aNbTir)
{
    return TplMap2DRobustInit<cElHomographie>(aPackFull,aPropRan,aNbTir,eTM2_Homogr,nullptr);
}

ElSimilitude  L2EstimSimHom(const ElPackHomologue & aPack)
{
   ElSimilitude aRes;
   L2EstimMapHom(&aRes,aPack);
   return aRes;
}



ElSimilitude SimilRobustInitGen(const ElPackHomologue & aPackFull,double aPropRan,int aNbTir,bool IsRot)
{
/*
    ELISE_ASSERT(  aPackFull.size()>=2,"SimilRobustInit not enough pint");
    cParamMap2DRobustInit aParam(eTM2_Simil,aNbTir,nullptr);
    aParam.mPropRan = aPropRan;
    Map2DRobustInit(aPackFull,aParam);
    ELISE_ASSERT( aParam.mRes!=0,"SimilRobustInit no result");
    // cXml_Map2D    aParam. ToXmlGen();

    ELISE_ASSERT(aParam.mRes->Type()==eTM2_Simil,"SimilRobustInit");
    ElSimilitude * aResSim = static_cast<ElSimilitude *>(aParam.mRes);

    ElSimilitude aSim = * aResSim;
*/
    std::vector<double> aVPds;
    ElSimilitude aSim = TplMap2DRobustInit<ElSimilitude>(aPackFull,aPropRan,aNbTir,eTM2_Simil,&aVPds);

    if (IsRot)
    {
        Pt2dr aSc1 = vunit(aSim.sc());
        int aK=0;
        double aSomP=0;
        Pt2dr aCdg1(0,0);
        Pt2dr aCdg2(0,0);
        for (ElPackHomologue::const_iterator it=aPackFull.begin(); it!=aPackFull.end(); it++)
        {
            double aPds = aVPds[aK++];
            aSomP += aPds;
            aCdg1 = aCdg1 + it->P1() * aPds;
            aCdg2 = aCdg2 + it->P2() * aPds;
        }
        aCdg1 = aCdg1 / aSomP;
        aCdg2 = aCdg2 / aSomP;
        Pt2dr aTr = aCdg2 - aCdg1*aSc1;

        aSim = ElSimilitude(aTr,aSc1);
    }

    return aSim;
}

ElSimilitude SimilRobustInit(const ElPackHomologue & aPackFull,double aPropRan,int aNbTir)
{
    return SimilRobustInitGen( aPackFull,aPropRan,aNbTir,false);
}

ElSimilitude RotationRobustInit(const ElPackHomologue & aPackFull,double aPropRan,int aNbTir)
{
    return SimilRobustInitGen( aPackFull,aPropRan,aNbTir,true);
}

//=====================================================================================

class cMapPol2d : public  cElMap2D
{
   public :
      std::vector<std::string> ParamAux() const;
      cMapPol2d(int aDeg,const Box2dr & aBox0,int aRabDegInv=2);
      Pt2dr operator () (const Pt2dr & aP) const;
      int Type() const ;
      static cMapPol2d Id(const Box2dr & aBox);

      ~cMapPol2d();
      int   NbUnknown() const;
      void  AddEq(Pt2dr & aCste,std::vector<double> & anEqX,std::vector<double> & anEqY,const Pt2dr & aP1,const Pt2dr & aP2 ) const;
      void  InitFromParams(const std::vector<double> &aSol);
      std::vector<double> Params() const;
      virtual bool Compatible(const cElMap2D *) const;
      cElMap2D * Duplicate() ;  // En gal retourne this, mais permet au vecteur a 1 de se simplifier
      cElMap2D * Identity() ;  // En gal retourne this, mais permet au vecteur a 1 de se simplifier
      cElMap2D * Map2DInverse() const;

      cXml_Map2D    ToXmlGen() ; // Peuvent renvoyer 0
      static cMapPol2d FromXml(const cXml_Map2dPol &);
      cXml_Map2dPol ToXmlPol() const; // Peuvent renvoyer 0


      double         Ampl() const;
      const Polynome2dReal & PolX() const;
      const Polynome2dReal & PolY() const;
      Polynome2dReal & PolX() ;
      Polynome2dReal & PolY() ;


   private :

       int            mDeg;
       int            mRabDegInv;
       Box2dr         mBox;
       double         mAmpl;
       Polynome2dReal mPolX;
       Polynome2dReal mPolY;
       int            mNbMon;
};

cElMap2D *  MapPolFromHom(const ElPackHomologue & aPack,const Box2dr & aBox,int aDeg,int aRabDegInv)
{
   cMapPol2d * aRes = new cMapPol2d(aDeg,aBox,aRabDegInv);
   L2EstimMapHom(aRes,aPack);
   return aRes;
}

std::vector<std::string>  cMapPol2d::ParamAux() const
{
   std::vector<std::string> aRes;

   aRes.push_back(ToString(mDeg));
   aRes.push_back(ToString(mBox._p0.x));
   aRes.push_back(ToString(mBox._p0.y));
   aRes.push_back(ToString(mBox._p1.x));
   aRes.push_back(ToString(mBox._p1.y));
   aRes.push_back(ToString(mRabDegInv));

   return aRes;
}

bool cMapPol2d::Compatible(const cElMap2D * aMap) const
{
    const cMapPol2d * aPol  = static_cast<const cMapPol2d *>(aMap);


    return    (mDeg==       aPol->mDeg)
           && (mRabDegInv== aPol->mRabDegInv)
           && (mBox._p0==   aPol->mBox._p0)
           && (mBox._p1==   aPol->mBox._p1) ;
}

double  cMapPol2d::Ampl() const {return mAmpl;}
const   Polynome2dReal & cMapPol2d::PolX() const {return mPolX;}
const   Polynome2dReal & cMapPol2d::PolY() const {return mPolY;}
Polynome2dReal & cMapPol2d::PolX() {return mPolX;}
Polynome2dReal & cMapPol2d::PolY() {return mPolY;}

cMapPol2d::~cMapPol2d()
{
}

cMapPol2d::cMapPol2d(int aDeg,const Box2dr & aBox,int aRabDegInv) :
  mDeg       (aDeg),
  mRabDegInv (aRabDegInv),
  mBox       (aBox),
  mAmpl      (ElMax(euclid(aBox._p0),euclid(aBox._p1))),
  mPolX      (mDeg,mAmpl),
  mPolY      (mDeg,mAmpl),
  mNbMon     (mPolX.NbMonome())
{
   if (aDeg>=1)
   {
      mPolX.SetDegre1(0,1,0);
      mPolY.SetDegre1(0,0,1);
   }
}

Pt2dr cMapPol2d::operator () (const Pt2dr & aP) const
{
    return Pt2dr(mPolX(aP),mPolY(aP));
}

int   cMapPol2d::NbUnknown() const
{
   return 2 * mNbMon;
}


void  cMapPol2d::AddEq(Pt2dr & aCste,std::vector<double> & anEqX,std::vector<double> & anEqY,const Pt2dr & aP1,const Pt2dr & aP2 ) const
{
     aCste = aP2;
     for (int aKm=0 ; aKm<mNbMon ; aKm++)
     {
          anEqX[aKm] = mPolX.KthMonome(aKm)(aP1);
          anEqX[aKm+mNbMon] = 0.0;
          anEqY[aKm] = 0.0;
          anEqY[aKm+mNbMon] = mPolY.KthMonome(aKm)(aP1);
     }
}

void  cMapPol2d::InitFromParams(const std::vector<double> &aSol)
{
    std::vector<double> aVX;
    std::vector<double> aVY;
    for (int aKm=0 ; aKm<mNbMon ; aKm++)
    {
       aVX.push_back(aSol[aKm]);
       aVY.push_back(aSol[aKm+mNbMon]);
    }
    mPolX = Polynome2dReal::FromVect(aVX,mAmpl);
    mPolY = Polynome2dReal::FromVect(aVY,mAmpl);
}

std::vector<double> cMapPol2d::Params() const
{
   std::vector<double> aRes =  mPolX.ToVect();
   std::vector<double> aPolY =  mPolY.ToVect();
   std::copy(aPolY.begin(),aPolY.end(),back_inserter(aRes));

   return aRes;
}
   

cElMap2D * cMapPol2d::Duplicate() 
{
    cMapPol2d * aRes = new cMapPol2d(mDeg,mBox,mRabDegInv);
    // cMapPol2d * aRes = Identity();
    aRes->mPolX = mPolX;
    aRes->mPolY = mPolY;
    return aRes;
}

int cMapPol2d::Type() const 
{
    return eTM2_Polyn;
}

cMapPol2d  cMapPol2d::Id(const Box2dr & aBox) 
{
   return cMapPol2d(1,aBox,0);
}
cElMap2D * cMapPol2d::Identity() 
{
   return new cMapPol2d(mDeg,mBox,mRabDegInv);
}
/*
*/
cElMap2D *  cMapPol2d::Map2DInverse() const
{
    Box2d<double> aBoxF = mBox.BoxImage(*this);
    cMapPol2d * aRes = new cMapPol2d(mDeg+mRabDegInv,aBoxF,0);

    ElPackHomologue aPack;
    int aNbPts = 3*(mDeg+mRabDegInv);
    for (int aKX=0 ; aKX<=aNbPts ; aKX++)
    {
       for (int aKY=0 ; aKY<=aNbPts ; aKY++)
       {
             Pt2dr aP2 = mBox.FromCoordLoc(Pt2dr(aKX/double(aNbPts),aKY/double(aNbPts)));
             Pt2dr aP1 = (*this)(aP2);
             aPack.Cple_Add(ElCplePtsHomologues(aP1,aP2,1.0));
       }
    }

    L2EstimMapHom(aRes,aPack);
    // void  L2EstimMapHom(cElMap2D * aRes,const ElPackHomologue & aPack);
    ELISE_ASSERT(false,"cMapPol2d::Map2DInverse");
    return aRes;
}

// cXml_FulPollXY  Xml2EL(const Polynome2dReal & aPol);
cXml_FulPollXY  El2Xml(const Polynome2dReal & aPol)
{
  cXml_FulPollXY aRes;
  aRes.Degre() = aPol.DMax();  
  aRes.Ampl() = aPol.Ampl() ;  
  aRes.Coeffs() = aPol.ToVect();  

  return aRes;
}

cXml_Map2dPol cMapPol2d::ToXmlPol() const
{
   cXml_Map2dPol aXMapPol;

   aXMapPol.Box() = mBox;

   if (mRabDegInv!=0)
   {
        aXMapPol.DegAddInv().SetVal(mRabDegInv);
   }

   aXMapPol.MapX()  = El2Xml(mPolX);
   aXMapPol.MapY()  = El2Xml(mPolY);
   return aXMapPol;
}

cXml_Map2D    cMapPol2d::ToXmlGen()
{
   cXml_Map2DElem aMapE;
   aMapE.Pol().SetVal(ToXmlPol());
   return cXml_Map2D(MapFromElem(aMapE));
}

Polynome2dReal Xml2EL(const cXml_FulPollXY & aXml)
{
  Polynome2dReal aRes = Polynome2dReal::FromVect(aXml.Coeffs(),aXml.Ampl());
  return aRes;
}

cMapPol2d cMapPol2d::FromXml(const cXml_Map2dPol & aXml)
{
    cMapPol2d aRes(aXml.MapX().Degre(),aXml.Box(),aXml.DegAddInv().ValWithDef(0));

    aRes.mPolX = Xml2EL(aXml.MapX());
    aRes.mPolY = Xml2EL(aXml.MapY());

    return aRes;
}

//=====================================================================================


ElAffin2D::ElAffin2D
(
     Pt2dr im00,  // partie affine
     Pt2dr im10,  // partie vecto
     Pt2dr im01
) :
    mI00 (im00),
    mI10 (im10),
    mI01 (im01)
{
}


ElAffin2D::ElAffin2D() :
    mI00 (0,0),
    mI10 (1,0),
    mI01 (0,1)
{
}

bool ElAffin2D::IsId() const
{
   return 
           (mI00==Pt2dr(0,0))
        && (mI10==Pt2dr(1,0))
        && (mI01==Pt2dr(0,1)) ;
}

ElAffin2D ElAffin2D::Id()
{
   return ElAffin2D();
}

ElAffin2D ElAffin2D::trans(Pt2dr aTr)
{
   return ElAffin2D(aTr,Pt2dr(1,0),Pt2dr(0,1));
}





ElAffin2D::ElAffin2D (const ElSimilitude & aSim) :
    mI00 (aSim(Pt2dr(0,0))),
    mI10 (aSim(Pt2dr(1,0)) -mI00),
    mI01 (aSim(Pt2dr(0,1)) -mI00)
{
}

ElAffin2D ElAffin2D::operator * (const ElAffin2D & sim2) const 
{
    return ElAffin2D
           (
              (*this)(sim2(Pt2dr(0,0))),
              IVect(sim2.IVect(Pt2dr(1,0))),
              IVect(sim2.IVect(Pt2dr(0,1)))
           );

}
ElAffin2D ElAffin2D::operator + (const ElAffin2D & sim2) const 
{
    return ElAffin2D
           (
               mI00 + sim2.mI00,
               mI10 + sim2.mI10,
               mI01 + sim2.mI01
           );

}

ElAffin2D ElAffin2D::CorrectWithMatch(Pt2dr aPt,Pt2dr aRes) const
{
    Pt2dr aGot = (*this) (aPt);

    return ElAffin2D
           (
               mI00 + aRes-aGot,
               mI10,
               mI01
           );
}


ElAffin2D ElAffin2D::inv () const
{
    REAL delta = mI10 ^ mI01;

    Pt2dr  Inv10 = Pt2dr(mI01.y,-mI10.y) /delta;
    Pt2dr  Inv01 = Pt2dr(-mI01.x,mI10.x) /delta;

    return  ElAffin2D
            (
                 -(Inv10*mI00.x+Inv01*mI00.y),
                 Inv10,
                 Inv01
            );
}

ElAffin2D ElAffin2D::TransfoImCropAndSousEch(Pt2dr aTr,Pt2dr aResol,Pt2dr * aSzInOut)
{
   ElAffin2D aRes
             (
                   -Pt2dr(aTr.x/aResol.x,aTr.y/aResol.y),
                   Pt2dr(1.0/aResol.x,0.0),
                   Pt2dr(0.0,1.0/aResol.y)
             );

   if (aSzInOut)
   {
      Box2dr aBoxIn(aTr, aTr+*aSzInOut);
      Box2dr aBoxOut  = aBoxIn.BoxImage(aRes);

      *aSzInOut = aBoxOut.sz();
       aRes = trans(-aBoxOut._p0) * aRes;
   }

   return aRes;
}

ElAffin2D  ElAffin2D::TransfoImCropAndSousEch(Pt2dr aTr,double aResol,Pt2dr * aSzInOut)
{
   return TransfoImCropAndSousEch(aTr,Pt2dr(aResol,aResol),aSzInOut);
}


ElAffin2D  ElAffin2D::L2Fit(const  ElPackHomologue & aPack,double *aResidu)
{
   ELISE_ASSERT(aPack.size()>=3,"Less than 3 point in ElAffin2D::L2Fit");

   static L2SysSurResol aSys(6);
   aSys.GSSR_Reset(false);


   //   C0 X1 + C1 Y1 +C2 =  X2     (C0 C1)  (X1)   C2
   //                               (     )  (  ) +
   //   C3 X1 + C4 Y1 +C5 =  Y2     (C3 C4)  (Y1)   C5

  double aCoeffX[6]={1,1,1,0,0,0};
  double aCoeffY[6]={0,0,0,1,1,1};


   for 
   (
        ElPackHomologue::const_iterator it=aPack.begin();
        it!=aPack.end();
        it++
   )
   {
       aCoeffX[0] = it->P1().x;
       aCoeffX[1] = it->P1().y;

       aCoeffY[3] = it->P1().x;
       aCoeffY[4] = it->P1().y;

       aSys.AddEquation(1,aCoeffX, it->P2().x);
       aSys.AddEquation(1,aCoeffY, it->P2().y);
   }

   Im1D_REAL8 aSol = aSys.Solve(0);
   double * aDS = aSol.data();

   Pt2dr aIm00(aDS[2],aDS[5]);
   Pt2dr aIm10(aDS[0],aDS[3]);
   Pt2dr aIm01(aDS[1],aDS[4]);


   ElAffin2D aRes(aIm00,aIm10,aIm01);

   if (aResidu)
   {
      *aResidu = 0;
      for 
      (
           ElPackHomologue::const_iterator it=aPack.begin();
           it!=aPack.end();
           it++
      )
      {
          *aResidu +=  euclid(aRes(it->P1()),it->P2()) ;
      }
      int aNbPt = aPack.size();
      if (aNbPt>3)
          *aResidu /= (aNbPt-3);
   }
   return aRes;
}


ElAffin2D ElAffin2D::FromTri2Tri
          (
               const Pt2dr & a0, const Pt2dr & a1, const Pt2dr & a2,
               const Pt2dr & b0, const Pt2dr & b1, const Pt2dr & b2
          )
{
     ElAffin2D aA(a0,a1-a0,a2-a0);
     ElAffin2D aB(b0,b1-b0,b2-b0);

     return aB * aA.inv();
}

cElHomographie ElAffin2D::ToHomographie() const
{
    cElComposHomographie aHX(mI10.x,mI01.x,mI00.x);
    cElComposHomographie aHY(mI10.y,mI01.y,mI00.y);
    cElComposHomographie aHZ(     0,     0,     1);

    return  cElHomographie(aHX,aHY,aHZ);
}

// -------------------- :: -------------------
cXml_Map2D MapFromElem(const cXml_Map2DElem & aMapE)
{
    cXml_Map2D aRes;
    aRes.Maps().push_back(aMapE);
    return aRes;
}

//--------------------------------------------


cElMap2D *  ElAffin2D::Map2DInverse() const
{
   return  new ElAffin2D(inv());
}

cXml_Map2D ElAffin2D::ToXmlGen()
{
   cXml_Map2DElem anElem;
   anElem.Aff().SetVal(El2Xml(*this));
   return cXml_Map2D(MapFromElem(anElem));
}

int   ElAffin2D::NbUnknown() const
{
    return 6;
}

std::vector<double> ElAffin2D::Params() const
{
   std::vector<double> aRes;

   aRes.push_back(mI10.x);
   aRes.push_back(mI01.x);
   aRes.push_back(mI00.x);
   aRes.push_back(mI10.y);
   aRes.push_back(mI01.y);
   aRes.push_back(mI00.y);

   return aRes;
}

void  ElAffin2D::InitFromParams(const std::vector<double> &aSol)
{
   mI10 = Pt2dr(aSol[0],aSol[3]);
   mI01 = Pt2dr(aSol[1],aSol[4]);
   mI00 = Pt2dr(aSol[2],aSol[5]);
}
//    A B  X  +  C
//    D E  Y     F
//   A C = P10 , CD = P01  EF = P00
void  ElAffin2D::AddEq
      (
           Pt2dr & aCste,
           std::vector<double> & anEqX,
           std::vector<double> & anEqY,
           const Pt2dr & aP1,
           const Pt2dr & aP2 
       ) const
{
    aCste.x  = aP2.x;
    anEqX[0] = aP1.x;
    anEqX[1] = aP1.y;
    anEqX[2] = 1;
    anEqX[3] = 0;
    anEqX[4] = 0;
    anEqX[5] = 0;


    aCste.y  = aP2.y;
    anEqY[0] = 0;
    anEqY[1] = 0;
    anEqY[2] = 0;
    anEqY[3] = aP1.x;
    anEqY[4] = aP1.y;
    anEqY[5] = 1;
}

cElMap2D * ElAffin2D::Duplicate() 
{
   return new ElAffin2D(*this);
}

cElMap2D * ElAffin2D::Identity() 
{
   return new ElAffin2D(ElAffin2D::Id());
}

int  ElAffin2D::Type()  const {return eTM2_Affine;}


/*****************************************************/
/*                                                   */
/*            ElSimilitude                           */
/*                                                   */
/*****************************************************/

cElMap2D * ElSimilitude::Map2DInverse() const
{
    return new ElSimilitude(inv());
}

cXml_Map2D  ElSimilitude::ToXmlGen()
{
   cXml_Map2DElem anElem;
   anElem.Sim().SetVal(El2Xml(*this));
   return cXml_Map2D(MapFromElem(anElem));
}


int   ElSimilitude::NbUnknown() const
{
    return 4;
}

//   _sc * aP = (s.x+i s.y) * (p.x + i p.y) = (s.x*p.x - s.y p.y) + i (s.y *p.x + s.x * p.y)
//    A -B  X  +  C
//     B A  Y     D
//    A = v1 B=v2 C=v3 D=v4
void  ElSimilitude::AddEq
      (
           Pt2dr & aCste,
           std::vector<double> & anEqX,
           std::vector<double> & anEqY,
           const Pt2dr & aP1,
           const Pt2dr & aP2 
       ) const
{
    aCste.x  = aP2.x;
    anEqX[0] = aP1.x;
    anEqX[1] = -aP1.y;
    anEqX[2] = 1;
    anEqX[3] = 0;


    aCste.y  = aP2.y;
    anEqY[0] = aP1.y;
    anEqY[1] = aP1.x;
    anEqY[2] = 0;
    anEqY[3] = 1;
}

std::vector<double> ElSimilitude::Params() const
{
   std::vector<double> aRes;

   aRes.push_back(_sc.x);
   aRes.push_back(_sc.y);
   aRes.push_back(_tr.x);
   aRes.push_back(_tr.y);

   return aRes;
}

void  ElSimilitude::InitFromParams(const std::vector<double> &aSol)
{
   _sc = Pt2dr(aSol[0],aSol[1]);
   _tr = Pt2dr(aSol[2],aSol[3]);
}

cElMap2D * ElSimilitude::Duplicate() 
{
   return new ElSimilitude(*this);
}

cElMap2D * ElSimilitude::Identity() 
{
   return new ElSimilitude();
}

int ElSimilitude::Type() const { return eTM2_Simil; }


/*****************************************************/
/*                                                   */
/*            cCamAsMap                              */
/*                                                   */
/*****************************************************/

cCamAsMap::cCamAsMap(CamStenope * aCam,bool aDirect)  :
     mCam   (aCam),
     mDirect (aDirect)
{
}

Pt2dr cCamAsMap::operator () (const Pt2dr & p) const
{
   return  mDirect  ? 
           mCam->DistDirecte(p) :  
           mCam->DistInverse(p);
}

cElMap2D * cCamAsMap::Map2DInverse() const
{
   return new cCamAsMap(mCam,!mDirect);
}

cXml_Map2D    cCamAsMap::ToXmlGen()
{
   cXml_MapCam aXmlCam;

   aXmlCam.Directe() = mDirect;
   aXmlCam.PartieCam() = mCam->ExportCalibInterne2XmlStruct(mCam->Sz());

   cXml_Map2DElem anElem;
   anElem.Cam().SetVal(aXmlCam);
   return cXml_Map2D(MapFromElem(anElem));
}

int cCamAsMap::Type() const { return eTM2_Cam; }


/*****************************************************/
/*                                                   */
/*            cElHomographie                         */
/*                                                   */
/*****************************************************/


cElMap2D * cElHomographie::Map2DInverse() const
{
    return new cElHomographie(Inverse());
}

cXml_Map2D   cElHomographie::ToXmlGen()
{
   cXml_Map2DElem anElem;
   anElem.Homog().SetVal(ToXml());
   return cXml_Map2D(MapFromElem(anElem));
}

Pt2dr  cElHomographie::operator() (const Pt2dr & aP) const
{
   return Direct(aP);
}

int   cElHomographie::NbUnknown() const
{
    return 8;
}

std::vector<double> cElHomographie::Params() const
{
   std::vector<double> aRes;

   aRes.push_back(mHX.CoeffX());
   aRes.push_back(mHX.CoeffY());
   aRes.push_back(mHX.Coeff1());

   aRes.push_back(mHY.CoeffX());
   aRes.push_back(mHY.CoeffY());
   aRes.push_back(mHY.Coeff1());

   aRes.push_back(mHZ.CoeffX());
   aRes.push_back(mHZ.CoeffY());


   return aRes;
}

void  cElHomographie::InitFromParams(const std::vector<double> &aSol)
{
   mHX = cElComposHomographie(aSol[0],aSol[1],aSol[2]);
   mHY = cElComposHomographie(aSol[3],aSol[4],aSol[5]);
   mHZ = cElComposHomographie(aSol[6],aSol[7],1.0);
}
//    A B  X1 +  C  ~   X2      A X1 + BY1 +C - X2 (GX1 + HY1 ) = X2
//    D E  Y1    F  ~   Y2      DX1  + EY1 +F - Y2 (GX1 + HY1 ) = Y2
//    G H  1  +  I  ~   1
//   A  B C=Hx ,  DEF=Hy  GHI=Hz
void  cElHomographie::AddEq
      (
           Pt2dr & aCste,
           std::vector<double> & anEqX,
           std::vector<double> & anEqY,
           const Pt2dr & aP1,
           const Pt2dr & aP2 
       ) const
{
    aCste.x  = aP2.x;
      anEqX[0] = aP1.x;
      anEqX[1] = aP1.y;
      anEqX[2] = 1;
      anEqX[3] = 0;
      anEqX[4] = 0;
      anEqX[5] = 0;
      anEqX[6] =  -aP2.x * aP1.x;
      anEqX[7] =  -aP2.x * aP1.y;

    aCste.y  = aP2.y;
      anEqY[0] = 0;
      anEqY[1] = 0;
      anEqY[2] = 0;
      anEqY[3] = aP1.x;
      anEqY[4] = aP1.y;
      anEqY[5] = 1;
      anEqY[6] =  -aP2.y * aP1.x;
      anEqY[7] =  -aP2.y * aP1.y;
}

cElMap2D * cElHomographie::Duplicate() 
{
   return new cElHomographie(*this);
}

cElMap2D * cElHomographie::Identity() 
{
   return new cElHomographie(cElHomographie::Id());
}

int cElHomographie::Type() const { return eTM2_Homogr; }

/*****************************************************/
/*                                                   */
/*            ElHomot, ElTrans                       */
/*                                                   */
/*****************************************************/
class cElHomotPure : public cElMap2D
{
   public :
        std::vector<std::string> ParamAux() const;
        cElHomotPure(const Pt2dr& aPInv,double aScale);
        Pt2dr operator () (const Pt2dr & p) const;
        cElHomotPure(const cXml_HomotPure &) ;
        virtual int Type() const ;
        cElHomotPure inv() const;
        virtual cElMap2D * Identity() ;
        virtual  cElMap2D * Map2DInverse() const;
        virtual cElMap2D * Duplicate() ;  
        virtual cXml_Map2D    ToXmlGen() ; // Peuvent renvoyer 0
        const Pt2dr & PInv() const ;// {return mPInv;}
        const double & Scale()  const;// {return mScale;}
        virtual int   NbUnknown() const;
        virtual void  AddEq(Pt2dr & aCste,std::vector<double> & anEqX,std::vector<double> & anEqY,const Pt2dr & aP1,const Pt2dr & aP2 ) const;
        virtual void  InitFromParams(const std::vector<double> &aSol);
        virtual bool Compatible(const cElMap2D *) const; // Pour l'affectation, peut faire un down cast 

   private :
        Pt2dr   mPInv;
        double  mScale;
};
cXml_HomotPure   EL2Xml(const  cElHomotPure & aHom);
cElHomotPure      Xml2EL(const cXml_HomotPure & aXml);
class cElTrans : public cElMap2D
{
   public :
        cElTrans(const Pt2dr& aTr);
        static cElTrans Id();
        Pt2dr operator () (const Pt2dr & p) const;
        cElTrans(const cXml_Trans &) ;
        virtual int Type() const ;
        cElTrans inv() const;
        virtual  cElMap2D * Map2DInverse() const;
        virtual cElMap2D * Duplicate() ;  
        virtual cElMap2D * Identity() ;
        virtual cXml_Map2D    ToXmlGen() ; // Peuvent renvoyer 0
        const Pt2dr & Trans() const ;// {return mPInv;}
        virtual int   NbUnknown() const;
        virtual void  AddEq(Pt2dr & aCste,std::vector<double> & anEqX,std::vector<double> & anEqY,const Pt2dr & aP1,const Pt2dr & aP2 ) const;
        virtual void  InitFromParams(const std::vector<double> &aSol);

   private :
        Pt2dr   mTrans;
};

// Accesseur
const Pt2dr &  cElHomotPure::PInv() const  {return mPInv;}
const double &  cElHomotPure::Scale()  const {return mScale;}
const Pt2dr &  cElTrans::Trans() const  {return mTrans;}

// Cstr
cElHomotPure::cElHomotPure(const Pt2dr & aPInv,double aScale) :
    mPInv  (aPInv),
    mScale (aScale)
{
}
cElTrans::cElTrans(const Pt2dr & aTrans) :
    mTrans  (aTrans)
{
}
// Xml Cstr
cElHomotPure::cElHomotPure(const cXml_HomotPure & aXml)  :
   mPInv  (aXml.PtInvar()),
   mScale (aXml.Scale())
{
}

cElTrans::cElTrans(const cXml_Trans  & aXml) :
   mTrans (aXml.Tr())
{
}
// Appel pour operer sur un point
Pt2dr cElHomotPure::operator () (const Pt2dr & p) const
{
   return mPInv  + (p-mPInv) * mScale;
}
Pt2dr cElTrans::operator () (const Pt2dr & p) const
{
   return mTrans + p;
}

// Typage dynamic
int cElHomotPure::Type() const {return int(eTM2_HomotPure);}
int cElTrans::Type() const     {return int(eTM2_Trans);}

// Inverse // Duplicate
cElHomotPure cElHomotPure::inv() const {return cElHomotPure(mPInv,1.0/mScale);}
cElMap2D * cElHomotPure::Map2DInverse() const {return new cElHomotPure(inv());}

cElTrans cElTrans::inv() const {return cElTrans(-mTrans);}
cElMap2D * cElTrans::Map2DInverse() const {return new cElTrans(inv());}

cElMap2D * cElHomotPure::Duplicate() { return new cElHomotPure(*this); }
cElMap2D * cElTrans::Duplicate() { return new cElTrans(*this); }

//  ToXmlGen
cXml_HomotPure   EL2Xml(const  cElHomotPure & aHom)
{
   cXml_HomotPure aXml;
   aXml.Scale() =  aHom.Scale();
   aXml.PtInvar() =  aHom.PInv();

   return aXml;
}
cXml_Trans   EL2Xml(const  cElTrans & aTrans)
{
    cXml_Trans aXml;
    aXml.Tr() = aTrans.Trans();
    return aXml;
}

cElHomotPure      Xml2EL(const cXml_HomotPure & aXml)
{
   return cElHomotPure(aXml.PtInvar(),aXml.Scale());
}
cElTrans      Xml2EL(const cXml_Trans & aXml)
{
   return cElTrans(aXml.Tr());
}


cXml_Map2D    cElHomotPure::ToXmlGen() 
{
   cXml_Map2DElem anElem;
   anElem.HomotPure().SetVal(EL2Xml(*this));
   return MapFromElem(anElem);
}
cXml_Map2D    cElTrans::ToXmlGen() 
{
   cXml_Map2DElem anElem;
   anElem.Trans().SetVal(EL2Xml(*this));
   return MapFromElem(anElem);
}

//  Systeme d'equation 
int   cElTrans::NbUnknown() const { return 2; }
int   cElHomotPure::NbUnknown() const { return 1; }

void  cElTrans::AddEq
      (
           Pt2dr & aCste,
           std::vector<double> & anEqX,
           std::vector<double> & anEqY,
           const Pt2dr & aP1,
           const Pt2dr & aP2 
      ) const
{
    aCste.x  = aP2.x - aP1.x;
    anEqX[0] = 1;
    anEqX[1] = 0;

    aCste.y  = aP2.y - aP1.y;
    anEqY[0] = 0;
    anEqY[1] = 1;
}

void  cElTrans::InitFromParams(const std::vector<double> &aSol)
{
    mTrans.x = aSol[0];
    mTrans.y = aSol[1];
}

// mPInv  + (p1-mPInv) * mScale = p2
void  cElHomotPure::AddEq
      (
           Pt2dr & aCste,
           std::vector<double> & anEqX,
           std::vector<double> & anEqY,
           const Pt2dr & aP1,
           const Pt2dr & aP2 
      ) const
{
    aCste.x  = aP2.x - mPInv.x;
    anEqX[0] = aP1.x - mPInv.x;

    aCste.y  = aP2.y - mPInv.y;
    anEqY[0] = aP1.y - mPInv.y;
}

bool cElHomotPure::Compatible(const cElMap2D * aMap) const
{
    const cElHomotPure * aH2  = static_cast<const cElHomotPure *>(aMap);
    return aH2->mPInv == mPInv;
}


void  cElHomotPure::InitFromParams(const std::vector<double> &aSol)
{
    mScale = aSol[0];
}


//  ======== Identity =============

cElTrans cElTrans::Id() {return cElTrans(Pt2dr(0,0));}
cElMap2D * cElTrans::Identity()  {return new cElTrans(cElTrans::Id());}


cElMap2D * cElHomotPure::Identity()  {return new cElHomotPure(mPInv,1.0);}

std::vector<std::string>  cElHomotPure::ParamAux() const
{
   std::vector<std::string> aRes;

   aRes.push_back(ToString(mPInv.x));
   aRes.push_back(ToString(mPInv.y));

   return aRes;
}

/*
//  A  X1  + B  = X2
//  A  Y1    C    Y2
void  ElHomot::AddEq
      (
           Pt2dr & aCste,
           std::vector<double> & anEqX,
           std::vector<double> & anEqY,
           const Pt2dr & aP1,
           const Pt2dr & aP2 
      ) const
{
    aCste.x  = aP2.x;
    anEqX[0] = aP1.x;
    anEqX[1] = 1;
    anEqX[2] = 0;


    aCste.y  = aP2.y;
    anEqY[0] = aP1.y;
    anEqY[1] = 0;
    anEqY[2] = 1;
}
*/



/*****************************************************/
/*                                                   */
/*            ElHomot                                */
/*                                                   */
/*****************************************************/
ElHomot::ElHomot(Pt2dr aTrans, double aScale) :
   mTr (aTrans),
   mSc (aScale)
{
}

ElHomot::ElHomot(const cXml_Homot & aXmlHomot) :
   mTr (aXmlHomot.Tr()),
   mSc (aXmlHomot.Scale())
{
}

ElHomot ElHomot::operator * (const ElHomot & aHom2) const
{
  return ElHomot ( mTr+aHom2.mTr*mSc   ,    mSc*aHom2.mSc );
}
int ElHomot::Type() const { return int(eTM2_Homot); }
cElMap2D * ElHomot::Map2DInverse() const 
{
   return new ElHomot(inv());
}
cElMap2D *  ElHomot::Duplicate() 
{
    return new ElHomot(mTr,mSc);
}

cElMap2D  * ElHomot::Identity() {return new ElHomot;}

cXml_Map2D  ElHomot::ToXmlGen()
{
   cXml_Map2DElem anElem;
   anElem.Homot().SetVal(EL2Xml(*this));
   return MapFromElem(anElem);
}

int  ElHomot::NbUnknown() const {return 3;}

//  A  X1  + B  = X2
//  A  Y1    C    Y2

void  ElHomot::AddEq
      (
           Pt2dr & aCste,
           std::vector<double> & anEqX,
           std::vector<double> & anEqY,
           const Pt2dr & aP1,
           const Pt2dr & aP2 
      ) const
{
    aCste.x  = aP2.x;
    anEqX[0] = aP1.x;
    anEqX[1] = 1;
    anEqX[2] = 0;


    aCste.y  = aP2.y;
    anEqY[0] = aP1.y;
    anEqY[1] = 0;
    anEqY[2] = 1;
}

std::vector<double> ElHomot::Params() const
{
   std::vector<double> aRes;

   aRes.push_back(mSc);
   aRes.push_back(mTr.x);
   aRes.push_back(mTr.y);

   return aRes;
}

void  ElHomot::InitFromParams(const std::vector<double> &aSol)
{
   mSc = aSol[0];
   mTr = Pt2dr(aSol[1],aSol[2]);
}

ElHomot ElHomot::inv () const
{
   return ElHomot ( (-mTr)/mSc, 1/mSc);
}

cXml_Homot   EL2Xml(const ElHomot & aHom)
{
   cXml_Homot aXml;
   aXml.Scale() =  aHom.Sc();
   aXml.Tr() =  aHom.Tr();

   return aXml;
}
ElHomot      Xml2EL(const cXml_Homot & aXml)
{
   return ElHomot(aXml.Tr(),aXml.Scale());
}




/*****************************************************/
/*                                                   */
/*            cElMap2D                               */
/*                                                   */
/*****************************************************/

cElMap2D * cElMap2D::IdentFromType(int aType, const std::vector<std::string> * aVAux)
{
    if ((aVAux==0) || (aVAux->size()==0))
    {
       if (aType == int(eTM2_Homot))    return new ElHomot;
       if (aType == int(eTM2_Simil))    return new ElSimilitude;
       if (aType == int(eTM2_Affine))   return new ElAffin2D(ElAffin2D::Id());
       if (aType == int(eTM2_Homogr))   return new cElHomographie(cElHomographie::Id());

       if (aType == int(eTM2_Trans))   return new cElTrans(cElTrans::Id());

       ELISE_ASSERT(false,"cElMap2D::IdentFromType");
    }

    if (aType == int(eTM2_HomotPure))
    {
       int aNb = aVAux->size();
       ELISE_ASSERT( (aNb==2)  ,"Bad size for Homot Pure Map2D args");
       double xi,yi;
       FromString(xi,(*aVAux)[0]);
       FromString(yi,(*aVAux)[1]);
       return new cElHomotPure(Pt2dr(xi,yi),1.0);
    }

    if (aType == int(eTM2_Polyn))
    {
       int aNb = aVAux->size();
       ELISE_ASSERT( (aNb>=5) && (aNb<=6) ,"Bad size for Polynomial Map2D args");

       int aDeg,aDegRabInv=2;
       double x1,y1,x2,y2;
       FromString(aDeg,(*aVAux)[0]);
       FromString(x1,(*aVAux)[1]);
       FromString(y1,(*aVAux)[2]);
       FromString(x2,(*aVAux)[3]);
       FromString(y2,(*aVAux)[4]);
       if (aNb>=6)
       {
          FromString(aDegRabInv,(*aVAux)[5]);
       }

       return new cMapPol2d(aDeg,Box2dr(Pt2dr(x1,y1),Pt2dr(x2,y2)),aDegRabInv);
    }

    ELISE_ASSERT(false,"cElMap2D::IdentFromType");
    return 0;
}


std::vector<std::string>  cElMap2D::ParamAux() const
{
    std::vector<std::string> aRes;
    return aRes;
}


std::vector<double> cElMap2D::Params() const
{
    ELISE_ASSERT(false,"cElMap2D::Params");
    return std::vector<double>();
}

bool  cElMap2D::Compatible(const cElMap2D *) const
{
   return true;
} 

void cElMap2D::Affect(const cElMap2D & aMap)
{
    ELISE_ASSERT(Type()==aMap.Type(),"Different type in cElMap2D::Affect");
    ELISE_ASSERT(Compatible(&aMap),"Incompatible Map2D from same type");
    std::vector<double>  aVP = aMap.Params();
    InitFromParams(aVP);
}

cElMap2D * cElMap2D::Map2DInverse() const
{
   ELISE_ASSERT(false,"No def cElMap2D::Map2DInverse");
   return 0;
}

cElMap2D * cElMap2D::Simplify() 
{
   return this;
}

cXml_Map2D      cElMap2D::ToXmlGen()
{
   ELISE_ASSERT(false,"No def cElMap2D::ToXmlGen");
   return cXml_Map2D();
}

void   cElMap2D::SaveInFile(const std::string & aName)
{
    cXml_Map2D aXml = ToXmlGen();
    MakeFileXML(aXml,aName);
}

cElMap2D *  Map2DFromElem(const cXml_Map2DElem & aXml)
{

   if (aXml.Homot().IsInit()) return new ElHomot(aXml.Homot().Val());
   if (aXml.Homog().IsInit()) return new cElHomographie(aXml.Homog().Val());
   if (aXml.Sim().IsInit()) return new ElSimilitude(Xml2EL(aXml.Sim().Val()));
   if (aXml.Aff().IsInit()) return new ElAffin2D(Xml2EL(aXml.Aff().Val()));
   if (aXml.Cam().IsInit())
   {
       CamStenope* aCS = Std_Cal_From_CIC(aXml.Cam().Val().PartieCam());
       return new cCamAsMap(aCS,aXml.Cam().Val().Directe());
   }

   if (aXml.Pol().IsInit())
   {
        return new cMapPol2d (cMapPol2d::FromXml(aXml.Pol().Val()));
   }

   ELISE_ASSERT(false,"Map2DFromElem");
   return 0;
}

cElMap2D *  cElMap2D::FromFile(const std::string & aName)
{
   cXml_Map2D aXml = StdGetFromSI(aName,Xml_Map2D);
   std::vector<cElMap2D *> aVMap;

   for (std::list<cXml_Map2DElem>::const_iterator itM=aXml.Maps().begin() ; itM!=aXml.Maps().end() ; itM++)
   {
      aVMap.push_back(Map2DFromElem(*itM));
   }


   return new cComposElMap2D(aVMap);
}

int   cElMap2D::NbUnknown() const
{
   ELISE_ASSERT(false,"cElMap2D::NbUnknown");
   return -1;
}
void  cElMap2D::AddEq(Pt2dr & ,std::vector<double> & ,std::vector<double> & ,const Pt2dr & aP1,const Pt2dr & aP2 ) const
{
   ELISE_ASSERT(false,"cElMap2D::AddEq");
}

void  cElMap2D::InitFromParams(const std::vector<double> &aSol)
{
   ELISE_ASSERT(false,"cElMap2D::InitFromParams");
}

cElMap2D * cElMap2D::Duplicate() 
{
   ELISE_ASSERT(false,"cElMap2D::AddEq");
   return 0;
}

cElMap2D * cElMap2D::Identity() 
{
   ELISE_ASSERT(false,"cElMap2D::AddEq");
   return 0;
}

/*****************************************************/
/*                                                   */
/*            cComposElMap2D                         */
/*                                                   */
/*****************************************************/

cComposElMap2D::cComposElMap2D(const std::vector<cElMap2D *>  & aVMap) :
   mVMap (aVMap)
{
}

Pt2dr cComposElMap2D::operator () (const Pt2dr & aP)  const
{
   Pt2dr aRes = aP;
   for (int aK=0 ; aK<int(mVMap.size()) ; aK++)
       aRes = (*(mVMap[aK]))(aRes);
   return aRes;
}

cElMap2D *  cComposElMap2D::Map2DInverse() const
{
   std::vector<cElMap2D *> aVInv;
   for (int aK=int(mVMap.size()-1) ; aK>=0 ; aK--)
      aVInv.push_back(mVMap[aK]->Map2DInverse());

   return new cComposElMap2D(aVInv);
}

cElMap2D * cComposElMap2D::Simplify() 
{
   if (mVMap.size()==1) 
      return mVMap[0];

   return this;
}


cXml_Map2D    cComposElMap2D::ToXmlGen()
{
   cXml_Map2D aRes;

   for (int aK=0 ; aK<int(mVMap.size()) ; aK++)
   {
        cXml_Map2D aXml = mVMap[aK]->ToXmlGen();
        for (std::list<cXml_Map2DElem>::const_iterator itM2=aXml.Maps().begin() ; itM2!=aXml.Maps().end() ; itM2++)
        {
            aRes.Maps().push_back(*itM2);
        }
   }
   return aRes;
}
int  cComposElMap2D::Type()  const {return eTM2_Compos;}

/*****************************************************/
/*                                                   */
/*                 ::                                */
/*                                                   */
/*****************************************************/

cElMap2D *  L2EstimMapHom(cElMap2D * aRes,const ElPackHomologue & aPack)
{
    int aNbUk = aRes->NbUnknown();
    std::vector<double> aVCoefX(aNbUk);
    std::vector<double> aVCoefY(aNbUk);
    L2SysSurResol aSys(aNbUk);

    for (ElPackHomologue::const_iterator itCpl=aPack.begin();itCpl!=aPack.end() ; itCpl++)
    {
        Pt2dr aQ;
        aRes->AddEq(aQ,aVCoefX,aVCoefY,itCpl->P1(),itCpl->P2());
        aSys.AddEquation(itCpl->Pds(),VData(aVCoefX),aQ.x);
        aSys.AddEquation(itCpl->Pds(),VData(aVCoefY),aQ.y);
    }
    Im1D_REAL8  aSol = aSys.Solve(0);
    aRes->InitFromParams(std::vector<double>(aSol.data(),aSol.data()+aNbUk));
    return aRes;
}

cElMap2D * L2EstimMapHom(eTypeMap2D aType,const ElPackHomologue & aPack,const std::vector<std::string> * aVAux)
{
    cElMap2D * aRes = cElMap2D::IdentFromType(aType,aVAux);
    L2EstimMapHom(aRes,aPack);
    return aRes;
}


int CPP_CalcMapAnalitik(int argc,char** argv)
{
    std::string aNameOneIm1,aNameOneIm2,aNameOut,aSH;
    bool aDeprExpTxt=false;
    std::string anExt="dat";

    std::string anOri;
    std::string aNameType;
    Pt2dr       aPerResidu(100,100);
    std::vector<double> aVRE; // Robust Estim
    std::vector<std::string>  aParamAux;
    std::vector<std::string>  aDeprParamPoly;
    bool IdX=false;
    bool IdY=false;

    // int NbTest =50;
    // double  Perc = 80.0;
    // int     NbMaxPts= 10000;
    eTypeMap2D aType;
    bool aModeHelp;
    bool ByKey=false;

    if ((argc>=2) && (std::string(argv[1]) == "-help"))
    {
        StdReadEnum(aModeHelp,aType,"-help",eTypeMap2D(eTM2_Homogr+1));
    }

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  <<  EAMC(aNameOneIm1,"Name Im1")
                    <<  EAMC(aNameOneIm2,"Name Im2")
                    <<  EAMC(aNameType,"Model in [Homot,Simil,Affine,Homogr,Polyn]")
                    <<  EAMC(aNameOut,"Name Out"),
        LArgMain()  <<  EAM(aSH,"SH",true,"Set of homologue")
                    <<  EAM(anExt,"Ext",true,"in [dat,txt,xml]")
                    <<  EAM(anOri,"Ori",true,"Directory to read distorsion")
                    <<  EAM(aPerResidu,"PerResidu",true,"Period for computing residual")
                    <<  EAM(aVRE,"PRE",true,"Param for robust estimation [PropInlayer,NbRan(500),NbPtsRan(+inf)]")
                    <<  EAM(aParamAux,"ParamAux",true,"Param , For Homot [XC,YC], for polygonal model [Deg,x0,y0,x1,y1]")
                    <<  EAM(ByKey,"ByKey",true,"When true multiple, Param is a pattern of Im1, param2 is a key of compute")
	            <<  EAM(aDeprExpTxt,"ExpTxt",true,"DEPRECATED !!! => use Ext (string not bool)")
                    <<  EAM(aDeprParamPoly,"ParPol",true,"Param for polygonal model [Deg,x0,y0,x1,y1]")
                    <<  EAM(IdX,"IdX",true,"Force P2.x=P1.x")
                    <<  EAM(IdY,"IdY",true,"Force P2.y=P1.y")
    );
    ELISE_ASSERT
    (
       ! EAMIsInit(&aDeprExpTxt),
       "ExpTxt is deprecated, use Ext instead"
    );
    ELISE_ASSERT
    (
       ! EAMIsInit(&aDeprParamPoly),
       "ParPol is deprecated, use ParamAux instead"
    );

    std::vector<std::string> aVIm1;
    std::vector<std::string> aVIm2;
    cElemAppliSetFile anEASF(aNameOneIm1);
    if (ByKey)
    {
        aVIm1 = *(anEASF.SetIm());
        for (int aK1=0 ; aK1<int(aVIm1.size()) ; aK1++)
        {
           aVIm2.push_back(anEASF.mICNM->Assoc1To1(aNameOneIm2,aVIm1[aK1],true));
        }
    }
    else
    {
       aVIm1.push_back(aNameOneIm1);
       aVIm2.push_back(aNameOneIm2);
    }

    StdReadEnum(aModeHelp,aType,std::string("TM2_")+aNameType,eTM2_NbVals);


    cInterfChantierNameManipulateur * anICNM = anEASF.mICNM;
    std::string aDir = anEASF.mDir;

    std::string aDirResidu = aDir+ "ResiduImDir/";
   


    CamStenope * aCS1=0,*aCS2=0;
    if (EAMIsInit(&anOri))
    {
         StdCorrecNameOrient(anOri,aDir);
         aCS1 = anICNM->GlobCalibOfName(aVIm1[0],anOri,false);
         aCS2 = anICNM->GlobCalibOfName(aVIm2[0],anOri,false);

         aCS1->Get_dist().SetCameraOwner(aCS1);
         aCS2->Get_dist().SetCameraOwner(aCS2);
    }

    ElPackHomologue aPackInGlob;
    ElPackHomologue aPackInitialGlob;
    // std::string anExt = aExpTxt ? "txt" : "dat";
    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(aSH)
                       +  std::string("@")
                       +  std::string(anExt);

    Pt2dr aP1Max(-1e20,-1e20);
    Pt2dr aP1Min( 1e20, 1e20);

    for (int aK=0 ; aK<int(aVIm1.size()) ; aK++)
    {
        std::string aNameIn = aDir + anICNM->Assoc1To2(aKHIn,aVIm1[aK],aVIm2[aK],true);
        ElPackHomologue aPackInLoc =  ElPackHomologue::FromFile(aNameIn);

        for (ElPackHomologue::iterator itCpl=aPackInLoc.begin();itCpl!=aPackInLoc.end() ; itCpl++)
        {
            if (IdX)
               itCpl->P2().x = itCpl->P1().x;
            if (IdY)
               itCpl->P2().y = itCpl->P1().y;

            aPackInitialGlob.Cple_Add(itCpl->ToCple());
            if (aCS1)
            {
                itCpl->P1() = aCS1->DistInverse(itCpl->P1());
                itCpl->P2() = aCS2->DistInverse(itCpl->P2());
            }

            aPackInGlob.Cple_Add(itCpl->ToCple());
            aP1Max = Sup(aP1Max, itCpl->P1());
            aP1Min = Inf(aP1Min, itCpl->P1());
        }
    }

    // std::cout << "P1Max " << aP1Max  << " " << aP1Min << "\n";

    Pt2di aSzResidu = round_up(aP1Max.dcbyc(aPerResidu));
    Im2D_REAL8 aImResX(aSzResidu.x,aSzResidu.y,0.0);
    Im2D_REAL8 aImResY(aSzResidu.x,aSzResidu.y,0.0);


    // double anEcart,aQuality;
    // bool Ok;

    cElMap2D * aMapCor = 0;
  
    if (EAMIsInit(&aVRE))
    {
       // <<  EAM(aVRE,"PRE",true,"Param for robust estimation [PercInlayer,NbRan(500),NbPtsRan(+inf)]")
       double aProp  =  aVRE[0];
       int aNbRan    = (aVRE.size()>1) ? aVRE[1] : 500;
       int aNbPtsRan = (aVRE.size()>2) ? aVRE[2] : 2e9;
       cParamMap2DRobustInit aParam(aType,aNbRan,&aParamAux);
       aParam.mPropRan = aProp;
       aParam.mNbMaxPtsRansac = aNbPtsRan;
       Map2DRobustInit(aPackInGlob,aParam);
       aMapCor = aParam.mRes;
    }
    else
    {
       aMapCor  =  L2EstimMapHom(aType,aPackInGlob,&aParamAux);
    }


    std::vector<cElMap2D *> aVMap;
    if (aCS1)
    {
       aVMap.push_back(new cCamAsMap(aCS1,false));
    }
    aVMap.push_back(aMapCor);
    if (aCS2)
    {
       aVMap.push_back(new cCamAsMap(aCS2,true));
    }

    cComposElMap2D aComp(aVMap);

    std::vector<double> aVDist;
    double aSomD2X=0.0;
    double aSomD2Y=0.0;
    for (ElPackHomologue::iterator itCpl=aPackInitialGlob.begin();itCpl!=aPackInitialGlob.end() ; itCpl++)
    {
        Pt2dr aRes = aComp(itCpl->P1())-itCpl->P2();
        double aD = euclid(aRes);
        aVDist.push_back(aD);
        aSomD2X += ElSquare(aRes.x);
        aSomD2Y += ElSquare(aRes.y);
        Pt2di aPInd = round_ni(itCpl->P1().dcbyc(aPerResidu));
        aImResX.SetR_SVP(aPInd,aRes.x);
        aImResY.SetR_SVP(aPInd,aRes.y);
        // if (BadNumber(aD)) std::cout << "KKKK " << aD << "\n";
    }
    MakeFileXML(aComp.ToXmlGen(),aNameOut);

    if (EAMIsInit(&aPerResidu))
    {
       ELISE_fp::MkDirSvp(aDirResidu);
       std::string aPref = "Res-" +  aNameType + "-" + aVIm1[0] + "-" + aVIm2[0] ;
       Tiff_Im::CreateFromIm(aImResX,aDirResidu+aPref+"-X.tif");
       Tiff_Im::CreateFromIm(aImResY,aDirResidu+aPref+"-Y.tif");
    }
    

    int aNbDist=10;
    for (int aK=0; aK<=aNbDist ; aK++)
    {
        double aProp = aK/double(aNbDist);
        std::cout << "  Residu at " << aProp*100 << " percentil = " << KthValProp(aVDist,aProp) << "\n";
    }
    aSomD2X /= aPackInGlob.size();
    aSomD2Y /= aPackInGlob.size();

    std::cout << "MoyD2 = " << sqrt(aSomD2X)  << " " << sqrt(aSomD2Y) << "\n";



    return EXIT_SUCCESS;

}


int CPP_SampleMap2D(int argc,char** argv)
{
    std::string aNameMap;
    Box2dr  aBox;
    std::string aNameOut;
    int aNbSample;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  <<  EAMC(aNameMap,"Name map")
                    <<  EAMC(aBox,"Box")
                    <<  EAMC(aNbSample,"Number of sample"),
        LArgMain()  <<  EAM(aNameOut,"Out",false,"Txt file ")
    );
    
    cElMap2D * aMap = cElMap2D::FromFile(aNameMap);

    if (!EAMIsInit(&aNameOut)) aNameOut = StdPrefix(aNameMap) + "-Samples.txt";

    FILE * aFP = 0;
    if (aNameOut!="NONE")
        aFP = FopenNN(aNameOut.c_str(),"w","CPP_SampleMap2D");

    Pt2dr aSomInit(0,0);
    Pt2dr aSomMap(0,0);
    double aSomP=0;

    for (int aX=0 ; aX<=aNbSample ; aX++)
    {
        for (int aY=0 ; aY<=aNbSample ; aY++)
        {
// std::cout << "XxYyy " << aX << " " << aY << "\n";
             Pt2dr aPInit = aBox.FromCoordLoc(Pt2dr(aX/double(aNbSample),aY/double(aNbSample)));
             Pt2dr aPMap = (*aMap)(aPInit);

             if (aFP!=0)
             {
                 fprintf(aFP,"%f %f %f %f\n",aPInit.x,aPInit.y,aPMap.x,aPMap.y);
             }
             aSomP++;
             aSomInit = aSomInit + aPInit;
             aSomMap = aSomMap + aPMap;
        }
    }
    if (aFP) fclose(aFP);
    aSomInit = aSomInit / aSomP;
    aSomMap = aSomMap / aSomP;

    std::cout << "Average   " << aSomInit << " => " <<  aSomMap << "\n";
    std::cout << "Trans : " << aSomMap - aSomInit << "\n";

    return EXIT_SUCCESS;
}


int CPP_ReechImMap(int argc,char** argv)
{
    std::string aNameIm,aNameMap;
    Pt2di aSzOut;
    std::string aNameOut;
    std::string aPrefixOut("Reech_");
    std::string aMAF;
    std::string aMAFOut;
    bool aDoImgReech=true;
    Pt2di aWinInt(5,5); 
	
	Tiff_Im * aTifOut = 0;
	std::vector<Im2DGen *> aVecImOut;
 
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  <<  EAMC(aNameIm,"Name Im")
                    <<  EAMC(aNameMap,"Name map"),
        LArgMain()  <<  EAM(aNameOut,"Out",false,"Tif file to write to, this file must already exist")
                    <<  EAM(aPrefixOut,"PrefixOut",false,"Prefix of output file, def 'Reech_'")
                    <<  EAM(aMAF,"MAF",false,"Xml file of Image Measures")
                    <<  EAM(aDoImgReech,"DoImgReech",false,"Generate Image Reech ; Def=true")
                    <<  EAM(aWinInt,"Win",false,"Interpolation window ; Def=[5,5]")
    );

    if (!EAMIsInit(&aNameOut))  
       aNameOut = DirOfFile(aNameIm) + aPrefixOut + NameWithoutDir(StdPrefix(aNameIm)) + ".tif";
     else
       aTifOut = new Tiff_Im(Tiff_Im::StdConvGen(aNameOut,-1,true));

    cElMap2D * aMap = cElMap2D::FromFile(aNameMap);
    
    if(aDoImgReech)
    {

		Tiff_Im aTifIn = Tiff_Im::StdConvGen(aNameIm,-1,true);


		std::vector<Im2DGen *>  aVecImIn =  aTifIn.ReadVecOfIm();
		int aNbC = aVecImIn.size();
		Pt2di aSzIn = aVecImIn[0]->sz();
		//if (! EAMIsInit(&aSzOut))
		//   aSzOut = aSzIn;
		if (!EAMIsInit(&aNameOut))
			aSzOut = aSzIn;
		else
			aSzOut = aTifOut->sz();

		if (!EAMIsInit(&aNameOut))
			aVecImOut =  aTifIn.VecOfIm(aSzOut);
		else
			aVecImOut = aTifOut->ReadVecOfIm();

		std::vector<cIm2DInter*> aVInter;
		for (int aK=0 ; aK<aNbC ; aK++)
		{
			aVInter.push_back(aVecImIn[aK]->SinusCard(aWinInt.x,aWinInt.y));
		}

		Pt2di aP;
		for (aP.x =0 ; aP.x<aSzOut.x ; aP.x++)
		{
			for (aP.y =0 ; aP.y<aSzOut.y ; aP.y++)
			{
				Pt2dr aQ = (*aMap)(Pt2dr(aP));
				for (int aK=0 ; aK<aNbC ; aK++)
				{
					double aV = aVInter[aK]->GetDef(aQ,0);
					if( aV != 0 )
						aVecImOut[aK]->SetR(aP,aV);
					else
					{
						aVecImOut[aK]->SetR(aP,aVecImOut[aK]->GetR(aP,0));
					}
				}
			}
		}

		if (!EAMIsInit(&aNameOut))
			aTifOut = new Tiff_Im
					(
						aNameOut.c_str(),
						aSzOut,
						aTifIn.type_el(),
						Tiff_Im::No_Compr,
						aTifIn.phot_interp()
					);

		ELISE_COPY(aTifOut->all_pts(),StdInPut(aVecImOut),aTifOut->out());
	}
    
    //if a xml MAF file is given
    if (EAMIsInit(&aMAF))
    {
		
		if (!EAMIsInit(&aMAFOut))
			aMAFOut = DirOfFile(aMAF) + NameWithoutDir(StdPrefix(aMAF)) + "_Reech_" + NameWithoutDir(aNameIm) + ".xml";
       
		//input
		cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(aMAF,SetOfMesureAppuisFlottants);
		std::list<cMesureAppuiFlottant1Im> & aLMAF = aDico.MesureAppuiFlottant1Im();
		
		//output
		cSetOfMesureAppuisFlottants aDicoOut;
		std::list<cMesureAppuiFlottant1Im> aLMAFOut;
		std::list<cOneMesureAF1I> aMesOut;
		
		for (std::list<cMesureAppuiFlottant1Im>::iterator iT1 = aLMAF.begin() ; iT1 != aLMAF.end() ; iT1++)
		{
			
			if(NameWithoutDir(aNameIm).compare(iT1->NameIm()) == 0)
			{
				std::list<cOneMesureAF1I> & aMes = iT1->OneMesureAF1I();
				for (std::list<cOneMesureAF1I>::iterator iT2 = aMes.begin() ; iT2 != aMes.end() ; iT2++)
				{
					cOneMesureAF1I aOMAF1I;
					aOMAF1I.NamePt() = iT2->NamePt();
					aOMAF1I.PtIm() = Pt2dr(iT2->PtIm())*2- (*aMap)(Pt2dr(iT2->PtIm()));
					
					aMesOut.push_back(aOMAF1I);
				}
			}
		}
		cMesureAppuiFlottant1Im   aMAF1Im;
		aMAF1Im.NameIm() = NameWithoutDir(aNameOut);
		aMAF1Im.OneMesureAF1I() = aMesOut;
		aLMAFOut.push_back(aMAF1Im);
		aDicoOut.MesureAppuiFlottant1Im() = aLMAFOut;
		MakeFileXML(aDicoOut,aMAFOut);
	}

    return EXIT_SUCCESS;
}

    //=====================================
    //     DMatch2Hom
    //=====================================

class cAppli_CPP_DenseMapToHom
{
   public :

    std::string mPref;
    std::string mName1;
    std::string mName2;
    std::string mNamePx1;
    std::string mNamePx2;
    std::string mNamePds;
    std::string mSH;
    std::string mExt;
    Pt2di       mSzIm;
    Tiff_Im *   mTifX;
    Tiff_Im *   mTifY;
    Fonc_Num    mFPond;
    double      mNbTile;
    Pt2di       mSz;
    Pt2di       mSzRed;

    Im2D_REAL8  mImX1;
    Im2D_REAL8  mImY1;
    Im2D_REAL8  mImX2;
    Im2D_REAL8  mImY2;
    Im2D_REAL8  mImP;
    double      mOverlap; // Si 1 les dalles se recouvrent juste

    cAppli_CPP_DenseMapToHom(int argc,char** argv);
};


cAppli_CPP_DenseMapToHom::cAppli_CPP_DenseMapToHom(int argc,char** argv) :
    mSH      ("DM"),
    mExt     ("dat"),
    mFPond   (1.0),
    mNbTile  (30),
    mImX1    (1,1),
    mImY1    (1,1),
    mImX2    (1,1),
    mImY2    (1,1),
    mImP     (1,1),
    mOverlap (1.0)
{

	bool aExpTxt=false;
	
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  <<  EAMC(mPref,"Pref where , Dir=MEC-${Pref}-{Im1}-{Im2}")
                    <<  EAMC(mName1,"Name Im1")
                    <<  EAMC(mName2,"Name Im2"),
                    // <<  EAMC(mNamePx1,"Name Px1 (dx)")
                    // <<  EAMC(mNamePx2,"Name Px2 (dy)"),
        LArgMain()  <<  EAM(mSH,"SH",true,"Set of homologue, def=DM")
					<<  EAM(aExpTxt,"ExpTxt",true,"Ascii format for in and out, def=false")
                    <<  EAM(mNbTile,"NbTiles",true,"Number of tile/side (will be slightly changed), Def=30")
                    <<  EAM(mNamePds,"Pds",true,"File for weighting, def W=1.0")
    );

    cStdNamePx2D aName2D(mPref,mName1,mName2) ;
    mNamePx1 = aName2D.mNX;
    mNamePx2 = aName2D.mNY;

    if (EAMIsInit(&mNamePds))
    {
       mFPond =  Tiff_Im(mNamePds.c_str()).in_proj();
    }

    cElemAppliSetFile anEASF(mName1);
    cInterfChantierNameManipulateur * anICNM = anEASF.mICNM;
    std::string aDir = anEASF.mDir;
	std::string mExt = aExpTxt ? "txt" : "dat";
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(mSH)
                       +  std::string("@")
                       +  std::string(mExt);

    std::string aNameOut = aDir + anICNM->Assoc1To2(aKHOut,mName1,mName2,true);

    mTifX =   new Tiff_Im(Tiff_Im::StdConvGen(mNamePx1,-1,true));
    mTifY =   new Tiff_Im(Tiff_Im::StdConvGen(mNamePx2,-1,true));

    mSz = mTifX->sz();

    double aLargTile = sqrt((mSz.x * double(mSz.y) ) / ElSquare(mNbTile));
    mSzRed = round_up(Pt2dr(mSz)/aLargTile);
    aLargTile = ElMax(mSz.x/double(mSzRed.x),mSz.y/double(mSzRed.y)) * mOverlap;
    Pt2dr mDemiTile(aLargTile/2.0,aLargTile/2.0);

    mImX1 =  Im2D_REAL8(mSzRed.x,mSzRed.y);
    mImY1 =  Im2D_REAL8(mSzRed.x,mSzRed.y);
    mImX2 =  Im2D_REAL8(mSzRed.x,mSzRed.y);
    mImY2 =  Im2D_REAL8(mSzRed.x,mSzRed.y);
    mImP  =  Im2D_REAL8(mSzRed.x,mSzRed.y,0.0);

    Pt2di aP;
    double aMaxP=0.0;
    for (aP.x=0 ; aP.x< mSzRed.x ; aP.x++)
    {
       for (aP.y=0 ; aP.y< mSzRed.y ; aP.y++)
       {
           Pt2dr aPC = Pt2dr(aP) + Pt2dr(0.5,0.5); // au cente de la dalle reduite
           aPC = aPC.dcbyc(Pt2dr(mSzRed)); // Entre 0 et 1
           aPC = aPC.mcbyc(Pt2dr(mSz));    // au centre des coordonnees pleines

           Pt2di aPIm0 = Sup(Pt2di(0,0),round_down(aPC-mDemiTile));
           Pt2di aPIm1 = Inf(mSz,round_down(aPC+mDemiTile));
           double aSom[5];
           ELISE_COPY
           (
               rectangle(aPIm0,aPIm1),
               Virgule(FX*mFPond,FY*mFPond,mTifX->in()*mFPond,mTifY->in()*mFPond,mFPond),
               sigma(aSom,5)
           );
           if (aSom[4] > mImP.GetR(aP)) 
           {
               mImX1.SetR(aP,aSom[0]/aSom[4]);
               mImY1.SetR(aP,aSom[1]/aSom[4]);
               mImX2.SetR(aP,aSom[2]/aSom[4]);
               mImY2.SetR(aP,aSom[3]/aSom[4]);
               mImP.SetR(aP,aSom[4]);
               ElSetMax(aMaxP,aSom[4]);
           }
       }
    }
    std::cout << "PerResidu=" << Pt2dr(mSz).dcbyc(Pt2dr(mSzRed)) << "\n";

    ElPackHomologue aPack;
    for (aP.x=0 ; aP.x< mSzRed.x ; aP.x++)
    {
       for (aP.y=0 ; aP.y< mSzRed.y ; aP.y++)
       {
           Pt2dr aPIm1(mImX1.GetR(aP),mImY1.GetR(aP));
           Pt2dr aPIm2(mImX2.GetR(aP),mImY2.GetR(aP));
           aPIm2  = aPIm1 + aPIm2;
           aPack.Cple_Add(ElCplePtsHomologues(aPIm1,aPIm2,mImP.GetR(aP)));
       }
    }
    aPack.StdPutInFile(aNameOut);
}


int CPP_DenseMapToHom(int argc,char** argv)
{
    cAppli_CPP_DenseMapToHom anAppli(argc,argv);

    return EXIT_SUCCESS;
}


    //=====================================
    //     FermDenseMap
    //=====================================

int CPP_FermDenseMap(int argc,char** argv)
{
    std::string mPref,mNameImA,mNameImB,mNameImC;
    int aNum=-1;
    double aSigmaPds=-1;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  <<  EAMC(mPref,"Pref where , Dir=MEC-${Pref}-{Im1}-{Im2}")
                    <<  EAMC(mNameImA,"ImA")
                    <<  EAMC(mNameImB,"ImB")
                    <<  EAMC(mNameImC,"ImC"),
        LArgMain()  
                    <<  EAM(aNum,"Num",true,"Num of Px, def=last")
                    <<  EAM(aSigmaPds,"SigmaP",true,"Sigma use for pds comp, Pds=1/(1+Sq(Res/Sig))")
    );
    bool UsePds = EAMIsInit(&aSigmaPds);

    cStdNamePx2D aNAB(mPref,mNameImA,mNameImB,aNum);
    cStdNamePx2D aNBC(mPref,mNameImB,mNameImC,aNum);

    Tiff_Im aTXab(aNAB.mNX.c_str());
    Pt2di aSz = aTXab.sz();
    Tiff_Im aTYab(aNAB.mNY.c_str());

    Tiff_Im aTXbc(aNBC.mNX.c_str());
    Tiff_Im aTYbc(aNBC.mNY.c_str());

    Fonc_Num aDifX = aTXab.in() + aTXbc.in();
    Fonc_Num aDifY = aTYab.in() + aTYbc.in();

    if (mNameImA!=mNameImC)
    {
       cStdNamePx2D aNAC(mPref,mNameImA,mNameImC,aNum);

       Tiff_Im aTXac(aNAC.mNX.c_str());
       Tiff_Im aTYac(aNAC.mNY.c_str());

       aDifX = aDifX - aTXac.in();
       aDifY = aDifY - aTYac.in();
    }

    Im2D_REAL4 aIX(aSz.x,aSz.y);
    Im2D_REAL4 aIY(aSz.x,aSz.y);
    ELISE_COPY(aIX.all_pts(),Virgule(aDifX,aDifY),Virgule(aIX.out(),aIY.out()));
    Im2D_REAL4 aIPds(aSz.x,aSz.y);

    std::vector<double> aVD;
    double aSomD = 0;
    Pt2di aP;
    for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
    {
       for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
       {
           Pt2dr aDif(aIX.GetR(aP),aIY.GetR(aP));
           double aDist = euclid(aDif);
           aVD.push_back(aDist);
           aSomD += aDist;
           if (UsePds)
           {
              double aPds = 1/(1+ElSquare(aDist/aSigmaPds));
              aIPds.SetR(aP,aPds);
           }
       }
    }

    std::cout << "Moy Euclid = " << aSomD /(aSz.x*double(aSz.y)) << "\n";
    int aNbStat = 12 ; 
    for (int aK=1; aK<=aNbStat  ; aK++)
    {
       double aProp = 1- aK / double(aNbStat);
       aProp = 1-ElSquare(aProp);
       std::cout << "Residu at " << (aProp*100.0) << "% = " << KthValProp(aVD,aProp) << "\n";
    }



    std::string aDirRes("Tmp-Ferm-" + mPref  +  (EAMIsInit(&aNum) ?  ("-Num"+ToString(aNum)): "" ) + "/");
    ELISE_fp::MkDirSvp(aDirRes);

    std::string aPref = aDirRes + "Residu-" + mNameImA + "-" + mNameImB + "-"+ mNameImC;
    
    Tiff_Im::CreateFromFonc(aPref+"-X.tif",aSz,aIX.in(),GenIm::real4);
    Tiff_Im::CreateFromFonc(aPref+"-Y.tif",aSz,aIY.in(),GenIm::real4);
    Tiff_Im::CreateFromFonc(aPref+"-N.tif",aSz,sqrt(Square(aIX.in())+Square(aIY.in())),GenIm::real4);

    if (UsePds)
    {
       Tiff_Im::CreateFromFonc(aPref+"-P.tif",aSz,aIPds.in(),GenIm::real4);
    }

    return EXIT_SUCCESS;
}


int CPP_CmpDenseMap(int argc,char** argv)
{
    std::string mDir,mNameIm1A,mNameIm2A,mNameIm1B,mNameIm2B;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  <<  EAMC(mDir,"Pref where , Dir=MEC-${Pref}-{Im1}-{Im2}")
                    <<  EAMC(mNameIm1A,"Im1A")
                    <<  EAMC(mNameIm2A,"Im2A")
                    <<  EAMC(mNameIm1B,"Im1B")
                    <<  EAMC(mNameIm2B,"Im2B"),
        LArgMain()  
    );

    cStdNamePx2D aNA(mDir,mNameIm1A,mNameIm2A);
    cStdNamePx2D aNB(mDir,mNameIm1B,mNameIm2B);

    std::cout << aNA.mNX << "\n";
    std::cout << aNA.mNY << "\n";


    Tiff_Im aX1(aNA.mNX.c_str());
    Tiff_Im aY1(aNA.mNY.c_str());
    Tiff_Im aX2(aNB.mNX.c_str());
    Tiff_Im aY2(aNB.mNY.c_str());

    Pt2di aSz = aX1.sz();

    double aS1,aS2;

    ELISE_COPY(aX1.all_pts(),Abs(aX1.in())+Abs(aY1.in()),sigma(aS1));
    ELISE_COPY(aX1.all_pts(),Abs(aX2.in())+Abs(aY2.in()),sigma(aS2));

    double R12 = aS1/aS2;
    double aDif;

    ELISE_COPY
    (
         aX1.all_pts(),
         Abs(aX2.in()*R12-aX1.in())+Abs(aY2.in()*R12-aY1.in()),
         sigma(aDif)
    );

    std::cout << "R12 = " << R12  << " DIF=" << (aDif / (double (aSz.x) * aSz.y))<< "\n";
    return EXIT_SUCCESS;
}






typedef  std::pair<Pt2dr,Pt2dr>  tPairP;
void  Map2DRobustInit(const ElPackHomologue & aPackFull,cParamMap2DRobustInit & aParam)
{
   aParam.mRes = 0;
   std::vector<tPairP> aVRansac;
   int aNbPtsTot = aPackFull.size();

   // Selection de mNbMaxPtsRansac points repartis regulierement
   {
       int aCpt = 0;
       double aPropCons = ElMin(aParam.mNbMaxPtsRansac/double(aNbPtsTot),1.0);

       for (ElPackHomologue::tCstIter itH=aPackFull.begin() ; itH!=aPackFull.end() ; itH++)
       {
            Pt2dr aP1 = itH->P1();
            Pt2dr aP2 = itH->P2();
            tPairP aPair(aP1,aP2);

            if  ( round_ni((aCpt-1)*aPropCons) != round_ni(aCpt*aPropCons) )
            {
                aVRansac.push_back(aPair);
            }
       }
   }
   int aNbPRan = aVRansac.size();

   // Calcul du Ransac
   cElMap2D * aTestMap = cElMap2D::IdentFromType(aParam.mType,&aParam.mVAux);
   cElMap2D * aBestSol = cElMap2D::IdentFromType(aParam.mType,&aParam.mVAux);
   double aBestScRan = 1e60;
   {
      int aNbUk = aBestSol->NbUnknown();
      int aNbPtsReq = (aNbUk+1)/2;
      if (aNbPtsTot < aNbPtsReq)
        return;

      for (int aKTirage = 0 ; aKTirage<aParam.mNbTirRans ; aKTirage++)
      {
           // aPackEstim must contain point that are relatively far from each others
           ElPackHomologue aPackEstim;
           for (int aKP=0 ; aKP<aNbPtsReq ; aKP++)
           {
                int aNbTest = 0;
                double aDMaxMin = -1;
                int aKMM=-1; // KMaxMin
                while (aNbTest <  aParam.mNbTestFor1P)
                {
                     int aKP = NRrandom3(aNbPRan);
                     Pt2dr aP1 = aVRansac[aKP].first;
                     double aDMin = 1e60;
                     for (ElPackHomologue::const_iterator itP = aPackEstim.begin(); itP!=aPackEstim.end() ; itP++)
                     {
                         ElSetMin(aDMin,euclid(itP->P1(),aP1));
                     }
                     if (aDMin>0)
                     {
                         aNbTest++;
                         aDMin = NRrandom3() * aDMin;
                         if (aDMin>aDMaxMin)
                         {
                            aDMaxMin = aDMin;
                            aKMM = aKP;
                         }
                     }
                }
                ELISE_ASSERT(aKMM>=0,"Incoh in Map2DRobustInit");
                aPackEstim.Cple_Add(ElCplePtsHomologues(aVRansac[aKMM].first,aVRansac[aKMM].second));
           }
  
           // Une fois connu le pack on estime la map, puis le score
           L2EstimMapHom(aTestMap,aPackEstim);
           std::vector<double> aVDist;
           for (int aKP=0 ; aKP< aNbPRan ; aKP++)
           {
               const tPairP & aPair = aVRansac[aKP];
               aVDist.push_back(euclid((*aTestMap)(aPair.first) - aPair.second));
           }
           double aScore = KthValProp(aVDist,aParam.mPropRan);
           if (aScore< aBestScRan)
           {
              aBestScRan = aScore;
              aBestSol->Affect(*aTestMap);
           }
      }
   }
   double aD2Std  = ElMax(1e-60,ElSquare(aBestScRan));
   
   for (int aKItL2=0 ; aKItL2<aParam.mNbIterL2; aKItL2++)
   {
       aParam.mVPdsSol.clear();
       ElPackHomologue aPackEstim;
       // std::vector<double> aVD2;
       for (ElPackHomologue::tCstIter itH=aPackFull.begin() ; itH!=aPackFull.end() ; itH++)
       {
            Pt2dr aP1 = itH->P1();
            Pt2dr aP2 = itH->P2();
            double aD2 = square_euclid((*aBestSol)(aP1)-aP2);
            double aPds   = 1/ (1+ (4.0*aD2)/aD2Std);
            aPackEstim.Cple_Add(ElCplePtsHomologues(aP1,aP2,aPds));
            aParam.mVPdsSol.push_back(aD2);
       }
       aD2Std  = KthValProp(aParam.mVPdsSol,aParam.mPropRan);
       L2EstimMapHom(aBestSol,aPackEstim);
   }
   delete aTestMap;
   aParam.mRes = aBestSol;
}


/**********************************************************************/
/*                                                                    */
/*                     TESTS                                          */
/*                                                                    */
/**********************************************************************/

Pt2dr PRanCInSquare(double aScale) {return Pt2dr(aScale*NRrandC(),aScale*NRrandC());}

double TestMap2D(cElMap2D & aMapInit,const ElPackHomologue & aPackInit,bool WithRand)
{
    ElPackHomologue  aPack = aPackInit;
    for (ElPackHomologue::iterator itCpl=aPack.begin();itCpl!=aPack.end() ; itCpl++)
    {
         itCpl->P2() = (aMapInit)(itCpl->P1());
    } 
    bool IsPolyn =  (aMapInit.Type() == eTM2_Polyn);

    // On test aussi la fonction d'affectation
    cElMap2D * aMap = IsPolyn                                          ?
                      L2EstimMapHom(aMapInit.Identity(),aPack)                   :
                      L2EstimMapHom(eTypeMap2D(aMapInit.Type()),aPack) ;

    cElMap2D * aMap2 =  IsPolyn                                    ?
                        aMapInit.Identity()                        :
                        cElMap2D::IdentFromType(aMapInit.Type())   ;

    aMap2->Affect(*aMap);

    double aSomD = 0;
    for (ElPackHomologue::iterator itCpl=aPack.begin();itCpl!=aPack.end() ; itCpl++)
    {
         aSomD += euclid(aMapInit(itCpl->P1()) - (*aMap2)(itCpl->P1()));
    }

    // Randomization
    if (WithRand)
    {
       int aCpt = 0;
       for (ElPackHomologue::iterator itCpl=aPack.begin();itCpl!=aPack.end() ; itCpl++)
       {
           if ((aCpt%8)==0) // 12.5 % d'erreur
               itCpl->P2() =  itCpl->P2() +  PRanCInSquare(100.0);
            else 
               itCpl->P2() =  itCpl->P2() +  PRanCInSquare(0.5);
           aCpt ++;
       }
       std::vector<std::string> aVAux = aMapInit.ParamAux();
       cParamMap2DRobustInit aParam(eTypeMap2D(aMapInit.Type()),200,&aVAux);
       Map2DRobustInit(aPack,aParam);
       cElMap2D & aMRob = *(aParam.mRes);

       std::vector<double> aVDistTh;
       std::vector<double> aVDistEmp;
       for (ElPackHomologue::iterator itCpl=aPack.begin();itCpl!=aPack.end() ; itCpl++)
       {
            aVDistTh.push_back(euclid(aMapInit(itCpl->P1()) - (aMRob)(itCpl->P1())));
            aVDistEmp.push_back(euclid( aMRob(itCpl->P1()) - itCpl->P2()));
       }
       std::cout << "MED " << KthValProp(aVDistTh,0.5) << " " <<  KthValProp(aVDistEmp,0.5) << "\n";
       std::cout << " 75 " << KthValProp(aVDistTh,0.75) << " " <<  KthValProp(aVDistEmp,0.75) << "\n";
       std::cout << " 90 " << KthValProp(aVDistTh,0.9) << " " <<  KthValProp(aVDistEmp,0.9) << "\n\n";
    }


    return aSomD;
}

cElComposHomographie RanCH(double aScaleXY,double Cste){return cElComposHomographie(NRrandC()*aScaleXY,NRrandC()*aScaleXY,1+Cste*NRrandC());}

void TestMap2D()
{
    ElPackHomologue aPackInit;
    for (int aK=0 ; aK< 1000 ; aK++)
        aPackInit.Cple_Add(ElCplePtsHomologues(PRanCInSquare(1000),PRanCInSquare(1000),1.0));

    for (int aK=0 ; aK< 10 ; aK++)
    {
         ElSimilitude aS(PRanCInSquare(100),PRanCInSquare(1));
         ElAffin2D anAff(PRanCInSquare(100),PRanCInSquare(1),PRanCInSquare(1));
         cElHomographie aHom(RanCH(1,1000),RanCH(1,1000),RanCH(1e-5,1e-5));
         ElHomot  aHomot(PRanCInSquare(100), NRrandC());

         int aDeg=3+aK/4;
         Box2dr aBox(PRanCInSquare(100),PRanCInSquare(100));
         cMapPol2d aMapPol(aDeg,aBox,2);
         Polynome2dReal & aPolX = aMapPol.PolX();
         Polynome2dReal & aPolY = aMapPol.PolY();
         double anAmpl = aMapPol.Ampl();
         for (int aKm = 0 ; aKm< aPolX.NbMonome() ; aKm++)
         {
              aPolX.SetCoeff(aKm,1e-2*NRrandC()*pow(anAmpl,-aPolX.DegreTot(aKm)));
              aPolY.SetCoeff(aKm,1e-2*NRrandC()*pow(anAmpl,-aPolX.DegreTot(aKm)));
         }



         if (aK==5)
            std::cout << "=============================================\n";
         if (aK< 5)
         {
            std::cout << " # "
                      << " SIM " << TestMap2D(aS   ,aPackInit,false) 
                      << " AFF " << TestMap2D(anAff,aPackInit,false) 
                      << " Hom " << TestMap2D(aHom,aPackInit,false) 
                      << " Homot " << TestMap2D(aHomot,aPackInit,false) 
                      << " Polyn " << TestMap2D(aMapPol,aPackInit,false) 
                      << "\n";
          }
          else
          {
             TestMap2D(aS,aPackInit,true);
             TestMap2D(anAff,aPackInit,true);
             TestMap2D(aHom,aPackInit,true);
             TestMap2D(aHomot,aPackInit,true);
             TestMap2D(aMapPol,aPackInit,true);
             std::cout << "  -  -  -  -  -  -  -  -  -\n";
          }

    }
    exit(EXIT_SUCCESS);
}


/*********************************************/
/*                                           */
/*       Map2Evol                            */
/*                                           */
/*********************************************/

// cXYMapPol2d
// P(T,x,y) =  Sum ( T^k P_k(x,y)) = 

     //   Calcul map evol

class cMapPolXYT
{
     public :
         cMapPolXYT(int aDegreT,const Pt2dr & aBoxT,int aDegXY,const Box2dr & aBox,const std::vector<std::string> & aPAux);
         void   AddEq(eTypeMap2D aType,L2SysSurResol & aSys,const ElPackHomologue & aPack,const double & aTemp);
         int NbUnknown() const;
         void  InitFromSol(double * aSol);
         cMapPol2d SolOfTemp(double aTemp) const;
    
         double Test(const cMapPol2d &,const std::string & aName,const ElPackHomologue & aPack,const double & aTemp);
         cXml_EvolMap2dPol ToXML() const;
         static cMapPolXYT FromXml(const cXml_EvolMap2dPol &);
     private :
         double ToTNorm(const double & aT) const;

         
         Pt2dr  mBoxT;
         double mT0;
         double mAmplT;

         Box2dr                   mBoxXY;
         int                      mDegXY;
         int                      mDegT;
         cMapPol2d                mMapPolTmp; // Buferr
         std::vector<cMapPol2d>   mVMaps;
         int                      mNbUXY;
         int                      mNbUnknown;
         std::vector<std::string> mUserParamAux;
};

//  cMapPolXYT(int aDegreT,const Pt2dr & aBoxT,int aDegXY,const Box2dr & aBox);
cMapPolXYT cMapPolXYT::FromXml(const cXml_EvolMap2dPol & aXml)
{
   std::vector<std::string> aParamAux;
   cMapPolXYT aRes(aXml.DegT(),aXml.IntervT(),aXml.DegXY(),aXml.BoxXY(),aParamAux);

   for (int aK=0 ; aK<int(aXml.PolOfT().size()) ; aK++)
   {
      aRes.mVMaps.push_back(cMapPol2d::FromXml(aXml.PolOfT()[aK]));
   }
   return aRes;
}

cXml_EvolMap2dPol cMapPolXYT::ToXML() const
{
    cXml_EvolMap2dPol aRes;

    aRes.DegT()    = mDegT;
    aRes.IntervT() = mBoxT;
    aRes.DegXY()   = mDegXY;
    aRes.BoxXY()   = mBoxXY;
    for (int aK=0 ; aK<int(mVMaps.size()) ; aK++)
    {
        aRes.PolOfT().push_back(mVMaps[aK].ToXmlPol());
    }

    return aRes;
}

cMapPolXYT::cMapPolXYT(int aDegreT,const Pt2dr & aBoxT,int aDegXY,const Box2dr & aBox,const std::vector<std::string> & aPAux) :
    mBoxT         (aBoxT),
    mT0           ((aBoxT.x+aBoxT.y)/2.0),
    mAmplT        (ElAbs(aBoxT.x-aBoxT.y)),
    mBoxXY        (aBox),
    mDegXY        (aDegXY),
    mDegT         (aDegreT),
    mMapPolTmp    (mDegXY,mBoxXY),
    mNbUXY        (mMapPolTmp.NbUnknown()),
    mNbUnknown    (mNbUXY * (1+mDegT)),
    mUserParamAux (aPAux)
{
}

int cMapPolXYT::NbUnknown() const
{
   return mNbUnknown;
}

double cMapPolXYT::ToTNorm(const double & aT) const
{
    return (aT-mT0) / mAmplT;
}

double  cMapPolXYT::Test(const cMapPol2d & aMap,const std::string & aName,const ElPackHomologue & aPack,const double & aTemp)
{
    double aSomDP=0;
    double aSomP=0;
    std::vector<double>  aVD;

    for (ElPackHomologue::const_iterator itP= aPack.begin() ; itP!=aPack.end() ; itP++)
    {
          double aD = euclid(aMap(itP->P1())-itP->P2());
          double aPds = itP->Pds();
          aSomDP += aD * aPds;
          aSomP  += aPds;
          aVD.push_back(aD);
    }

    std::cout << "Residual, For " << aName 
              << " moy=" << aSomDP/aSomP 
              << " med=" << KthValProp(aVD,0.5)
              << " %80=" << KthValProp(aVD,0.8)
              << "\n";
   return aSomDP/aSomP;
}


void   cMapPolXYT::AddEq(eTypeMap2D aType,L2SysSurResol & aSys,const ElPackHomologue & aPackInit,const double & aTemp)
{
    ElPackHomologue aPack = aPackInit;
    // cMapPol2d aMapPol(mDegXY,mBoxXY,2);

    //int aDeg,const Box2dr & aBox0,int aRabDegInv=2);
    //std::vector<std::string>  cMapPol2d::ParamAux() const
    if (aType!=eTM2_NbVals)
    {
        std::vector<std::string> aParAux ;
        if (aType ==eTM2_Polyn) 
        {
           aParAux =   mMapPolTmp.ParamAux();
        }
        else if (aType == eTM2_HomotPure)
        {
           aParAux = mUserParamAux;
        }
        
        cElMap2D * aMapEstim = L2EstimMapHom(aType,aPackInit,&aParAux);
        for (ElPackHomologue::iterator itP=aPack.begin(); itP!=aPack.end() ; itP++)
        {
             itP->P2() = (*aMapEstim) (itP->P1());
        }
        delete aMapEstim;
    }
    std::vector<double> aX_xy(mNbUXY);
    std::vector<double> aY_xy(mNbUXY);
    std::vector<double> aX_xyt(mNbUnknown);
    std::vector<double> aY_xyt(mNbUnknown);

    for (ElPackHomologue::iterator itP=aPack.begin(); itP!=aPack.end() ; itP++)
    {
         Pt2dr aCste;
         mMapPolTmp.AddEq(aCste,aX_xy,aY_xy,itP->P1(),itP->P2());
         for (int aD=0 ; aD<=mDegT ; aD++)
         {
             double aCoefT = pow(ToTNorm(aTemp),aD);
             int aInd = aD * mNbUXY;
             for (int aK=0 ; aK<mNbUXY ; aK++)
             {
                 aX_xyt[aK+aInd] = aCoefT * aX_xy[aK];
                 aY_xyt[aK+aInd] = aCoefT * aY_xy[aK];
             }
         }
         aSys.AddEquation(itP->Pds(),VData(aX_xyt),itP->P2().x);
         aSys.AddEquation(itP->Pds(),VData(aY_xyt),itP->P2().y);
    }
}

void  cMapPolXYT::InitFromSol(double * aSol)
{
    mVMaps.clear();
    for (int aD=0 ; aD<= mDegT ; aD++)
    {
        cMapPol2d aMap = mMapPolTmp;
        std::vector<double> aVD(aSol+aD*mNbUXY,aSol+(aD+1)*mNbUXY);
        aMap.InitFromParams(aVD);
        mVMaps.push_back(aMap);
    }
}

cMapPol2d cMapPolXYT::SolOfTemp(double aTemp) const
{
    std::vector<double> aVP(mNbUXY,0.0);

    for (int aD=0 ; aD<= mDegT ; aD++)
    {
        std::vector<double>  aVt = mVMaps[aD].Params();
        double aCoefT = pow(ToTNorm(aTemp),aD);
        for (int aKxy=0 ; aKxy<mNbUXY ; aKxy++)
        {
            aVP[aKxy] += aVt[aKxy] * aCoefT;
        }
    }
    cMapPol2d aRes = mMapPolTmp;
    aRes.InitFromParams(aVP);
    return aRes;
}

class cAppliCalcMapXYT
{
     public :
         cAppliCalcMapXYT(int,char **);

     private :
         void AddIm(const std::string &,bool isMaster);
         std::string   mPat;
         std::string   mMaster;
         std::string   mKeyCalT;
         std::string   mSH;
         std::string   mExt;
         std::string   mNameType;
         bool          mFirstIm;
         Pt2di         mSz;
         std::vector<ElPackHomologue> mVPack;
         std::vector<ElPackHomologue> mVPackNorm;
         std::vector<double>          mVTemp;
         std::vector<string>          mVNames;
         std::set<string>             mSetIm;
         cInterfChantierNameManipulateur * mICNM;
         double                            mPdsMaster;
         eTypeMap2D                        mType;
         int                               mDegreT;
         int                               mDegreXY;
         double                            mTempMin;
         double                            mTempMax;
         std::string                       mNameOut;
         std::string                       mOriCmpRot;
         std::vector<std::string>          mUserParamAux;

};

void cAppliCalcMapXYT::AddIm(const std::string & aNameIm,bool isMaster)
{
    if (BoolFind(mSetIm,aNameIm)) 
       return;
    if ( isMaster &&  (aNameIm!=mMaster))
       return;

    mSetIm.insert(aNameIm);
    Tiff_Im aTifIm = Tiff_Im::StdConvGen(aNameIm,-1,true);
    Pt2di aSz = aTifIm.sz();


    if (isMaster)
    {
       ElPackHomologue aNewPack;
       for (int aKp=0 ; aKp<int(mVPack.size()) ; aKp++)
       {
            const ElPackHomologue & aPack = mVPack[aKp];
            for (ElPackHomologue::const_iterator itP= aPack.begin() ; itP!=aPack.end() ; itP++)
            {
                 aNewPack.Cple_Add(ElCplePtsHomologues(itP->P1(),itP->P1(),itP->Pds()*mPdsMaster/mVPack.size()));
            }
       }
       mVPack.push_back(aNewPack);
    }
    else
    {
       std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@") +  std::string(mSH) +  std::string("@") +  std::string(mExt);

       std::string aNameH = mICNM->Assoc1To2(aKHIn,mMaster,aNameIm,true);
       mVPack.push_back(ElPackHomologue::FromFile(aNameH));
    }

    mVNames.push_back(aNameIm);

    if ((mKeyCalT=="THOM") ||  (mKeyCalT=="NUM") ||  (mKeyCalT=="VEC"))
    {
        std::cout << "For key= " << mKeyCalT << "\n";
        ELISE_ASSERT(false,"Cannot compute special key in CalcMapXYT");
    }
    else 
    {
         std::string aStrTemp = mICNM->Assoc1To1(mKeyCalT,aNameIm,true);
         double aTemp;
         FromString(aTemp,aStrTemp);
         mVTemp.push_back(aTemp);
    }

    if (mFirstIm)
    {
       mSz = aSz;
       mTempMin = mVTemp.back();
       mTempMax = mVTemp.back();
    }
    else
    {
       mSz = Sup(aSz,mSz);
       ElSetMin(mTempMin,mVTemp.back());
       ElSetMax(mTempMax,mVTemp.back());
    }

    mFirstIm = false;
}

ElPackHomologue   CompenseRotationFromPoints(const ElPackHomologue & aPack,const CamStenope & aCam1,const CamStenope & aCam2);


cAppliCalcMapXYT::cAppliCalcMapXYT(int argc,char ** argv) :
    mSH         (""),
    mExt        ("dat"),
    mFirstIm    (true),
    mPdsMaster  (1.0),
    mType       (eTM2_NbVals),
    mNameOut    ("PolOfTXY.xml")
{
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  
                    <<  EAMC(mMaster,"Name master")
                    <<  EAMC(mPat,"Pattern of im")
                    <<  EAMC(mDegreT,"Degre of polynom for temp")
                    <<  EAMC(mDegreXY,"Degre of polynom for XY")
                    <<  EAMC(mKeyCalT,"Key to calc T (THOM->read thom mtd, NUM->order,VEC->use VT args )")
                    <<  EAMC(mSH,"Set of homologue"),
        LArgMain()  <<  EAM(mNameType,"Model",true,"Model in [Homot,Simil,Affine,Homogr,Polyn]")
                    <<  EAM(mPdsMaster,"PdsM",true,"Pds for master, def=1")
                    <<  EAM(mNameOut,"Out",true,"file for result, def=PolOfTXY.xml")
                    <<  EAM(mOriCmpRot,"OriCmpRot",true,"Orientation folder, to compense rotation")
                    <<  EAM(mUserParamAux,"ParamAux",true,"Param Aux, [Xc,Yc] for HomotPure")
    );

    cElemAppliSetFile anEASF(mPat);
    mICNM = anEASF.mICNM;

    const cInterfChantierNameManipulateur::tSet * aSet = anEASF.SetIm();

    for (int aK=0 ; aK<int(aSet->size()) ; aK++)
    {
        AddIm((*aSet)[aK],false);
    }
    AddIm(mMaster,true);
    if (EAMIsInit(&mNameType))
    {
        bool aModeHelp;
        StdReadEnum(aModeHelp,mType,std::string("TM2_")+mNameType,eTM2_NbVals);
    }

    cMapPolXYT aMapXYT(mDegreT,Pt2dr(mTempMin,mTempMax),mDegreXY,Box2dr(Pt2dr(0,0),Pt2dr(mSz)),mUserParamAux);

    L2SysSurResol aSys(aMapXYT.NbUnknown());
    for (int aK=0 ; aK<int(mVPack.size()) ; aK++)
    {
         ElPackHomologue aPack = mVPack[aK];
         if (EAMIsInit(&mOriCmpRot))
         {
            // mICNM->StdNameCalib(mOriCmpRot,mVNames[aK]);
            // mICNM->StdNameCalib(mOriCmpRot,mMaster);
            CamStenope * aCam1 = mICNM->GlobCalibOfName(mMaster    ,mOriCmpRot,false);
            CamStenope * aCam2 = mICNM->GlobCalibOfName(mVNames[aK],mOriCmpRot,false);

            aPack = CompenseRotationFromPoints(aPack,*aCam1,*aCam2);
         }
         mVPackNorm.push_back(aPack);
         aMapXYT.AddEq(mType,aSys,aPack,mVTemp[aK]);
         std::cout << " NAME " << mVNames[aK] << " TMP=" << mVTemp[aK] << "\n";
    }

    Im1D_REAL8 aSol = aSys.GSSR_Solve((bool*)0);
    aMapXYT.InitFromSol(aSol.data());

    
    double aSomD = 0;
    for (int aK=0 ; aK<int(mVPack.size()) ; aK++)
    {
        cMapPol2d aMap = aMapXYT.SolOfTemp(mVTemp[aK]);
        // aMap.SaveInFile("XML-"+mVNames[aK] + ".xml");
        aSomD += aMapXYT.Test(aMap,mVNames[aK],mVPackNorm[aK],mVTemp[aK]);
    }
    std::cout << " *** MOY DIST GLOB = " << aSomD /mVPack.size() << " ***\n";
    MakeFileXML(aMapXYT.ToXML(),mNameOut);
// cMapPolXYT(int aDegreT,const Pt2dr & aBoxT,int aDegXY,const Box2dr & aBox);
}

int CPP_CalcMapXYT(int argc,char ** argv)
{
   cAppliCalcMapXYT anAppli(argc,argv);

   return EXIT_SUCCESS;
}


     //   ================  use map evol  ==============


int CPP_MakeMapEvolOfT(int argc,char ** argv)
{
    std::string  aNameMapEvol, aNameOut;
    double aTemp;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  
                    <<  EAMC(aNameMapEvol,"Name of map evol")
                    <<  EAMC(aTemp,"Value of \"Temperature\""),
        LArgMain()  
                    <<  EAM(aNameOut,"Out",true,"Name for result")
    );

    cXml_EvolMap2dPol aXml_MapE= StdGetFromSI(aNameMapEvol,Xml_EvolMap2dPol);
    cMapPolXYT  aMapEv = cMapPolXYT::FromXml(aXml_MapE);
    cMapPol2d aMap = aMapEv.SolOfTemp(aTemp);

    if (!EAMIsInit(&aNameOut)) 
       aNameOut = StdPrefix(aNameMapEvol) + "-" + ToString(aTemp) + ".xml";

    aMap.SaveInFile(aNameOut);

    return EXIT_SUCCESS;
}


int CPP_PolynOfImageStd(int argc,char ** argv)
{
    std::string aNameIm;
    std::string aMasq;
    std::string aNameOut="FitPolyIm.tif";
    std::string aNameMapOut="FitPolyIm.xml";

    Pt2di       aP0(100,100);
    Pt2di       aP1(1000,1000);

    int         aNb;
    int         aDeg=2;
    Box2dr      aBox(aP0,aP0+aP1);


//    eTypeMap2D aType="eTM2_Polyn";

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aNameIm,"Image name")
                   << EAMC(aNb,"Number of points in X (and Y respectively)"),
        LArgMain() << EAM(aP0,"P0",true,"P0 of the bounding box")
                   << EAM(aP1,"P1",true,"P1 of the bounding box")
                   << EAM(aDeg,"Deg",true,"Polynom degree")
                   << EAM(aNameOut,"Out",true,"Name of the output image")
                   << EAM(aMasq,"Masq",true,"Name of the mask image")
    );


    //lecture d'une image
    Tiff_Im aTifIn = Tiff_Im::StdConvGen(aNameIm,-1,true);
    Pt2di   aTifSz = aTifIn.sz();

    Im2D_REAL4 aImR(aTifSz.x, aTifSz.y);
    ELISE_COPY
    (
        aImR.all_pts(),
        aTifIn.in(),
        aImR.out()
    );


    Im2D_REAL4 aMasqIm(aTifSz.x,aTifSz.y,1.0);
    if (EAMIsInit(&aMasq))
    {
        Tiff_Im aMasqTif = Tiff_Im::StdConvGen(aMasq,-1,true);
        
        ELISE_COPY
        (
            aMasqIm.all_pts(),
            aMasqTif.in(),
            aMasqIm.out()
        );
    }

    Im2D_REAL8 aImRes(aTifSz.x,aTifSz.y,0.0);

    Pt2di aPas(floor(double(aP1.x-aP0.x)/aNb), floor(double(aP1.y-aP0.y)/aNb));

    ElPackHomologue aPack;
    for (int aK1=aP0.x; aK1<aP1.x; aK1=aK1+aPas.x)
    {
        for (int aK2=aP0.y; aK2<aP1.y; aK2=aK2+aPas.y)
        {
            Pt2dr aP(aK1,aK2);
	    if(aMasqIm.Val(aP.x,aP.y))
	    {

		if(ERupnik_MM())
		    std::cout << "* aK1=" << aK1 << ", aK2" << aK2 << ", aPas=" << aPas << " ---- ImR=" << aImR.Val(aP.x,aP.y) <<  "\n";


                double aD(aImR.Val(aP.x,aP.y));
                aPack.Cple_Add(ElCplePtsHomologues(aP,aP+Pt2dr(aD,aD),aMasqIm.Val(aP.x,aP.y)));  
	    }
        }
    }


    cMapPol2d aMapPol(aDeg,aBox,2);
    std::vector<std::string> aVAux = aMapPol.ParamAux();
    cParamMap2DRobustInit aParam(eTypeMap2D(aMapPol.Type()),200,&aVAux);
    Map2DRobustInit(aPack,aParam);

    cElMap2D * aMapCor= aParam.mRes;
    std::vector<cElMap2D *> aVMap;
    aVMap.push_back(aMapCor);
    cComposElMap2D aComp(aVMap);

    for (int aK1=aP0.x; aK1<aP1.x; aK1++)
    {
        for (int aK2=aP0.y; aK2<aP1.y; aK2++)
        {
	    if(aMasqIm.Val(aK1,aK2))
            {
                Pt2dr  aP(aK1,aK2);
                double aRes  = aComp(aP).x - aP.x;

                aImRes.SetR_SVP(Pt2di(aP.x,aP.y),aRes);
	    }
        }
    }
    MakeFileXML(aComp.ToXmlGen(),aNameMapOut);

    Tiff_Im::CreateFromIm(aImRes,aNameOut);

    return EXIT_SUCCESS;




}
  
/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
