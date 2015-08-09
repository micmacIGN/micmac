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

class cBGC3_Modif2D ; //   : public cBasicGeomCap3D
class cPolynomial_BGC3M2D ;//  : public cBGC3_Modif2D
class cPolynBGC3M2D_Formelle; // : public cGenPDVFormelle
class cOneEq_PBGC3M2DF;

class cBGC3_Modif2D  : public cBasicGeomCap3D
{
      public : 
           cBGC3_Modif2D(cBasicGeomCap3D * aCam0);




           virtual ElSeg3D  Capteur2RayTer(const Pt2dr & aP) const ;
           virtual Pt2dr    Ter2Capteur   (const Pt3dr & aP) const ;
           virtual Pt2di    SzBasicCapt3D() const ;
           virtual double ResolSolOfPt(const Pt3dr &) const ;
           virtual bool  CaptHasData(const Pt2dr &) const ;
           virtual bool     PIsVisibleInImage   (const Pt3dr & aP) const ;

  // Optical center 
           virtual bool     HasOpticalCenterOfPixel() const; // 1 - They are not alway defined
  // When they are, they may vary, as with push-broom, Def fatal erreur (=> Ortho cam)
           virtual Pt3dr    OpticalCenterOfPixel(const Pt2dr & aP) const ;

           inline Pt2dr CamInit2CurIm(const Pt2dr & aP) const{return aP+DeltaCamInit2CurIm(aP);}
           inline Pt2dr CurIm2CamInit(const Pt2dr & aP) const{return aP+DeltaCurIm2CamInit(aP);}

      protected  : 
            cBasicGeomCap3D * mCam0;
            Pt2di  mSz;

      private  : 

            // Ter2Cam (x) = x + DifCorTer2Cal(x)

            virtual Pt2dr DeltaCamInit2CurIm(const Pt2dr & aP) const = 0;
            Pt2dr   DeltaCurIm2CamInit(const Pt2dr & aP) const ;
            

};


class cPolynomial_BGC3M2D  : public cBGC3_Modif2D
{
      public : 
           cPolynomial_BGC3M2D(cBasicGeomCap3D * aCam0,int aDegree);
           Pt2dr DeltaCamInit2CurIm(const Pt2dr & aP) const ;
           inline Pt2dr ToPNorm(const Pt2dr aP) const {return (aP-mCenter)/mAmpl;}
           inline Pt2dr FromPNorm(const Pt2dr aP) const {return aP*mAmpl + mCenter;}

           std::vector<double> & Cx();
           std::vector<double> & Cy();
           inline int DegX(int aK) const {return mDegX.at(aK);}
           inline int DegY(int aK) const {return mDegY.at(aK);}
           inline int DegreMax()   const {return mDegreMax;}
      private : 
           void Show() const;
           void Show(const std::string & aMes,const std::vector<double> & aCoef) const;
           void ShowMonome(const std::string & , int aDeg) const;
           void SetPow(const Pt2dr & aPN) const;
 
           int                 mDegreMax;
           Pt2dr               mCenter;
           double              mAmpl;
           std::vector<double> mCx;
           std::vector<double> mCy;

           static std::vector<int> mDegX;
           static std::vector<int> mDegY;


           static std::vector<double> mPowX;
           static std::vector<double> mPowY;
           mutable Pt2dr  mCurPPow;
};
/*
mPTerrain       (   isPTerrainFixe ?
                        Pt3d<Fonc_Num>(cVarSpec(0,mNameTerX),cVarSpec(0,mNameTerY),cVarSpec(0,mNameTerZ)):
                        mEqP3I->PF()
                    ),
*/

class cOneEq_PBGC3M2DF : public cElemEqFormelle,
                         public cObjFormel2Destroy

{
    public :
       cOneEq_PBGC3M2DF(cPolynBGC3M2D_Formelle &,std::vector<double > &);

       Fonc_Num  FormProjCor(Pt2d<Fonc_Num> aP);
       
   private :
       std::vector<Fonc_Num>     mVFCoef;
       cPolynBGC3M2D_Formelle *  mPF;
       cPolynomial_BGC3M2D*      mCamCur;
};



class cPolynBGC3M2D_Formelle : public cGenPDVFormelle
{

    public  :
         friend class cOneEq_PBGC3M2DF;

         cPolynBGC3M2D_Formelle(cSetEqFormelles & aSet,cPolynomial_BGC3M2D aCam0,bool GenCode);
         void GenerateCode();
    private :
         cPolynBGC3M2D_Formelle(const cPolynBGC3M2D_Formelle &); // N.I.


   // ==> To unvirtualize cGenPDVFormelle 
         const cBasicGeomCap3D * GPF_CurBGCap3D() const ;
         cBasicGeomCap3D * GPF_NC_CurBGCap3D() ;
         Pt2d<Fonc_Num>  FormProj();
         Pt2dr AddEqAppuisInc(const Pt2dr & aPIm,double aPds, cParamPtProj &,bool IsEqDroite);
         cIncListInterv & IntervAppuisPtsInc() ;

         cPolynomial_BGC3M2D mCam0;
         cPolynomial_BGC3M2D mCamCur;



         cEqfP3dIncTmp * mEqP3I;

         cVarEtat_PhgrF    mFAmpl;
         cP2d_Etat_PhgrF   mFCentr;

         cP3d_Etat_PhgrF   mFP3DInit;
         cP2d_Etat_PhgrF   mFProjInit;

         cP2d_Etat_PhgrF   mFGradX;
         cP2d_Etat_PhgrF   mFGradY;
         cP2d_Etat_PhgrF   mFGradZ;

         cOneEq_PBGC3M2DF    mCompX;
         cOneEq_PBGC3M2DF    mCompY;
         std::string         mNameType;
         cIncListInterv      mLInterv;
  
};


/***************************************************************/
/*                                                             */
/*               cOneEq_PBGC3M2DF                              */
/*            cPolynBGC3M2D_Formelle                           */
/*                                                             */
/***************************************************************/

               // ============ cOneEq_PBGC3M2DF =================

cOneEq_PBGC3M2DF::cOneEq_PBGC3M2DF(cPolynBGC3M2D_Formelle & aPF,std::vector<double > & aCoef) :
   cElemEqFormelle (aPF.Set(),false),
   mPF     (&aPF),
   mCamCur (&(aPF.mCamCur))
{
    for (int aK=0 ; aK<int(aCoef.size()) ; aK++)
    {
        mVFCoef.push_back(aPF.Set().Alloc().NewF(&(aCoef[aK])));
    }

    CloseEEF();
    aPF.Set().AddObj2Kill(this);

}


Fonc_Num  cOneEq_PBGC3M2DF::FormProjCor(Pt2d<Fonc_Num> aP)
{
   //  ELISE_ASSERT(false,"cOneEq_PBGC3M2DF::FormProjCor 2 complete");
   Fonc_Num aRes = 0;
   for (int aK=0 ; aK<int(mVFCoef.size()) ; aK++)
   {
       aRes = aRes + mVFCoef[aK] * PowI(aP.x,mCamCur->DegX(aK)) *  PowI(aP.y,mCamCur->DegX(aK));
   }
   return aRes;
}


               // ============ cPolynBGC3M2D_Formelle =================



cPolynBGC3M2D_Formelle::cPolynBGC3M2D_Formelle
(
        cSetEqFormelles &   aSet,
        cPolynomial_BGC3M2D aCam0,
        bool                GenCode
) :
   cGenPDVFormelle (aSet),
   mCam0           (aCam0),
   mCamCur         (aCam0),
   mEqP3I          (mSet.Pt3dIncTmp()),
   mFAmpl          ("Ampl"),
   mFCentr         ("Centr"),
   mFP3DInit       ("PTerInit"),
   mFProjInit      ("ProjInit"),
   mFGradX         ("GradX"),
   mFGradY         ("GradY"),
   mFGradZ         ("GradZ"),
   mCompX          (*this,mCamCur.Cx()),
   mCompY          (*this,mCamCur.Cy()),
   mNameType       ("cGen2DBundleEgProj_Deg"+ToString(mCamCur.DegreMax()))
{
    mCompX.IncInterv().SetName("CX");
    mCompY.IncInterv().SetName("CY");
    mLInterv.AddInterv(mEqP3I->IncInterv());
    mLInterv.AddInterv(mCompX.IncInterv());
    mLInterv.AddInterv(mCompY.IncInterv());

    if (GenCode)
    {
        GenerateCode();
        return;
    } 
}

cBasicGeomCap3D * cPolynBGC3M2D_Formelle::GPF_NC_CurBGCap3D() { return & mCamCur; }
const cBasicGeomCap3D *  cPolynBGC3M2D_Formelle::GPF_CurBGCap3D() const { return & mCamCur; }

Pt2d<Fonc_Num>  cPolynBGC3M2D_Formelle::FormProj()
{
   Pt3d<Fonc_Num>  aPTerUnknown  = mEqP3I->PF();
   Pt3d<Fonc_Num>  aDeltaPTU = aPTerUnknown-mFP3DInit.PtF();
 

   Fonc_Num aDptUX = aDeltaPTU.x;
   Pt2d<Fonc_Num>  aProj  =   mFProjInit.PtF() 
                            + mFGradX.PtF().mul(aDeltaPTU.x) 
                            + mFGradY.PtF().mul(aDeltaPTU.y) 
                            + mFGradZ.PtF().mul(aDeltaPTU.z);

   

   Pt2d<Fonc_Num>  aPPN = (aProj-mFCentr.PtF()).div(mFAmpl.FN());

   return    aProj 
          +  Pt2d<Fonc_Num>
             (
                 mCompX.FormProjCor(aPPN),
                 mCompY.FormProjCor(aPPN)
             );
}

void  cPolynBGC3M2D_Formelle::GenerateCode()
{
    Pt2d<Fonc_Num>  aFProj =  FormProj();
    std::vector<Fonc_Num> aV;
    aV.push_back(aFProj.x);
    aV.push_back(aFProj.y);

    cElCompileFN::DoEverything
    (
        DIRECTORY_GENCODE_FORMEL,  // Directory ou est localise le code genere
        mNameType,  // donne les noms de fichier .cpp et .h ainsi que les nom de classe
        aV,  // expressions formelles 
        mLInterv  // intervalle de reference
    );

}



Pt2dr cPolynBGC3M2D_Formelle::AddEqAppuisInc(const Pt2dr & aPIm,double aPds, cParamPtProj &,bool IsEqDroite)
{
    ELISE_ASSERT(false,"cPolynBGC3M2D_Formelle::AddEqAppuisInc still un implemanted");
    return Pt2dr(0,0);
}


cIncListInterv & cPolynBGC3M2D_Formelle::IntervAppuisPtsInc() 
{
   return mLInterv;
}

/***************************************************************/
/*                                                             */
/*                    cBGC3_Modif2D                            */
/*                                                             */
/***************************************************************/

cBGC3_Modif2D::cBGC3_Modif2D(cBasicGeomCap3D * aCam0) :
    mCam0 (aCam0),
    mSz (mCam0->SzBasicCapt3D())
{
}

ElSeg3D  cBGC3_Modif2D::Capteur2RayTer(const Pt2dr & aP) const
{
    return mCam0->Capteur2RayTer(CurIm2CamInit(aP));
}

Pt2dr  cBGC3_Modif2D::Ter2Capteur(const Pt3dr & aP) const
{
    return CamInit2CurIm(mCam0->Ter2Capteur(aP));
}


Pt2di  cBGC3_Modif2D::SzBasicCapt3D() const
{
    return mSz;
}

double cBGC3_Modif2D::ResolSolOfPt(const Pt3dr & aP) const 
{
   return mCam0->ResolSolOfPt(aP);
}


bool  cBGC3_Modif2D::CaptHasData(const Pt2dr &aP) const
{
    return  mCam0->CaptHasData(CurIm2CamInit(aP));
}

bool      cBGC3_Modif2D::PIsVisibleInImage(const Pt3dr & aP) const
{
    return mCam0->PIsVisibleInImage(aP);
}

bool      cBGC3_Modif2D::HasOpticalCenterOfPixel() const
{
   return mCam0->HasOpticalCenterOfPixel();
}


Pt3dr    cBGC3_Modif2D::OpticalCenterOfPixel(const Pt2dr & aP) const 
{
   return mCam0->OpticalCenterOfPixel(CurIm2CamInit(aP));
}


          // A AFFINER PLUS TARD !!!!!  Version de base du point fixe a 1 iter ...
Pt2dr   cBGC3_Modif2D::DeltaCurIm2CamInit(const Pt2dr & aP) const
{
    Pt2dr aSol = -DeltaCamInit2CurIm(aP);


    Pt2dr aTest = CamInit2CurIm(aP+aSol);
    aSol = aSol + (aP-aTest);


    return aSol;
}

// ==========================================================

/***************************************************************/
/*                                                             */
/*                    cPolynomial_BGC3M2D                      */
/*                                                             */
/***************************************************************/



std::vector<int> cPolynomial_BGC3M2D::mDegX;
std::vector<int> cPolynomial_BGC3M2D::mDegY;

std::vector<double> cPolynomial_BGC3M2D::mPowX;
std::vector<double> cPolynomial_BGC3M2D::mPowY;

std::vector<double> & cPolynomial_BGC3M2D::Cx() {return mCx;}
std::vector<double> & cPolynomial_BGC3M2D::Cy() {return mCy;}

void cPolynomial_BGC3M2D::SetPow(const Pt2dr & aPN) const
{
     if (aPN==mCurPPow) return;
     mCurPPow = aPN;
     for (int aD=1 ; aD<= mDegreMax ; aD++)
     {
           mPowX[aD] = mPowX[aD-1] * aPN.x;
           mPowY[aD] = mPowY[aD-1] * aPN.y;
     }
}


Pt2dr cPolynomial_BGC3M2D::DeltaCamInit2CurIm(const Pt2dr & aP0) const 
{
      Pt2dr aPN = ToPNorm(aP0); 
      SetPow(aPN);
      double aSx=0 ;
      double aSy=0 ;

      for (int aK=0 ; aK<int(mCx.size()) ; aK++)
      {
          double aPXY = mPowX[mDegX[aK]] * mPowY[mDegY[aK]] ;

          aSx += mCx[aK] * aPXY;
          aSy += mCy[aK] * aPXY;
      }


      return Pt2dr(aSx,aSy);
}



cPolynomial_BGC3M2D::cPolynomial_BGC3M2D(cBasicGeomCap3D * aCam0,int aDegreeMax) :
    cBGC3_Modif2D (aCam0),
    mDegreMax     (aDegreeMax),
    mCenter       (Pt2dr(mSz)/2.0),
    mAmpl         (euclid(mCenter)),
    mCurPPow      (0.0,0.0)
{

     int aCpt=0;
     for (int  aDegreeTot=0 ; aDegreeTot<=aDegreeMax ; aDegreeTot++)
     {
          for (int aDegX=0 ; aDegX<= aDegreeTot ; aDegX++)
          {
              int aDegY=aDegreeTot - aDegX;
              mCx.push_back(0);
              mCy.push_back(0);
              if (aCpt>=int(mDegX.size()))
              {
                  mDegX.push_back(aDegX);
                  mDegY.push_back(aDegY);
              }
              aCpt++;
          }
          if (int(mPowX.size()) <= aDegreeTot)
          {
              mPowX.push_back(1.0);
              mPowY.push_back(1.0);
          }
     }
     if (0)
     {
         Show();
     }
}

void cPolynomial_BGC3M2D::Show() const
{
    std::cout << "#### DMax= " << mDegreMax 
              << "  SizPow=" << mPowX.size()
              << "\n";
    Show("CoefX",mCx);
    Show("CoefY",mCy);
}

void cPolynomial_BGC3M2D::ShowMonome(const std::string & aVar , int aDeg) const
{
    if (aDeg==0) return;
    std::cout << "*" << aVar;
    if (aDeg==1) return;
    std::cout << "^" << aDeg ;
}
  

void cPolynomial_BGC3M2D::Show(const std::string & aMes,const std::vector<double> & aCoef) const
{
     std::cout << " -*-*-*- " << aMes << " -*-*-*-\n";
     for (int aK=0 ; aK<int(aCoef.size()) ; aK++)
     {
          std::cout << "    ";
          std::cout << ((aK==0) ? "  " : "+ ");
          std::cout << aCoef[aK] ;
          ShowMonome("X",mDegX[aK]);
          ShowMonome("Y",mDegY[aK]);
          // if (mDegX[aK]) std::cout << "X^" << mDegX[aK];
          // if (mDegY[aK]) std::cout << "Y^" << mDegY[aK];
          std::cout << "\n";
     }
}


void TestBGC3M2D()
{
   std::string aName =  "/media/data2/Jeux-Test/Dino/Ori-Martini/Orientation-_MG_0140.CR2.xml";
   CamStenope * aCS = BasicCamOrientGenFromFile(aName);

   
   cPolynomial_BGC3M2D  aP1(aCS,1);
   cPolynomial_BGC3M2D  aP2(aCS,0);
   cPolynomial_BGC3M2D  aP2Bis(aCS,3);
   cPolynomial_BGC3M2D  aP3(aCS,2);
   cPolynomial_BGC3M2D  aP4(aCS,1);
}

void GenCodeEqProjGen(int aDeg)
{
    cSetEqFormelles  * aSet = new cSetEqFormelles(cNameSpaceEqF::eSysPlein);
    std::vector<double> aPAF;
    CamStenopeIdeale aCSI(false,1.0,Pt2dr(0,0),aPAF);
    aCSI.SetSz(Pt2di(100,100));

    cPolynomial_BGC3M2D aPolCSI(&aCSI,aDeg);

    new cPolynBGC3M2D_Formelle(*aSet,aPolCSI,true);
}






/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
