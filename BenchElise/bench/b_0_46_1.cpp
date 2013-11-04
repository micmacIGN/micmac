/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/


#include "StdAfx.h"
#include "bench.h"

#define WDRF 1
#if (WDRF)
typedef cParamIFDistRadiale tDRF;
#else
typedef cParamIntrinsequeFormel tDRF;
#endif

#define TheNbDivTri 10
//=======================================================

class cSC_Chantier;
typedef enum
{
    eCamDist,   // Camera faite pour une calibration de la dist initiale
    eCamGen    // Camera generique
} eTyCam;

// Le mur est dans le plan  Oxy
//
//
//

class cCS_CamDist
{
     public :
	     Pt2dr M2C(const Pt2dr &) const ;
	     Pt2dr C2M(const Pt2dr &) const ;
	     cCS_CamDist ();
	     REAL & Focale();
	     Pt2dr & PP();
	     const ElDistRadiale_PolynImpair & DRad() const;
	     ElDistRadiale_PolynImpair & DRad() ;

	     void C2M(ElPackHomologue &) const;
	     void C2M(const cCS_CamDist &,ElPackHomologue &) const;
     private :
             REAL                      mFocale;
             Pt2dr                     mPP;
             ElDistRadiale_PolynImpair mDRad;
};





class cCS_ParamCamPhys
{
      public :
         Pt2dr   M2C(const Pt2dr &,bool & isInside) const; // Monde to Cam

         cCS_ParamCamPhys (const std::string & aName);

	 Pt2dr  SzIm() const;
	 Box2dr BoxIm() const {return mBoxIm;}    
	 // Est-ce qu'un point sort du champs physique de la camera
	 bool Inside(const Pt2dr &) const;

         const ElDistRadiale_PolynImpair & DRadVraie() const;
         const ElDistRadiale_PolynImpair & DRadEstim() const;
	 REAL  DiagReelle() const;
         void SetDRadtEstim(const ElDistRadiale_PolynImpair &);

	 void C2MEstim(ElPackHomologue &) const;

	 REAL & FocaleEstim();
	 Pt2dr & PPEstim();

      private :
         friend class cSC_ParamOriCam;


         Pt2dr   C2MVraie(const Pt2dr &) const; // Cam to Monde

      /*
         mSzIm = typiquement (1,1) pour une taille carre
	 Definit un domaine image centre en 0,0

         Si mSzIm= (1,1),  mFocale =1.0,  pour un 1/2 angle de 45 Deg
      */

	 Pt2dr                      mSzIm;    
	 Box2dr                     mBoxIm;    
         REAL                       mDiagReelle;

         cCS_CamDist                mDistVraie;
         cCS_CamDist                mDistEstim;
	 INT                        mWithTri;

};




class cSC_ParamOriCam
{
      public :

	  cSC_ParamOriCam
          (
	       cCS_ParamCamPhys &,
	       Pt3dr aCentre,
	       REAL a01,
	       REAL a02,
	       REAL a12
	  );
	  
	  Pt2dr InterOxy(Pt2dr) const;
	  std::vector<Pt2dr> CadreOnOxy() const;


          Pt3dr M2CP3(Pt3dr aP) {return mRotM2C.ImAff(aP);}

	  Pt2dr M2C(Pt3dr aP,bool & isInside) const;
	  const cCS_ParamCamPhys & PCP() const;
	  cCS_ParamCamPhys & PCP() ;

	  ElRotation3D Ori();

	  cCameraFormelle   * mCamF;
	  cRotationFormelle * mTriRotF;


          // Specifique a la gestion de equations d'appuis sur grille
          cAppuiGridEq      * mGrEqAp;
          void InitEqGrid(cSC_Chantier & aCH);
          std::list<Appar23>  mListApGrid;
          INT                  mNum;

      protected :
      private :
          cCS_ParamCamPhys &  mCamPh;
	  Pt3dr               mCentre;
	  ElRotation3D        mRotC2M;
	  ElRotation3D        mRotM2C;
          static INT          TheNum;
};

class cCameraSec : public cSC_ParamOriCam
{
	public :
	  cCameraSec
          (
	       cSC_Chantier   &,
	       Pt3dr aCentre,
	       REAL a01,
	       REAL a02,
	       REAL a12,
	       eTyCam
	  );
	   ElPackHomologue &  PackDist();
	   cCameraFormelle &  CamF();
	   cRotationFormelle & TriRotF();
	private :
	  ElPackHomologue mPackDist;

};



class cSC_Chantier
{
     public :
	     cSC_Chantier (const std::string & aName,cSC_ParamOriCam  & aCC);
             void ShowCadre(const cSC_ParamOriCam &,INT);

	     ElPackHomologue  PackCple
		              (
			           cSC_ParamOriCam &,
				   cSC_ParamOriCam &,
				   cSC_ParamOriCam * pCC
			      );

	     Pt3dr GetPt();
	     cSC_ParamOriCam    & CC();
	     cCameraFormelle *    CCF();
	     cRotationFormelle *  TriCCR();

	     
             void AddCamDist(cCameraSec *);
             void AddCamGen(cCameraSec *);
	     void WClear();

	     void CalcInitDist();
	     tDRF & PIDRF();
	     cSetEqFormelles &     DRadSetEq();
	     void OneStepSolveDR();
	     void OneStepSolveTri();

	     void AddLiaison(cSC_ParamOriCam &, cSC_ParamOriCam &);
	     void AddLiaison(cCpleGridEq *,cCpleCamFormelle * ,ElPackHomologue &);

	     cSetEqFormelles &  TriSetEq() {return  mTriSetEq;}
             cTriangulFormelle & Tri () {return *mTri;}

              void AddCamApGrid(cSC_ParamOriCam *);

     private :

	     void TestTri(Pt2dr aP);

	     REAL  ZofXY(const Pt2dr & aP) const;

	     cSC_ParamOriCam    &           mCC;
	     cCS_ParamCamPhys &             mPCP;
	     REAL                           mDiag;
	     Video_Win *                    pWMur;
	     Box2dr                         mBoxMurCC;
	     REAL                           mCourb;
	     REAL                           mErreurPixel;
	     std::list<cCameraSec *>        mLCamDist;
	     std::list<cCameraSec *>        mLCamGen;
             std::list<cSignedEqFPtLiaison *>  mLiaisDRad;
             std::list<cCpleGridEq *>          mLiaisTri;

	     cSetEqFormelles            mDRadSetEq;
	     tDRF *                     mPIDRF;
	     cSetEqFormelles            mTriSetEq;
             cTriangulFormelle *        mTri;
             std::list<cSC_ParamOriCam *>     mLCamApGrid;
};


extern void ShowMatr(const char * mes, ElMatrix<REAL> aMatr);

/******************************************************/
/*                                                    */
/*             cCS_CamDist                            */
/*                                                    */
/******************************************************/

cCS_CamDist::cCS_CamDist() :
   mFocale (1.0),
   mPP (0,0),
   mDRad(1e5,Pt2dr(0,0))
{
}

Pt2dr    cCS_CamDist::M2C(const Pt2dr & aP) const
{
     return  mDRad.Inverse(mPP+ aP*mFocale);
}
Pt2dr    cCS_CamDist::C2M(const Pt2dr & aP) const
{
   return (mDRad.Direct(aP)-mPP)/mFocale;
}
REAL & cCS_CamDist::Focale()
{
   return mFocale;
}
Pt2dr & cCS_CamDist::PP()
{
   return mPP;
}
ElDistRadiale_PolynImpair & cCS_CamDist::DRad()
{
    return mDRad;
}
const ElDistRadiale_PolynImpair & cCS_CamDist::DRad() const
{
    return mDRad;
}

void cCS_CamDist::C2M(  const cCS_CamDist & aCam2,
		        ElPackHomologue & aPack
                     ) const
{
   for 
   (
        ElPackHomologue::iterator it=aPack.begin();
        it != aPack.end();
        it++
   )
   {
	   it->P1() = C2M(it->P1());
	   it->P2() = aCam2.C2M(it->P2());
   }
}

void cCS_CamDist::C2M( ElPackHomologue & aPack) const
{
    C2M(*this,aPack);
}

/******************************************************/
/*                                                    */
/*        cCS_ParamCamPhys                            */
/*                                                    */
/******************************************************/

Pt2dr    cCS_ParamCamPhys::M2C(const Pt2dr & aP,bool & isInside) const
{
   Pt2dr aRes = mDistVraie.M2C(aP);
   isInside = Inside(aRes);
   return aRes;
}

Pt2dr    cCS_ParamCamPhys::C2MVraie(const Pt2dr & aP) const
{
   return  mDistVraie.C2M(aP);
}

cCS_ParamCamPhys::cCS_ParamCamPhys(const std::string & aName) :
    mDistVraie (),
    mDistEstim (),
    mWithTri   (0)
{
     std::vector<REAL> aVCoeffsDist;
     LArgMain anArgObl;
     LArgMain anArgFac;
     ElInitArgMain
     (
         aName,
	 anArgObl,
	 anArgFac    << EAM(mDistVraie.Focale(),"Focale",false)
	             << EAM(mDistVraie.PP(),"PP",false)
                     << EAM(mDistEstim.Focale(),"FocEstInit",false)
	             << EAM(mDistEstim.PP(),"PPEstInit",false)
	             << EAM(mSzIm,"SzIm",false)
	             << EAM(aVCoeffsDist,"CoeffsDist",false)
	             << EAM(mDiagReelle,"DiagReelle",false)
	             << EAM(mWithTri,"WithTri",false)
     );

     if (mWithTri)
     {
          mDistVraie.PP() = Pt2dr(0,0);
     }
     
     mBoxIm = Box2dr(-mSzIm,mSzIm);
     mDistVraie.DRad().PushCoeff(aVCoeffsDist);
     cout << mDistVraie.Focale()  << " " 
          << mSzIm << mDistVraie.PP() << "\n";
}

void  cCS_ParamCamPhys::C2MEstim(ElPackHomologue & aPack) const
{
    mDistEstim.C2M(aPack);
}

REAL cCS_ParamCamPhys::DiagReelle() const
{
   return mDiagReelle;
}

Pt2dr  cCS_ParamCamPhys::SzIm() const
{
   return mSzIm;
}

bool cCS_ParamCamPhys::Inside(const Pt2dr & aP) const
{
	return mBoxIm.inside(aP);
}

const ElDistRadiale_PolynImpair & cCS_ParamCamPhys::DRadVraie() const
{
   return mDistVraie.DRad();
}
const ElDistRadiale_PolynImpair & cCS_ParamCamPhys::DRadEstim() const
{
   return mDistEstim.DRad();
}

void  cCS_ParamCamPhys::SetDRadtEstim
      (const ElDistRadiale_PolynImpair & aDist)
{
      mDistEstim.DRad() = aDist;
}

REAL &  cCS_ParamCamPhys::FocaleEstim() {return mDistEstim.Focale();}
Pt2dr & cCS_ParamCamPhys::PPEstim()     {return mDistEstim.PP();}

/******************************************************/
/*                                                    */
/*        cSC_ParamOriCam                             */
/*                                                    */
/******************************************************/

INT cSC_ParamOriCam::TheNum = 0;

cSC_ParamOriCam::cSC_ParamOriCam
(
     cCS_ParamCamPhys & aCamPh,Pt3dr aCentre,REAL a01,REAL a02,REAL a12
) :
     mCamF     (0),
     mTriRotF  (0),
     mGrEqAp   (0),
     mNum      (TheNum++),
     mCamPh    (aCamPh),
     mCentre   (aCentre),
     mRotC2M   (aCentre,a01,a02,a12),
     mRotM2C   (mRotC2M.inv())
{
}



void cSC_ParamOriCam::InitEqGrid(cSC_Chantier & aCH)
{
   mGrEqAp = aCH.TriSetEq().NewEqAppuiGrid(aCH.Tri(),*mTriRotF);
   aCH.PackCple(*this,*this,&(aCH.CC()));
   aCH.AddCamApGrid(this);
}



ElRotation3D cSC_ParamOriCam::Ori()
{
   return mRotM2C;
}

const cCS_ParamCamPhys & cSC_ParamOriCam ::PCP() const
{
   return mCamPh;
}
cCS_ParamCamPhys & cSC_ParamOriCam ::PCP() 
{
   return mCamPh;
}


Pt2dr cSC_ParamOriCam::InterOxy(Pt2dr aP) const
{
    aP = mCamPh.C2MVraie(aP);
    Pt3dr aRay(aP.x,aP.y,1.0);
    aRay = mRotC2M.ImVect(aRay);

    Pt3dr aRes = mCentre - aRay * (mCentre.z/aRay.z);
    return Pt2dr(aRes.x,aRes.y);
}


std::vector<Pt2dr>  cSC_ParamOriCam::CadreOnOxy() const
{
   Pt2dr aSZ = mCamPh.SzIm();
   std::vector<Pt2dr> aRes;
   
   aRes.push_back(InterOxy(Pt2dr( aSZ.x, aSZ.y)));
   aRes.push_back(InterOxy(Pt2dr(-aSZ.x, aSZ.y)));
   aRes.push_back(InterOxy(Pt2dr(-aSZ.x,-aSZ.y)));
   aRes.push_back(InterOxy(Pt2dr( aSZ.x,-aSZ.y)));

   return aRes;
}

Pt2dr ProjPerspCannonique(const Pt3dr & aP) {return Pt2dr(aP.x/aP.z,aP.y/aP.z);}


Pt2dr cSC_ParamOriCam::M2C(Pt3dr aP,bool & isInside) const
{
    return  mCamPh.M2C(ProjPerspCannonique(mRotM2C.ImAff(aP)),isInside);
}

/******************************************************/
/*                                                    */
/*           cCameraSec                               */
/*                                                    */
/******************************************************/

cCameraSec::cCameraSec
(
     cSC_Chantier   & aCh,
     Pt3dr aCentre,
     REAL a01,
     REAL a02,
     REAL a12,
     eTyCam aType
)  :
    cSC_ParamOriCam(aCh.CC().PCP(),aCentre,a01,a02,a12),
    mPackDist  ()
{
     aCh.ShowCadre(*this,P8COL::cyan);

     if (aType  == eCamDist)
     {
         mPackDist  = aCh.PackCple(aCh.CC(),*this,0);
         aCh.AddCamDist(this);
     }
     if (aType  == eCamGen)
     {
        static bool First = true;
	ElPackHomologue aPack = aCh.PackCple(aCh.CC(),*this,0);
        ElPackHomologue aPackCorr = aPack;
        aCh.CC().PCP().C2MEstim(aPackCorr);

        ElRotation3D aSol = Ori() * aCh.CC().Ori().inv() ;
	REAL DBaseReelle =  euclid (aSol.inv().tr());


	cResMepRelCoplan aSolCopl = aPackCorr.MepRelCoplan(DBaseReelle,true);
	ElRotation3D aRot = aSolCopl.BestSol().Rot();


	cout << aRot.teta01() << " "
             << aRot.teta02() << " "
             << aRot.teta12() << "\n";

	// ShowMatr("Cam2",Ori().Mat());
	// ShowMatr("Calc",aRot.Mat());

        cout << "BASE R " <<  DBaseReelle
	     << vunit(aRot.tr()) << vunit(aSol.tr()) << "\n";

	mCamF = aCh.PIDRF().NewCam
		(
		    First ? cNameSpaceEqF::eRotBaseU : cNameSpaceEqF::eRotLibre,
		    aRot.inv(),
		    aCh.CCF()
		);

	mTriRotF = aCh.TriSetEq().NewRotation
		  (
		      First ? cNameSpaceEqF::eRotBaseU : cNameSpaceEqF::eRotLibre,
		      aRot.inv(),
		      aCh.TriCCR()
		  );

        aCh.AddLiaison
        (
	     aCh.TriSetEq().NewCpleGridEq(aCh.Tri(),*aCh.TriCCR(),aCh.Tri(),*mTriRotF),
	     aCh.DRadSetEq().NewCpleCam(*aCh.CCF(),*mCamF),
	     aPack
	 );
        aCh.AddCamGen(this);

        First = false;
        InitEqGrid(aCh);
     }
}

ElPackHomologue & cCameraSec::PackDist()
{
	return mPackDist;
}

cCameraFormelle & cCameraSec::CamF()
{
    return *mCamF;
}
cRotationFormelle & cCameraSec::TriRotF()
{
    return *mTriRotF;
}

/******************************************************/
/*                                                    */
/*         cSC_Chantier                               */
/*                                                    */
/******************************************************/

void cSC_Chantier::AddLiaison
     (
          cCpleGridEq * aGrid,
	  cCpleCamFormelle * aCpl,
	  ElPackHomologue & aPack
      )
{
   aCpl->StdPack() = aPack;
   mLiaisDRad.push_back(aCpl);

   aGrid->StdPack() = aPack;
   mLiaisTri.push_back(aGrid);
}

void cSC_Chantier::AddLiaison(cSC_ParamOriCam & aCam1, cSC_ParamOriCam & aCam2)
{
    ElPackHomologue aPack = PackCple(aCam1,aCam2,0);
    AddLiaison
    (
	 TriSetEq().NewCpleGridEq(*mTri,*(aCam1.mTriRotF),*mTri,*(aCam2.mTriRotF)),
         DRadSetEq().NewCpleCam(*(aCam1.mCamF),*(aCam2.mCamF)),
	 aPack
   );
}

void  cSC_Chantier::AddCamApGrid(cSC_ParamOriCam * pCam)
{
   mLCamApGrid.push_back(pCam);
}




void cSC_Chantier::ShowCadre(const cSC_ParamOriCam & aPh,INT aCoul)
{
     if (! pWMur) 
        return;
     std::vector<Pt2dr> Pts = aPh.CadreOnOxy() ;

     for (INT aK=0 ;aK<INT(Pts.size()) ; aK++)
         pWMur->draw_seg
         (
             Pts[aK],
             Pts[(aK+1)%(Pts.size())],
	     Line_St(pWMur->pdisc()(aCoul),2)
         );
}

cSC_Chantier::cSC_Chantier
(
      const std::string & aName,
      cSC_ParamOriCam  & aCC
) :
    mCC   (aCC),
    mPCP  (mCC.PCP()),
    mDiag (aCC.PCP().DiagReelle()),
    mPIDRF (0),
    mTri   (0)
{
    std::vector<Pt2dr> VP = mCC.CadreOnOxy();

    Pt2dr aP0 = VP[0];
    Pt2dr aP1 = VP[0];
    for (INT aK=0; aK<INT(VP.size()) ; aK++)
    {
        aP0.SetInf(VP[aK]);
        aP1.SetSup(VP[aK]);
    }
    mBoxMurCC = Box2dr(aP0,aP1);

    REAL   aRabV = 1.2;
    Pt2di  aSzMaxVisu(300,300);
    LArgMain anArgObl;
    LArgMain anArgFac;
    ElInitArgMain
    (
         aName,
	 anArgObl,
	 anArgFac    << EAM(aRabV,"RabV",true)
	             << EAM(aSzMaxVisu,"SzMaxVisu",true)
		     << EAM(mCourb,"Courb",false)
		     << EAM(mErreurPixel,"Noise",true)
    );

    Pt2dr aSz = (aP1-aP0) * aRabV;
    REAL dR = (aRabV-1)/2.0;
    aP0 = aP0 - aSz * dR;
    aP1 = aP1 + aSz * dR;

    REAL aZoom = ElMin(aSzMaxVisu.y/aSz.x,aSzMaxVisu.y/aSz.y);

    //pWMur  = Video_Win::PtrWStd(aSz*aZoom,true);
    pWMur  = Video_Win::PtrWStd(Pt2di(aSz*aZoom),true); // __NEW
    pWMur = pWMur->PtrChc(aP0,Pt2dr(aZoom,aZoom),true);

    ShowCadre(mCC,P8COL::red);
}

cSC_ParamOriCam &cSC_Chantier::CC() {return mCC;}

void cSC_Chantier::WClear()
{
     if (pWMur) 
        pWMur->clear();
}

bool RANZ = false;
REAL cSC_Chantier::ZofXY(const Pt2dr & aPMur) const 
{

     Pt2dr aP = (aPMur-mBoxMurCC.milieu())/(mBoxMurCC.diam()/2.0);
     aP += Pt2dr(0.5,0.5);
     REAL aRes =     mCourb * ( square_euclid(aP)) ; 

     if (RANZ) aRes += 2* NRrandC();

     return aRes;
/*
     cout << aRes << "\n";

     return 0.0 * NRrandC();
*/
}

Pt3dr cSC_Chantier::GetPt()
{
    Pt2dr aP = pWMur->clik_in()._pt;
    pWMur->draw_circle_abs(aP,2.0, pWMur->pdisc()(P8COL::red));

    return Pt3dr(aP.x,aP.y,ZofXY(aP));
}


void   cSC_Chantier::AddCamDist(cCameraSec * aCam)
{
   mLCamDist.push_back(aCam);
}
void   cSC_Chantier::AddCamGen(cCameraSec * aCam)
{
   mLCamGen.push_back(aCam);
}

ElPackHomologue cSC_Chantier::PackCple
                (
                    cSC_ParamOriCam & aCamA,
                    cSC_ParamOriCam & aCamB,
                    cSC_ParamOriCam *       pCC
                )
{
        INT   aNb = 400;
	ElPackHomologue aRes;
	REAL aStep = sqrt(mBoxMurCC.surf() / aNb);
        INT aNbX =  round_ni(mBoxMurCC.largeur() / aStep);
        INT aNbY =  round_ni(mBoxMurCC.hauteur() / aStep);

	// cout << "PACK Nb " << aNbX << " " << aNbY << "\n";
	// cout << "ERR " << mErreurPixel/mDiag << "\n";

        bool First = true;
	for (INT aKX=0 ; aKX<aNbX ; aKX++)
	{
	    for (INT aKY=0 ; aKY<aNbY ; aKY++)
	    {
		Pt2dr aPds((aKX+0.5)/aNbX,(aKY+0.5)/aNbY);
                Pt2dr aPM2 =   mBoxMurCC._p0
			     + mBoxMurCC.sz().mcbyc(aPds);
		Pt3dr aPM3(aPM2.x,aPM2.y,ZofXY(aPM2));

                bool oKA,oKB;
		Pt2dr aPA = aCamA.M2C(aPM3,oKA);
		Pt2dr aPB = aCamB.M2C(aPM3,oKB);


                aPA += Pt2dr(NRrandC(),NRrandC())*mErreurPixel/mDiag;
		bool oK = oKA && oKB;

                if (pCC)
                {
                   if (oKA)
                   {
                         // Pt3dr aQ = pCC->M2CP3(aPM3);
                         Pt3dr aQ = aPM3;
                          if (First) 
                             cout << "APPUI ["  <<  aCamA.mNum << "] : " << aPA << " " << aQ << "\n";
                         aCamA.mListApGrid.push_back(Appar23(aPA,aQ));
                         First = false;
                   }
                }
                else
                {
		   if (oK)
		   {
		       //aRes.add(ElCplePtsHomologues(aPA,aPB,1.0));
		       aRes.Cple_Add(ElCplePtsHomologues(aPA,aPB,1.0));
		   }
                }

		if (pWMur)
		{
                    pWMur->draw_circle_abs
		    (
		        aPM2,
			1.0,
			pWMur->pdisc()(oK ?P8COL::green:P8COL::magenta)
                    );
                }
	    }
	}
	return aRes;
}

void cSC_Chantier::CalcInitDist()
{
     bool CFiged = true;
     cLEqHomOneDist anEq(mDiag);
     for
     (
           std::list<cCameraSec *>::iterator it=mLCamDist.begin();
	   it != mLCamDist.end();
	   it++
     )
     {
            anEq.AddCple((*it)->PackDist());
     }
     anEq.CloseSet();

     anEq.NStepOpt(10,true);
     if (! CFiged)
        anEq.NStepOpt(10,false);

    ElDistRadiale_PolynImpair aDCur = anEq.DRF()->DistCur();
    mCC.PCP().SetDRadtEstim(aDCur);

    cout << "CENTRE " << aDCur.Centre() << "\n";
    ElDistRadiale_PolynImpair aDVraie  = mCC.PCP().DRadVraie();
    for (INT aK=0 ; aK<5 ; aK++)
    {
            cout << " Coeff Dist = " 
                 << aDCur.Coeff(aK) << " " 
                 << aDVraie.CoeffGen(aK) << " " 
                 <<"\n";
    }


#if (WDRF)
    //mPIDRF = mDRadSetEq.NewIntrDistRad(mPCP.FocaleEstim(),mPCP.PPEstim(),0,aDCur);
    cCamStenopeDistRadPol camera(false/*isDistC2M*/,mPCP.FocaleEstim(),mPCP.PPEstim(),aDCur,vector<double>() ); // __NEW
    mPIDRF = mDRadSetEq.NewIntrDistRad(false/*isDistC2M*/,&camera,0);                                           // __NEW
#else
    mPIDRF = mDRadSetEq.NewParamIntrNoDist(mPCP.FocaleEstim(),mPCP.PPEstim());
#endif

    mCC.mCamF = mPIDRF->NewCam
	         (
	             cNameSpaceEqF::eRotFigee,
		     ElRotation3D(Pt3dr(0,0,0),0,0,0),
                     0
	         );

    mPIDRF->SetFocFree(true);
    static_cast<cParamIntrinsequeFormel *>(mPIDRF)->SetPPFree(true);



    //mTri = mTriSetEq.NewTriangulFormelle(mPCP.BoxIm(),TheNbDivTri,1.8/TheNbDivTri);
    mTri = mTriSetEq.NewTriangulFormelle(2/*aDim*/,mPCP.BoxIm(),TheNbDivTri,1.8/TheNbDivTri); // __NEW
    mCC.mTriRotF = mTriSetEq.NewRotation
	           (
		         cNameSpaceEqF::eRotFigee,
                         ElRotation3D(Pt3dr(0,0,0),0,0,0),
                         0
		   );

     mCC.InitEqGrid(*this);
}


tDRF & cSC_Chantier::PIDRF()
{
  return *mPIDRF;
}

cSetEqFormelles & cSC_Chantier::DRadSetEq()
{
   return mDRadSetEq;
}

cCameraFormelle *  cSC_Chantier::CCF()
{
   return mCC.mCamF ;
}

cRotationFormelle *  cSC_Chantier::TriCCR()
{
   return mCC.mTriRotF ;
}

void CorrectDist
     (
		  ElPackHomologue & aPack,
		  ElDistRadiale_PolynImpair aD
     )
{
	for 
        (
            ElPackHomologue::iterator itP = aPack.begin();
            itP != aPack.end();
	    itP++
        )
	{
		itP->P1() =  aD.Direct(itP->P1());
		itP->P2() =  aD.Direct(itP->P2());
	}
}

void CorrectPhotogram
     (
           Box2dr aBox,
           ElPackHomologue & aPack,
	   REAL              aFoc,
	   Pt2dr             aPP
     )
{
	ElPackHomologue aRes;
	for 
        (
            ElPackHomologue::iterator itP = aPack.begin();
            itP != aPack.end();
	    itP++
        )
	{
		Pt2dr aP1 = (itP->P1()-aPP)/aFoc;
		Pt2dr aP2 = (itP->P2()-aPP)/aFoc;
		if (aBox.inside(aP1) && aBox.inside(aP2))
                {
			//aRes.add(ElCplePtsHomologues(aP1,aP2));
			aRes.Cple_Add(ElCplePtsHomologues(aP1,aP2)); // __NEW
                }
	}

	aPack = aRes;
}


void cSC_Chantier::TestTri(Pt2dr aP)
{
    cout << aP <<  mTri->Direct(aP) ;
    if (euclid(aP) > 1e-2)
        cout << mTri->Direct(aP)/ aP ;
    cout << "\n";
}

void cSC_Chantier::OneStepSolveTri()
{
   mTri->Show();
   static bool First = true;
   if (First)
   {
	   mTriSetEq.SetClosed();
	   First = false;
   }



   mTriSetEq.AddContrainte(mTri->ContraintesRot(),true/*Strictes*/);
   mTriSetEq.AddContrainte(TriCCR()->StdContraintes(),true/*Strictes*/);

   for
   (
           std::list<cCameraSec *>::iterator it=mLCamGen.begin();
	   it != mLCamGen.end();
	   it++
   )
   {
       ElRotation3D   R1 = TriCCR()->CurRot();
       ElRotation3D   R2 = (*it)->TriRotF().CurRot();
       cout << "D COPT " << euclid(R1.tr()-R2.tr()) << R1.tr() << R2.tr() <<"\n";
       mTriSetEq.AddContrainte((*it)->TriRotF().StdContraintes(),true/*Strictes*/);
   }


   // LIAISONS 

   for
   (
           std::list<cCpleGridEq *>::iterator it=mLiaisTri.begin();
	   it != mLiaisTri.end();
	   it++
   )
   {
       ElPackHomologue aPack = (*it)->StdPack();
       CorrectDist(aPack,mPCP.DRadEstim());
       CorrectPhotogram(mPCP.BoxIm(),aPack,mPCP.FocaleEstim(),mPCP.PPEstim());
       

       mTri->Show(aPack);
       REAL Ec =(*it)->AddPackLiaisonP1P2(aPack,false);
       cout << "TRI ECART = " << Ec << "\n";
   }

   // APPUIS
   for
   (
           std::list<cSC_ParamOriCam *>::iterator itC=mLCamApGrid.begin();
           itC != mLCamApGrid.end();
           itC++
   )
   {
          for
          (
                  std::list<Appar23>::iterator itP = (*itC)->mListApGrid.begin();
                  itP != (*itC)->mListApGrid.end();
                  itP++
          )
          {
if (itP == (*itC)->mListApGrid.begin())
{
   bool OK;
   cout <<  (*itC)->mGrEqAp->ResiduAppui(itP->pter,itP->pim) << "\n";
   cout << itP->pter << " "
        << (*itC)->M2C(itP->pter,OK)
        << itP->pim << " "
        << (*itC)->mTriRotF->CurCOpt() << "\n";
}
          }
          // cAppuiGridEq      * mGrEqAp;
          // std::list<Appar23>  mListApGrid;
   }
                                                                                            
   getchar();



   mTriSetEq.SolveResetUpdate();

   TestTri(Pt2dr(-0.5,-0.5));
   TestTri(Pt2dr(-0.5, 0.5));
   TestTri(Pt2dr( 0.5,-0.5));
   TestTri(Pt2dr( 0.5, 0.5));

   TestTri(Pt2dr( 0.0, 0.0));
   TestTri(Pt2dr( -0.7071, 0.0));
   TestTri(Pt2dr(  0.7071, 0.0));
   TestTri(Pt2dr(  0.4   , 0.0));
   TestTri(Pt2dr( -0.4   , 0.0));


}


void cSC_Chantier::OneStepSolveDR()
{
	static bool First = true;
	if (First)
	{
	   mDRadSetEq.SetClosed();
	   First = false;
	}
   cout << "PARAM INTERNE "
	<< mPIDRF->CurFocale() << " " 
	<< mPIDRF->CurPP() << "\n";

   mDRadSetEq.AddContrainte(mPIDRF->StdContraintes(),true/*Strictes*/);
   mDRadSetEq.AddContrainte(CCF()->RF().StdContraintes(),true/*Strictes*/);

   for
   (
           std::list<cCameraSec *>::iterator it=mLCamGen.begin();
	   it != mLCamGen.end();
	   it++
   )
   {
       mDRadSetEq.AddContrainte((*it)->CamF().RF().StdContraintes(),true/*Strictes*/);
   }

#if (WDRF)
   ElDistRadiale_PolynImpair aD =  mPIDRF->DistCur();
   cout << "DIST " << aD.Centre() 
	<< aD.Coeff(0) << " " << aD.Coeff(1) << "\n";
#endif

   for
   (
           std::list<cSignedEqFPtLiaison *>::iterator it=mLiaisDRad.begin();
	   it != mLiaisDRad.end();
	   it++
   )
   {
       ElPackHomologue aPack = (*it)->StdPack();
       if (! WDRF)
           CorrectDist(aPack,mPCP.DRadEstim());
       REAL Ec =(*it)->AddPackLiaisonP1P2(aPack,false);
       cout << "ECART = " << Ec << "\n";
   }

   mDRadSetEq.SolveResetUpdate();

#if (WDRF)
   aD =  mPIDRF->DistCur();
   cout << "DIST " << aD.Centre() 
        << aD.Coeff(0) << " " << aD.Coeff(1) << "\n";
#endif
}


//=======================================================

/*
     Convergente X
	cSC_ParamOriCam aCam2(aPh,Pt3dr(3.0,0.0,-10),0.0,0.3,0.0);

     Convergente Y
	cSC_ParamOriCam aCam2(aPh,Pt3dr(0.0,3.0,-10),0.0,0.0,0.3);
*/



void bench_mep_relative()
{



	// cCpleCamFormelle::GenAllCode() ;

	cCS_ParamCamPhys aPh
        (
              "SzIm=[0.7071,0.7271]"
              " DiagReelle=2828"
              // " Focale=2.05 PP=[0.02,0.04]"
              " Focale=1.25 PP=[0.0,0.0]"
               " CoeffsDist=[0.045366,-0.0223375,0.00266689]"
              // " CoeffsDist=[-0.00388893,-8.59644e-05,2.70663e-05]"
	      " WithTri=1"
              " FocEstInit=1.0 PPEstInit=[0.0,0.0]"
              // " FocEstInit=1.23 PPEstInit=[0.02,0.04]"
        );
 
        ElDistRadiale_PolynImpair aDVraie  = aPh.DRadVraie();

	cSC_ParamOriCam aCamCentr(aPh,Pt3dr(0,0,-10),0,0,0);
        cSC_Chantier aChantier
                    (
                         "SzMaxVisu=[500,500]"
                         " Courb=0.00"
	                 " Noise=0.00",
                         aCamCentr
                    );

	//
	// CALCUL DE LA DISTORTION RADIALE initiale
	//

	cCameraSec aCDRG
		   (
		       aChantier,
		       Pt3dr(0.1,0.0,-10),
		       0.0,0.3,0.0,
		       eCamDist
		   );
	cCameraSec aCDRD
		   (
		       aChantier,
		       Pt3dr(-0.1,0.0,-10),
		       0.0,-0.3,0.0,
		       eCamDist
		   );
	cCameraSec aCDRH
		   (
		       aChantier,
		       Pt3dr(0.0,0.1,-10),
		       0.0,0.0,0.3,
		       eCamDist
		   );
	cCameraSec aCDRB
		   (
		       aChantier,
		       Pt3dr(0.0,-0.1,-10),
		       0.0,0.0,-0.3,
		       eCamDist
		   );
	aChantier.CalcInitDist();

	//  CALCUL DES POSITION  DE CAMERA
	aChantier.WClear();

	// RANZ = true;

	cCameraSec aC1
		   (
		       aChantier,
		       Pt3dr(5.0,0.0,-10),
		       0.0,0.5,0.0,
		       eCamGen
		   );
	cCameraSec aC2
		   (
		       aChantier,
		       Pt3dr(-5.0,0.0,-10),
		       0.0,-0.5,0.0,
		       eCamGen
		   );

	 // Avec deplacement vertical
	/*
	cCameraSec aC3
		   (
		       aChantier,
		       Pt3dr(0.0,5.0,-10),
		       0.0,0.0,0.5,
		       eCamGen
		   );

	cCameraSec aC4
		   (
		       aChantier,
		       Pt3dr(0.0,-5.0,-10),
		       0.0,0.0,-0.5,
		       eCamGen
		    );
		    */
		    

        // Avec deplacement horizonatl
	
	while(0)
	{
            REAL a,b,c;
	    cin >> a >> b >> c;
	    aChantier.WClear();
            aChantier.ShowCadre(aCamCentr,P8COL::red);
	    cCameraSec aCTest
		   (
		       aChantier,
		       Pt3dr(4.0,4.0,-10),
		       a,b,c,
		       eCamGen
		       );
	}
	cCameraSec aC3
		   (
		       aChantier,
		       Pt3dr(0.0,5.0,-10),
		       0.0,0.0,0.5,
		       eCamGen
		   );
	cCameraSec aC4
		   (
		       aChantier,
		       Pt3dr(0.0,4.0,-10),
		       0.0,3.14,3.14-0.3,
		       eCamGen
		   );
		  

	/*
	cCameraSec aC3
		   (
		       aChantier,
		       Pt3dr(-2.0,0.0,-10),
		       1.507,0.0,0.2,
		       eCamGen
		   );
	cCameraSec aC4
		   (
		       aChantier,
		       Pt3dr( 2.5,0.0,-10),
		       -1.507,0.0,-0.4,
		       eCamGen
		   );

       */


        aChantier.AddLiaison(aC1,aC2);
        aChantier.AddLiaison(aC1,aC3);
        aChantier.AddLiaison(aC2,aC3);

        aChantier.AddLiaison(aC1,aC4);
        aChantier.AddLiaison(aC2,aC4);
        aChantier.AddLiaison(aC3,aC4);


	/*
        aChantier.AddLiaison(aC1,aC3B);
        aChantier.AddLiaison(aC1,aC4B);

        aChantier.AddLiaison(aC2,aC3B);
        aChantier.AddLiaison(aC2,aC4B);
	*/
/*

	cCameraSec aC5
		   (
		       aChantier,
		       Pt3dr(0.5,-0.0,-10),
		       3.0,0.0,-0.0,
		       eCamGen
		    );
*/

       aChantier.OneStepSolveTri(); 
       aChantier.OneStepSolveTri(); 
       aChantier.OneStepSolveTri(); 
       aChantier.OneStepSolveTri(); 
       aChantier.OneStepSolveTri(); 
       aChantier.OneStepSolveTri(); 
       aChantier.OneStepSolveTri(); 
       aChantier.OneStepSolveTri(); 
       aChantier.OneStepSolveTri(); 
       getchar();

 
	aChantier.OneStepSolveDR();
	aChantier.OneStepSolveDR();
	aChantier.OneStepSolveDR();
	aChantier.OneStepSolveDR();
	aChantier.OneStepSolveDR();
	aChantier.OneStepSolveDR();
	aChantier.OneStepSolveDR();
	aChantier.OneStepSolveDR();
	aChantier.OneStepSolveDR();
	aChantier.OneStepSolveDR();
	aChantier.OneStepSolveDR();
	aChantier.OneStepSolveDR();
	aChantier.OneStepSolveDR();
	getchar();

	//cSC_ParamOriCam aCamDrad1(aPh,Pt3dr(0.0,0.1,-10),0.0,0.3,0.03);


	// cSC_ParamOriCam aCamDrad1(aPh,Pt3dr(0.0,0.1,-10),0.0,0.3,0.03);

	// cSC_ParamOriCam aCam2(aPh,Pt3dr(0.0,0.01,-10),0.0,0.3,0.0);
	//cSC_ParamOriCam aCam2(aPh,Pt3dr(10.0,0.00,-10),0.0,0.8,0.0);
	// cSC_ParamOriCam aCam2(aPh,Pt3dr(0.0,0.0,-12),0.0,0.0,0.0);



	// cSC_ParamOriCam aCam2(aPh,Pt3dr(3.0,0.0,-10),0.0,0.3,0.0);


	//cSC_ParamOriCam aCam2(aPh,Pt3dr(2.5,0,-10),0.3,0.2,0.1);
	// cSC_ParamOriCam aCam2(aPh,Pt3dr(0.0,0,7),0.0,0.0,0);




         


	//============================
	// aPack = aChantier.PackCple(400,aCam1,aCam2, 0.0);
	
	/*
        aPh.C2MEstim(aPack);
	cResMepRelCoplan aSolCopl = aPack.MepRelCoplan(1.0,true);
	ElRotation3D aRot = aSolCopl.BestSol().Rot();

        ElRotation3D aSol = aCamDrad1.Ori() * aCamCentr.Ori().inv() ;

	cout << aRot.teta01() << " "
             << aRot.teta02() << " "
             << aRot.teta12() << "\n";

	ShowMatr("Cam2",aCamDrad1.Ori().Mat());
	ShowMatr("Calc",aRot.Mat());

        cout << vunit(aRot.tr()) << vunit(aSol.tr()) << "\n";


	while (1)
	{
		bool oK;
             Pt3dr aPM3 = aChantier.GetPt();
	     cout << aPM3 << aCamCentr.M2C(aPM3,oK) 
		          << aCamDrad1.M2C(aPM3,oK) << "\n";
	}
	*/

	getchar();
	cout << "EXIT \n";
	exit(-1);
}



