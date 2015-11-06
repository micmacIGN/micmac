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
#include "StdAfx.h"



std::string iToString(const int & i)
{
  return ToString(i);
}


/************************************************/
/*                                              */
/*             cParamPointeInit                 */
/*                                              */
/************************************************/


cParamPointeInit::~cParamPointeInit()
{
}


cParamPointeInit::cParamPointeInit(CamStenope * aCam) :
    mCam     (aCam)
{
}

Pt2dr cParamPointeInit::Im2Norm(Pt2dr aP)
{
      Pt3dr aRay =  mCam->F2toDirRayonL3(aP);
      return Pt2dr(aRay.x,aRay.y)/ aRay.z;
}

Pt2dr cParamPointeInit::Norm2Im(Pt2dr aP)
{
      return  mCam->L3toF2(Pt3dr(aP.x,aP.y,1.0));
}

Pt3dr cParamPointeInit::P3ofIm(Pt2dr aP)
{
    ELISE_ASSERT(false,"No  cParamPointeIni::P3ofIm");
    return Pt3dr(0,0,0);
}

void cParamPointeInit::ConvPointeImagePolygone(Pt2dr &)
{
}

const cPolygoneEtal::tContCible & cParamPointeInit::CiblesInit() const
{
    // EN FAIT CETTE FONCTION, virtelle, n'est jamais appelee
     static cPolygoneEtal::tContCible aRes;
     return aRes;
}

/************************************************/
/*                                              */
/*             cWPointeData                     */
/*                                              */
/************************************************/


cWPointeData::cWPointeData
(
        bool modeIm,
	Video_Win aW,
	const std::string & aName
) :
   mW (aW),
   mV (aW,StdPalOfFile(aName,aW),Pt2di(10,10)),
   pScr (ElImScroller::StdPyramide (mV,aName))
{
}

/************************************************/
/*                                              */
/*             cWPointe                         */
/*                                              */
/************************************************/


cWPointe::cWPointe
(
      std::list<cPt2Im> &  aLPt,
      bool modeIm,
      Video_Win aW,
      const std::string & aName,
      eTyEtat             aEtatDepAff,
      cPointeInit  &      aPI
) :
    cWPointeData            (modeIm,aW,aName),
    EliseStdImageInteractor (cWPointeData::mW,*pScr,2),
    mPt2Is                  (aLPt),
    mName                   (aName),
    mModeIm                 (modeIm),
    mSz                     (aW.sz()),
    mEtatDepAff             (aEtatDepAff),
    mTifFile                (Tiff_Im::StdConvGen(aName,-1,false)),
    mPI                     (aPI)
{
        // SetModeReplicationPixel();
	INT aDyn = 256;
	if (mTifFile.type_el() == GenIm::u_int2)
		aDyn = 2500;
	W().SetInteractor(this);
	EtalD(aDyn);
}

INT cWPointe::GetRadiom(Pt2di aP)
{
   INT aVal;

   ELISE_COPY
   (
         rectangle(aP,aP+Pt2di(1,1)),
         mTifFile.in(),
         sigma(aVal)
   );
   
   return aVal;
}


Video_Win cWPointe::W() {return cWPointeData::mW;}

void cWPointe::Refresh()
{
    pScr->LoadAndVisuIm(false);
    ShowVect();
}




/************************************************/
/*                                              */
/*             cEllipse_Proj                    */
/*                                              */
/************************************************/

// mDirH et mDirV pas necessairement orthognaux
// car projection des directions verticales et horizontales
// Normale mDirH mDirV : repere directe




/************************************************/
/*                                              */
/*             cPointeInit                      */
/*                                              */
/************************************************/


class cParamSaisiePoly
{
    public :

       cParamSaisiePoly(const  std::vector<Pt2dr>  & aLpt) :
          mCont (aLpt)
       {
       }

        std::vector<Pt2dr> mCont;
};

class cPointeInit  : public Grab_Untill_Realeased
{
	public :
	 
	        cParamEllipse InitParamEllipe(const cCibleCalib & aC,double aRay);
	        cParamEllipse InitParamEllipe(const cCibleCalib & aC);
		bool OkForEllipse() ;
		void ShowEllipe
		     (
			 const cParamEllipse &,
			 int aCoul1,
			 int aCoul2,
			 bool  isSortant,
			 const cParamEllipse * aMasq
                     );

		cPointeInit
	        (
		   cParamPointeInit &  aPPI,
                   Pt2di aSzWIm,
                   Pt2di aSzWPolyg
                );


		void Test();
		virtual ~cPointeInit() {}


	private :
                void RecherchePosAutomCDD(cPt2Im & aC);
                cPt2Im *  GetPointeOfCible(const cCiblePolygoneEtal * aCPE);
                cPt2Im *  Nearest(bool InIm,Pt2dr aP,bool AcceptSelected = true);
                cPt2Im *  ClikNearest();
                void InitPtTmp();
                void Refresh();
		std::list<cPt2Im> InitListePt2Is();
		void End();
		void                    Sauv();
		const cCiblePolygoneEtal * ShowProj(Pt2dr *,cParamSaisiePoly *);
		void                    MakeCam();
		void  GUR_query_pointer(Clik,bool);
		CaseGPUMT * StdCase(const std::string &,Pt2di aCase);
		BoolCaseGPUMT * BoolStdCase(const std::string & NameTrue,
		                        const std::string & NameFalse,
					Pt2di aCase
					);

		void  AddAPointe(cPt2Im * aCPP);
	        void SaisiePolyg(const  std::vector<Pt2dr>  &);

                cParamPointeInit & mPPI;
		std::string        mNameFilePointe;

		std::list<cPt2Im> mPt2Is;

		cWPointe   mWIm;
		cWPointe   mPolyg;

                Tiff_Im                 mTifMenu;
	        Pt2di                   mSzCase;
		bool                    mModeCurIsPopUp;
		bool                    mModeCurIsDyn;
                GridPopUpMenuTransp     mPopUp;
		Pt2dr                   mP0Cl;
		REAL                    VDyn0;

		CaseGPUMT *             mCaseOkCur;
		CaseGPUMT *             mCaseSetCur;
		CaseGPUMT *             mCaseKillCur;

		CaseGPUMT *             mCaseProj;
		CaseGPUMT *             mCaseDyn;
		CaseGPUMT *             mCaseKillLiais;

		CaseGPUMT *             mCaseExit;
		CaseGPUMT *             mCasePolyg;

		BoolCaseGPUMT *         mBCaseBilin;

                CamStenopeIdeale         mCam;
		bool                     mCamUp2Date;
   
                static const int aNbDiscEl = 160;
                Pt2dr            mVPt[aNbDiscEl+1];

		bool                     mWithIm;
		Im2D_REAL4               mIm;

		cRechercheCDD *         mRCdd;
};


void cWPointe::ShowCible(cCibleCalib & aCC,const Pt2dr & aP)
{
    cParamEllipse * aPrec = 0;
    if (! aCC.Ponctuel())
    {
         cParamEllipse anEl = mPI.InitParamEllipe(aCC);
	 anEl.SetCentre(aP);

         mPI.ShowEllipe
         (
             anEl,P8COL::red,
             P8COL::red,
	     aCC.ReliefIsSortant(),
             0
         );
	 aPrec = new  cParamEllipse(anEl);
    }

    std::vector<cCercleRelief> aVCR =  aCC.CercleRelief();
    for (int aK=0 ; aK<int(aVCR.size()) ; aK++)
    {
         
	 cParamEllipse * anEl2 = new cParamEllipse(mPI.InitParamEllipe(aCC,aVCR[aK].Rayon()));
	 anEl2->SetCentre(aP+anEl2->ProjN()*-aVCR[aK].Profondeur());

          std::cout << aP << anEl2->ProjN()*-aVCR[aK].Profondeur() << "\n";

         mPI.ShowEllipe
         (
             *anEl2,
             P8COL::blue,
             P8COL::green,
	     aCC.ReliefIsSortant(),
             aPrec
        );
	delete aPrec;
	aPrec = anEl2;
   }
   delete aPrec;
}

void cWPointe::ShowVect()
{
      for 
      (
	     std::list<cPt2Im>::iterator iT = mPt2Is.begin();
	     iT !=  mPt2Is.end();
	     iT ++
      )
      {
          cPt1IM & aPI1 = mModeIm ?  iT->mPtIm : iT->mPtPol ;
	  if (aPI1.mEtat >= mEtatDepAff)
          {
              Pt2dr aP = U2W(aPI1.mPt);
	      if ((aP.x>0) && (aP.y>0) && (aP.x<mSz.x) && (aP.y < mSz.y))
	      {
	           if (true) // (aPI1.mEtat == eNonSelec)
		   {
		       INT Coul = P8COL::red; // eCur
		       if (aPI1.mEtat == eNonSelec)
                          Coul =  P8COL::black;
		       if (aPI1.mEtat == eSelected)
                          Coul =  P8COL::blue;
		       if (aPI1.mEtat == eAutreImPointes)
                          Coul =  P8COL::green;

                       W().draw_circle_abs(aP,2.0,W().pdisc()(Coul));
		       if (iT->pCible)
		       {
		          W().fixed_string
                          (
			    aP + (mModeIm? Pt2dr(10,-10) : Pt2dr(0,0)),
			    iT->mName.c_str(),
			    W().pdisc()(Coul),
			    true
                          );
                          if (mModeIm && mPI.OkForEllipse())
                          {
		             if ((aPI1.mEtat ==eCur) || (SC() >1))
			     {
				 ShowCible(*(iT->pCible->CC()),aPI1.mPt);
			     }
                          }
		       }
                   }
	      }
          }
      }
}





void cPointeInit::Refresh()
{
     mWIm.Refresh();
     mPolyg.Refresh();
}

void cPointeInit::End()
{
    Sauv();
    exit(0);
}

void cPointeInit::MakeCam()
{
   if (mCamUp2Date) return;

   std::list<Appar23>  l23;
   for 
   (
      std::list<cPt2Im>::iterator iT= mPt2Is.begin();
      iT != mPt2Is.end();
      iT++
   )
   {
       if (     ((iT->mPtIm.mEtat == eSelected) || (iT->mPtIm.mEtat == eAutreImPointes))
            && (iT->pCible !=0))
       {
           l23.push_back
           (
              Appar23(mPPI.Im2Norm(iT->mPtIm.mPt),iT->pCible->Pos())
           );
       }
   }
   if (l23.size() < 4)
   {
	  return;
   }
   // CamStenopeIdeale aCam(1.0,Pt2dr(0,0));
   REAL DMin;
   ElRotation3D aRot = mCam.CombinatoireOFPA(true,18,l23,&DMin);
   cout << "DMIN = " << DMin << "\n";
   mCam.SetOrientation(aRot);
   mCamUp2Date = true;
}

bool cPointeInit::OkForEllipse() 
{
   MakeCam();

   return     mPPI.PC() &&  mCamUp2Date;
}

void cPointeInit::ShowEllipe
     (
          const cParamEllipse & anEl,
          int aCoul1,
          int aCoul2,
	  bool  isSortant,
	  const cParamEllipse  * aMasq
     )
{
   for (int aK = 0 ; aK< aNbDiscEl ; aK++)
   {
      Pt2dr aP1 = anEl.Centre()+anEl.KiemV(aK);
      Pt2dr aP2 = anEl.Centre()+anEl.KiemV(aK+1);
      bool aDraw = true;
      if (aMasq)
      {
         if ((! aMasq->VecInEllipe(aP1 -aMasq->Centre())) ^ isSortant)
            aDraw = false;
      }
      if (aDraw)
          mWIm.W().draw_seg(mWIm.U2W(aP1),mWIm.U2W(aP2),mWIm.W().pdisc()(aCoul1));
   }
}


cParamEllipse cPointeInit::InitParamEllipe(const cCibleCalib & aCible)
{
    return InitParamEllipe(aCible,-1.0);
}


cParamEllipse cPointeInit::InitParamEllipe(const cCibleCalib & aCible,double aRay)
{
   Pt3dr aN = vunit(aCible.Normale());
   // Pt3dr aH = vunit(aN ^ Pt3dr(0,0,1));
   Pt3dr aH = vunit(OneDirOrtho(aN));
   Pt3dr aV = aN ^ aH ;

   Pt3dr aC3 = aCible.Position();
   Pt2dr aC2 = mPPI.Norm2Im(mCam.R3toF2(aC3));
   // Ce sont des diametre en mm , d'ou le / 2000
   if (aRay < 0)
      aRay =  aCible.Rayons()[0] ;
   aRay /= (1000.0 * 2);

   Pt2dr aDirV =  mPPI.Norm2Im(mCam.R3toF2(aC3+aV*aRay))-aC2;
   Pt2dr aDirH =  mPPI.Norm2Im(mCam.R3toF2(aC3+aH*aRay))-aC2;
   Pt2dr aProjN =  mPPI.Norm2Im(mCam.R3toF2(aC3+aN*(1/1000.0)))-aC2;


   cParamEllipse aRes(aNbDiscEl,aC2,aDirH,aDirV,aProjN);

   aRes.Compute();

   return aRes;
}

cPt2Im *  cPointeInit::GetPointeOfCible(const cCiblePolygoneEtal * aCPE)
{
   for 
   (
      std::list<cPt2Im>::iterator iT= mPt2Is.begin();
      iT != mPt2Is.end();
      iT++
   )
      if (iT->pCible == aCPE)
         return & (*iT);
    

   return 0;
}

void  cPointeInit::SaisiePolyg(const  std::vector<Pt2dr>  &)
{
}


/*
void cPointeInit::AddAPointe(cPt2Im * aCPP)
{
    if (aCPP->mPtPol.mEtat != eSelected)
    {
        mPt2Is.back().mPtPol.mPt = aCPP->mPtPol.mPt;
        mPt2Is.back().mPtPol.mEtat = eCur;
        mPt2Is.back().pCible = aCPP->pCible;
        mPt2Is.back().mName = aCPP->mName;
        if (mPPI.PseudoPolyg())
        {
            aCPP->mPtPol.mPt = aPU;
            mPt2Is.back().mPtPol.mPt = aPU;
            const_cast<cCiblePolygoneEtal *>(mPt2Is.back().pCible)->SetPos(mPPI.P3ofIm(aPU));
        }
   }

}
*/

const cCiblePolygoneEtal * cPointeInit::ShowProj(Pt2dr * aPU2Get,cParamSaisiePoly * aPSP)
{
   bool Visu = (aPU2Get==0) && (aPSP==0);
   MakeCam();
   if (! mCamUp2Date)
      return 0;

   mPPI.SauvRot(mCam.Orient());
   const cCiblePolygoneEtal * aRes = 0;
   double aDistMin = 1e20;

   const cPolygoneEtal::tContCible & Cible3d = mPPI.CiblesInit() ;
   for 
   (
         cPolygoneEtal::tContCible::const_iterator itC=Cible3d.begin();
         itC!=Cible3d.end();
	 itC++
   )
   {
       bool OkDet=true;
       cCibleCalib * aCC =(*itC)->CC();
       if (OkForEllipse())
       {
            cParamEllipse anEl = InitParamEllipe(*aCC,1.0);
	    double aDet = anEl.DirH() ^ anEl.DirV();
	    OkDet = (aDet < 0);
       }
       int aCoul = P8COL::red;
       if (OkDet)
       {

            Pt2dr aPU = mCam.R3toF2((*itC)->Pos());
            aPU = mPPI.Norm2Im(aPU);
            Pt2dr aPW =  mWIm.U2W(aPU);
	    if (aPU2Get)
	    {
	        double aD = euclid(aPU,*aPU2Get);
		if (aD<aDistMin)
		{
		   aDistMin = aD;
		   aRes = *itC;
		}
	    }
	    if (Visu)
	    {
                mWIm.W().draw_circle_abs(aPW,5.0,mWIm.W().pdisc()(aCoul));
      

                std::string  aStr = ToString((*itC)->Ind());
                mWIm.W().fixed_string
                (
			    aPW,
			    aStr.c_str(),
			    mWIm.W().pdisc()(aCoul),
			    true
               );
	   }
      }
   }

   return aRes;
}

void cPointeInit::Sauv()
{
     std::vector<FILE *> aVFP;
     aVFP.push_back(ElFopen(mNameFilePointe.c_str(),"w"));

     if (mPPI.SauvInterm())
        aVFP.push_back(ElFopen(mPPI.NamePointeInterm().c_str(),"w"));

     if (mPPI.SauvFinal())
        aVFP.push_back(ElFopen(mPPI.NamePointeFinal().c_str(),"w"));

     int aNbFP = aVFP.size();
   // mNameFilePointe (mPPI.NamePointeInit()),


     FILE * fPol = 0;
     FILE * fPointe = 0;
     if (mPPI.PseudoPolyg())
     {
         fPol =  ElFopen(mPPI.NamePolygone().c_str(),"w");
	 fPointe = ElFopen(mPPI.NamePointePolygone().c_str(),"w");
	 cout << mNameFilePointe << "\n";
	 cout << mPPI.NamePolygone() << "\n";
	 cout << mPPI.NamePointePolygone() << "\n";
     }
     for 
     (
        std::list<cPt2Im>::iterator iT = mPt2Is.begin();
        iT !=  mPt2Is.end();
        iT ++
     )
     {
        if ((iT->mPtPol.mEtat != eInexistant) && (iT->mPtPol.mEtat != eCur))
	{
            if ((iT->mPtIm.mEtat == eSelected) || (iT->mPtIm.mEtat == eAutreImPointes))
            {
	        for (int aK=0; aK<aNbFP ; aK++)
                  fprintf
                  (
                     aVFP[aK],"%d %f %f\n",
                     iT->pCible->Ind(),
                     iT->mPtIm.mPt.x,
                     iT->mPtIm.mPt.y
                  );
            }
	    if (fPol)
            {
                Pt3dr P3 = iT->pCible->Pos();
                fprintf(fPol,"%d %f %f %f M7 0\n",iT->pCible->Ind(),P3.x,P3.y,P3.z);
            }
	    if (fPointe)
            {
                fprintf
                (
                   fPointe,"%d %f %f\n",
                   iT->pCible->Ind(),
                   iT->mPtPol.mPt.x,
                   iT->mPtPol.mPt.y
                );
            }
	}

     }
     for (int aK=0; aK<aNbFP ; aK++)
         ElFclose(aVFP[aK]);
     if (fPol) 
        ElFclose(fPol);
     if (fPointe) 
        ElFclose(fPointe);
     ShowProj(0,0);
}


void cPointeInit::InitPtTmp()
{
     mPt2Is.back().mPtIm.mEtat = eInexistant;
     mPt2Is.back().mPtPol.mEtat = eInexistant;
     mPt2Is.back().pCible = 0;
     mPt2Is.back().mName = "";
}



CaseGPUMT * cPointeInit::StdCase(const std::string & aName,Pt2di aCase)
{
// std::cout << "MMMMDDD " << MMDir() << "\n";
	std::string aFN = MMDir() + std::string("data/") + aName +std::string(".tif");

	return  new CaseGPUMT
		    (
		         mPopUp,aName,aCase,
			Tiff_Im(aFN.c_str()).in(0) *255
		    );

}

BoolCaseGPUMT * cPointeInit::BoolStdCase
            (
                const std::string & aNameT,
                const std::string & aNameF,
                Pt2di aCase
            )
{
	std::string aFNT =  MMDir() + std::string("data/") + aNameT +std::string(".tif");
	std::string aFNF = MMDir() + std::string("data/") + aNameF +std::string(".tif");

	return  new BoolCaseGPUMT
		    (
		         mPopUp,aNameT,aCase,
			Tiff_Im(aFNT.c_str()).in(0) *255,
			Tiff_Im(aFNF.c_str()).in(0) *255,
			true
		    );

}






std::list<cPt2Im> cPointeInit::InitListePt2Is()
{

    std::list<cPt2Im> aRes;
    cSetPointes1Im PPol(mPPI.Polygone(),mPPI.NamePointePolygone());


    for
    (
        cSetPointes1Im::tCont::iterator itP = PPol.Pointes().begin();
        itP != PPol.Pointes().end();
        itP++
    )
    {
	    Pt2dr aP = itP->PosIm();
	    mPPI.ConvPointeImagePolygone(aP);
            aRes.push_back(cPt2Im(&itP->Cible(),aP));
    }



    aRes.push_back(cPt2Im(0,Pt2dr(0,0)));

    return aRes;
}

cPt2Im *  cPointeInit::ClikNearest()
{
     Clik aCl = mWIm.clik_press();

     bool InIm = (aCl._w ==  mWIm.W());
     Pt2dr aP = InIm ? mWIm.W2U(aCl._pt) :  mPolyg.W2U(aCl._pt) ;

     return Nearest(InIm,aP);
}

cPt2Im *  cPointeInit::Nearest(bool InIm,Pt2dr aP,bool AcceptSelected)
{
   cPt2Im * aRes = 0;

   REAL aDmin = 1e20;

   std::list<cPt2Im>::iterator End = mPt2Is.end();
   End--;
   for (std::list<cPt2Im>::iterator itP = mPt2Is.begin(); itP!= End ; itP++)
   {
       Pt2dr aQ = InIm ? itP->mPtIm.mPt : itP->mPtPol.mPt;
       eTyEtat aState =  InIm ? itP->mPtIm.mEtat : itP->mPtPol.mEtat;
       if  ((AcceptSelected || (aState!=eSelected)) && (aState!=eAutreImPointes))
       {
           REAL aD = euclid(aQ,aP);
           if (aD<aDmin)
           {
              aDmin = aD;
	      aRes = &(*itP);
           }
       }
   }

   return aRes;
}

static std::vector<double> NoParAdd;

cPointeInit::cPointeInit
(
     cParamPointeInit & aPPI,
     Pt2di aSzWIm,
     Pt2di aSzWPolyg
) :
   mPPI     (aPPI),
   mNameFilePointe (mPPI.NamePointeInit()),

   mPt2Is (InitListePt2Is()),
   mWIm   (
                mPt2Is,
		true,
                Video_Win::WStd(aSzWIm,1.0),
                mPPI.NameImageCamera(),
		mPPI.EtatDepAff(),
		*this
          ),
   mPolyg (
               mPt2Is,
               false,
               // Video_Win::WStd(aSzWPolyg,1.0),
               Video_Win(mWIm.W().disp(),mWIm.W().sop(),Pt2di(0,0),aSzWPolyg),
               mPPI.NameImagePolygone(),
	       mPPI.EtatDepAff(),
	       *this
          ),
   mTifMenu      (MMIcone("Loupe")),
   mSzCase       (mTifMenu.sz()),
   mPopUp        (
                          mWIm.W(),
                          mSzCase,
                          Pt2di(5,5),
                          Pt2di(1,1)
                 ),
    mCaseOkCur   (StdCase("OkCur",Pt2di(0,0))),
    mCaseSetCur  (StdCase("SetCur",Pt2di(0,1))),
    mCaseKillCur (StdCase("KillCur",Pt2di(0,2))),

    mCaseProj    (StdCase("Proj",Pt2di(1,0))),
    mCaseDyn     (StdCase("Dyn",Pt2di(1,1))),
    mCaseKillLiais (StdCase("TDM",Pt2di(1,2))),

    mCaseExit    (StdCase("Exit",Pt2di(4,4))) ,
    mCasePolyg   (StdCase("Polyg",Pt2di(2,1))) ,
    mBCaseBilin  (BoolStdCase("2Lin","PPv",Pt2di(2,0))),
    mCam         (true,1.0,Pt2dr(0,0),NoParAdd),
    mCamUp2Date   (false),
    mWithIm       (true),
    mIm           (1,1),
    mRCdd         (0)
{
    if (mWithIm)
    {
       // Tiff_Im aTF = Tiff_Im::BasicConvStd(mPPI.NameImageCamera());
       Tiff_Im aTF = Tiff_Im::StdConvGen(mPPI.NameImageCamera(),-1,false);
       Pt2di aSz =aTF.sz();
       mIm = Im2D_REAL4(aSz.x,aSz.y);
       ELISE_COPY(aTF.all_pts(),aTF.in(),mIm.out());

       Video_Win * aWIm = new Video_Win(mWIm.W(),Video_Win::eDroiteH,Pt2di(500,500));
       mRCdd = new cRechercheCDD(mIm,aWIm);
    }
    InitPtTmp();

    if (ELISE_fp::exist_file(mNameFilePointe.c_str()))
    {
	    cout << "OK FOUND FILE " << mNameFilePointe << " \n";
	    cSetPointes1Im aSet = mPPI.SetPointe(mNameFilePointe);
            for 
            (
               cSetPointes1Im::tCont::iterator itP = aSet.Pointes().begin();
               itP != aSet.Pointes().end();
               itP++
            )
	    {
//std::cout << "  KK " << itP->Cible().Ind() << "\n";
//bool aTest = (itP->Cible().Ind()==1051);
	         bool aGot = false;
	          for 
                  (
	                 std::list<cPt2Im>::iterator iT = mPt2Is.begin();
	                 iT !=  mPt2Is.end();
	                 iT ++
                  )
		  {
//if (aTest)
//{
    //std::cout << "  YY " << iT->pCible << "\n";
    //std::cout << "   --  " << iT->pCible->Ind() << "\n";
//}
			  if (   iT->pCible 
			      && (iT->pCible->Ind()==itP->Cible().Ind())
			      )
                          {
			          aGot = true;
				  iT->mPtPol.mEtat = eSelected;
				  iT->mPtIm.mEtat = eSelected;
				  iT->mPtIm.mPt = itP->PosIm();
			  }
		  }
/*
   Ceux ci correspondent aux cibles qui ne peuvent pas etre pointees par les
   sur le IGNPointePoly
   
*/
		  if (! aGot)
		  {
		       cPt2Im aNew(&(itP->Cible()),Pt2dr(-1,-1));
		       aNew.mPtPol.mEtat =eAutreImPointes;
		       aNew.mPtIm.mEtat =eAutreImPointes;
		       aNew.mPtIm.mPt = itP->PosIm();
		       mPt2Is.push_front(aNew);
		  }
            }
    }
}

void  cPointeInit::GUR_query_pointer(Clik aCl,bool)
{
   if (mModeCurIsPopUp)
   {
       mPopUp.SetPtActif(Pt2di(aCl._pt));
   }
   if (mModeCurIsDyn)
   {
       REAL dY = aCl._pt.y-mP0Cl.y;
       REAL aFact = pow(2.0,dY/100.0);
       mWIm.EtalD(VDyn0 * aFact);
   }
}
	        
		
void cPointeInit::RecherchePosAutomCDD(cPt2Im & aC)
{
    if (aC.pCible->CC()->Ponctuel())
       return;
    cParamEllipse  anEl = InitParamEllipe(*(aC.pCible->CC()));
    anEl.SetCentre(aC.mPtIm.mPt);
    mRCdd->RechercheInit(anEl,20.0);
    mRCdd->RechercheCorrel(anEl);
    

    aC.mPtIm.mPt = anEl.Centre();

    mRCdd->Show(anEl);
}



void cPointeInit::Test()
{
     Clik aCl = mWIm.clik_press();
     mModeCurIsPopUp = false;
     mModeCurIsDyn = false;

     VDyn0 = mWIm.VDyn();

     if ((aCl._w == mWIm.W()) && (aCl._b==3))
     {
         mModeCurIsPopUp = true;
         mPopUp.UpCenter(Pt2di(aCl._pt));
	 mWIm.W().grab(*this);
	 CaseGPUMT * aCase = mPopUp.PopAndGet();
         mModeCurIsPopUp = false;

         if (aCase == mCaseKillCur)
         {
	     InitPtTmp();
         }
         if (aCase == mCasePolyg)
         {
             std::vector<Pt2dr> aLpt = mWIm.GetPolyg(P8COL::red,P8COL::green);
	     SaisiePolyg(aLpt);
         }
	 if (aCase == mBCaseBilin)
         {
             mWIm.SetModeReplicationPixel(! mBCaseBilin->Val());
         }
	 if (aCase == mCaseExit)
         {
             End();
         }
	 if (aCase == mCaseDyn)
	 {
            mModeCurIsDyn = true;
	    // aCl = mWIm.W().disp().clik_press();
            mP0Cl = aCl._pt;
	    mWIm.W().grab(*this);
            mModeCurIsDyn = false;
	 }
	 if (aCase == mCaseOkCur)
	 {
	     mCamUp2Date = false;
             cPt2Im & aP2 = mPt2Is.back();
	     if ((aP2.mPtPol.mEtat==eCur) && (aP2.mPtIm.mEtat == eCur))
             {
                  
                 cPt2Im * aQ2 = Nearest(false,aP2.mPtPol.mPt);
		 if (aQ2)
		 {
		     aQ2->mPtPol.mEtat = eSelected;
		     aQ2->mPtIm.mEtat = eSelected;
		     aQ2->mPtIm.mPt = aP2.mPtIm.mPt;
		     InitPtTmp();
		     Refresh();
		 }
	     }
	 }
	 if (aCase == mCaseKillLiais)
	 {
	     mCamUp2Date = false;
             cPt2Im * aP2  =ClikNearest();
	     if (aP2)
	     {
	        if (aP2->mPtPol.mEtat == eSelected)
		{
                   aP2->Reinit();
		   Refresh();
		}
	     }
	 }
	 if (aCase == mCaseSetCur)
	 {
	     mCamUp2Date = false;
             cPt2Im * aP2  =ClikNearest();
	     if (aP2)
	     {
	        if (aP2->mPtPol.mEtat == eSelected)
		{
                   mPt2Is.back() = *aP2;
                   mPt2Is.back().mPtIm.mEtat = eCur;
                   mPt2Is.back().mPtPol.mEtat = eCur;
                   aP2->Reinit();
		   Refresh();
		}
	     }
	 }
	 if (aCase == mCaseProj)
	 {
		 ShowProj(0,0);
	 }

     }

     if (aCl._b==1)
     {
	cPt2Im * aCPP = 0;  // Cible la + proche
        if (aCl._w == mWIm.W())
        {
            Pt2dr aPU =  mWIm.W2U(aCl._pt);
	    cPt2Im & aC = mPt2Is.back();
            aC.mPtIm.mPt = aPU;
            aC.mPtIm.mEtat = eCur;
	    if (aC.mPtPol.mEtat == eInexistant) // ShowProj
	    {

                   const cCiblePolygoneEtal *  aCPE = ShowProj(&aPU,0);
		   if (aCPE)
		   {
		      aCPP = GetPointeOfCible(aCPE);
		   }
	    }

        //    INT aRad =  mWIm.GetRadiom(aPU);
        }
        if ((aCPP != 0) || (aCl._w == mPolyg.W()))
	{
	    Pt2dr aPU ;
	    if (aCPP ==0)
	    {
               bool NAS = ! mPPI.PseudoPolyg();
	       aPU = mPolyg.W2U(aCl._pt);
               aCPP = Nearest(false,aPU,NAS);
	    }
	    if (aCPP)
	    {
	        if (aCPP->mPtPol.mEtat != eSelected)
	        {
                   mPt2Is.back().mPtPol.mPt = aCPP->mPtPol.mPt;
                   mPt2Is.back().mPtPol.mEtat = eCur;
	           mPt2Is.back().pCible = aCPP->pCible;
	           mPt2Is.back().mName = aCPP->mName;
		   if (mPPI.PseudoPolyg())
		   {
                       aCPP->mPtPol.mPt = aPU;
                       mPt2Is.back().mPtPol.mPt = aPU;
		       const_cast<cCiblePolygoneEtal *>(mPt2Is.back().pCible)
			       ->SetPos(mPPI.P3ofIm(aPU));
		   }
	        }
	    }
	}

	if (aCl.shifted())
	{
	    cPt2Im & aC = mPt2Is.back();
	    if (aC.pCible != 0)
	    {
	        if (mRCdd)
	           RecherchePosAutomCDD(aC);
	    }
	}

	Refresh();
     }

	// mPolyg.clik_press();
}

/************************************************/
/*                                              */
/*               ::                             */
/*                                              */
/************************************************/

void PointesInitial
     (
          cParamPointeInit & aPPI,
          Pt2di aSzWIm,
          Pt2di aSzWPolyg
     ) 
{
   cPointeInit aPointe(aPPI,aSzWIm,aSzWPolyg);

   while (1) 
      aPointe.Test();
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
