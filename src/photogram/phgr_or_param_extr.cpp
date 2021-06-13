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




/************************************************************/
/*                                                          */
/*               cIncParamExtrinseque                       */
/*                                                          */
/************************************************************/

cIncParamExtrinseque::~cIncParamExtrinseque()
{
}
extern cSetEqFormelles&  NOSET();
extern cSetEqFormelles*  PtrNOSET();


cIncParamExtrinseque::cIncParamExtrinseque
(
             tPosition                  aTPos,
             AllocateurDInconnues  &    anAlloc,
             cIncEnsembleCamera    *    apEns
)  :
   mAlloc     (anAlloc),
   mTPos      (aTPos),
   mIncInterv (false,"toto",NOSET()),
   mpEns      (apEns)
{
}

AllocateurDInconnues & cIncParamExtrinseque::Alloc()
{
   return mAlloc;
}


std::string   cIncParamExtrinseque::NameType()
{
    switch(mTPos)
    {
         case ePosFixe :       return "Fixe";
         case ePosBaseUnite :  return "BUnite";
         case ePosLibre :      return "Libre";
    }
    ELISE_ASSERT(false,"cIncParamExtrinseque::NameTPos");
    return "toto";
}


cIncParamExtrinseque::tPosition cIncParamExtrinseque::TPos() const
{
   return mTPos;
}

void cIncParamExtrinseque::EndInitIPE()
{
     mIncInterv.Close();
     for (INT aK=0 ; aK<INT(mFCO.size()) ; aK++)
     {
        mFC0Adr.push_back(mFCO[aK]->FoncSetVarAdr());
        mFCO[aK]->SetCoordCur(mAlloc.ValsVar());
     }

    SetRappelCOInit();
}


const  cIncIntervale & cIncParamExtrinseque::IntervInc() const
{
   return mIncInterv;
}
cIncIntervale & cIncParamExtrinseque::IntervInc() 
{
   return mIncInterv;
}

cIncEnsembleCamera * cIncParamExtrinseque::Ensemble()
{
    return mpEns;
}


std::vector<cElCompiledFonc *> &
     cIncParamExtrinseque::FoncteurRappelCentreOptique ()
{
   return mFCO;
}

          
/************************************************************/
/*                                                          */
/*               cParamExtrinsequeRigide                    */
/*                                                          */
/************************************************************/

class cParamExtrinsequeRigide : public cIncParamExtrinseque
{
      public  :
         ElRotation3D CurRot ();
         cParamExtrinsequeRigide
         (AllocateurDInconnues &,ElRotation3D,cIncEnsembleCamera *);

      private :

         void SetRappelCOInit() {}

         class cNumVarLoc
         {
           public :
              cNumVarLoc() : fOmega(3,3) {}

              void Init(std::string aNum);

              std::string            mNum;
              std::string            mNameMatr[3][3];

              std::string            mNameTrx;
              std::string            mNameTry;
              std::string            mNameTrz;

              Pt3d<Fonc_Num>         fTr;
              ElMatrix<Fonc_Num>     fOmega;
         };

         ElRotation3D          mCurRot; 
         cNumVarLoc            mNVL[2];


         ElMatrix<Fonc_Num> & Omega(INT aNum)  ;
         Pt3d<Fonc_Num>     & Tr(INT aNum)  ;
         void InitFoncteur(cElCompiledFonc &,INT aNum) ;
};

void cParamExtrinsequeRigide::cNumVarLoc::Init(std::string aNum)
{
   mNum = aNum;

   mNameTrx = "COptX_"+aNum;
   mNameTry = "COptY_"+aNum;
   mNameTrz = "COptZ_"+aNum;

   fTr = Pt3d<Fonc_Num>(cVarSpec(0,mNameTrx),cVarSpec(0,mNameTry),cVarSpec(0,mNameTrz));

   for (INT x=0; x<3 ; x++)
       for (INT y=0; y<3 ; y++)
       {
             mNameMatr[x][y] = "Matr_" +ToString(x)+ToString(y) + aNum;
             fOmega(x,y) = cVarSpec(0,mNameMatr[x][y]);
       }
}

ElMatrix<Fonc_Num> & cParamExtrinsequeRigide::Omega(INT aNum)
{
   return mNVL[aNum].fOmega;
}

Pt3d<Fonc_Num>     & cParamExtrinsequeRigide::Tr(INT aNum)
{
   return mNVL[aNum].fTr;
}


void cParamExtrinsequeRigide::InitFoncteur(cElCompiledFonc & aFoncteur,INT aNum) 
{
   for (INT x=0; x<3 ; x++)
       for (INT y=0; y<3 ; y++)
       {
             *(aFoncteur.RequireAdrVarLocFromString(mNVL[aNum].mNameMatr[x][y])) = mCurRot.Mat()(x,y);
       }
   
   *(aFoncteur.RequireAdrVarLocFromString(mNVL[aNum].mNameTrx)) = mCurRot.tr().x;
   *(aFoncteur.RequireAdrVarLocFromString(mNVL[aNum].mNameTry)) = mCurRot.tr().y;
   *(aFoncteur.RequireAdrVarLocFromString(mNVL[aNum].mNameTrz)) = mCurRot.tr().z;
}

cParamExtrinsequeRigide::cParamExtrinsequeRigide
(
     AllocateurDInconnues & anAlloc,
     ElRotation3D aRot,
     cIncEnsembleCamera * apEns
) :
     cIncParamExtrinseque(ePosFixe,anAlloc,apEns),
     mCurRot (aRot)
{
   mNVL[0].Init("Extr1");
   mNVL[1].Init("Extr2");
}



ElRotation3D  cParamExtrinsequeRigide::CurRot ()
{
   return mCurRot;
}

cIncParamExtrinseque * cIncParamExtrinseque::IPEFixe
                       (
                            AllocateurDInconnues & anAlloc,
                            ElRotation3D aRot,
                            cIncEnsembleCamera * apEns
                       ) 
{
    cParamExtrinsequeRigide * Res =  
           new cParamExtrinsequeRigide(anAlloc,aRot,apEns);
    Res->EndInitIPE();
    return Res;
}

/************************************************************/
/*                                                          */
/*               cIncRotatVectLibre                         */
/*               cIncVectUnite                              */
/*                                                          */
/*             (Classes support)                            */
/*                                                          */
/************************************************************/

class cIncRotatVectLibre
{
   public  :



      ElMatrix<Fonc_Num>   &FRVLRot() {return fOmega;}
      
      REAL A01() const {return mA01;}
      REAL A02() const {return mA02;}
      REAL A12() const {return mA12;}


      cIncRotatVectLibre(AllocateurDInconnues & anAlloc,ElRotation3D  anOrientInit) :
          mA01       (anOrientInit.teta01()),
          fAlpha01   (anAlloc.NewF("cIncRotatVectLibre","A01",&mA01)),
          mA02       (anOrientInit.teta02()),
          fAlpha02   (anAlloc.NewF("cIncRotatVectLibre","A02",&mA02)),
          mA12       (anOrientInit.teta12()),
          fAlpha12   (anAlloc.NewF("cIncRotatVectLibre","A12",&mA12)),
          fOmega     (ElMatrix<Fonc_Num>::Rotation(fAlpha01,fAlpha02,fAlpha12))
      {
      }

   private :

      REAL               mA01;
      Fonc_Num           fAlpha01;
      REAL               mA02;
      Fonc_Num           fAlpha02;
      REAL               mA12;
      Fonc_Num           fAlpha12;
      ElMatrix<Fonc_Num> fOmega;

};

class cIncVectUnite 
{
   public :

       Pt3d<Fonc_Num>   & FVUTr(INT aNum) {return mNVL[aNum].fTr;}
       Pt3d<REAL>         VUTr () {return mCRot+Pt3d<REAL>::TyFromSpherique(mRho,mTeta,mPhi);}

       void InitFoncteur(cElCompiledFonc & aFoncteur,INT aNum)
       {
             *(aFoncteur.RequireAdrVarLocFromString(mNVL[aNum].mNameRho)) = mRho;
             *(aFoncteur.RequireAdrVarLocFromString(mNVL[aNum].mNameCRx)) = mCRot.x;
             *(aFoncteur.RequireAdrVarLocFromString(mNVL[aNum].mNameCRy)) = mCRot.y;
             *(aFoncteur.RequireAdrVarLocFromString(mNVL[aNum].mNameCRz)) = mCRot.z;
       }



      cIncVectUnite
      (
	       AllocateurDInconnues & anAlloc,
	       Pt3dr aCRot,
	       ElRotation3D  anOrientInit,
               std::vector<cElCompiledFonc *> &aFCO
      ) :
           mCRot      (aCRot),
           mRho       (ToSpherique(anOrientInit.tr()-aCRot ,mRho,mTeta,mPhi)),
           //mTeta      (mTeta),
           //mPhi       (mPhi),
           mTeta0     (mTeta),
           mPhi0      (mPhi),
	   mIndTeta   (anAlloc.CurInc()),
           fTeta      (anAlloc.NewF("cIncVectUnite","Teta",&mTeta)),
           fPhi       (anAlloc.NewF("cIncVectUnite","Phi",&mPhi))
      {
          mNVL[0].Init("Extr1",fTeta,fPhi);
          mNVL[1].Init("Extr2",fTeta,fPhi);
          aFCO.push_back(cElCompiledFonc::FoncSetVar(PtrNOSET(),mIndTeta));
          aFCO.push_back(cElCompiledFonc::FoncSetVar(PtrNOSET(),mIndTeta+1));
      }
 
      const REAL Teta0() {return mTeta0;}
      const REAL Phi0()   {return mPhi0;}

   private :

      class cNumVarLoc
      {
          public :
              std::string mNameRho;
              Fonc_Num    fRho;
              Pt3d<Fonc_Num> fTr;

              std::string mNameCRx;
              std::string mNameCRy;
              std::string mNameCRz;
              Fonc_Num    fCRx;
              Fonc_Num    fCRy;
              Fonc_Num    fCRz;
              void Init(const std::string & aNum,Fonc_Num  fTeta,Fonc_Num  fPhi)

              {
                     mNameRho = "BU_COptRho_" +aNum;
		     mNameCRx = "BU_CRotX_" + aNum;
		     mNameCRy = "BU_CRotY_" + aNum;
		     mNameCRz = "BU_CRotZ_" + aNum;
                     fRho = cVarSpec(0,mNameRho);
                     fCRx = cVarSpec(0,mNameCRx);
                     fCRy = cVarSpec(0,mNameCRy);
                     fCRz = cVarSpec(0,mNameCRz);
                     fTr  =    Pt3d<Fonc_Num>(fCRx,fCRy,fCRz)
			     + Pt3d<Fonc_Num>::TyFromSpherique(fRho,fTeta,fPhi);
              }
      };

      Pt3dr      mCRot;
      REAL       mTeta;
      REAL       mPhi;
      REAL       mRho;
      REAL       mTeta0;
      REAL       mPhi0;

      INT        mIndTeta;
      Fonc_Num   fTeta;
      Fonc_Num   fPhi;

      cNumVarLoc mNVL[2];
     
};

class cIncVectLibre 
{
   public :

         Pt3d<Fonc_Num> & FVLTr() {return fPt;}
         Pt3d<REAL>       VLTr () {return mPt;}


      cIncVectLibre
      (
           AllocateurDInconnues & anAlloc,
	   ElRotation3D  anOrientInit,
           std::vector<cElCompiledFonc *> &aFCO
      ) :
           mPt        (anOrientInit.tr()),
           mPt0       (mPt),
           mIndX      (anAlloc.CurInc()),
           fPt        (anAlloc.NewPt3("cIncVectLibre",mPt))
      {
          aFCO.push_back(cElCompiledFonc::FoncSetVar(PtrNOSET(),mIndX));
          aFCO.push_back(cElCompiledFonc::FoncSetVar(PtrNOSET(),mIndX+1));
          aFCO.push_back(cElCompiledFonc::FoncSetVar(PtrNOSET(),mIndX+2));
      }

      Pt3dr Pt0() const {return mPt0;}

   private :

      Pt3d<REAL>     mPt;
      Pt3d<REAL>     mPt0;
      INT            mIndX;
      Pt3d<Fonc_Num> fPt;
};



/************************************************************/
/*                                                          */
/*               cParamExtrinsequeBaseUnite                 */
/*                                                          */
/************************************************************/


class cParamExtrinsequeBaseUnite : public cIncParamExtrinseque
{
      public  :

         ElRotation3D CurRot ();
         cParamExtrinsequeBaseUnite
          (AllocateurDInconnues &,Pt3dr aCRot,ElRotation3D,cIncEnsembleCamera *);

      private :

         cIncRotatVectLibre  mRVL;
         cIncVectUnite       mVU;

         ElMatrix<Fonc_Num> & Omega(INT aNum)  {return mRVL.FRVLRot();}
         Pt3d<Fonc_Num>     & Tr(INT aNum)   {return mVU.FVUTr(aNum);}
         void InitFoncteur(cElCompiledFonc & aFoncteur,INT aNum) 
         {
              mVU.InitFoncteur(aFoncteur,aNum);
         }
         void SetRappelCOInit()
         {
            *(mFC0Adr[0]) =  mVU.Teta0();
            *(mFC0Adr[1]) =  mVU.Phi0();
         }

};

cParamExtrinsequeBaseUnite::cParamExtrinsequeBaseUnite
(
    AllocateurDInconnues & anAlloc,
    Pt3dr         aCRot,
    ElRotation3D  aRot,
    cIncEnsembleCamera * apEns
) :
        cIncParamExtrinseque(ePosBaseUnite,anAlloc,apEns),
        mRVL(anAlloc,aRot),
        mVU (anAlloc,aCRot,aRot,mFCO)
{
}

ElRotation3D cParamExtrinsequeBaseUnite::CurRot ()
{
   return ElRotation3D
	  (
	       mVU.VUTr(),
               mRVL.A01(),mRVL.A02(),mRVL.A12()
          );
}


cIncParamExtrinseque * cIncParamExtrinseque::IPEBaseUnite
                      (
                           AllocateurDInconnues & anAlloc,
			   Pt3dr  aCRot,
                           ElRotation3D aRInit,
                           cIncEnsembleCamera *  apEns
                      )
{
   cParamExtrinsequeBaseUnite * Res =  
        new cParamExtrinsequeBaseUnite(anAlloc,aCRot,aRInit,apEns);
   Res->EndInitIPE();
   return Res;
}


/************************************************************/
/*                                                          */
/*               cParamExtrinsequeLibre                     */
/*                                                          */
/************************************************************/


class cParamExtrinsequeLibre : public cIncParamExtrinseque
{
      public  :

         ElRotation3D CurRot ();
         cParamExtrinsequeLibre
             (AllocateurDInconnues &,ElRotation3D,cIncEnsembleCamera *);

      private :

           void InitFoncteur(cElCompiledFonc & aFoncteur,INT aNum){}
           ElMatrix<Fonc_Num> & Omega(INT )  {return mRVL.FRVLRot();}
           Pt3d<Fonc_Num>     & Tr(INT )   {return mVL.FVLTr();}

           cIncRotatVectLibre  mRVL;
           cIncVectLibre       mVL;

           void SetRappelCOInit()
           {
              Pt3dr aP0 = mVL.Pt0();

              *(mFC0Adr[0]) =  aP0.x;
              *(mFC0Adr[1]) =  aP0.y;
              *(mFC0Adr[2]) =  aP0.z;
           }

};


cParamExtrinsequeLibre::cParamExtrinsequeLibre
(
    AllocateurDInconnues & anAlloc,
    ElRotation3D  aRot,
    cIncEnsembleCamera * apEns
) :
        cIncParamExtrinseque(ePosLibre,anAlloc,apEns),
        mRVL(anAlloc,aRot),
        mVL (anAlloc,aRot,mFCO)
{
}

ElRotation3D cParamExtrinsequeLibre::CurRot ()
{
   return ElRotation3D
	  (
	       mVL.VLTr(),
               mRVL.A01(),mRVL.A02(),mRVL.A12()
          );
}


cIncParamExtrinseque * cIncParamExtrinseque::IPELibre
                       (
                            AllocateurDInconnues & anAlloc,
                            ElRotation3D aRInit,
                            cIncEnsembleCamera * apEns
                       )
{
    cParamExtrinsequeLibre * Res = 
          new cParamExtrinsequeLibre(anAlloc,aRInit,apEns);
    Res->EndInitIPE();
    return Res;
}


cIncParamExtrinseque *  cIncParamExtrinseque::Alloc
(
                     tPosition  aPos,
                     AllocateurDInconnues & anAlloc,
                     ElRotation3D aRInit,
                     cIncEnsembleCamera * apEns
)
{
     switch (aPos)
     {
         case ePosFixe :       return   IPEFixe      (anAlloc,aRInit,apEns);
         case ePosBaseUnite :  return   IPEBaseUnite (anAlloc,Pt3dr(0,0,0),aRInit,apEns);
         case ePosLibre :      return   IPELibre     (anAlloc,aRInit,apEns);
    }
    ELISE_ASSERT(false,"cIncParamExtrinseque::Alloc");
    return 0;
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
