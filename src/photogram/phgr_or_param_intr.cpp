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
/*               cIncParamIntrinseque                       */
/*                                                          */
/************************************************************/

    //======================================
    // cIncParamIntrinseque::cNumVarLoc
    //======================================

void cIncParamIntrinseque::cNumVarLoc::Init(cIncParamIntrinseque & aParam,std::string aNum)
{
   mNum = aNum;
   mNameFocale = aParam.mNameFocale + mNum;
   mNamePPx    = aParam.mNamePPx + mNum;
   mNamePPy    = aParam.mNamePPy + mNum;

   fFocale = aParam.mIsFFree ? aParam.fFocale :  cVarSpec(0,mNameFocale) ;
   fPP     = aParam.mIsPPFree ? aParam.fPP : Pt2d<Fonc_Num>(cVarSpec(0,mNamePPx),cVarSpec(0,mNamePPy));

}


    //======================================
    // cIncParamIntrinseque 
    //======================================

REAL cIncParamIntrinseque::CurFocale() const
{
   return mFocale;
}
Pt2dr cIncParamIntrinseque::CurPP() const
{
   return mPP;
}

ElDistRadiale_PolynImpair   cIncParamIntrinseque::CurDR() const
{
    return mDR;
}
ElDistRadiale_PolynImpair   cIncParamIntrinseque::DRInit() const
{
    return mDRInit;
}



void cIncParamIntrinseque::InitFoncteur(cElCompiledFonc & aFoncteur,INT aNum)
{
  if (! mIsFFree)
  {
     *(aFoncteur.RequireAdrVarLocFromString(mNVL[aNum].mNameFocale)) = mFocale;
  }

  if (! mIsPPFree)
  {
      *(aFoncteur.RequireAdrVarLocFromString(mNVL[aNum].mNamePPx)) = mPP.x;
      *(aFoncteur.RequireAdrVarLocFromString(mNVL[aNum].mNamePPy)) = mPP.y;
  }
}

cIncParamIntrinseque::~cIncParamIntrinseque()
{

   for (INT aK=0 ; aK<INT(mFoncsAux.size()) ; aK++)
       delete mFoncsAux[aK];
}

cSetEqFormelles*  PtrNOSET()
{
  ELISE_ASSERT(false,"NOSET-NOSET");
  return (cSetEqFormelles *) 0;
}

cSetEqFormelles&  NOSET()
{
  return *PtrNOSET();
}

cIncParamIntrinseque::cIncParamIntrinseque
(
      AllocateurDInconnues  &     anAlloc,
      REAL                        aFocale,
      bool                        isFocaleFree,
      Pt2dr                       aPP,
      bool                        isPPFree,
      cIncEnsembleCamera *        apEns,
      ElDistRadiale_PolynImpair * pDistRad
)   :
    mAlloc        (anAlloc),
    mIncInterv    (false,"toto",NOSET()),
    mpEns         (apEns),
    mNameFocale   ("Focale"),
    mNamePPx      ("PPx"),
    mNamePPy      ("PPy"),
    mIsFFree      (isFocaleFree),
    mFocale       (aFocale),
    mFocaleInit   (mFocale),
    mIsPPFree     (isPPFree),
    mPP           (aPP),
    mIndFoc       (anAlloc.CurInc()),
    fFocale       (isFocaleFree ?  anAlloc.NewF("cIncParamIntrinseque","F",&mFocale)   : 0),
    fPP           (isPPFree     ?  anAlloc.NewPt2("cIncParamIntrinseque:PP",mPP) : Pt2d<Fonc_Num>(0,0)),
    mWithDR       (pDistRad != 0),
    mDR           (mWithDR ? (*pDistRad) :  ElDistRadiale_PolynImpair(1.0,Pt2dr(0,0))),
    mDRInit       (mDR),
    mIndCrd       (anAlloc.CurInc()),
    mCentreDR     (mWithDR ? anAlloc.NewPt2("cIncParamIntrinseque:CD",mDR.Centre()) :  Pt2d<Fonc_Num>(0,0)),
    mFoncRCD      (),
    mFoncRFoc     (0),
    mFoncsAux     ()
{

   INT anIndCoeffDr = anAlloc.CurInc();
   if (mWithDR)
   {
      for (INT aK=0 ; aK<INT(mDR.NbCoeff()) ; aK++)
      {
           mCoeffDR.push_back(anAlloc.NewF("cIncParamIntrinseque","R"+ToString(3+2*aK),&mDR.Coeff(aK)));
      }
   }

   mNVL[0].Init(*this,"Intr1");
   mNVL[1].Init(*this,"Intr2");
   mIncInterv.Close();

    if (mWithDR)
    {

        // Foncteur de Rappels du centre de distorsion
        for (INT aK=0; aK<2 ; aK++)
        {
            mFoncRCD.push_back(cElCompiledFonc::FoncSetVar(PtrNOSET(),mIndCrd+aK));
            mFoncsAux.push_back(mFoncRCD[aK]);
            mFRCDAdr.push_back(mFoncRCD[aK]->FoncSetVarAdr());
        }
        CurSetRappelCrd();
        for (INT aK=0; aK<2 ; aK++)
            mFoncRCD[aK]->SetCoordCur(Alloc().ValsVar());

        for (INT aK=0 ; aK<INT(mDR.NbCoeff()) ; aK++)
        {
            mFoncRCoeffD.push_back(cElCompiledFonc::FoncSetVar(PtrNOSET(),anIndCoeffDr+aK));
            mFoncsAux.push_back(mFoncRCoeffD[aK]);
            mFRCoeffDAdr.push_back(mFoncRCoeffD[aK]->FoncSetVarAdr());
	    CurSetRappelCoeffDist(aK);
            mFoncRCoeffD[aK]->SetCoordCur(Alloc().ValsVar());
        }

    }

    if (mIsFFree)
    {
         mFoncRFoc  = cElCompiledFonc::FoncSetVar(PtrNOSET(),mIndFoc);
         mFoncsAux.push_back(mFoncRFoc);
         mFRadrFoc = mFoncRFoc->RequireAdrVarLocFromString(cElCompiledFonc::NameFoncSetVar);

         CurSetRappelFocal();
         mFoncRFoc->SetCoordCur(Alloc().ValsVar());
    }
}


std::vector<cElCompiledFonc *> &  cIncParamIntrinseque::FoncteurRappelCentreDist ()
{
    ELISE_ASSERT(mWithDR,"No Dist Rad in FoncteurRappelCentreDist");
    
    return mFoncRCD;
}
std::vector<cElCompiledFonc *> & cIncParamIntrinseque::FoncteursAuxiliaires()
{
   return mFoncsAux;
}



void cIncParamIntrinseque::SetRappelCrd(Pt2dr aC)
{
    FoncteurRappelCentreDist();
    *mFRCDAdr[0] = aC.x;
    *mFRCDAdr[1] = aC.y;
}
void cIncParamIntrinseque::CurSetRappelCrd()
{
    SetRappelCrd(mDR.Centre());
}
void cIncParamIntrinseque::InitSetRappelCrd()
{
    SetRappelCrd(mDRInit.Centre());
}

void cIncParamIntrinseque::SetRappelCoeffDist(INT aK,REAL aVal)
{
   ELISE_ASSERT((aK>=0) && (aK<INT(mFRCoeffDAdr.size())),"SetRappelCoeffDist");
   *(mFRCoeffDAdr[aK]) = aVal;
}

void  cIncParamIntrinseque::CurSetRappelCoeffDist(INT aK)
{
     SetRappelCoeffDist(aK,mDR.Coeff(aK));
}
void  cIncParamIntrinseque::InitSetRappelCoeffDist(INT aK)
{
     SetRappelCoeffDist(aK,mDRInit.Coeff(aK));
}
std::vector<cElCompiledFonc *> &   cIncParamIntrinseque::FoncteurRappelCoeffDist ()     
{
    return mFoncRCoeffD;
}


cElCompiledFonc * cIncParamIntrinseque::FoncteurRappelFocal()
{
    ELISE_ASSERT(mIsFFree,"No FFre in cIncParamIntrinseque::SetRappelFocal");
    return mFoncRFoc;
}

void cIncParamIntrinseque::SetRappelFocal(REAL aFoc)
{
    FoncteurRappelFocal();
    *mFRadrFoc = aFoc;
}
void cIncParamIntrinseque::CurSetRappelFocal()
{
    SetRappelFocal(mFocale);
}
void cIncParamIntrinseque::InitSetRappelFocal()
{
    SetRappelFocal(mFocaleInit);
}






cIncParamIntrinseque  * cIncParamIntrinseque::NewOneNoDist
         (
             AllocateurDInconnues  &     anAlloc,
             REAL                        aFocale,
             bool                        isFocaleFree,
             Pt2dr                       aPP,
             bool                        isPPFree,
             cIncEnsembleCamera *        apEns,
             ElDistRadiale_PolynImpair * aDist
         ) 
{
    return new cIncParamIntrinseque(anAlloc,aFocale,isFocaleFree,aPP,isPPFree,apEns,aDist);
}



std::string   cIncParamIntrinseque::NameType()
{
   std::string aRes =   std::string("F") + ToString(mIsFFree)
                      + std::string("PP") + ToString(mIsPPFree);

   if (mWithDR)
   {
       aRes = aRes + "DR" + ToString(INT(1+2*mCoeffDR.size()));
   }
   return aRes;
}

Pt2d<Fonc_Num>   cIncParamIntrinseque::DistInv  (Pt2d<Fonc_Num> aP) 
{
   if (mWithDR)
   {
       Pt2d<Fonc_Num> fEc = aP -mCentreDR;
       Fonc_Num fRho2 = ElSquare(fEc.x) +ElSquare(fEc.y);
       Fonc_Num fSom = 1;
       Fonc_Num fRho2N  = 1;

       for (int aK=0 ; aK<(INT) mCoeffDR.size() ; aK++)
       {
           fRho2N = fRho2N * fRho2;
           fSom  = fSom + fRho2N * mCoeffDR[aK];
       }
       return mCentreDR +  Pt2d<Fonc_Num>(fEc.x*fSom,fEc.y*fSom);
   }
   return aP;
}

Pt3d<Fonc_Num>  cIncParamIntrinseque::DirRayon(Pt2d<Fonc_Num> aPCam,INT num)
{

    aPCam = DistInv(aPCam) - mNVL[num].fPP;
    Fonc_Num F = mNVL[num].fFocale;
    return Pt3d<Fonc_Num> (aPCam.x/F,aPCam.y/F,1);
    // return Pt3d<Fonc_Num> (aPCam.x,aPCam.y,F);
    // return Pt3d<Fonc_Num> (aPCam.x/sqrt(F),aPCam.y/sqrt(F),sqrt(F));
}

bool cIncParamIntrinseque::operator == (const cIncParamIntrinseque & aParam) const
{
   return this == &aParam;
}

const  cIncIntervale & cIncParamIntrinseque::IntervInc() const
{
   return mIncInterv;
}
cIncIntervale & cIncParamIntrinseque::IntervInc()
{
   return mIncInterv;
}

AllocateurDInconnues & cIncParamIntrinseque::Alloc()
{
   return mAlloc;
}

cIncEnsembleCamera * cIncParamIntrinseque::Ensemble()
{
    return mpEns;
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
