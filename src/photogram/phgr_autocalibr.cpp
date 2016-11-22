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


/*************************************************/
/*                                               */
/*        cElFaisceauDr2D::cFaisceau             */
/*                                               */
/*************************************************/

cElFaisceauDr2D::cFaisceau::cFaisceau
(
    Pt2dr aP0,Pt2dr aDir,REAL aPds
)  :
   SegComp  (aP0,aP0+aDir),
   mPds     (aPds)
{
}

REAL cElFaisceauDr2D::cFaisceau::Pds() const
{
   return mPds;
}

/*************************************************/
/*                                               */
/*        cElFaisceauDr2D                        */
/*                                               */
/*************************************************/

void  cElFaisceauDr2D::AddFaisceau(Pt2dr aP0,Pt2dr aDir,REAL aPds)
{
	mPckDr.push_back(cFaisceau(aP0,aDir,aPds));
}

void cElFaisceauDr2D::PtsConvergence
     (REAL  & teta0,REAL & phi0,bool OptimPhi)
{
    INT aDim = OptimPhi ? 2  : 1;

    Fonc_Num fTeta =  kth_coord(IndTeta);
    Fonc_Num fPhi  =   OptimPhi ? kth_coord(IndPhi) : phi0;
    Pt3d<Fonc_Num> PPF =  Pt3d<Fonc_Num>::TyFromSpherique(1,fTeta,fPhi);

    SystLinSurResolu  aSys(aDim,(int) mPckDr.size());


    PtsKD aPts(aDim);
    aPts(0) = teta0;
    if (OptimPhi)
        aPts(1) = phi0;

    for (tIter anIt=Begin() ; anIt!=End() ; anIt++)
    {
        aSys.PushDifferentialEquation
        (
	     anIt->ordonnee(PPF) * anIt->Pds(),
	     aPts
	);
    }

   Im1D_REAL8  aSol = aSys.L1Solve();
   teta0 = aSol.data()[IndTeta];
   if (OptimPhi)
      phi0 = aSol.data()[IndPhi];
}

REAL  cElFaisceauDr2D::ResiduConvergence(REAL  teta,REAL phi)
{
    Pt3dr PP =  Pt3dr::TyFromSpherique(1,teta,phi);

    REAL aRes = 0;
    REAL aSomP = 0;
    for (tIter anIt=Begin() ; anIt!=End() ; anIt++)
    {
       REAL aP = anIt->Pds();
       aRes += ElAbs(anIt->ordonnee(PP)*aP);
       aSomP += aP;
    }
    return aRes/ aSomP;
}


void  cElFaisceauDr2D::PtsConvergenceItere 
     (
         REAL  & teta0,REAL & phi0,INT NbStep,
	 REAL Epsilon,bool OptimPhi,REAL DeltaRes
     )
{
     REAL LastResidu = 0;
     for(;(NbStep>0) ; NbStep--)
     {
	 REAL aResidu = ResiduConvergence(teta0,phi0);

	 if ((aResidu< Epsilon) || ElAbs(aResidu-LastResidu)<DeltaRes)
            return;
	 PtsConvergence(teta0,phi0,OptimPhi);
     }
}

REAL  cElFaisceauDr2D::TetaDirectionInf()
{
    Pt2dr aDir(0,0);
    for (tIter anIt=Begin() ; anIt!=End() ; anIt++)
    {
       aDir += anIt->tangente().Square() * anIt->Pds();
    }

    REAL teta = angle(aDir);

    return teta/2.0;
}

extern  Pt2d<Fonc_Num> operator * (Pt2d<Fonc_Num> ,Fonc_Num);

void cElFaisceauDr2D::CalibrDistRadiale
     (
         Pt2dr   &            aC0,
	 bool                 CentreMobile,
	 REAL    &            TetaEpip,
	 REAL    &            PhiEpip,
	 std::vector<REAL> &  Coeffs
     )
{
    std::string NameGroupe = "GrpElFaisceauDr2D"; // Groupe Bidon pour classe desuete
    AllocateurDInconnues anAlloc;

    Pt2d<Fonc_Num> fCentre = CentreMobile ? 
                             anAlloc.NewPt2(NameGroupe,&aC0.x,&aC0.y) :  
                             Pt2d<Fonc_Num>(aC0.x,aC0.y);

    Fonc_Num fTeta = anAlloc.NewF(NameGroupe,"Teta",&TetaEpip);
    Fonc_Num fPhi = anAlloc.NewF(NameGroupe,"Phi",&PhiEpip);

    Pt3d<Fonc_Num> fPPEpi =  Pt3d<Fonc_Num>::TyFromSpherique(1,fTeta,fPhi);
    Pt2d<Fonc_Num> fEpi(fPPEpi.x,fPPEpi.y);

    std::vector<Fonc_Num> fCoeff;
    for (INT aK=0 ; aK <INT(Coeffs.size()) ; aK++)
    {
        fCoeff.push_back(anAlloc.NewF(NameGroupe,"Coef:"+ToString(aK),&Coeffs[aK]));
    }

    PtsKD aPInit = anAlloc.PInits();
    SystLinSurResolu aSys(aPInit.Dim(),NbDr());

    for (tIter anIt=Begin() ; anIt!=End() ; anIt++)
    {
        cVarSpec fCx (anIt->p0().x);
        cVarSpec fCy (anIt->p0().y);

        Pt2d<Fonc_Num>  fPt (fCx,fCy);

        Pt2d<Fonc_Num>  fEcart = fPt-fCentre;
        Fonc_Num        fRho2  = square_euclid(fEcart);
        Pt2d<Fonc_Num> fImDist = fPt;

        for (INT aK = 0 ; aK< INT(Coeffs.size()); aK++)
        {
             fImDist = fImDist + fEcart * PowI(fRho2,aK+1) * fCoeff[aK];
        }

        ElMatrix<Fonc_Num>   fDiff(2,true); // initialise a Id

        Pt2d<Fonc_Num> fDistDx(fImDist.x.deriv(fCx),fImDist.y.deriv(fCx));
        Pt2d<Fonc_Num> fDistDy(fImDist.x.deriv(fCy),fImDist.y.deriv(fCy));

        SetCol(fDiff,0,fDistDx);
        SetCol(fDiff,1,fDistDy);


        Pt2d<Fonc_Num> fTgtInit (anIt->tangente().x,anIt->tangente().y);

        Pt2d<Fonc_Num> fImTgt = fDiff * fTgtInit;
        Fonc_Num EqDroite = (fImTgt^fEpi) -(fImTgt^fImDist) *fPPEpi.z;

        aSys.PushDifferentialEquation(EqDroite,aPInit,anIt->Pds());
    }

    Im1D_REAL8  aSol = aSys.L1Solve();
    anAlloc.SetVars(aSol.data());
}




/*

void cElFaisceauDr2D::CalibrDistRadiale ()
{
     REAL aTeta0 = TetaDirectionInf();
     REAL aPhi0  = 0;
     PtsConvergenceItere
     (
          aTeta0,aPhi0,10,
	  1e-10,true,-1
     );

     
}

*/





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
