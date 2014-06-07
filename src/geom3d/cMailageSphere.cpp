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



/***********************************************/
/*                                             */
/*             cMailageSphere                  */
/*                                             */
/***********************************************/

cMailageSphere::cMailageSphere(Pt2dr aStep,Pt2dr aMin,Pt2dr aMax,bool Inv) :
   mStep (aStep),
   mMin  (aMin),
   mMax  (aMax),
   mInv  (Inv)
{
}

void cMailageSphere::WriteFile(const std::string & aNameFile)
{
    FILE * fp = ElFopen(aNameFile.c_str(),"w");
    ELISE_ASSERT(fp!=0,"Cannot open in cMailageSphere::WriteFile");
    fprintf(fp,"StepTetaPhi=[%e,%e]\n",mStep.x,mStep.y);
    fprintf(fp,"TetaPhiMin=[%lf,%lf]\n",mMin.x,mMin.y);
    fprintf(fp,"TetaPhiMax=[%lf,%lf]\n",mMax.x,mMax.y);
    fprintf(fp,"Inv=%d\n",mInv);
    ElFclose(fp);
}

cMailageSphere cMailageSphere::FromFile(const std::string & aNameFile)
{
      cMailageSphere  aMail(Pt2dr(0,0),Pt2dr(0,0),Pt2dr(0,0),0);
      StdInitArgsFromFile
      (
            aNameFile,
            LArgMain()
                 <<  EAM(aMail.mStep,"StepTetaPhi")
                 <<  EAM(aMail.mMin,"TetaPhiMin")
                 <<  EAM(aMail.mMax,"TetaPhiMax")
                 <<  EAM(aMail.mInv,"Inv")
      );

      cout << aMail.mStep << aMail.mMin << aMail.mMax << aMail.mInv << "\n";

      return aMail;
}


void cMailageSphere::SetStep(Pt2dr aStep) {mStep = aStep;}
void cMailageSphere::SetMax(Pt2dr aMax)   {mMax = aMax;}
void cMailageSphere::SetMin(Pt2dr aMin)   {mMin = aMin;}

Pt2dr cMailageSphere::Pix2Spherik(Pt2dr aIndTP)   
{
     return mInv  ?
	       mMax-aIndTP.mcbyc(mStep)   :
	       mMin+aIndTP.mcbyc(mStep)   ;
}
				 
Pt2dr cMailageSphere::Spherik2PixR(Pt2dr aTetaPhi)
{
      return mInv  ?
	     (mMax-aTetaPhi).dcbyc(mStep)   :
	     (aTetaPhi-mMin).dcbyc(mStep)   ;
}
Pt2di cMailageSphere::Spherik2PixI(Pt2dr aTetaPhi)
{
	return round_ni(Spherik2PixR(aTetaPhi));
}


Pt2di cMailageSphere::SZEnglob()
{
    return 
	Pt2di(1,1)
     +  Sup(Spherik2PixI(mMax),Spherik2PixI(mMin));
}
// OK

Pt3dr cMailageSphere::DirMoy()
{
    Pt3dr aP1 = Pt3dr::TyFromSpherique(1.0,mMax.x,mMax.y);
    Pt3dr aP2 = Pt3dr::TyFromSpherique(1.0,mMin.x,mMin.y);

    return vunit(aP1+aP2);
}
#if (0)

Pt2dr cMailageSphere::DirMoyH()
{
    Pt2dr aP1 = Pt2dr::FromPolar(1.0,mMax.x);
    Pt2dr aP2 = Pt2dr::FromPolar(1.0,mMin.x);

    return vunit(aP1+aP2);
}
#endif





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
