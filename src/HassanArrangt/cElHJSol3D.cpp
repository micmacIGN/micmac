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


/******************************************/
/*                                        */
/*       cTrapuFace                       */
/*                                        */
/******************************************/

cTrapuFace::cTrapuFace
(
    INT aNum
)  :
   mNum  (aNum)
{
}

void cTrapuFace::AddSom(INT aNum)
{
   mSoms.push_back(aNum);
}

/******************************************/
/*                                        */
/*       cTrapuBat                        */
/*                                        */
/******************************************/


cTrapuBat::cTrapuBat() :
   mFirstPt(true)
{
}

INT  cTrapuBat::GetNumSom(Pt3dr aP,REAL aEpsilon)
{
   REAL aDMin = aEpsilon*2+1;
   INT  aKMin = -1;

   for (INT aK=0 ; aK<INT(mSoms.size()) ; aK++)
   {
        REAL aDist = euclid(mSoms[aK]-aP);
        if (aDist<aDMin)
        {
             aDMin = aDist;
             aKMin = aK;
        }
   }

   if (aDMin<aEpsilon)
      return aKMin;

   mSoms.push_back(aP);
   return (int) mSoms.size()-1;
}

INT cTrapuBat::NbFaces() const
{
   return (INT) mFaces.size();
}


std::vector<Pt3dr>  & cTrapuBat::Soms()
{
   return mSoms;
}

Pt3dr & cTrapuBat::P0() {return mP0;}
Pt3dr & cTrapuBat::P1() {return mP1;}

std::vector<Pt3dr> cTrapuBat::PtKiemeFace(INT aK) const
{
  const std::vector<INT> & aVI = mFaces[aK].mSoms;
  std::vector<Pt3dr> aRes;

  for (INT aK=0; aK<INT(aVI.size()); aK++)
      aRes.push_back(mSoms[aVI[aK]]);
  return aRes;
}

void  cTrapuBat::AddFace
      (
           const std::vector<Pt3dr> & aVP,
           REAL  aEps,
           INT   aNum
      )          
{
   mFaces.push_back(cTrapuFace(aNum));

    INT aNb = (int) aVP.size();
    for (INT aK=0 ; aK<aNb ; aK++)
    {
        Pt3dr aP0 = aVP[(aK+aNb-1)%aNb];
        Pt3dr aP1 = aVP[aK];
        Pt3dr aP2 = aVP[(aK+1)%aNb];

	ElSeg3D aSeg(aP0,aP2);
	if  (
		 (aSeg.AbscOfProj(aP1) < aSeg.AbscOfProj(aP0) - aEps)
              || (aSeg.AbscOfProj(aP1) > aSeg.AbscOfProj(aP2) + aEps)
	      || (aSeg.DistDoite(aP1) > aEps)
	    )
        {
            mFaces.back().AddSom(GetNumSom(aP1,aEps));
        }

        if (mFirstPt)
        {
           mFirstPt = false;
           mP0 = aP1;
           mP1 = aP1;
        }
        else
        {
           mP0 = Inf(mP0,aP1);
           mP1 = Sup(mP1,aP1);
        }
    }
}

Pt3dr  cTrapuBat::P0() const {return mP0;}
Pt3dr  cTrapuBat::P1() const {return mP1;}


void cTrapuBat::PutXML(class cElXMLFileIn & aFile)
{
    cElXMLFileIn::cTag  aTagGlob(aFile,"BatimentTrapu");
    
    {
        cElXMLFileIn::cTag  aTagPoints(aFile,"TableauPoints3D");
        for (INT aK=0 ; aK<INT(mSoms.size()) ; aK++)
            aFile.PutPt3dr(mSoms[aK]);
        aTagPoints.NoOp();
    }
    {
          cElXMLFileIn::cTag  aTagPoints(aFile,"EnsembleDeFaces");
          for (INT aK=0 ; aK<INT(mFaces.size()) ; aK++)
              aFile.PutTabInt(mFaces[aK].mSoms,"UneFace");
    }

    aTagGlob.NoOp();

}

/******************************************/
/*                                        */
/*       cElImagesOfTrapu                 */
/*                                        */
/******************************************/


Im2D_INT1 cElImagesOfTrapu::ImLabel()
{
   return mImLabel;
}
Im2D_REAL4 cElImagesOfTrapu::ImZ()
{
   return mImZ;
}
Im2D_U_INT1 cElImagesOfTrapu::ImShade()
{
   return mImShade;
}

Pt2di cElImagesOfTrapu::Dec() const {return mDec;}



cElImagesOfTrapu::cElImagesOfTrapu
(
     const cTrapuBat & aBat,INT aRab,bool Sup,Pt2di * aDec
)  :
     mImLabel (1,1),
     mImZ     (1,1),
     mImShade (1,1)
{
   Pt3dr aP0 =  aBat.P0();
   Pt3dr aP1 =  aBat.P1();

   Pt2di aSz(round_ni(aP1.x-aP0.x+2*aRab),round_ni(aP1.y-aP0.y+2*aRab));
   mDec = Pt2di(round_ni(aP0.x-aRab),round_ni(aP0.y-aRab));
   if (aDec)
      mDec = *aDec;

   mImLabel.Resize(aSz);
   mImZ.Resize(aSz);
   ELISE_COPY(mImLabel.all_pts(),-1,mImLabel.out());
   ELISE_COPY(mImZ.all_pts(),0.0,mImZ.out());
   INT1 ** aDLab = mImLabel.data();
   REAL4 ** aDZ = mImZ.data();

   for (INT aKF=0 ; aKF<aBat.NbFaces() ; aKF++)
   {
	   // cout << "aKF " << aKF << "\n";
        std::vector<Pt3dr> aVP3 = aBat.PtKiemeFace(aKF);

        std::vector<Pt2dr> aVP2;
	Pt2di aP0( 100000, 100000);
	Pt2di aP1(-100000,-100000);
        for (INT aKP =0; aKP<INT(aVP3.size()); aKP++)
        {
             Pt3dr aP3 = aVP3[aKP];
	     Pt2dr aP2(aP3.x,aP3.y);
	     aVP2.push_back(aP2);
	     aP0.SetInf(Pt2di(aP2));
	     aP1.SetSup(Pt2di(aP2));
        }

	aP0 -= Pt2di(3,3);
	aP1 += Pt2di(3,3);


	cElPlan3D aPl(aVP3,0);

	for (INT aX=aP0.x; aX<=aP1.x; aX++)
        {
	    for (INT aY=aP0.y; aY<=aP1.y; aY++)
	    {
		Pt2di aP(aX,aY);
                if (PointInPoly(aVP2,Pt2dr(aP)))
		{
                    REAL4 aNewZ = (REAL4) aPl.ZOfXY(Pt2dr(aP));
                    REAL4& aZCur = aDZ[aY-mDec.y][aX-mDec.x];
                    if (
                             (aDLab[aY-mDec.y][aX-mDec.x]==-1)
			  || (Sup && (aNewZ>aZCur))
			  || ((!Sup) && (aNewZ<aZCur))
		       )
                    {
                        aZCur = aNewZ;
                        aDLab[aY-mDec.y][aX-mDec.x] = aKF;
                    }
		}
	    }
        }
   }

   mImShade = Shading(aSz,mImZ.in()/1.0,8,0.7);
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
