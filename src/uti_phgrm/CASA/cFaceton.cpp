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

#include "Casa.h"



/*************************************************/
/*                                               */
/*               cFaceton                        */
/*                                               */
/*************************************************/

cFaceton::cFaceton(double aPds,Pt2dr anIndex,Pt3dr aCdg,Pt3dr aNorm) :
   mPds     (aPds),
   mIndex   (anIndex),
   mCentre  (aCdg),
   mNormale (aNorm),
   mOk      (true)
{
}

cFaceton::cFaceton() :
  mOk (false)
{
}
 
bool cFaceton::Ok() const
{
   return mOk;
}

const Pt2dr & cFaceton::Index() const
{
   return mIndex;
}

const Pt3dr & cFaceton::Centre() const
{
   return mCentre;
}

const Pt3dr & cFaceton::Normale() const
{
     return mNormale;
}
 

ElSeg3D cFaceton::DroiteNormale() const
{
   return ElSeg3D (mCentre,mCentre+mNormale);
}

bool cFaceton::IsFaceExterne(const cInterfSurfaceAnalytique & anISA) const
{
   double aEps = 1e-6;
   Pt3dr aP0 = anISA.E2UVL(mCentre); 
   Pt3dr aP1 = anISA.E2UVL(mCentre + mNormale * aEps); 
   // On est une face externe si le rayon rentre dans la surface, donc z decroit
   return aP1.z < aP0.z;
}


int cFaceton::GetIndMoyen(const std::vector<cFaceton> & aVF)
{
    double aDMin = 1e30;
    int aKMin = -1;

    for (int aK1=0 ; aK1 <int(aVF.size()) ; aK1++)
    {
        double aSomD = 0.0;
        Pt3dr aC1 = aVF[aK1].mCentre;;
        for (int aK2=0 ; aK2 <int(aVF.size()) ; aK2++)
        {
            Pt3dr aV12 = aVF[aK2].mCentre-aC1;
            aSomD += aVF[aK2].mPds * square_euclid(aV12);
        }
        if (aSomD<aDMin)
        {
           aDMin = aSomD;
           aKMin = aK1;
        }
    }    
    ELISE_ASSERT(aKMin>=0,"cFaceton::GetMoyen");
    return aKMin;
}

/*************************************************/
/*                                               */
/*               cAccumFaceton                   */
/*                                               */
/*************************************************/


cAccumFaceton::cAccumFaceton () :
    mSomPds  (0),
    mSomPt   (0,0,0),
    mSomInd   (0,0),
    mMoment   (3,3,0.0)
{
}


void cAccumFaceton::Add(const Pt2dr & anIndex,const Pt3dr  & aPt,double aPds)
{
   mSomPds += aPds;
   mSomPt  = mSomPt + aPt * aPds;
   mSomInd = mSomInd + anIndex * aPds;
   double aC[3];
   aPt.to_tab(aC);

   for (int aX=0 ; aX<3 ; aX++)
      for (int aY=0 ; aY<3 ; aY++)
      {
          mMoment(aX,aY) += aC[aX] * aC[aY] * aPds;
  // std::cout  << aPt <<  mMoment(aX,aY) << " " << aC[aX] << " " << aC[aY] << "\n";
      }
}


cFaceton   cAccumFaceton::CompileF(const cElNuage3DMaille & aNuage)
{
   Pt3dr aCdg = mSomPt / mSomPds;
   ElMatrix<double> aM2 = mMoment;
   double aC[3];
   aCdg.to_tab(aC);

   for (int aX=0 ; aX<3 ; aX++)
      for (int aY=0 ; aY<3 ; aY++)
      {
          aM2(aX,aY) = aM2(aX,aY) / mSomPds -  aC[aX] * aC[aY];
      }

  ElMatrix<double> aValP(3,3),aVecp(1,3);
  std::vector<int>  aVInd = jacobi_diag(aM2,aValP,aVecp);

  Pt3dr aU;

  int i0 = aVInd[0];
  int i1 = aVInd[1];
  int i2 = aVInd[2];

  // ELISE_ASSERT(0<=aValP(i0,i0), "Erreur in jacobi");
  // ELISE_ASSERT(aValP(i0,i0) <= aValP(i1,i1), "Erreur in jacobi");
  // ELISE_ASSERT(aValP(i1,i1) <= aValP(i2,i2), "Erreur in jacobi");

  if (   (aValP(i0,i0) <0) || (aValP(i0,i0) > aValP(i1,i1)) || (aValP(i1,i1) > aValP(i2,i2)))
  {
     cElWarning::JacobiInCasa.AddWarn("",__LINE__,__FILE__);
     return cFaceton();
  }



  aVecp.GetCol(i0,aU);

  cBasicGeomCap3D * aCam = aNuage.Cam();

  Pt2dr aPIm = aCam->Ter2Capteur(aCdg);
  // Pt3dr aDir = aCam->F2toDirRayonR3(aPIm);
  Pt3dr aDir = aCam->Capteur2RayTer(aPIm).Tgt(); // RPCNuage
  // std::cout << scal(aDir,aU) << "\n";
  if (scal(aDir,aU) < 0) 
     aU = -aU;

  return cFaceton(mSomPds,mSomInd/mSomPds,aCdg,aU);

  // std::cout << aU << "\n";

  // return  cFaceton(Pt2dr(0,0),Pt3dr
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
