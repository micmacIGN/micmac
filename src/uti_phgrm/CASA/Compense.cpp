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


/***************************************************/
/*                                                 */
/*               cOneSurf_Casa                     */
/*                                                 */
/***************************************************/

cOneSurf_Casa::cOneSurf_Casa() :
   mISAF (0),
   mW    (0),
   mFMoy (0)
{
}

bool cOneSurf_Casa::IsFaceExterne(const cInterfSurfaceAnalytique & anISA,double aTol) const
{
   int aNbExt=0;
   int aNbInt=0;
   for (int aK=0 ; aK<int(mVF.size()) ; aK++)
   {
       bool Ext = mVF[aK].IsFaceExterne(anISA);
       // std::cout << "EXT = " << Ext << "\n";
       if (Ext) 
           aNbExt++;
       else
          aNbInt++;
   }
   double aRatio = aNbExt / double(aNbExt+aNbInt);
   std::cout << "RATIO Ext/Int : " << aRatio << "\n";
   aRatio *= 100.0;
   if (aRatio<50) aRatio = 100 -aRatio;
   if (aRatio<aTol)
   {
       std::cout << "COHERENCE, exigee : " << aTol << " , obtenue : " << aRatio << "\n";
       ELISE_ASSERT(false,"Orientation surface incoherente");
   }
   return aNbExt >  aNbInt;
}

void cOneSurf_Casa::ActiveContrainte(cSetEqFormelles & aSet)
{
   aSet.AddContrainte(mISAF->StdContraintes(),true);
}

void  cOneSurf_Casa::Compense(const cCasaEtapeCompensation & anEtape,bool First)
{
     std::vector<double> aVEcart;
     const cInterfSurfaceAnalytique & aSan= mISAF->CurSurf();
     double aSP=0.0;
     double aSEc2=0.0;

     for (int aK=0 ; aK<int(mVF.size()) ; aK++)
     {
        Pt3dr aP = mVF[aK].Centre();
        double aPds = 1.0;
        double anEc = aSan.E2UVL(aP).z;

        if (! First)
        {
           if (anEc > mREC.mMoyHaut)
              aPds = 0.0;
        }


        mISAF->AddObservRatt(aP,aPds);

        aSP +=  aPds;
        aSEc2 += aPds*ElSquare(anEc);
        aVEcart.push_back(ElAbs(anEc));
     }

     aSEc2 /= aSP;

     mREC.mMoyHaut =  KthValProp(aVEcart,0.99);
     mREC.mMoyQuad = sqrt(aSEc2);


     cout  << "  Compense :" << mName << " " << mREC.mMoyQuad  << " ECAR a 99 " <<  mREC.mMoyHaut << "\n";
}


/***************************************************/
/*                                                 */
/*               cOneSurf_Casa                     */
/*                                                 */
/***************************************************/


void cAppli_Casa::OneEtapeCompense(const cCasaEtapeCompensation & anEtape)
{
     for (int aKIter=0; aKIter<anEtape.NbIter().Val() ; aKIter++)
     {
         for (int aK=0 ; aK<int(mVSC.size()) ; aK++)
         {
             mVSC[aK]->ActiveContrainte(mSetEq);
         }
         mSetEq.SetPhaseEquation();
         for (int aK=0 ; aK<int(mVSC.size()) ; aK++)
         {
             mVSC[aK]->Compense(anEtape,(aKIter==0));
         }
         mSetEq.SolveResetUpdate();
     }


     if (anEtape.Export().IsInit())
     {
         cXmlModeleSurfaceComplexe aXmlModele;
         for (int aK=0 ; aK<int(mVSC.size()) ; aK++)
         {
             const cInterfSurfaceAnalytique * aSurf =  &(mVSC[aK]->mISAF->CurSurf());
             bool IsExt = mVSC[aK]->IsFaceExterne(*aSurf,mParam.PercCoherenceOrientation().Val());
             aSurf = const_cast<cInterfSurfaceAnalytique *>(aSurf)->DuplicateWithExter(IsExt);
             aSurf =  UsePts(aSurf);
             cXmlOneSurfaceAnalytique aXmlSurf;
             aXmlSurf.XmlDescriptionAnalytique() = aSurf->Xml();
             aXmlSurf.Id() = mVSC[aK]->mName;
             aXmlSurf.VueDeLExterieur() = IsExt;
             aXmlModele.XmlOneSurfaceAnalytique().push_back(aXmlSurf);
         }
         MakeFileXML
         (
             aXmlModele,
             mDC+anEtape.Export().Val()
         );
     }

}

void cAppli_Casa::Compense(const cCasaSectionCompensation & aSC)
{
   for 
   (
      std::list<cCasaEtapeCompensation>::const_iterator itEC = aSC.CasaEtapeCompensation().begin();
      itEC != aSC.CasaEtapeCompensation().end();
      itEC++
   )
   {
        OneEtapeCompense(*itEC);
   }
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
