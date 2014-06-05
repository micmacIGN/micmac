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

#include "general/all.h"
#include "private/all.h"
#include "Digeo.h"

namespace NS_ParamDigeo
{

/****************************************/
/*                                      */
/*             cAppliDigeo              */
/*                                      */
/****************************************/

 
cAppliDigeo::cAppliDigeo
( 
   cResultSubstAndStdGetFile<cParamDigeo> aParam,
   cAppliDigeo *                          aMaterAppli,
   cModifGCC *                            aModif,
   bool                                   IsLastGCC

) :
    cParamDigeo  (*(aParam.mObj)),
    mDC          (aParam.mDC),
    mICNM        (aParam.mICNM),
    mMaster      (aMaterAppli),
    mModifGCC    (aModif),
    mLastGCC     (IsLastGCC),
    mFileGGC_H   (aMaterAppli ? aMaterAppli->mFileGGC_H : 0),
    mFileGGC_Cpp (aMaterAppli ? aMaterAppli->mFileGGC_Cpp : 0)
{
   InitConvolSpec();
}

void cAppliDigeo::AllocImages()
{
   for 
   (
       std::list<cImageDigeo>::const_iterator itI=ImageDigeo().begin();
       itI!=ImageDigeo().end();
       itI++
   )
   {
       std::list<std::string> aLS = mICNM->StdGetListOfFile(itI->KeyOrPat());
       for 
       (
            std::list<std::string>::const_iterator itS=aLS.begin();
            itS!=aLS.end();
            itS++
       )
       {
           mVIms.push_back(new cImDigeo(mVIms.size(),*itI,*itS,*this));
       }
   }
}

bool cAppliDigeo::MultiBloc() const
{
  return DigeoDecoupageCarac().IsInit();
}



void cAppliDigeo::DoCarac()
{
  if (! ComputeCarac())
      return;
  if (GenereCodeConvol().IsInit() && (mFileGGC_H ==0))
  {
      cGenereCodeConvol & aGCC = GenereCodeConvol().Val();
      std::string aName = aGCC.File().Val();
      std::string aDir = aGCC.Dir().Val();
      mFileGGC_H = FopenNN(aDir+aName+".h","w","cAppliDigeo::DoCarac");
      mFileGGC_Cpp = FopenNN(aDir+aName+".cpp","w","cAppliDigeo::DoCarac");

      fprintf(mFileGGC_H,"#include \"general/all.h\"\n");
      fprintf(mFileGGC_H,"#include \"private/all.h\"\n");
      fprintf(mFileGGC_H,"#include \"Digeo.h\"\n\n");
      fprintf(mFileGGC_H,"namespace NS_ParamDigeo {\n");


      fprintf(mFileGGC_Cpp,"#include \"%s.h\"\n\n",aName.c_str());
      fprintf(mFileGGC_Cpp,"namespace NS_ParamDigeo {\n");
      fprintf(mFileGGC_Cpp,"void cAppliDigeo::InitConvolSpec()\n");
      fprintf(mFileGGC_Cpp,"{\n");
      fprintf(mFileGGC_Cpp,"    static bool theFirst = true;\n");
      fprintf(mFileGGC_Cpp,"    if (! theFirst) return;\n");
      fprintf(mFileGGC_Cpp,"    theFirst = false;\n");
      fprintf(mFileGGC_Cpp,"\n");

  }

  Box2di aBox = mVIms[0]->BoxIm();
  Pt2di aSzGlob = aBox.sz();
  int    aBrd=0;
  int    aSzMax = aSzGlob.x + aSzGlob.y;
  if (DigeoDecoupageCarac().IsInit())
  {
     aBrd = DigeoDecoupageCarac().Val().Bord();
     aSzMax = DigeoDecoupageCarac().Val().SzDalle();
  }

  cDecoupageInterv2D aDec (aBox,Pt2di(aSzMax,aSzMax),Box2di(aBrd));

  // Les images s'itialisent en fonction de la Box
  for (int aKI=0 ; aKI<int(mVIms.size()) ; aKI++)
  {
      for (int aKB=0; aKB<aDec.NbInterv() ; aKB++)
      {
          mVIms[aKI]->NotifUseBox(aDec.KthIntervIn(aKB));
      }
      mVIms[aKI]->AllocImages();
  }


  for (int aKB=0; aKB<aDec.NbInterv() ; aKB++)
  {
      for (int aKI=0 ; aKI<int(mVIms.size()) ; aKI++)
      {
         
          mVIms[aKI]->LoadImageAndPyram(aDec.KthIntervIn(aKB));
          mVIms[aKI]->DoExtract();
      }
  }

  if (GenereCodeConvol().IsInit() &&  mLastGCC)
  {
      fprintf(mFileGGC_H,"}\n");
      ElFclose(mFileGGC_H);

      fprintf(mFileGGC_Cpp,"}\n");
      fprintf(mFileGGC_Cpp,"}\n");
      ElFclose(mFileGGC_Cpp);
  }
}



void cAppliDigeo::DoAll()
{
     AllocImages();
     ELISE_ASSERT(mVIms.size(),"NoImage selected !!");
     DoCarac();
}


     // cOctaveDigeo & GetOctOfDZ(int aDZ);


        //   ACCESSEURS BASIQUES


const std::string & cAppliDigeo::DC() const { return mDC; }

cInterfChantierNameManipulateur * cAppliDigeo::ICNM() {return mICNM;}


FILE *  cAppliDigeo::FileGGC_H()
{
   return mFileGGC_H;
}

FILE *  cAppliDigeo::FileGGC_Cpp()
{
   return mFileGGC_Cpp;
}


cModifGCC * cAppliDigeo::ModifGCC() const 
{ 
   return mModifGCC; 
}

};



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
