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

#ifdef MAC
// Modif Greg pour avoir le nom de la machine dans les log
#include <sys/utsname.h>
#endif


// Test Mercurial

namespace NS_ParamMICMAC
{

static bool WithWTieP=true ;
static Video_Win *  WTiePCor = 0;


/********************************************************************/
/*                                                                  */
/*                  Gestion des cellules                            */
/*                                                                  */
/********************************************************************/



class cCelTiep
{
   public :
      cCelTiep() :
         mCor      (Cor2I(-1)),
         mHeapInd  (-1)
      {
      }
/*
*/

     int & HeapInd() {return mHeapInd;}
     int Cor() const {return mCor;}
      
   private :

       static INT2 Cor2I(double aCor) 
       {
            return round_ni(ElMax(-1.0,ElMin(1.0,aCor))*1e4);
       }
       INT2  mZ;
       INT2  mCor;
       INT2  mX;
       INT2  mY;
       int   mHeapInd;
};


typedef cCelTiep * cCelTiepPtr;


class cHeap_CTP_Index
{
    public :
     static void SetIndex(cCelTiepPtr & aCTP,int i) 
     {
        aCTP->HeapInd() = i;
     }
     static int  Index(const cCelTiepPtr & aCTP)
     {
             return aCTP->HeapInd();
     }

};

class cHeap_CTP_Cmp
{
    public :
         bool operator () (const cCelTiepPtr & aCTP1,const cCelTiepPtr & aCTP2)
         {
               return aCTP1->Cor() > aCTP2->Cor();
         }
};

static std::vector< std::vector<cCelTiep> > mMatrCTP;
static Pt2di mP0Tiep;
static Pt2di mSzTiep;

/********************************************************************/
/********************************************************************/
/********************************************************************/

void  cAppliMICMAC::DoMasqueAutoByTieP(const Box2di& aBox,const cMasqueAutoByTieP & aMATP)
{
#ifdef ELISE_X11
   if (WithWTieP)
   {
       WTiePCor= Video_Win::PtrWStd(aBox.sz());
   }
#endif 
   mTP3d = StdNuage3DFromFile(WorkDir()+aMATP.FilePt3D());

   mP0Tiep = aBox._p0;
   mSzTiep = aBox.sz();

   std::cout << "== cAppliMICMAC::DoMasqueAutoByTieP " << aBox._p0 << " " << aBox._p1 << " Nb=" << mTP3d->size() << "\n"; 
   std::cout << " =NB Im " << mVLI.size() << "\n";

   cXML_ParamNuage3DMaille aXmlN =  mCurEtape->DoRemplitXML_MTD_Nuage();


   cElNuage3DMaille *  aNuage = cElNuage3DMaille::FromParam(aXmlN,FullDirMEC());


   mMatrCTP = std::vector< std::vector<cCelTiep> > (mSzTiep.y);
   for (int anY = 0 ; anY<mSzTiep.y ; anY++)
      mMatrCTP[anY] =  std::vector<cCelTiep>(mSzTiep.x);



   for (int aK=0 ; aK<int(mTP3d->size()) ; aK++)
   {
       Pt3dr aPE = (*mTP3d)[aK];
       Pt3dr aPL = aNuage->Euclid2ProfAndIndex(aPE);
       Pt3dr aPL2 = aNuage->Euclid2ProfPixelAndIndex(aPE);

       int aXIm = round_ni(aPL2.x);
       int aYIm = round_ni(aPL2.y);
       int aZIm = round_ni(aPL2.z);


       for (int aKIm=0 ; aKIm<int(mVLI.size()) ; aK++)
       {
           mVLI[aKIm]->MakeDeriv(aXIm,aYIm,aZIm);
       }

       std::cout << aPE << aPL << aPL2 << "\n";
       getchar();

    
   }


   //cElNuage3DMaille * aNuage = ;
   getchar();
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
