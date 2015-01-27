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
#include "XML_GEN/all.h"


/*
     Algorithmie de Conu
*/

/**********************************************************/
/*                                                        */
/*                cNuage                                  */
/*                                                        */
/**********************************************************/

class cNuage
{
    public :
      cNuage(const std::string &);
      void TestShowMasq();
      void TestShowZ();

      void VisuHom(const cNuage & aN2);
    private :
      cElNuage3DMaille *  mN1Resol1;
      cElNuage3DMaille *  mN1;
      Pt2di               mSz1;
      Video_Win *         mW;
};

cNuage::cNuage(const  std::string & aName) :
    mN1Resol1      (cElNuage3DMaille::FromFileIm(aName)),
    mN1            (mN1Resol1->ReScaleAndClip(2.0)),
    mSz1           (mN1->Sz()),
    mW             (Video_Win::PtrWStd(mSz1))
{
}

void cNuage::VisuHom(const cNuage & aN2)
{
    Clik  aCl = mW->clik_in();

    Pt2di aPIm1 = aCl._pt;

    if (! mN1->IndexHasContenu(aPIm1))
    {
       std::cout << "En Dehors \n";
      return;
    }

    mW->draw_circle_loc(aPIm1,2.0,mW->pdisc()(P8COL::red));

    Pt3dr aPTer = mN1->PtOfIndex(aPIm1);
    std::cout << aPIm1 <<  mN1->Terrain2Index(aPTer) << "\n";
    Pt2dr aP2 = aN2.mN1->Terrain2Index(aPTer);
    aN2.mW->draw_circle_loc(aP2,2.0,aN2.mW->pdisc()(P8COL::green));

}


void cNuage::TestShowMasq()
{
//  AFFICHAFE DU MASQUE 
    Im2D_U_INT1 aIm1(mSz1.x,mSz1.y,0);
    std::cout << "TAILLE " << mN1->Sz() << "\n";

    // On cree une image de masque :
    //    0 -> en dehors
    //    1 -> dedans
    for
    (
       cElNuage3DMaille::tIndex2D it=mN1->Begin();
       it!=mN1->End();
       mN1->IncrIndex(it)
    )
    {
       aIm1.data()[it.y][it.x] = 1; 
    }

   // Visualise le contenu de l'image ds la fenetre
    ELISE_COPY(mW->all_pts(),aIm1.in(),mW->odisc());
    getchar();
}

void cNuage::TestShowZ()
{
    ElSeg3D aSeg  = mN1->FaisceauFromIndex(mSz1/2.0);

    Im2D_U_INT1 aImMasq(mSz1.x,mSz1.y,0);
    Im2D_REAL4 aImZ(mSz1.x,mSz1.y,0.0);
    ELISE_COPY(mW->all_pts(),P8COL::yellow,mW->odisc());
    double aZMin = 1e30;
    double aZMax = -1e30;
    // Vi
    for
    (
       cElNuage3DMaille::tIndex2D it=mN1->Begin();
       it!=mN1->End();
       mN1->IncrIndex(it)
    )
    {
       Pt3dr aPTer = mN1->PtOfIndex(it);
       double aZ = aSeg.AbscOfProj(aPTer);
       mW->draw_circle_loc(it,1.0, mW->pcirc()(aZ*1000));
       aZMin = ElMin(aZ,aZMin);
       aZMax = ElMax(aZ,aZMax);
       aImZ.data()[it.y][it.x] = aZ;
       aImMasq.data()[it.y][it.x] = 1;
    }

   // Visualise le contenu de l'image ds la fenetre
    // ELISE_COPY(mW->all_pts(),aIm1.in(),mW->odisc());


      // mW->draw_circle_loc(it,1.0, mW->pdisc()(P8COL::red));

   ELISE_COPY
   (
       select(aImZ.all_pts(),aImMasq.in()),
       (aImZ.in()-aZMin) * (255.0 /(aZMax-aZMin)),
       mW->ogray()
   );
}

/**********************************************************/
/*                                                        */
/*             cAppliTestELA                              */
/*                                                        */
/**********************************************************/

class cAppliTestELA
{
    public :
        cAppliTestELA(int argc,char ** argv);

    private :

      std::string mDir;
      std::string mNameF1;
      std::string mNameF2;
      cNuage      mN1;
      cNuage      mN2;


};

cAppliTestELA::cAppliTestELA(int argc,char ** argv) :
    mDir  (argv[1]),
    mNameF1  (argv[2]),   
    mNameF2  (argv[3]),   
    mN1      (mDir+mNameF1),
    mN2      (mDir+mNameF2)
{
   mN1.TestShowZ();
   mN2.TestShowZ();

   while (1)
     mN1.VisuHom(mN2);
   getchar();
}


/****************************************************/
/*                                                  */
/*            main                                  */
/*                                                  */
/****************************************************/

int main(int argc,char ** argv)
{
    for (int aK =0 ; aK< argc ; aK++)
       std::cout << "ARGV[" << aK << "]=" << argv[aK] << "\n";
    ELISE_ASSERT(argc>=4,"Pas assez d'arguments");
    cAppliTestELA aAP(argc,argv);
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
