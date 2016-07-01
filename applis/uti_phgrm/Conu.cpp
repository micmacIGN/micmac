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


void Banniere_Conu()
{
   std::cout << "\n";
   std::cout <<  " *********************************\n";
   std::cout <<  " *     CO-herence de             *\n";
   std::cout <<  " *     NU-age de points          *\n";
   std::cout <<  " *********************************\n";

}


/****************************************************/
/*                                                  */
/*          HEADER                                  */
/*                                                  */
/****************************************************/


class cAppliConu
{
    public :
         cAppliConu(int argc,char ** argv);

	 void NoOp(){}
    private :

        Pt2di        mSzIm;
	Im2D_REAL4    mImP;
	Im2D_REAL4    mImX;
	Im2D_REAL4    mImY;
	Im2D_REAL4    mImZ;

        std::string         mDir;
        std::string         mName1;
        std::string         mName2;
        cElNuage3DMaille *  mN1;
        cElNuage3DMaille *  mN2;
};


/****************************************************/
/*                                                  */
/*          cSubWPin                                */
/*                                                  */
/****************************************************/


cAppliConu::cAppliConu(int argc,char ** argv):
   mImP  (1,1),
   mImX  (1,1),
   mImY  (1,1),
   mImZ  (1,1)
{
    std::string aTF;
    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAM(mDir) 
                      << EAM(mName1) 
                      << EAM(mName2) ,
           LArgMain() << EAM(aTF,"TF",true)
    );

std::cout << "AAAAAAAAAAAAa\n";
    mN1 = cElNuage3DMaille::FromFileIm(mDir+mName1);
std::cout << "BBBB\n";
    mN2 = cElNuage3DMaille::FromFileIm(mDir+mName2);
    if (aTF!="")
       mN2->Std_AddAttrFromFile(mDir+aTF);
std::cout << "CCCCC\n";
    Pt2dr aSz =mN2->Sz();
    Pt2dr aTr (10,50);
    cElNuage3DMaille *  mScTr2 = mN2->ReScaleAndClip(Box2dr(aTr,aSz-aTr),4.0);

   if (aTF!="")
   {
        const std::vector<cLayerNuage3DM> & aVL = mScTr2->Attrs();
        std::vector<Im2DGen> aVI;
        aVI.push_back(*(aVL[0].Im()));
        aVI.push_back(*(aVL[1].Im()));
        aVI.push_back(*(aVL[2].Im()));

        Tiff_Im::CreateFromIm(aVI,"tata.tif");
   }

   std::list<std::string> aLC;
   aLC.push_back("Create by MicMac");
   aLC.push_back("www.micmac.ign.fr");
   mScTr2->PlyPutFile("toto.ply",aLC,false);
 
    
std::cout << "EEEEE " << mSzIm << "\n";


    Video_Win aW = Video_Win::WStd(mSzIm,0.3);
    Im2D_U_INT1 aRes(mSzIm.x,mSzIm.y,0);
    TIm2D<U_INT1,INT> aTRes(aRes);

    for (cElNuage3DMaille::tIndex2D anI=mN1->Begin(); anI!=mN1->End() ; mN1->IncrIndex(anI))
    {
         bool OK;
         double aD = mN1->DiffDeSurface(OK,anI,*mN2);

         Pt3dr aQ = mN1->PtOfIndex(anI);

         if (OK)
         {
              int aC = ElMax(1,ElMin(254,round_ni(128 + aD *100)));
              aW.draw_circle_abs(anI,1.0,aW.pgray()(aC));
              aTRes.oset(anI,aC);

              double aD2 = mN1->DiffDeSurface(OK,anI,*mScTr2);
              if (OK)
                 std::cout << ElAbs(aD-aD2) << "\n";
              else
                 std::cout << " ???? " << "\n";
         }
         else
         {
              aW.draw_circle_abs(anI,1.0,aW.pdisc()(P8COL::red));
              aTRes.oset(anI,255);
         }

         Pt3dr aP(
                 mImX.data()[anI.y][anI.x],
                 mImY.data()[anI.y][anI.x],
                 mImZ.data()[anI.y][anI.x]
               );

         // std::cout << euclid(aP-aQ) << "\n";
    }
    getchar();
    Tiff_Im::CreateFromIm(aRes,"CoheNuage.tif");
std::cout << "FFFFF\n";
}



/****************************************************/
/*                                                  */
/*            main                                  */
/*                                                  */
/****************************************************/

int main(int argc,char ** argv)
{
    cAppliConu aAP(argc,argv);
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
