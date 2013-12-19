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

using namespace NS_ParamChantierPhotogram;

/*********************************************/
/*                                           */
/*                ::                         */
/*                                           */
/*********************************************/


void Banniere_Marc()
{
   std::cout << "\n";
   std::cout <<  " *********************************\n";
   std::cout <<  " * (1) M-ethode                  *\n";
   std::cout <<  " *     A-utomatique de           *\n";
   std::cout <<  " *     R-egistration par         *\n";
   std::cout <<  " *     C-orrelation              *\n";
   std::cout <<  " *********************************\n";

   std::cout <<  " \n";
   std::cout <<  " ---    (1) Marc de Bourgogne, bien entendu, \n";
   std::cout <<  " ---        aucune personalisation !\n";
}

/*********************************************/
/*                                           */
/*            cAppliMarc                     */
/*                                           */
/*********************************************/

class cAppliMarc : public cAppliBatch
{
    public :
       cAppliMarc(int argc,char ** argv);

       std::string  CurCom() const;

       void Exec();
    private :
       std::string mMode;

       std::string mI2FromI1Key;
       int         mI2I1Dir;
       std::string mCple2HomAp;
       std::string mIm2Calib;

       std::string mFPMicM;
       double      mIncInit;
       int         mGrad;
       int         mMoy;
       int         mUsePolyn;
};


void  cAppliMarc::Exec()
{
    System(CurCom(),false);
}


std::string cAppliMarc::CurCom() const
{
   std::string aRes = 
                     std::string("bin/MICMAC ")
		   + mFPMicM +  std::string(" ")
		   + ComCommune()
		   + std::string(" @I2FromI1Key=") + mI2FromI1Key
		   + std::string(" @I2FromI1SensDirect=") + ToString(mI2I1Dir)
		   + std::string(" @FCND_CalcHomFromI1I2=") + mCple2HomAp
		   + std::string(" @FCND_GeomCalc=") + mIm2Calib
		   + std::string(" @Px1IncCalc=") + ToString(mIncInit)
		   + std::string(" @Px2IncCalc=") + ToString(mIncInit)
		   + std::string(" \%GradIm=") + ToString(mGrad)
		   + std::string(" \%Moyenneur=") + ToString(mMoy)
		   + std::string(" \%DegModZ4=") + ToString(mUsePolyn?3:-1)
		   + std::string(" \%ExportPolynD3=") + ToString(mUsePolyn?1:0) 
		   + std::string(" \%ExportPolynD5=") + ToString(mUsePolyn?1:0) ;


  return aRes;
    
}


cAppliMarc::cAppliMarc(int argc,char ** argv) :
   cAppliBatch
   (
        argc,argv,
        4,   // Trois argument obligatoires
	1,   // 1 Argument est un fichier
	"Micmac-FUSION"
   ),
   mIncInit (50.0)
{

   for (int aK=0 ; aK<ARGC(); aK++)
      std::cout << ARGV()[aK] << "\n";
   std::cout << "-------------------------------\n";


    AddPatSauv("SupModele*.xml");
    // AddPatSauv("XML_Homol*.xml"); 
    mCple2HomAp =  "Cple2HomAp";
    mI2FromI1Key = "";
    mI2I1Dir     = 1;
    mIm2Calib    = "Im2Calib";

    std::string aTypeApp ;

    ElInitArgMain
    (
           ARGC(),ARGV(),
           LArgMain() << EAM(mI2FromI1Key) 
	              << EAM(aTypeApp),
           LArgMain() << EAM(mCple2HomAp,"C2H",true)
	              << EAM(mI2I1Dir,"I2I1Dir",true)
	              << EAM(mIm2Calib,"I2C",true)
	              << EAM(mIncInit,"Inc",true)
    );


    mMoy = 0;

    if (aTypeApp=="Std")
    {
       mFPMicM  = "param-Superspos-D3.xml";
       mGrad = 0;
       mUsePolyn = 0;
    }
    else if (aTypeApp=="GrD3")
    {
       mFPMicM  = "param-Superspos-D3.xml";
       mGrad = 1;
       mUsePolyn = 1;
    }
    else if (aTypeApp=="GrHom")
    {
       mFPMicM  = "param-Superspos-D3.xml";
       mGrad = 1;
       mUsePolyn = 0;
    }
    else
    {
       ELISE_ASSERT(false,"Unknown Mode in appar");
    }
    mFPMicM = "applis/XML-Pattron/" + mFPMicM;

}


   //===========================================

int main(int argc,char ** argv)
{
    cAppliMarc aAP(argc,argv);

    aAP.DoAll();

    // std::cout << aAP.CurCom() << "\n";


    Banniere_Marc();

    return 0;
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
