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

/************************************************/
/*                                              */
/*             cWPointeData                     */
/*                                              */
/************************************************/






     //  =========   cParamPointeInitEtalonnage ===================


class cParamPointeInitEtalonnage : public cParamPointeInit
{
      public :
         cParamPointeInitEtalonnage(const std::string &,cParamEtal);
      private :
         cSetPointes1Im    SetPointe(const std::string &) ;
     std::string       NamePointeInit();

     std::string       NamePointeInterm()
     {
             return mEtalon.NamePointeResult(mNameIm,true,false);
     }
     std::string       NamePointeFinal()
     {
             return mEtalon.NamePointeResult(mNameIm,false,false);
     }

     bool SauvInterm()
     {
         return mEtalon.Param().ParamRechInit().mUseCI == eUCI_Only;
     }
     bool SauvFinal()
     {
         return mEtalon.Param().ParamRechDRad().mUseCI == eUCI_Only;
     }




     std::string       NameImageCamera();
     std::string       NameImagePolygone();
     std::string       NamePointePolygone();
     std::string       NamePolygone() const {return mParamEtal.NameCible3DPolygone();}
         const cPolygoneEtal &   Polygone() const ;
         eTyEtat EtatDepAff() const {return eNonSelec;}
         bool PseudoPolyg() const { return false;}
     void SauvRot(ElRotation3D) const {}
     void    ConvPointeImagePolygone(Pt2dr&);

     const cPolygoneEtal::tContCible & CiblesInit() const
     {
         return Polygone().ListeCible();
     }

     NS_ParamChantierPhotogram::cPolygoneCalib * PC() const
     {
         return Polygone().PC();
     }


     std::string        mNameIm;
     cParamEtal         mParamEtal;
         cEtalonnage        mEtalon;
     Tiff_Im            mFileImPolygone;
     Pt2di              mSzImPolyg;
};

static std::vector<double>  NoParAdd;

cParamPointeInitEtalonnage::cParamPointeInitEtalonnage
(
       const std::string & aNameIm,
       cParamEtal aParam
) :
    cParamPointeInit  (
             new CamStenopeIdeale(true,aParam.FocaleInit(),aParam.SzIm()/2.0,NoParAdd)
                      ),
    mNameIm           (aNameIm),
    mParamEtal        (aParam),
    mEtalon           (false,aParam),
    mFileImPolygone   (Tiff_Im::BasicConvStd(mParamEtal.NameImPolygone())),
    mSzImPolyg        (mFileImPolygone.sz())
{
}


cSetPointes1Im cParamPointeInitEtalonnage:: SetPointe(const std::string & aName)
{
   return cSetPointes1Im(mEtalon.Polygone(),aName);
}

std::string cParamPointeInitEtalonnage::NamePointeInit()
{
    return mEtalon.NamePointeInit(mNameIm);
}

std::string cParamPointeInitEtalonnage::NameImageCamera()
{
     return  mParamEtal.NameTiff(mNameIm);
}


std::string  cParamPointeInitEtalonnage::NameImagePolygone()
{
    return mParamEtal.NameImPolygone();
}

std::string  cParamPointeInitEtalonnage::NamePointePolygone()
{
     return mParamEtal.NamePointePolygone();
}

const cPolygoneEtal &   cParamPointeInitEtalonnage::Polygone() const
{
   return mEtalon.Polygone();
}

void    cParamPointeInitEtalonnage::ConvPointeImagePolygone(Pt2dr& aP)
{
    if (mParamEtal.EgelsConvImage())
       aP.y = mSzImPolyg.y-aP.y;
}

/*


std::list<cPt2Im> cParamPointeInitEtalonnage::InitListePt2Is()
{
    std::list<cPt2Im> aRes;
    cSetPointes1Im PPol(mEtalon.Polygone(),mEtalon.Param().NamePointePolygone());

    for
    (
       cSetPointes1Im::tCont::iterator itP = PPol.Pointes().begin();
       itP != PPol.Pointes().end();
       itP++
    )
    {
        aRes.push_back(cPt2Im(&itP->Cible(),itP->PosIm()));
    }

    aRes.push_back(cPt2Im(0,Pt2dr(0,0)));

    return aRes;
}
*/


int PointeInitPolyg_main(int argc,char ** argv)
{
   MMD_InitArgcArgv(argc,argv);

   ELISE_ASSERT(argc>=3,"Not Enough Arg");
   std::string aNameIm = argv[2];


   argv[2] = argv[1];
   cParamEtal aParam(argc-1,argv+1);

   // cParamEtal aParam(2,argv);

   cParamPointeInit * aPPI  = new cParamPointeInitEtalonnage(aNameIm,aParam);


   PointesInitial(*aPPI,Pt2di(600,600),Pt2di(500,500));

   // cEtalonnage::TestGrid(argc,argv,"GridTriangul");

   return 0;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
