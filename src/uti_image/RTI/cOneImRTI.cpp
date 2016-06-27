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

#include "RTI.h"

/*****************************************************************/
/*                                                               */
/*              cOneIm_RTI                                       */
/*                                                               */
/*****************************************************************/

template <class Type> Type & MessageInit(const std::string & aMes,Type & aVal)
{
      std::cout << aMes << "\n" ;
    return aVal;
}
template <class Type> const Type & MessageInit(const std::string & aMes,const Type & aVal)
{
      std::cout << aMes << "\n" ;
    return aVal;
}

cOneIm_RTI::cOneIm_RTI(cAppli_RTI & anAppli,const std::string & aName,bool isMaster) :
    mAppli       (anAppli),
    mName        (aName),
    mMaster      (isMaster),
    mWithRecal   ((!mMaster) && mAppli.WithRecal()),
    mNameIS      (mAppli.Dir() + (mWithRecal ? cAppli_RTI::ThePrefixReech:"") + aName +  (mWithRecal ? ".tif" : "" )),
    mNameISPan   (NameFileStd(mNameIS,1,true)),
    mNameISR     ( cAppli_RTI::ThePrefixReech  + NameWithoutDir(mNameISPan) + "_Scaled.tif"),
    mNameMasq    (StdPrefix(mName) + "_Masq.tif"),
    mNameMasqR   (cAppli_RTI::ThePrefixReech + StdPrefix(mNameMasq) + "_Scaled.tif"),
    mNameDif     (cAppli_RTI::ThePrefixReech + "ImDif-" + StdPrefix(mName) + ".tif"),
    mXml         (0),
    mHasExp      (false)
{
}

void cOneIm_RTI::SetXml(cXml_RTI_Im* aXml)
{
    mXml = aXml;
    mHasExp = mXml->Export().IsInit();
    if (mHasExp)
    {
       mCenterLum =  mXml->Export().Val().PosLum();
    }
}

const Pt3dr & cOneIm_RTI::CenterLum() const
{
   ELISE_ASSERT(mHasExp,"cOneIm_RTI::CenterLum");
   return mCenterLum;
}

//cXml_RTI_Im*&   cOneIm_RTI::Xml() {return mXml;}


const std::string & cOneIm_RTI::NameDif() {return mNameDif;}
bool cOneIm_RTI::IsMaster() const {return mMaster;}

Im2D_U_INT2  cOneIm_RTI::MemImFull()
{
    return Im2D_U_INT2::FromFileStd(mNameISPan);
}

Im2D_U_INT2  cOneIm_RTI::MemImRed()
{
    return Im2D_U_INT2::FromFileStd(mNameISR);
}

Im2D_Bits<1> cOneIm_RTI::ImMasqFull()
{
    return MasqFromFile(mNameMasq);
}


Im2D_Bits<1> cOneIm_RTI::MasqRed(Im2D_U_INT2 anIm)
{
    return MasqFromFile(mNameMasqR,anIm.sz());
/*
   if (ELISE_fp::exist_file(mNameMasqR))
   {
       Tiff_Im aTif(mNameMasqR.c_str());
       Pt2di aSz = aTif.sz();
       Im2D_Bits<1> aRes(aSz.x,aSz.y);
       ELISE_COPY(aTif.all_pts(),aTif.in(),aRes.out());
       return aRes;
   }
   Pt2di aSz = anIm.sz();
   return Im2D_Bits<1>(aSz.x,aSz.y,1);
*/
}


Tiff_Im  cOneIm_RTI::DoImReduite()
{
   Tiff_Im::StdConvGen(mNameIS,1,true);


   if (! ELISE_fp::exist_file(mNameISR))
   {
        std::string aCom =   MM3dBinFile("ScaleIm ") 
                           + mNameISPan 
                           +  " "  +  ToString(mAppli.Param().ScaleSSRes())
                           +  " Out=" + mNameISR;

   
        System(aCom);
   }
   if (ELISE_fp::exist_file(mNameMasq) && (! ELISE_fp::exist_file(mNameMasqR)))
   {
        std::string aCom =    MM3dBinFile("ScaleIm ") 
                           +  mNameMasq 
                           +  " "  +  ToString(mAppli.Param().ScaleSSRes())
                           +  " Out=" + mNameMasqR
                           +  " ModMasq=true " ;

   
        System(aCom);
   }


   return Tiff_Im(mNameISR.c_str());
}

Tiff_Im   cOneIm_RTI::FileImFull(const std::string & aName) 
{
    if (aName=="ImDif") return Tiff_Im(mNameDif.c_str());
    if (aName=="") return Tiff_Im(mNameISPan.c_str());

    ELISE_ASSERT(false,"cOneIm_RTI::FileImFull");
    return Tiff_Im(mNameISPan.c_str());
}

const std::string & cOneIm_RTI::Name() const {return mName;}

Tiff_Im   cOneIm_RTI::MasqFull() {return Tiff_Im(mNameMasq.c_str());}

/*****************************************************************/
/*                                                               */
/*                 cOneIm_RTI_Slave                              */
/*                                                               */
/*****************************************************************/

cOneIm_RTI_Slave::cOneIm_RTI_Slave(cAppli_RTI & anAppli,const std::string & aName) :
   cOneIm_RTI      (anAppli,aName,false)
{
}

Tiff_Im cOneIm_RTI_Slave::DoImReduite()
{
   Tiff_Im aRes = cOneIm_RTI::DoImReduite();
   return aRes;
}

const std::string &  cOneIm_RTI_Slave::NameMasq()  const {return mNameMasq;}
const std::string &  cOneIm_RTI_Slave::NameMasqR() const {return mNameMasqR;}



/*****************************************************************/
/*                                                               */
/*                 cOneIm_RTI_Master                              */
/*                                                               */
/*****************************************************************/

cOneIm_RTI_Master::cOneIm_RTI_Master(cAppli_RTI & anAppli,const std::string & aName) :
   cOneIm_RTI(anAppli,aName,true)
{
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
