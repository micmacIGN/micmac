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
#ifndef __cModuleImageLoader_H__
#define __cModuleImageLoader_H__

#include <string> 
#include <complex>
#include <vector>

/*
   Il semble qu'il y ait une incompatibilite entre la definition
  initiale dans le namespace NS_ParamMICMAC  pour la correlation
  et d'autres besoins externes apparus ensuite. A discuter et clarifier.

     En attendant je restitue la version initiale qui permet de compile
    cStdTiffModuleImageLoader.cpp
*/
namespace NS_ParamMICMAC
{
class cAppliMICMAC;
template <class Type> 
struct sLowLevelIm
{
    Type *            mDataLin; 
    Type **           mData; 
    std::complex<int> mSzIm;  
  
    sLowLevelIm
    (
         Type *            aDataLin, 
         Type **           aData, 
         std::complex<int> aSzIm
    ) :
        mDataLin (aDataLin),
        mData    (aData),
        mSzIm    (aSzIm)
    {
    }
};

#define DeclareMembreLoadN(aType)\
      virtual void LoadNCanaux\
                   (\
                       const std::vector<sLowLevelIm<aType> > & aVImages,\
                       int              mFlagLoadedIms,\
                       int              aDeZoom,\
                       tPInt            aP0Im,\
                       tPInt            aP0File,\
                       tPInt            aSz\
                   ) = 0;\


#define DefinitMembreLoadCorrel(aType)\
      virtual void LoadCanalCorrel\
                   (\
                       const sLowLevelIm<aType> & anIm,\
                       int              aDeZoom,\
                       tPInt            aP0Im,\
                       tPInt            aP0File,\
                       tPInt            aSz\
                   )\
{\
   std::vector<sLowLevelIm<aType> > aV;\
   aV.push_back(anIm);\
   LoadNCanaux(aV,1,aDeZoom,aP0Im,aP0File,aSz);\
}

typedef enum
{
     eUnsignedChar,
     eSignedShort,
     eUnsignedShort,
     eFloat,
     eOther
} eTypeNumerique;


class cInterfModuleImageLoader
{
   public:

      virtual std::string NameFileOfResol(int aDeZoom) const; // Def erre faataale

      typedef std::complex<int> tPInt;

      DeclareMembreLoadN(unsigned char);
      DeclareMembreLoadN(short);
      DeclareMembreLoadN(unsigned short);
      DeclareMembreLoadN(float);

      DefinitMembreLoadCorrel(unsigned char);
      DefinitMembreLoadCorrel(short);
      DefinitMembreLoadCorrel(unsigned short);
      DefinitMembreLoadCorrel(float);

      virtual ~cInterfModuleImageLoader() {}
      // 1 
      virtual eTypeNumerique PreferedTypeOfResol(int aDeZoom ) const = 0 ;
      virtual tPInt Sz(int aDeZoom)  const= 0;
      virtual int NbCanaux () const = 0;

      // Pour que en cas d'execution paralelle, il 
      // y ait une preparation de la pyramide
      virtual void PreparePyram(int aDeZoom) =0;


      // Par defaut genere une erreurs, utilise pour compatibilite
      // avec d'anciens services tels que ValSpecNotImage
      virtual std::string  NameTiffImage() const;
       


   protected:
      cInterfModuleImageLoader():mAppli(0){}
      // Attenti
      cAppliMICMAC & Appli();
   private:
      friend class cAppliMICMAC;
      void SetAppli(cAppliMICMAC *);
      cAppliMICMAC * mAppli;
};

#if __TEST__
/*************************************************/
/*                                               */
/*              cInterfModuleImageLoader         */
/*                                               */
/*************************************************/

// A mettre dans le .h pour utilisation hors MICMAC
/*
    cInterfModuleImageLoader::cInterfModuleImageLoader() :
    mAppli (0)
{
}
*/
cAppliMICMAC & cInterfModuleImageLoader::Appli()
{
   //ELISE_ASSERT(mAppli!=0,"cInterfModuleImageLoader, Appli Not Init");
   return *mAppli;
}

void  cInterfModuleImageLoader::SetAppli(cAppliMICMAC * anAppli)
{
    mAppli = anAppli;
}

std::string  cInterfModuleImageLoader::NameTiffImage() const
{
    //ELISE_ASSERT(false,"Pas de cInterfModuleImageLoader::NameTiffImage");
    return "";
}


std::string cInterfModuleImageLoader::NameFileOfResol(int aDeZoom) const
{
    //ELISE_ASSERT(false,"Pas de cInterfModuleImageLoader::NameFileOfResol");
    return "";
}
#endif

};




#endif // __cModuleImageLoader_H__


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
