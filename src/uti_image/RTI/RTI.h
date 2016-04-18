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

#ifndef _RTI_H_
#define _RTI_H_

#include "StdAfx.h"


class cOneIm_RTI;
class cAppli_RTI;


typedef enum
{
     eRTI_Test,
     eRTI_Med,
     eRTI_Grad,
     eRTI_OldRecal,
     eRTI_RecalBeton_1Im,
     eRTI_RecalBeton_1AllIm,
     eRTI_PoseFromOmbre
} eModeRTI;


class cOneIm_RTI
{
    public :
      cOneIm_RTI(cAppli_RTI &,const std::string & aName,bool Master);
      virtual Tiff_Im DoImReduite();
      Tiff_Im      FileImFull(const std::string & aNameIm);

      const std::string & Name() const;
      Im2D_U_INT2  MemImRed();
      Im2D_U_INT2  MemImFull();
      Im2D_Bits<1> MasqRed(Im2D_U_INT2);
      const std::string & NameDif();
      bool IsMaster() const;
      void DoPoseFromOmbre(const cDicoAppuisFlottant &,const cSetOfMesureAppuisFlottants &);

      void  SetXml(cXml_RTI_Im*);
      const Pt3dr & CenterLum() const;
      Tiff_Im   MasqFull();
      Im2D_Bits<1> ImMasqFull();

    protected :
      cAppli_RTI &   mAppli;
      std::string    mName;
      bool           mMaster;
      bool           mWithRecal;
      std::string    mNameIS;  // Name Image Superpose
      std::string    mNameISPan;  // Name Image Superpose Panchro ?
      std::string    mNameISR; // IS Reduced
      std::string    mNameMasq;  // Name Image Superpose
      std::string    mNameMasqR; // IS Reduced
      string         mNameDif;
      cXml_RTI_Im*   mXml;
      bool           mHasExp;
      Pt3dr          mCenterLum;
};

class cOneIm_RTI_Slave : public cOneIm_RTI
{
    public :
       cOneIm_RTI_Slave(cAppli_RTI &,const std::string & aName);
       Tiff_Im DoImReduite();
       const std::string & NameMasq() const;
       const std::string & NameMasqR() const;
    private :
};

class cOneIm_RTI_Master : public cOneIm_RTI
{
    public :
       cOneIm_RTI_Master(cAppli_RTI &,const std::string & aName);
    protected :
};




class cAppli_RTI
{
    public :
       static const std::string ThePrefixReech;
       cAppli_RTI(const std::string & aFullNameParam,eModeRTI aMode,const std::string & aName2);
       void CreateSuperpHom();
       void CreatHom();
       const cXml_ParamRTI & Param() const;
       const std::string & Dir() const;
       cOneIm_RTI_Slave * UniqSlave();
       cOneIm_RTI_Master * Master();
       void MakeImageMed(const std::string & aNameIm);
       void MakeImageGrad();
       bool  WithRecal() const;

       void DoOneRecalRadiomBeton();
       void DoPoseFromOmbre(const cDicoAppuisFlottant &,const cSetOfMesureAppuisFlottants &);

       CamStenope *    OriMaster();
       const cXml_RTI_Ombre & Ombr() const;
       void  FiltrageGrad();


    private :
       Im2D_REAL8  OneItereRecalRadiom
                   (
                       double & aScaleRes,bool L1,Im2D_U_INT2,Im2D_Bits<1>,Im2D_U_INT2,
                       int aNbCase,int aDeg
                   );


       void MakeImageMed(const Box2di & aBox,const std::string & aNameIm);
       void MakeImageGrad(const Box2di&);

       CamStenope *                    mOriMaster;

       cXml_ParamRTI                    mParam;
       bool                             mWithRecal;
       std::string                      mFullNameParam;
       std::string                      mDir;
       cInterfChantierNameManipulateur  *mICNM;
       std::string                      mNameParam;
       bool                             mTest;
       std::vector<cOneIm_RTI *>        mVIms;
       std::vector<cOneIm_RTI_Slave *>  mVSlavIm;
       std::map<std::string,cOneIm_RTI*> mDicoIm;
       cOneIm_RTI_Master *              mMasterIm;
       cElemAppliSetFile                mEASF;
       std::string                      mNameImMed;
       std::string                      mNameImGx;
       std::string                      mNameImGy;
};




#endif // _RTI_H_

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
aooter-MicMac-eLiSe-25/06/2007*/
