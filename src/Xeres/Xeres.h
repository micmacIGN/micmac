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

#ifndef _Xeres_H_
#define _Xeres_H_

#include "StdAfx.h"



class cXeres_NumSom;  // Represente la numerotation "topologique" des camara


class cXeres_Cam;

class cAppliXeres;

//=================================


class cXeres_NumSom
{
    public :
       static cXeres_NumSom CreateFromStdName_V0(const std::string &);

       const std::string & Name() const;
       const int & NumParal() const;
       const int & NumMer() const;
       const double & Teta() const;
       const Pt2dr & DirT() const;
       const bool  & Porte() const;

    private :
       cXeres_NumSom(int aNumParal,int aNumMer,double aTeta,const std::string & aName,bool aPorte);
 
       std::string mName;
       int         mNumParal;  // Numero de cercle
       int         mNumMer;  // Numero ds le cercle
       double      mTeta;
       Pt2dr       mDirT;
       bool        mPorte;
       
};


class cXeres_Cam
{
     public :
         cXeres_Cam(const std::string &,cAppliXeres &);
         double & TmpDist() ;
         const cXeres_NumSom & NS() const;
         const bool & HasIm() const;
         const std::string & NameIm() const;
     private :
         cXeres_Cam(const cXeres_Cam &); // N.I. 

         cAppliXeres &   mAppli;
         std::string     mId;
         cXeres_NumSom   mNS;
         std::string     mNameIm;
         bool            mHasIm;

         double          mTmpDist;
           
};

//=================================

class cAppliXeres
{
     public :
         cAppliXeres(const std::string & aDir,const std::string & aSeq,cElRegex * aFilter =0);
         std::string NameOfId(const std::string &);

         void TestInteractNeigh();

         void CalculTiePoint(int aSz,int aNBHom,const std::string & aNameAdd="");
         void CalculHomMatch(const std::string & anOri);

         static void FusionneHom(const std::vector<cAppliXeres *>,const std::string & aPostOut);

         std::string ExtractId(const std::string & aNameIm);
         std::string Make2CurSeq(const std::string & aNameIm);

         bool NameInFilter(const std::string & aName) const;

     private :

         void ExeTapioca(const std::string & aFile);
         std::vector<cXeres_Cam *> GetNearestNeigh(cXeres_Cam *,int aDL,int aNb);
         std::vector<cXeres_Cam *> GetNearestExistingNeigh(cXeres_Cam *,int aDL,int aNb);
         void  AddNearestExistingNeigh(std::vector<cXeres_Cam *> & aRes,cXeres_Cam *,int aDL,int aNb);

         std::vector<cXeres_Cam *> NeighVois(cXeres_Cam *,int aKV);
         std::vector<cXeres_Cam *> NeighPtsHom(cXeres_Cam *);
         std::vector<cXeres_Cam *> NeighMatch(cXeres_Cam *);

         void TestOneNeigh(const std::string & aName,int aDeltaV);


         void AddCam(const  std::string & anId);

         cInterfChantierNameManipulateur *   mICNM;
         std::map<std::string,cXeres_Cam *>  mMapCam;
         std::vector<cXeres_Cam *>           mVCam;
         std::string mDir;
         std::string mSeq;
         std::string mPost;
         std::string mNameCpleXml;
         int         mSzTapioca;
         cElRegex *  mFilter;
};

#endif // _Xeres_H_

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
