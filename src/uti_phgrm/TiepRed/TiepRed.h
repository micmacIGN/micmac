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

#ifndef _TiepRed_H_
#define _TiepRed_H_

#include "StdAfx.h"

class cCameraTiepRed;
class cAppliTiepRed;
class cLnk2ImTiepRed;


typedef cVarSizeMergeTieP<Pt2df>  tMerge;
typedef cStructMergeTieP<tMerge>  tMergeStr;

class cCameraTiepRed
{
    public :
        cCameraTiepRed(cAppliTiepRed & anAppli,const std::string &,CamStenope *);
        const std::string NameIm() const;
 
        //  Intersection of bundles in ground geometry
        Pt3dr BundleIntersection(const Pt2df & aP1,const cCameraTiepRed & aCam2,const Pt2df & aP2,double & Precision) const;

        CamStenope  & CS();
        bool  SelectOnHom2Im() const;
        const int &   NbPtsHom2Im() const;

        void LoadHom(cCameraTiepRed & aCam2);
        void SetNum(int aNum);
        const int & Num() const;

        Pt2dr Hom2Cam(const Pt2df & aP) const;


    private :
        cCameraTiepRed(const cCameraTiepRed &); // Not Implemented


        cAppliTiepRed & mAppli;
        std::string mNameIm;
        CamStenope * mCS;
        int          mNbPtsHom2Im;
        int          mNum;
};

class cLnk2ImTiepRed
{
     public :
        cLnk2ImTiepRed(cCameraTiepRed * ,cCameraTiepRed *);
        cCameraTiepRed &     Cam1();
        cCameraTiepRed &     Cam2();
        std::vector<Pt2df>&  VP1();
        std::vector<Pt2df>&  VP2();

        void Add2Merge(tMergeStr *);
     private :
        cCameraTiepRed *    mCam1;
        cCameraTiepRed *    mCam2;
        std::vector<Pt2df>  mVP1;
        std::vector<Pt2df>  mVP2;
};


class cPMulTiepRed
{
     public :
       cPMulTiepRed(tMerge *,cAppliTiepRed &);
       const Pt2dr & Pt() const {return mP;}
     private :
       Pt2dr  mP;
       double mZ;
       double mPrec;
       double mGain;
};
class cP2dGroundOfPMul
{
    public :
          Pt2dr operator()(cPMulTiepRed * aPM) {return aPM->Pt();}
};



class cAppliTiepRed 
{
     public :
          cAppliTiepRed(int argc,char **argv); 
          void Exe();
          cVirtInterf_NewO_NameManager & NM();
          const cXml_ParamBoxReducTieP & ParamBox() const;
          const double & ThresoldPrec2Point() const;
          const double & ThresholdPrecMult() const;
          const int    & ThresholdNbPts2Im() const;
          const int    & ThresholdTotalNbPts2Im() const;
          void AddLnk(cLnk2ImTiepRed *);
          cCameraTiepRed * KthCam(int aK);

     private :

          void GenerateSplit();
          void DoReduceBox();
          cAppliTiepRed(const cAppliTiepRed &); // N.I.

          static const std::string TheNameTmp;

          std::string NameParamBox(int aK) const;

          double mPrec2Point; // Threshold on precision for a pair of tie P
          double mThresholdPrecMult; // Threshold on precision for multiple points
          int    mThresholdNbPts2Im;
          int    mThresholdTotalNbPts2Im;
          int    mSzTile;    //  Number of pixel / tiles

          std::string  mDir;
          std::string  mPatImage;
          std::string  mCalib;

          std::map<std::string,cCameraTiepRed *> mMapCam;
          std::vector<cCameraTiepRed *>          mVecCam;
          std::set<std::string>          * mSetFiles;
          cVirtInterf_NewO_NameManager *   mNM ;
          bool                             mCallBack;
          int                              mKBox;
          Box2dr                           mBoxGlob;
          Box2dr                           mBoxLoc;
          double                           mResol;
          cXml_ParamBoxReducTieP           mXmlParBox;
          std::list<cLnk2ImTiepRed *>      mLnk2Im;
          tMergeStr *                      mMergeStruct;
          const std::list<tMerge *> *      mLMerge;
          std::list<cPMulTiepRed *>        mLPMul;

          cP2dGroundOfPMul                            mPMul2Gr;
          ElQT<cPMulTiepRed*,Pt2dr,cP2dGroundOfPMul>  *mQT;
};


#endif // _TiepRed_H_

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
