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


#ifndef _ELISE_CASA_ALL_H_
#define _ELISE_CASA_ALL_H_


#include "general/all.h"
#include "private/all.h"
#include "im_tpl/image.h"
#include "ext_stl/tab2D_dyn.h"
#include "graphes/graphe.h"
#include "algo_geom/qdt.h"
#include <map>
#include <set>

#include "ext_stl/numeric.h"

#include "XML_GEN/all.h"
using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;

namespace NS_AJ
{

class cAppli_Traj_AJ;
class cTAj2_OneLayerIm;
class cTAj2_OneImage;
class cTAj2_OneLogIm;
class cTAj2_LayerLogIm;

typedef enum
{
    eMatchParfait,
    eNoMatch,
    eMatchDifForte,
    eMatchAmbigu,
    eMatchIncoh
} eTypeMatch;

template <class TypeMatch,class TypeThis> class cLinkMatch
{
      public :
          void Reset()
          {
               mBestMatch = 0;
               mNbMatch = 0;
               mDifMin = 1e30;
          }
          cLinkMatch() {Reset();}
          void Update(TypeMatch * aMatch,double aDif)
          {
             mNbMatch++;
             if (aDif<mDifMin)
             {
                mBestMatch= aMatch;
                mDifMin = aDif;
             }
          }

          eTypeMatch QualityMatch(double aDif,TypeThis * aThis)
          {
              if (mBestMatch==0)  return eNoMatch;
              if (mNbMatch>1)     return eMatchAmbigu;
              if (mDifMin > aDif) return eMatchDifForte;
              if (mBestMatch->BestMatch() != aThis) return eMatchIncoh;

              return eMatchParfait;
          }
          

          TypeMatch * mBestMatch;
          int    mNbMatch;
          double mDifMin;
};



class cTAj2_OneImage
{
     public :
         void SetLinks(cTAj2_OneImage * aPrec);
         void EstimVitesse();
         cTAj2_OneImage(cAppli_Traj_AJ &,cTAj2_OneLayerIm &,const std::string &);
         void InitT0(const cTAj2_OneImage &);
         double  T0() const;
         int     Num() const;
         void    SetNum(int aNum);
         const std::string & Name() const;
         void ResetMatch();
         void UpdateMatch(cTAj2_OneLogIm *,double);
 //         bool MatchAmbigu() const;

         cTAj2_OneLogIm *BestMatch();
         eTypeMatch QualityMatch(double aDif);
         void SetDefQualityMatch(eTypeMatch);
         eTypeMatch  DefQualityMatch();
         bool VitOK() const;
         Pt3dr Vitesse() const;

         
     private :
         cAppli_Traj_AJ &     mAppli;
         cTAj2_OneLayerIm &   mLayer;
         std::string          mName;
         cMetaDataPhoto *     mMDP;
         double               mTime2I0;
         int                  mNum;
         eTypeMatch           mDQM;
         
         cLinkMatch<cTAj2_OneLogIm,cTAj2_OneImage>   mLKM;
         cTAj2_OneImage *     mNext;
         cTAj2_OneImage *     mPrec;
         bool                 mVitOK;
         Pt3dr                mVitessse;
};


class cTAj2_OneLayerIm
{
     public :
        cTAj2_OneLayerIm(cAppli_Traj_AJ &,const cTrAJ2_SectionImages &);
        void AddIm(const std::string &);
        void Finish();
        void  FinishT0();
        cTAj2_OneImage * ImOfName(const std::string &);
        int  NbIm() const;
        cTAj2_OneImage * KthIm(int aK) const;
        void ResetMatch();
        std::vector<cTAj2_OneImage *> & Ims();
        const cTrAJ2_SectionImages & SIm() const;
        const std::vector<cTAj2_OneImage *> & MatchedIms() const;
        void AddMatchedIm(cTAj2_OneImage *);
     private :
        void InitT0();
        cAppli_Traj_AJ &               mAppli;
        const cTrAJ2_SectionImages &   mSIm;
        bool                           mFinishT0;
        std::vector<cTAj2_OneImage *>  mIms;
        std::vector<cTAj2_OneImage *>  mMatchedIms;
        std::map<std::string,cTAj2_OneImage *>  mDicIms;
};


class cTAj2_OneLogIm
{
    public :
        cTAj2_OneLogIm(cAppli_Traj_AJ &,int aKLine,cTAj2_LayerLogIm &,const cTrAJ2_SectionLog&,const std::string &);
        double Time() const;
        const std::string & KeyIm() const;
        double T0() const;
        void InitT0(const cTAj2_OneLogIm &);
        double Teta(int aK) const;
        Pt3dr  PCBrut() const;
        Pt3dr  PGeoC() const;
        void ResetMatch();
        int  KLine() const;
        void UpdateMatch(cTAj2_OneImage *,double);
        cTAj2_OneImage *BestMatch();
        eTypeMatch QualityMatch(double aDif);
   //      bool MatchAmbigu() const;
        const ElMatrix<double>  &   MatI2C() const;
    private :
        cAppli_Traj_AJ &          mAppli;
        int                       mKLine;
        cTAj2_LayerLogIm &        mLayer;
        const cTrAJ2_SectionLog & mParam;
        std::string               mLine;

        double                 mTime;
        bool                   mTimeIsInit;
        std::string            mKeyIm;
        bool                   mKeyImIsInit;


        double                 mT0;
        double                 mCoord[3];
        double                 mTetas[3];
        cLinkMatch<cTAj2_OneImage,cTAj2_OneLogIm> mLKM;
        bool                   mHasTeta;
        ElMatrix<double>       mMatI2C;
};


class cTAj2_LayerLogIm
{
    public :
        cTAj2_LayerLogIm(cAppli_Traj_AJ &,const cTrAJ2_SectionLog&);
        cElRegex & Autom();
        const cSysCoord *  SC();
        int  NbLog() const;
        cTAj2_OneLogIm * KthLog(int aK) const;
        void ResetMatch();
        std::vector<cTAj2_OneLogIm *> & Logs();
    private :
        void GenerateOneExample(const cGenerateTabExemple &);
         cAppli_Traj_AJ &               mAppli;
         const cTrAJ2_SectionLog&       mSL; 
         std::vector<cTAj2_OneLogIm *>  mLogs;
         cElRegex *                     mRegEx;
         const cSysCoord *              mSC;

         Pt3dr                          mCoordMin;
         Pt3dr                          mCoordMax;
};


class cTAj2_LayerAppuis
{
    public :
        cTAj2_LayerAppuis(cAppli_Traj_AJ &,const cTrAJ2_ConvertionAppuis &);

    private :
         void AddFile(const cTraJ2_FilesInputi_Appuis & aFIn,const std::string & aNameFile);

         cAppli_Traj_AJ &               mAppli;
         const cTrAJ2_ConvertionAppuis & mSAp;
         cElRegex *                     mCom;
         const cSysCoord *              mSIn;
         const cSysCoord *              mSOut;
         std::map<std::string,cMesureAppuiFlottant1Im>   mMapMesIm;
         std::map<std::string,cOneAppuisDAF>              mMapPtsAp;
};


class cAppli_Traj_AJ
{
    public :
       cAppli_Traj_AJ( cResultSubstAndStdGetFile<cParam_Traj_AJ> aParam);
       void DoAll();
       const std::string & DC();
       cInterfChantierNameManipulateur * ICNM();

       bool TraceImage(const cTAj2_OneImage &) const;
       bool TraceLog(const cTAj2_OneLogIm &) const;

    private :

    // Learn Offset
       double LearnOffset(cTAj2_OneLayerIm*,cTAj2_LayerLogIm*,const cLearnOffset &);

       double LearnOffsetByExample(cTAj2_OneLayerIm*,cTAj2_LayerLogIm*,const cLearnByExample &);
       double LearnOffsetByStatDiff(cTAj2_OneLayerIm*,cTAj2_LayerLogIm*,const cLearnByStatDiff &);

       int Avance(const std::vector<double> & aV,int aK0,double aVMax);

       double OneGainStat(const std::vector<double> & aV,int aK,double MaxEcart);

       void DoAlgoMatch(cTAj2_OneLayerIm*,cTAj2_LayerLogIm*, const cAlgoMatch & );
       void DoMatchNearest(cTAj2_OneLayerIm*,cTAj2_LayerLogIm*, const   cMatchNearestIm &);
       void DoAlgoMatchByName( cTAj2_OneLayerIm* aLIm, cTAj2_LayerLogIm* aLLog, const cMatchByName & aMN);

        


       void GenerateOrient(cTAj2_OneLayerIm*,const cTrAJ2_SectionMatch &,const cTrAJ2_GenerateOrient &);
    
    // -------
       cTAj2_OneLayerIm * ImLayerOfId(const std::string &);
       cTAj2_LayerLogIm * LogLayerOfId(const std::string &);
       void InitImages();
       void InitOneLayer(const cTrAJ2_SectionImages &);
       void InitLogs();
       void InitLogs(const cTrAJ2_SectionLog &);
       void InitAppuis();
       void InitOneAppuis(const  cTrAJ2_ConvertionAppuis &anAp);

       void TxtExportProjImage();
       void TxtExportProjImage(const cTrAJ2_ExportProjImage & anEPI);


       void DoMatch();
       void DoOneMatch(const cTrAJ2_SectionMatch &);
       void DoEstimeVitesse(cTAj2_OneLayerIm *,const cTrAJ2_ModeliseVitesse &);

       cParam_Traj_AJ                              mParam;
       cInterfChantierNameManipulateur *           mICNM;
       std::string                                 mDC;
       std::vector<cTAj2_OneLayerIm *>             mLayIms;
       std::map<std::string,cTAj2_OneLayerIm *>    mDicLIms;
       std::vector<cTAj2_LayerLogIm *>             mLayLogs;
       std::map<std::string,cTAj2_LayerLogIm *>    mDicLogs;
       std::map<std::string,cTAj2_LayerAppuis *>   mDicApp;
       double                                      mCurOffset;
       bool                                        mIsInitCurOffset;
};



};

#endif //  _ELISE_CASA_ALL_H_




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
