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


#ifndef _ELISE_SAISIEPTS_ALL_H_
#define _ELISE_SAISIEPTS_ALL_H_

using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;

namespace NS_SaisiePts
{

class cSP_PointeImage;
class cSP_PointGlob;
class cImage;
class cWinIm ;
class cAppli_SaisiePts;
class cSP_PointeImage;
class cSP_PointGlob;
class cCaseNamePoint;

typedef enum
{
   eModeScale,
   eModeTr,
   eModePopUp,
   eModeSaisiePts
} eModeWinIm;



class cSP_PointeImage
{
     public :

        cSP_PointeImage(cOneSaisie * aSIm,cImage * ,cSP_PointGlob *);
        cOneSaisie * Saisie();
        cImage * Image();
        cSP_PointGlob * Gl();
        bool  & Visible() ;
        bool BuildEpipolarLine(Pt2dr &pt1, Pt2dr &pt2);
private :
         cSP_PointeImage(const cSP_PointeImage &); // N.I.


        cOneSaisie *    mSIm;
        cImage *        mIm;
        cSP_PointGlob * mGl;
        bool            mVisible;
};


class cSP_PointGlob
{
     public:
          cSP_PointGlob(cAppli_SaisiePts &,cPointGlob * aPG);
          cPointGlob * PG();
          void AddAPointe(cSP_PointeImage *);

          void ReCalculPoints();
          void SuprDisp();
          bool & HighLighted();
          void SetKilled();

          bool IsPtAutom() const;
          void Rename(const std::string & aNewName);

     private:
          cSP_PointGlob(const cSP_PointGlob &) ; // N.I.

          cAppli_SaisiePts & mAppli;
          cPointGlob * mPG;
          std::map<std::string,cSP_PointeImage *>   mPointes;

          bool   mHighLighted;
};


class cImage
{
     public :
        void InitCameraAndNuage();
        Pt2dr PointArbitraire() const;
        cImage(const std::string & aName,cAppli_SaisiePts &);

        Fonc_Num  FilterImage(Fonc_Num,eTypePts,cPointGlob *);

        void SetSPIM(cSaisiePointeIm * aSPIM);
        Tiff_Im &  Tif() const;
        cCapture3D *       Capt3d();
        ElCamera *         CaptCam();
        cElNuage3DMaille * CaptNuage();
        void AddAPointe(cOneSaisie *,cSP_PointGlob *,bool FromFile);
        const std::string & Name() const;
        bool InImage(const Pt2dr & aP);
        Pt2di  SzIm() const;
        const std::vector<cSP_PointeImage *> &  VP();
        cSP_PointeImage * PointeOfNameGlobSVP(const std::string & aNameGlob);

        void SetWAff(cWinIm *);
        cWinIm * WAff() const;
        
        double CalcPriority(cSP_PointGlob * aPP) const;
        double Prio() const;
        void   SetPrio(double aPrio);
        bool PtInImage(const Pt2dr aPt);

        void CreatePGFromPointeMono(Pt2dr ,eTypePts,double aSz,cCaseNamePoint *);

     private :

           cAppli_SaisiePts &                        mAppli;

           std::string                               mName;
           mutable Tiff_Im *                         mTif;
           cCapture3D *                              mCapt3d;
           // mCapt3d est l'un ou l'autre
           ElCamera *                                mCaptCam;
           cElNuage3DMaille *                        mCaptNuage;
           cSaisiePointeIm *                         mSPIm;
           std::map<std::string,cSP_PointeImage *>   mPointes;
           std::vector<cSP_PointeImage *>            mVP;
           mutable Pt2di                             mSzIm;
           cWinIm *                                  mWAff;
           double                                    mPrio;
           bool                                      mInitCamNDone;
};

typedef cImage * tImPtr;


class cWinIm : public Grab_Untill_Realeased
{
public :
    cWinIm(cAppli_SaisiePts&, Video_Win aW, Video_Win aWTitle, cImage & aIm0);
    Video_Win W();
    void    GrabScrTr(Clik);
    void    ShowVect();

    bool    WVisible(const Pt2dr & aP);
    bool    WVisible(const Pt2dr & aP, eEtatPointeImage aState);
    bool    WVisible(cSP_PointeImage & aPIm);

    cSP_PointeImage * GetNearest(const Pt2dr & aPW,double aDSeuil);
    void    SetPt(Clik aClk);
    void    SetZoom(Pt2dr aP,double aFactZ);

    void    Redraw();

    void    MenuPopUp(Clik aClk);

    void    SetNewImage(cImage *);
    BoolCaseGPUMT *      BCaseVR();
    void    SetTitle();
    void    ShowPoint(const Pt2dr aP,eEtatPointeImage aState,cSP_PointGlob * PInfoHL,cSP_PointeImage *);
    void    ShowInfoPt(cSP_PointeImage *,bool Compl);

    void    SetNoImage();


    void    SetImage(cImage *);
    static const Pt2dr  PtsEchec;
    Box2dr  BoxImageVisible() const;

    cImage* Image() { return mCurIm; }

private :

    void    CreatePoint(const Pt2dr& aP,eTypePts,double aSz);
    Pt2dr   FindPoint(const Pt2dr &aPIm,eTypePts,double aSz,cPointGlob *);

    void    GUR_query_pointer(Clik,bool);
    void    RedrawGrabSetPosPt();



    cAppli_SaisiePts & mAppli;

    Video_Win               mW;
    Video_Win               mWT;
    eModeWinIm              mMode;
    Pt2dr                   mOldPt;
    Pt2dr                   mNewPt;
    eEtatPointeImage        mStatePtCur;

    VideoWin_Visu_ElImScr   mVWV;
    ElImScroller *          mScr;
    cImage *                mCurIm;
    Pt2dr                   mLastPGrab;
    Pt2dr                   mP0Grab;
    bool                    mModeRelication;
    Pt2di                   mSzW;

    Pt2di                   mSzCase;
    GridPopUpMenuTransp*    mPopUpBase;
    GridPopUpMenuTransp*    mPopUpShift;
    GridPopUpMenuTransp*    mPopUpCtrl;
    GridPopUpMenuTransp*    mPopUp1Shift;
    GridPopUpMenuTransp*    mPopUpCur;


    CaseGPUMT *             mCaseExit;
    CaseGPUMT *             mCaseVide;
    CaseGPUMT *             mCaseTDM;
    CaseGPUMT *             mCaseInterrog;
    CaseGPUMT *             mCaseSmile;
    BoolCaseGPUMT *         mBCaseVisiRefut;
    BoolCaseGPUMT *         mBCaseShowDet;
    CaseGPUMT *             mCaseHighLight;

    CaseGPUMT *             mCaseUndo;
    CaseGPUMT *             mCaseRedo;

    CaseGPUMT *             mCaseAllW;
    CaseGPUMT *             mCaseThisW;
    CaseGPUMT *             mCaseThisPt;

    CaseGPUMT *             mCaseNewPt;
    CaseGPUMT *             mCaseKillPt;
    CaseGPUMT *             mCaseRenamePt;
    CaseGPUMT *             mCaseMin3;
    CaseGPUMT *             mCaseMin5;
    CaseGPUMT *             mCaseMax3;
    CaseGPUMT *             mCaseMax5;


};

class cUndoRedo
{
    public :
       cUndoRedo(cOneSaisie aS,cImage *);
       const cOneSaisie & S() const;
       cImage *           I() const;
    private :
       cOneSaisie  mS;
       cImage*     mI;
};

typedef enum
{
    eCaseStd,
    eCaseCancel,
    eCaseAutoNum,
    eCaseSaisie
} eTypeCasePt;

class cCaseNamePoint
{
    public :
      cCaseNamePoint(const std::string &,eTypeCasePt);

      std::string mName;
      eTypeCasePt mTCP;
      bool        mFree;
      //  bool        mAutoNum;
      //  bool        mVraiCase;
};

class cVirtualInterface
{
    public:

    cVirtualInterface(){}
    ~cVirtualInterface(){}

    virtual void        RedrawAllWindows()=0;

    virtual void        SetInvisRef(bool aVal)=0;         // sert � rendre les points r�fut�s invisibles ou visibles
    bool                RefInvis() const    { return mRefInvis; }

    void                ChangeFreeNamePoint(const std::string &, bool SetFree);

    void                DeletePoint(cSP_PointGlob *aSG);

    void                Save();

    virtual cCaseNamePoint * GetIndexNamePoint() = 0 ;

    int                 GetNumCasePoint()          { return mVNameCase.size(); }
    cCaseNamePoint &    GetCaseNamePoint(int aK)   { return mVNameCase[aK];    }

//     virtual  cFenMenu *      MenuNamePoint()=0;

    virtual std::pair<int,std::string> IdNewPts(cCaseNamePoint * aCNP)=0;

    bool                Visible(eEtatPointeImage aState);

    void                ChangeState(cSP_PointeImage* aPIm, eEtatPointeImage aState);

    void                UpdatePoints(cSP_PointeImage* aPIm, Pt2dr pt);

    virtual void        AddUndo(cOneSaisie *)=0;

    virtual void        Redraw()=0;

protected:

    void                      InitNbWindows();

    void                      InitVNameCase();

    cAppli_SaisiePts*         mAppli;
    const cParamSaisiePts*    mParam;

    Pt2di                     mNb2W;        //window nb (col, raw)
    int                       mNbW;         //total window nb (col x raw)

    bool                      mRefInvis;

    std::vector <cCaseNamePoint>        mVNameCase;

    std::map<std::string,cCaseNamePoint *>  mMapNC;

private:

    virtual void              Init()=0;
};

#if ELISE_windows == 0

class cX11_Interface : public cVirtualInterface
{
public :

    cX11_Interface(cAppli_SaisiePts &appli);
    ~cX11_Interface();

    void            TestClick(Clik aCl);

    void            RedrawAllWindows();

    void            Redraw();

    void            BoucleInput();

    void            DrawZoom(const Pt2dr & aPGlob); //fenetre zoom

    const std::vector<cWinIm *> &  WinIms() { return mWins; }

    cFenMenu *      MenuNamePoint()         { return mMenuNamePoint; }

    cCaseNamePoint* GetIndexNamePoint();


    std::pair<int,std::string> IdNewPts(cCaseNamePoint * aCNP);

    void            ChangeFreeNamePoint(const std::string &, bool SetFree);

    void            _DeletePoint(cSP_PointGlob *);

    void            SetInvisRef(bool aVal);         // sert � rendre les points r�fut�s visibles ou non

    void            AddUndo(cOneSaisie * aSom);

private:

    void            Init();

    cWinIm *        WinImOfW(Video_Win);

    cWinIm *        mCurWinIm;

    std::vector<cWinIm *> mWins;

    Video_Display *       mDisp;

    Video_Win *           mWZ;
    cFenOuiNon *          mZFON;
    cFenMenu *            mMenuNamePoint;
    Video_Win *           mWEnter;

};
#endif 


class cAppli_SaisiePts
{
    public :

    cAppli_SaisiePts( cResultSubstAndStdGetFile<cParamSaisiePts> aParam, bool instanceInterface = true);
    const cParamSaisiePts &             Param() const;
    const std::string &                 DC()    const;     // directory chantier
    cInterfChantierNameManipulateur *   ICNM()  const;

    void ErreurFatale(const std::string &);

    cImage *                ImageOfNameSVP(const std::string & aName);
    cSP_PointGlob *         PGlobOfNameSVP(const std::string & aName);
    cSetOfSaisiePointeIm  & SOSPI();
    bool                    HasOrientation() const;

    void Undo();
    void Redo();
    void Sauv();
    void Exit();

    void AddUndo(cOneSaisie,cImage *);

    void ChangeImages
    (
        cSP_PointGlob * PointPrio,
        const std::vector<cWinIm *>  &  W2Ch
    );


    double  StatePriority(eEtatPointeImage aState);
    bool    Visible(cSP_PointeImage &);


    const Pt2di &       SzRech() const;
    Pt2di &             DecRech();
    Im2D_INT4           ImRechVisu() const;
    Im2D_INT4           ImRechAlgo() const;



    // 0 si existe deja
    cSP_PointGlob *     AddPointGlob(cPointGlob aPG,bool OkRessucite=false,bool Init=false,bool ReturnAlways=false);
    void                AddPGInAllImage(cSP_PointGlob * aSPG);

    void HighLightSom(cSP_PointGlob *);

    bool & ShowDet();

    void GlobChangStatePointe(const std::string & aName,const eEtatPointeImage aState);

    void ChangeName(std::string  anOldName,std::string  aNewName);

    cVirtualInterface * Interface() { return mInterface; }

    void SetInterface( cVirtualInterface * interf ) { mInterface = interf ;}

    int             nbImages()  { return mNbIm; }

    int             GetCptMax() const;

    cImage*         images(int aK) { return mImages[aK]; }

    std::vector< cSP_PointGlob * > PG() { return mPG; }

    private :

         void RenameIdPt(std::string &);


         void UndoRedo(std::vector<cUndoRedo>  & ToExe ,std::vector<cUndoRedo>  & ToPush); //UTILISE L'INTERFACE ReaffAllW();


         void InitImages();

         void InitInPuts();
         void AddOnePGInImage(cSP_PointGlob * aSPG,cImage & anI);


         void InitPG();
         void IniPointeIm();

         cParamSaisiePts &                     mParam;
         cVirtualInterface*                    mInterface;

         cInterfChantierNameManipulateur *     mICNM;
         std::string                           mDC;
         std::vector<cImage *>                 mImages;
         std::map<std::string,cImage *>        mMapIms;

         cSetPointGlob                         mSPG;
         std::vector<cSP_PointGlob *>          mPG;
         std::map<std::string,cSP_PointGlob *> mMapPG;


         cSetOfSaisiePointeIm              mSOSPI;

         int                               mNbIm;

         std::string                       mNameSauvPtIm;
         std::string                       mDupNameSauvPtIm;

         std::string                       mNameSauvPG;
         std::string                       mDupNameSauvPG;

         bool                              mShowDet;

         std::vector<cUndoRedo>            mStackUndo;
         std::vector<cUndoRedo>            mStackRedo;

         Pt2di                             mSzRech;
         Pt2di                             mDecRech;
         Im2D_INT4                         mImRechVisu;
         Im2D_INT4                         mImRechAlgo;
          
};



}

#endif //  _ELISE_SAISIEPTS_ALL_H_




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
