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

#ifndef _VINO_H_
#define _VINO_H_

#include "StdAfx.h"

#if (ELISE_X11)

#define TheNbMaxChan  10


   //======================== A externaliser  ====================

std::string StrNbChifSignNotSimple(double aVal,int aNbCh);
std::string StrNbChifSign(double aVal,int aNbCh);
std::string SimplString(std::string aStr);
std::string StrNbChifApresVirg(double aVal,int aNbCh);

Im2D_U_INT1 Icone(const std::string & aName,const Pt2di & aSz,bool Floutage,bool Negatif);
void PutFileText(Video_Win,const std::string &);


void CorrectRect(Pt2di &  aP0,Pt2di &  aP1,const Pt2di & aSz);
void FillStat(cXml_StatVino & aStat,Flux_Pts aFlux,Fonc_Num aFonc);
bool TreeMatchSpecif(const std::string & aNameFile,const std::string & aNameSpecif,const std::string & aNameObj);



class cCaseX11Xml
{
    public :
       void Efface();
       void Efface(int aCoul);
       void Efface(Col_Pal aCoul);
       static cCaseX11Xml * Alloc(Video_Win aW,Box2di aBox,int aCoul);
       void string(int aPos,const std::string & );
       bool Inside(const Pt2di &) const;
       static int  GetCase(const std::vector<cCaseX11Xml *> &,const Pt2di &);
       Pt2di P0Line() ;
       Clik clik_in();
    private :
       cCaseX11Xml(Video_Win aW,Box2di aBox,int aCoul);
       Video_Win mW;
       Box2di    mBox;
       int       mCoul;
};



class cWXXInfoCase
{
    public :
        cWXXInfoCase(cElXMLTree * aTree,cElXMLTree * aFilter);

        cElXMLTree  * mTree;
        cElXMLTree  * mFilter;
        bool          mModified;
};


class cWXXTreeSelector
{
    public :
       virtual bool SelectTree(cElXMLTree *);
};

class cWindowXmlEditor
{
     public :
         cWindowXmlEditor(Video_Win aW,bool aXmlMode,cElXMLTree * aTree,cWXXTreeSelector * aSelector,cElXMLTree * aFilter=0);
         Box2di  TopDraw();
         void Interact();
     private :


         void ShowQuit();
         void ShowWarn(const std::string& aMes1, const std::string& aMes2);

         int  EndXOfLevel(int aLevel);
         Box2di  PrintTag(Pt2di aP0,cElXMLTree *,int aMode,int aLev,cElXMLTree * aFilter) ; // 0 => terminal, 1 ouvrant , 2 fermant
         Box2di  Draw(Pt2di ,cElXMLTree * aTree ,int aLev,cElXMLTree * aFilter);
         void ModifyCase(cCaseX11Xml * aCase,int aK);

         bool                      mFirstDraw;
         Video_Win                 mW;
         bool                      mXmlMode;
         cElXMLTree *              mTreeGlob;
         cWXXTreeSelector *        mSelector;
         cElXMLTree *              mFilterGlob;
         // std::vector<cElXMLTree *> mTrees;

         Pt2di                     mPRab;
         std::vector<cCaseX11Xml*> mVCase;
         std::vector<cWXXInfoCase> mVInfoCase;
         cCaseX11Xml *             mCaseQuit;
         cCaseX11Xml *             mCaseWarn;
         int                       mGrayFond;
         int                       mGrayTag;
         int                       mSpaceTag;
         int                       mDecalX;
         bool                      mModeCreate;
};





   //======================== Specif Vino ====================


class cWXXVinoSelector : public cWXXTreeSelector
{
     public  :
           cWXXVinoSelector(const std::string & aName);
           bool SelectTree(cElXMLTree *);
     private :
          std::string mName;
};







typedef enum
{
   eModeGrapZoomVino,
   eModeGrapTranslateVino,
   eModeGrapAscX,
   eModeGrapAscY,
   eModeGrapShowRadiom,
   eModeVinoPopUp
}  eModeGrapAppli_Vino;


class cPopUpMenuMessage : public PopUpMenuTransp
{
   public :
      cPopUpMenuMessage(Video_Win aW,Pt2di aSz) ;
      void ShowMessage(const std::string & aName, Pt2di aP,Pt3di aCoul);
      void Hide();

};


class cAppli_Vino;
template <class Type> class cAppli_Vino_TplChgDyn
{
    public :
       static void SetDyn(cAppli_Vino &,int * anOut,const Type * anInput,int aNb);
};



class cAppli_Vino : public cXml_EnvVino,
                    public Grab_Untill_Realeased ,
                    public cElScrCalcNameSsResol,
                    public cImgVisuChgDyn
{
     public :
        friend class cAppli_Vino_TplChgDyn<double>;
        friend class cAppli_Vino_TplChgDyn<int>;


        bool Floutage() {return false;} // A mettre dans cXml_EnvVino,
        cAppli_Vino(int,char **);
        void PostInitVirtual();
        void  Boucle();
        cXml_EnvVino & EnvXml() {return static_cast<cXml_EnvVino &> (*this);}


     private :
        Box2di PutMessage(Pt2dr ,const std::string & aMes,int aCoulText,Pt2dr aSzRelief = Pt2dr(-1,-1),int aCoulRelief=-1);
        void   PutMessageRelief(int aK,const std::string & aMes);
        
        void ChgDyn(int * anOut,const int * anInput,int aNb) ;
        void ChgDyn(int * anOut,const double * anInput,int aNb) ;
        void SaveState();
        void  MenuPopUp();
        void InitMenu();
        void GrabShowOneVal();
        void ShowOneVal(Pt2dr aP);
        void EffaceMessageVal();
        void EffaceMessageRelief();
        void EffaceMessages(std::vector<Box2di> &);
        void Efface(const Box2di & aBox);
        void HistoSetDyn();
        void Refresh();
        void InitTabulDyn();
        void ZoomRect();
        void Help();
        void EditData();
        void DoHistoEqual(Flux_Pts aFlux);

        ElList<Pt2di> GetPtsImage(bool GlobScale,bool ModeRect,bool AcceptPoint);


        bool OkPt(const Pt2di & aPt);
        void End();
        CaseGPUMT * CaseBase(const std::string&,const Pt2di aNumCase);
        ChoixParmiCaseGPUMT * CaseChoix( ChoixParmiCaseGPUMT * aCaseBase,const std::string&,const Pt2di aNumCase,int aNumVal);


        void SetInterpoleMode(eModeInterpolation,bool DoRefresh);



        cXml_StatVino  StatRect(Pt2di &  aP0,Pt2di &  P1);

        std::string NamePyramImage(int aZoom);
        std::string  CalculName(const std::string & aName, INT InvScale); // cElScrCalcNameSsResol

        void  GUR_query_pointer(Clik,bool);
        void ExeClikGeom(Clik);
        void ZoomMolette();
        void ShowAsc();
        Pt2dr ToCoordAsc(const Pt2dr & aP);

        std::string               mNameXmlOut;
        std::string               mNameXmlIn;
        std::string               mDir;
        std::string               mNameIm;
        Tiff_Im  *                mTiffIm;
        std::string               mNameTiffIm;
        Pt2di                     mTifSz;
        bool                      mCoul;
        int                       mNbChan;
        double                    mNbPix;
        double                    mRatioFul;
        Pt2dr                     mRatioFulXY;

        Pt2di                     mSzIncr;
        Video_Win *               mWAscH;
        Video_Win *               mW;
        Video_Win *               mWHelp;
        Video_Win *               mWAscV;
        Video_Display *           mDisp;
        std::string               mTitle;
        Visu_ElImScr *            mVVE;
        ElImScroller *            mScr;
        std::vector<INT>          mVEch;
        Pt2dr                     mP0Click;
        std::vector<Box2di>       mVBoxMessageVal;
        std::vector<Box2di>       mVBoxMessageRelief;
        double                    mScale0;
        Pt2dr                     mTr0;
        int                       mBut0;
        bool                      mCtrl0;
        bool                      mShift0;
        eModeGrapAppli_Vino       mModeGrab;

        double                    mNbPixMinFile;
        double                    mSzEl;


         // Menus contextuels

        Pt2di                   mSzCase;
        GridPopUpMenuTransp*    mPopUpBase;
        CaseGPUMT *             mCaseExit;
        CaseGPUMT *             mCaseZoomRect;
        CaseGPUMT *             mCaseEdit;
        ChoixParmiCaseGPUMT *   mCaseInterpPpv;
        ChoixParmiCaseGPUMT *   mCaseInterpBilin;
        CaseGPUMT *             mCaseHStat;
        CaseGPUMT *             mCaseHMinMax;
        CaseGPUMT *             mCaseHEqual;

        GridPopUpMenuTransp*    mPopUpCur;
        CaseGPUMT *             mCaseCur;
        eModeInterpolation      mMode;
        cXml_StatVino *         mCurStats;
        bool                    mStatIsInFile;

        // Etalement  des tabulation
        bool                    mTabulDynIsInit;
        double                  mV0TabulDyn;
        double                  mStepTabulDyn;
        std::vector<int>        mTabulDyn;

        int         mNbHistoMax;
        int         mNbHisto;
        double      mVMaxHisto;
        Im1D_REAL8  mHisto;
        Im1D_REAL8  mHistoLisse;
        Im1D_REAL8  mHistoCum;
        std::string mNameHisto;
};

#endif



#endif // _VINO_H_

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
