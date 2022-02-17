#ifndef INITOUTIL_H
#define INITOUTIL_H

#include <stdio.h>
#include "Triangle.h"
#include "Pic.h"



/* ** PlyFile.h est maintenante inclus dans StdAfx.f du MicMac, dans include/general */
/*
 * *IL FAULT MISE INCLUDE DU OPENCV AVANT INCLUDE DU StdAfx.h
 * IL FAULT DESACTIVE L'OPTION WITH_HEADER_PRECOMP DANS MICMAC CMAKE
 * Si il trouve pas OpenCV sharedlib => export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
 */

typedef struct CplString
{
    string img1;
    string img2;
}CplString;

typedef struct CplPic
{
    pic * pic1;
    pic * pic2;
}CplPic;

typedef struct AJobCorel
{
    triangle * tri;
    pic * picM;
    pic * pic2nd;
    vector<pic*> lstPic3rd;
}AJobCorel;

typedef bool(*dsPt2drCompFunc)( Pt2dr const &,  Pt2dr const &);
typedef bool(*dsPt2diCompFunc)( Pt2di const &,  Pt2di const &);

extern void dispTriSurImg(Tri2d TriMaitre, pic * ImgMaitre ,Tri2d Tri2nd, pic * Img2nd, Pt2dr centre, double size, vector<Pt2dr> & listPtsInteret, bool dispAllPtsInteret = false);
extern Video_Win * display_image( Im2D<U_INT1,INT4> *ImgIm2D,
                    string nameFenetre, Video_Win *thisVWin, double zoomF=1, bool click=true);
extern void display_2image( Im2D<U_INT1,INT4> *Img1,
                            Im2D<U_INT1,INT4> *Img2,
                            Video_Win *Win1, Video_Win *Win2,
                            double zoomF);
extern Video_Win * draw_polygon_onVW(vector<Pt2dr> pts, Video_Win* VW, Pt3di color=Pt3di(0,255,0), bool isFerme = true, bool click = true);
extern Video_Win * draw_polygon_onVW(Tri2d &aTri, Video_Win* VW, Pt3di color=Pt3di(0,255,0), bool isFerme = true, bool click = true);
extern Video_Win* draw_polygon_onVW(Pt2dr ptHGCaree, int szCaree, Video_Win* VW, Pt3di color=Pt3di(0,255,0), bool isFerme = true, bool click = true);
extern vector<double> parse_dParam(vector<string> dParam);
extern Video_Win* draw_pts_onVW(vector<Pt2dr> lstPts, Video_Win* VW, Pt3di color=Pt3di(0,255,255));
extern Video_Win* draw_pts_onVW(Pt2dr aPts, Video_Win* VW, string colorName);
extern bool comparatorPt2dr ( Pt2dr const &l,  Pt2dr const &r);
extern bool comparatorPt2drY ( Pt2dr const &l,  Pt2dr const &r);
extern bool comparatorPt2drAsc ( Pt2dr const &l,  Pt2dr const &r);
extern bool comparatorPt2drYAsc ( Pt2dr const &l,  Pt2dr const &r);
extern void sortDescendPt2drX(vector<Pt2dr> & input);
extern void sortDescendPt2drY(vector<Pt2dr> & input);
extern void sortAscendPt2drX(vector<Pt2dr> & input);
extern void sortAscendPt2drY(vector<Pt2dr> & input);


extern std::string intToString ( int number );
extern void MakeXMLPairImg(vector<CplString> & cplImg, string & outXML);
extern vector<CplString> fsnPairImg(vector<CplString> & cplImg);

class InitOutil
{
	public:
        InitOutil           (string aFullPattern, string aOriInput,
                             string aTypeD, vector<double> aParamD,
                             string aHomolOutput,
                             int SzPtCorr, int SzAreaCorr,
                             double corl_seuil_glob, double corl_seuil_pt,
                             bool disp, bool aCplPicExistHomol, double pas = 0.5, bool assume1er=false);
        InitOutil           (string aFullPattern, string aOriInput, string aHomolInput = "Homol");
        InitOutil           (string aMeshToRead);
        PlyFile* read_file  (string pathPlyFileS);
        vector<pic*> load_Im();
        vector<triangle *> load_tri ();
        void reprojectAllTriOnAllImg();
        void initHomoPackVide(bool creatLikeHomoPackInit);
        void initAll(string pathPlyFileS = "NO");
        vector<pic*> getmPtrListPic(){return mPtrListPic;}
        vector<triangle*> getmPtrListTri(){return mPtrListTri;}
		
        string getPrivMember(string aName);
        cInterfChantierNameManipulateur * getPrivmICNM(){return mICNM;}
        vector<double> getParamD() {return mParamD;}

        void addToExistHomolFile(   pic * pic1,
                                    pic * pic2,
                                    vector<ElCplePtsHomologues> ptsHomo,
                                    string aHomolOut, bool addInverse = false);
        void writeToHomolFile(   pic * pic1,
                                 pic * pic2,
                                 vector<ElCplePtsHomologues> ptsHomo,
                                 string aHomolOut);
        vector<CplPic> getmCplHomolExist(){return mCplHomolExist;}
        vector<CplPic> loadCplPicExistHomol(); 
        void getLstJobCorrel(vector<AJobCorel> & jobCorrel){jobCorrel = mLstJobCorrel;}
        void creatJobCorrel(double angleF , vector<cXml_TriAngulationImMaster> &lstJobTriAngulationImMaster);
        void creatJobCorrel(double angleF);

        int mSzPtCorr;
        int mSzAreaCorr;
        double mPas;
        double mCorl_seuil_glob;
        double mCorl_seuil_pt;
        bool mDisp;
        bool mDebugByClick;
        bool mAssume1er;
	private:
        string mOriInput;
        string mFullPattern;
        string mDirImages, mPatIm;
        string mHomolOutput, mNameHomol, mKHOut, mKHOutDat, mKHIn;
        cInterfChantierNameManipulateur * mICNM;
        string mTypeD; vector<double> mParamD;

        vector<Face*> mFaceList;         /* intensite, list vert_ind, n° vertex ind*/
        vector<Vertex*> mVertexList;    /* 3-space position of a vertex */
        PlyFile *mPly;

        vector<string> mSetIm;
        vector<pic*> mPtrListPic;
        vector<triangle*> mPtrListTri;
        vector<CplPic> mCplHomolExist;

        bool mCplPicExistHomol;

        vector<AJobCorel> mLstJobCorrel; //list process for method correlation with verification by 3rd img
};


#endif
