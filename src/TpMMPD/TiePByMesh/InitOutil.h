#ifndef INITOUTIL_H
#define INITOUTIL_H

#include <stdio.h>
#include "StdAfx.h"
#include "Triangle.h"
#include "Pic.h"

/* ** PlyFile.h est maintenante inclus dans StdAfx.f du MicMac, dans include/general */
/*
 * *IL FAULT MISE INCLUDE DU OPENCV AVANT INCLUDE DU StdAfx.h
 * IL FAULT DESACTIVE L'OPTION WITH_HEADER_PRECOMP DANS MICMAC CMAKE
 * Si il trouve pas OpenCV sharedlib => export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
 */

extern void dispTriSurImg(Tri2d TriMaitre, pic * ImgMaitre ,Tri2d Tri2nd, pic * Img2nd, Pt2dr centre, double size, vector<Pt2dr> & listPtsInteret, bool dispAllPtsInteret = false);
extern std::string intToString ( int number );
class InitOutil
{
	public:
        InitOutil           (string aFullPattern, string aOriInput,
                             string aTypeD, vector<double> aParamD);
        PlyFile* read_file  (string pathPlyFileS);
        vector<pic*> load_Im();
        vector<triangle *> load_tri ();
        void reprojectAllTriOnAllImg();
        void initHomoPackVide(bool creatLikeHomoPackInit);
        void initAll(string pathPlyFileS);

        vector<pic*> getmPtrListPic(){return mPtrListPic;}
        vector<triangle*> getmPtrListTri(){return mPtrListTri;}
		
        string getPrivMember(string aName);
        cInterfChantierNameManipulateur * getPrivmICNM(){return mICNM;}
        vector<double> getParamD() {return mParamD;}
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
};


#endif
