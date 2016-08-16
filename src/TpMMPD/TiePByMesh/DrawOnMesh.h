#ifndef DRAWONMESH_H
#define DRAWONMESH_H

#include <stdio.h>
#include "StdAfx.h"
#include "InitOutil.h"

class DrawOnMesh
{
public:
    DrawOnMesh(InitOutil *aChain);
    vector<Pt3dr> drawPt3DByPt2DAndTriangle(triangle * tri, vector<PtInteretInTriangle> ptsInteret2DInTri, pic *img);
    Pt3dr drawPt3DByInterPts2DManyImgs(vector<Pt2dr> pts2D, vector<pic*> img);
    vector<Pt3dr> drawPackHomoOnMesh(ElPackHomologue aPack, pic* pic1, pic* pic2, Pt3dr color, string suffix);
    double countPtsSurImg(pic* img);
    void drawTri3DAndAllPts3DInTri(triangle * tri, pic *img);
    Pt3dr InterFaisce_cpy_cPI_Appli(
                                    const std::vector<CamStenope *> & aVCS,
                                    const std::vector<Pt2dr> & aNPts2D
                                    );   //copy appli Mehdi
private:
    void creatPLYPts3D  (
                            vector<Pt3dr> pts3DAllTri,
                            vector<Pt3dr> listFace3D,
                            string fileName,
                            Pt3dr colorRGB
                        );
    void creatPLYPts3D  (
                           vector<Pt3dr> pts3DAllTri,
                           string fileName,
                           Pt3dr colorRGB
                        );

    vector<triangle*> mPtrListTri;
    vector<pic*> mPtrListPic;
    InitOutil * mChain;

};

#endif
