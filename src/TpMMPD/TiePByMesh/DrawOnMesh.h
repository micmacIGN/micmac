#ifndef DRAWONMESH_H
#define DRAWONMESH_H

#include "InitOutil.h"
#include <stdio.h>

class DrawOnMesh
{
public:
    DrawOnMesh(InitOutil *aChain);
    DrawOnMesh();
    vector<Pt3dr> drawPt3DByPt2DAndTriangle(triangle * tri, vector<PtInteretInTriangle> ptsInteret2DInTri, pic *img);
    Pt3dr drawPt3DByInterPts2DManyImgs(vector<Pt2dr> pts2D, vector<pic*> img);
    vector<Pt3dr> drawPackHomoOnMesh(ElPackHomologue aPack, pic* pic1, pic* pic2, Pt3dr color, string suffix);
    double countPtsSurImg(pic* img);
    void drawTri3DAndAllPts3DInTri(triangle * tri, pic *img);
    Pt3dr InterFaisce_cpy_cPI_Appli(
                                    const std::vector<CamStenope *> & aVCS,
                                    const std::vector<Pt2dr> & aNPts2D
                                   );   //copy appli Mehdi
    void drawEdge_p1p2  (   vector<Pt3dr> &p1Vec,
                            vector<Pt3dr> &p2Vec,
                            string filename,
                            Pt3dr colorRGBVer,
                            Pt3d<double> colorRGBEdge);
    void drawListTriangle(
                           vector<triangle*> listTri,
                           string fileName,
                           Pt3dr colorRGB
                          );
    void drawListTriangle(  vector<cXml_Triangle3DForTieP> listTri,
                            string fileName,
                            Pt3dr colorRGB
                         );
    void drawListPtsOnPly(
                            vector<Pt3dr> lstPts,
                            string filename,
                            Pt3dr colorRGB
                         );
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
    void drawListTriangle(
                            vector<vector<Pt3dr> > listTri,
                            string fileName,
                            Pt3dr colorRGB
                         );

private:
    vector<triangle*> mPtrListTri;
    vector<pic*> mPtrListPic;
    InitOutil * mChain;

};

#endif
