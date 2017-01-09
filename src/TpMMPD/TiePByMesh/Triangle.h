#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <stdio.h>
#include "StdAfx.h"
#include "../kugelhupf.h"

/* ** PlyFile.h est maintenante inclus dans StdAfx.f du MicMac, dans include/general */
/*
 * *IL FAULT MISE INCLUDE DU OPENCV AVANT INCLUDE DU StdAfx.h
 * IL FAULT DESACTIVE L'OPTION WITH_HEADER_PRECOMP DANS MICMAC CMAKE
 * Si il trouve pas OpenCV sharedlib => export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
 */

class pic;
struct Vertex {
    float x,y,z;             /* the usual 3-space position of a vertex */
};
typedef struct Tri2d {
    Pt2dr sommet1[3];
    bool insidePic;
}Tri2d;

struct Face {
    unsigned char intensity; /* this user attaches intensity to faces */
    unsigned char nverts;    /* number of vertex indices in list */
    int *verts;              /* vertex index list */
};

extern PlyProperty face_props[2];
extern bool comparatorPt2dr ( Pt2dr const &l,  Pt2dr const &r);
extern bool comparatorPt2drY ( Pt2dr const &l,  Pt2dr const &r);
extern void sortPt2drDescendx(vector<Pt2dr> & input);
extern void sortPt2drDescendY(vector<Pt2dr> & input);

typedef struct matAffine {
    double el_00;double el_01;double el_02;
    double el_10;double el_11;double el_12;
} matAffine;

extern Pt2dr ApplyAffine (Pt2dr &Pts_Origin, matAffine &matrixAffine);

typedef struct PtInteretInTriangle {
    Pt2dr aPoint;
    pic * imageContainPtAndTriangle;
}PtInteretInTriangle;

typedef struct ImgetOfTri {
    Tri2d aTri;     //coor 2D of this tri
    pic * aPic;     //in which image this tri2D is exprime
    Pt2dr centre_geo;   //centre-geo of this tri in coor of aPic
    Pt2dr ptOriginImaget;   //pts origin of this imaget in coor of aPic (pt Haut Gauche)
    cCorrelImage * imaget;  //imaget of this tri
    int szW;    //size of imaget -- need *2+1 to have whole size
}ImgetOfTri;

/* ==== TRIANGLE : ==========
 * class pour decrire une triangle dans le fichier mesh
 * contient des méthode pour reprojecter le triangle vers image
*/
class triangle
{
public:
    triangle(Face* aFace, vector<Vertex*> VertexList, int num_pic, int index);      //definir triangle par ELEMENT face et VERTEX list
    void reproject(pic *aPic, bool &reprOK, Tri2d &result, int ind);    //reprojecter triangle 3d ver un pic
    Pt3dr getSommet(int i) {return mSommet[i];}                     //get sommet i dans cette triangle (une triangle a 3 sommet)
    //Vertex getSommetV(int i) {return *this->mPtrSommetV[i]; }             //get sommet i dans cette triangle sous type Vertex
    Tri2d **getReprSurImg() {return mResultTriReprSurAllImg;}       //pointer vers bloque de memoire contient reprojection de cette triange sur tout les images.
    ImgetOfTri create_Imagette_autour_triangle (pic* aPic);
    ImgetOfTri create_Imagette_autour_triangle_A2016 (pic* aPic);
    cCorrelImage* create_Imagette_adapt_triangle    ( pic * PicMaitre, Tri2d &triangle,
                                                      Pt2dr &centre_geo, bool & ok_in,
                                                      Pt2dr &PtOrigin);
    //pic* getPicPlusProche(vector<pic*>PtrListPic, Pt3dr & VecNormal, Pt3dr & PtsOrg, vector<pic*>&ptrListPicViewable, bool assum1er);
    //chercher image plus proche au vector normal du triangle 3d
    Pt3dr CalVecNormal(Pt3dr & returnPtOrg, double mulFactor);
    matAffine CalAffine(pic* pic1 , pic* pic2, bool & affineResolu);

    ImgetOfTri get_Imagette_by_affine_n(ImgetOfTri &ImgetMaitre,
                                        pic* Img2nd,
                                        matAffine &matrixAffine,
                                        bool &getImgetteFalse);
    double calAngle(Pt3dr Vec1, Pt3dr Vec2);
    vector<Pt3dr> ptsInTri2Dto3D(vector<Pt2dr> pts2DinTri, pic *img);
    //void savePtInteret2D(Pt2dr aPoint, pic*img);
    //vector<PtInteretInTriangle> getPtsInteret2DInImagetteDuTri(){return mPtsInteret2DInImagetteDuTri;}

    //bool check_inside_triangle (Pt2dr v, Tri2d aTri2D);
    bool check_inside_triangle_A2016 (Pt2dr aPt, Tri2d & aTri);
    double angleToVecNormalImg(pic* aPic);
    int mIndex;                                                           //index of this triangle in PtrListTri
private:
    Pt2dr expPtInRepTri2D(Pt2dr aPt, Tri2d & aTri);
    cCorrelImage * create_Imagette_autour_triangle_i(pic * PicMaitre,
                                                   Tri2d & triangle,
                                                   Pt2dr &centre_geo,
                                                   bool & ok_in, Pt2dr &PtOrigin);
    cCorrelImage * get_Imagette_by_affine(Tri2d &TriReprMaitre,
                                          cCorrelImage *Imgette_Maitre,
                                          pic* Img2nd,
                                          matAffine &matrixAffine, Pt2dr &centre_geo_master,
                                          bool &getImgetteFalse);         //get imagette of 2nd pic by affine and imagette from pic master
    //vector<Vertex> toTriangle(Face* aFace, vector<Vertex*> VertexList ); //creer triangle a partir de face qui a plus de 3 vertex
    double det(Pt2dr u, Pt2dr v);


    Pt3dr mSommet[3];                                                     //sommet du triangle en Pt3d
    //Vertex* mPtrSommetV[3];                                               //sommet du triangle en Vertex
    //vector<Pt2dr*> resultAffine;                                         //matrix de affine transformation sur tout les pic
    Tri2d ** mResultTriReprSurAllImg;                                     //pointer vers bloque de memoire contient reprojection de cette triange sur tout les images
    //vector<PtInteretInTriangle> mPtsInteret2DInImagetteDuTri;
};
/*
typedef struct Rect {
    Pt2dr diag[2];  //2 point du diagonal de la rectangle
    vector<triangle*> desTriInRect;
}Rect;
*/

/*======= Les fonction supplementaire =======*/

#endif
