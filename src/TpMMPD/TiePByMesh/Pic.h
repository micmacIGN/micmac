#ifndef PIC_H
#define PIC_H

#include <stdio.h>
#include "StdAfx.h"
#include "Triangle.h"

/* ** PlyFile.h est maintenante inclus dans StdAfx.f du MicMac, dans include/general */
/*
 * *IL FAULT MISE INCLUDE DU OPENCV AVANT INCLUDE DU StdAfx.h
 * IL FAULT DESACTIVE L'OPTION WITH_HEADER_PRECOMP DANS MICMAC CMAKE
 * Si il trouve pas OpenCV sharedlib => export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
 */
typedef struct PackHomo
{
    ElPackHomologue aPack;
    int indPic1;
    int indPic2;
}PackHomo;

/* ===== PIC : ====== PIC : ========= PIC : ======== PIC : ======== PIC : ======== PIC : ======= PIC :======
 * class pour decrire une image
 * header tif, path, info, taille...
 * affichier img
 * test si point se situe dans image
*/
class pic
{
public:
    pic (    const string * nameImg, string nameOri,
             cInterfChantierNameManipulateur * aICNM,
             int indexInListPic);                     //definir pic avec orientation

    bool checkInSide(Pt2dr aPoint);                   //verifie si point est dans pic
    string getNameImgInStr(){return mNameImg->c_str();}
    void AddPtsToPack(pic* Pic2nd, const Pt2dr &Pts1, const Pt2dr &Pts2);

    vector<Pt2dr> getPtsHomoInThisTri(triangle* aTri);

    int mIndex;
    CamStenope * mOriPic;                                                                    //orientation
    cInterfChantierNameManipulateur * mICNM;                                   //name manipulator
    const string * mNameImg;
    vector<triangle*> allTriangle;     //tout les triangle dans le mesh

    Pt2di mImgSz;
    Tiff_Im *mPicTiff;
    TIm2D<U_INT1,INT4> *mPic_TIm2D;
    Im2D<U_INT1,INT4> *mPic_Im2D;

    vector<Pt2dr> mListPtsInterestFAST;
    ElPackHomologue mPackFAST;
    vector<PackHomo> mPackHomoWithAnotherPic;
private:
    vector<Rect> mDesGrill;             //des grill d'image et index des rectangles dans chaque grill
};
#endif
