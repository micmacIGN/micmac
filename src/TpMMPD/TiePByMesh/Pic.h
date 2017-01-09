#ifndef PIC_H
#define PIC_H

#include <stdio.h>
#include "Triangle.h"
#include <iterator>

/* ** PlyFile.h est maintenante inclus dans StdAfx.f du MicMac, dans include/general */
/*
 * *IL FAULT MISE INCLUDE DU OPENCV AVANT INCLUDE DU StdAfx.h
 * IL FAULT DESACTIVE L'OPTION WITH_HEADER_PRECOMP DANS MICMAC CMAKE
 * Si il trouve pas OpenCV sharedlib => export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
 */
extern void StdCorrecNameHomol_G(std::string & aNameH,const std::string & aDir);
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

    bool checkInSide(Pt2dr aPoint,int aRab=0);                   //verifie si point est dans pic
    string getNameImgInStr(){return mNameImg->c_str();}
    void AddPtsToPack(pic* Pic2nd, const Pt2dr &Pts1, const Pt2dr &Pts2);
    void AddVectorPtsToPack(pic* Pic2nd, vector<Pt2dr> & Pts1, vector<Pt2dr> & Pts2);
    void AddVectorCplHomoToPack(pic* Pic2nd, vector<ElCplePtsHomologues> aHomo);
    void getPtsHomoOfTriInPackExist(string aKHIn,
                                         triangle * aTri, pic * pic1st, pic* pic2nd ,
                                         vector<ElCplePtsHomologues> & result);
    triangle * whichTriangle(Pt2dr & ptClick, bool & found);
    double calAngleViewToTri(triangle *aTri);
    void getTriVisible(vector<triangle *> &lstTri, double angleF, bool Zbuf = false);
    void getTriVisibleWithPic(vector<triangle*> & lstTri, double angleF,
                                   pic * pic2, vector<triangle *> &triVisblEnsmbl, bool Zbuf = false);
    void getPtsHomoInThisTri(triangle* aTri , vector<Pt2dr> & lstPtsInteret, vector<Pt2dr> & result);
    void roundPtInteret();
    double distDuTriauCtrOpt(triangle * aTri);
    void whichTrianglecontainPt(Pt2dr aPt, vector<triangle *> lstTri, vector<triangle *> result, bool & found);
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
    vector<triangle*>triVisible;
    vector<double> triVisibleInd;
private:
    //vector<Rect> mDesGrill;             //des grill d'image et index des rectangles dans chaque grill
};
#endif
