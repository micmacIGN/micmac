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

#ifndef __SCHNAPS_H__
#define __SCHNAPS_H__

#include "StdAfx.h"
#include <map>

/**
 * Schnaps : reduction of homologue points in image geometry
 *    S trict
 *    C hoice of
 *    H omologous
 *    N eglecting 
 *    A ccumulations on
 *    P articular
 *    S pots
 * 
 *
 * Inputs:
 *  - pattern of images
 *  - Homol dir
 *  - minimal number of searching windows in each picture
 *  - new homol dir name
 *
 * Output:
 *  - new homol dir
 *
 * Call example:
 *   mm3d Schnaps ".*.tif" NbWin=100 HomolOut=Homol_mini100
 *
 * Info: jmmuller
 * 
 * */

//#define ReductHomolImage_DEBUG
//#define ReductHomolImage_VeryStrict_DEBUG


//----------------------------------------------------------------------------
// PicSize class: only to compute windows size for one given picture size
class cPicSize
{
  public:
    cPicSize(Pt2di aSz);
    Pt2di getPicSz(){return mPicSz;}
    Pt2di getWinSz(){return mWinSz;}
    Pt2di getNbWin(){return mNbWin;}
    int getUsageBuffer(){return mUsageBuffer;}

    static int mTargetNumWindows;//the actual number of windows will depend on the picture ratio
  protected:
    Pt2di mPicSz;
    Pt2di mWinSz;
    Pt2di mNbWin;
    int mUsageBuffer;//buffer for image usage check
  };


//----------------------------------------------------------------------------
// PointOnPic class: an Homol point with coordinates on several pictures
class cPic;
class cHomol;
class cPointOnPic
{
  public:
    cPointOnPic();
    cPointOnPic(const cPointOnPic &o);
    cPointOnPic(cPic *aPic,Pt2dr aPt,cHomol* aHomol);
    int getId(){return mId;}
    cPic* getPic(){return mPic;}
    Pt2dr& getPt(){return mPt;}
    cHomol* getHomol(){return mHomol;}
    void setHomol(cHomol* aHomol){mHomol=aHomol;}
    void print();
  protected:
    int mId;//unique id
    static int mPointOnPicCounter;
    cPic* mPic;
    Pt2dr mPt;
    cHomol* mHomol;//the Homol it came from
    
};

//----------------------------------------------------------------------------
// Homol class: an Homol point with coordinates on several pictures
class cHomol
{
  public:
    cHomol();
    ~cHomol();
    int getId(){return mId;}
    bool alreadyIn(cPic * aPic, Pt2dr aPt);
    bool add(cPic * aPic, Pt2dr aPt);//return true if not already in
    void add(cHomol *aHomol);//merge with an other homol
    void print();
    bool isBad(){return mBad;}
    void setBad(){mBad=true;}
    cPointOnPic* getPointOnPic(cPic * aPic);
    cPointOnPic* getPointOnPic(unsigned int i){return (mPointOnPics.at(i));}
    unsigned int getPointOnPicsSize(){return mPointOnPics.size();}
    //part for strict filtering
    void addAppearsOnCouple(cPic * aPicA,cPic * aPicB);
    bool appearsOnCouple(cPic * aPicA,cPic * aPicB);//exactly this order
    bool appearsOnCouple2way(cPic * aPicA,cPic * aPicB);//both ways
    bool appearsOnCouple1way(cPic * aPicA,cPic * aPicB);//one way is enougth
    cPic* getAppearsOnCoupleA(unsigned int i){return mAppearsOnCoupleA[i];}
    cPic* getAppearsOnCoupleB(unsigned int i){return mAppearsOnCoupleB[i];}
    int getAppearsOnCoupleSize(){return mAppearsOnCoupleA.size();}
    bool checkMerge(cHomol* aHomol);
  protected:
    int mId;//unique id
    static int mHomolCounter;
    std::vector<cPointOnPic*> mPointOnPics;
    bool mBad;
    std::vector<cPic*> mAppearsOnCoupleA;//record in which couple A-B it has been seen
    std::vector<cPic*> mAppearsOnCoupleB;
};


//----------------------------------------------------------------------------
// Pic class: a picture with its homol points
class cPic
{
  public:
    cPic(std::string aDir, std::string aName);
    cPic(cPic* aPic);
    std::string getName(){return mName;}
    cPicSize * getPicSize(){return mPicSize;}
    bool removeHomolPoint(cPointOnPic* aPointOnPic);
    void printHomols();
    bool addSelectedPointOnPicUnique(cPointOnPic* aPointOnPic);
    float getPercentWinUsed(int nbWin);
    void setWinUsed(int _x,int _y);
    cPointOnPic * findPointOnPic(Pt2dr & aPt);
    void addPointOnPic(cPointOnPic* aPointOnPic);
    int getAllPointsOnPicSize(){return mAllPointsOnPic.size();}
    int getAllSelectedPointsOnPicSize(){return mAllSelectedPointsOnPic.size();}
    std::map<double,cPointOnPic*>  * getAllSelectedPointsOnPic(){return &mAllSelectedPointsOnPic;}
    std::map<cPic*, long> * getNbRawLinks(){return &nbRawLinks;}
    void selectHomols();
    void selectAllHomols();
    void fillPackHomol(cPic* aPic2,string & aDirImages,cInterfChantierNameManipulateur * aICNM,std::string & aKHOut);
    std::vector<int> getStats(bool before=true);//computes repartition in homol-2, homol-3, homol-4 etc... export it as vector indexed by multiplicity
    long getId(){return mId;}
    static std::vector<cPicSize>* getAllSizes(){return &mAllSizes;}
  protected:
    std::string mName;
    cPicSize * mPicSize;
    std::map<double,cPointOnPic*> mAllPointsOnPic;//key: x+mPicSize->getPicSz().x*y
    std::map<double,cPointOnPic*> mAllSelectedPointsOnPic;//key: x+mPicSize->getPicSz().x*y
    std::map<cPic*, long> nbRawLinks;//number of common points with other pictures
    std::vector<bool> mWinUsed;//to check repartition of homol points (with buffer)
    double makePOPKey(Pt2dr & aPt){return aPt.x*100+mPicSize->getPicSz().x*100*aPt.y*100;}
    long mId;
    static long mNbIm;
    static std::vector<cPicSize> mAllSizes;
};

//----------------------------------------------------------------------------

class CompiledKey2
{
  public:
    CompiledKey2(cInterfChantierNameManipulateur * aICNM,std::string aKH);
    std::string get(std::string param1,std::string param2);
    std::string getDir(std::string param1,std::string param2);
    std::string getFile(std::string param1,std::string param2);
    std::string getSuffix(){return mPart3;}
  protected:
    cInterfChantierNameManipulateur * mICNM;
    std::string mKH,mPart1,mPart2,mPart3;
};




void computeAllHomol(std::string aDirImages,
                     std::string aPatIm,
                     const std::vector<std::string> &aSetIm,
                     std::list<cHomol> &allHomolsIn,
                     CompiledKey2 &aCKin,
                     std::map<std::string,cPic*> &allPics,
                     bool veryStrict,
                     int aNumWindows);

#endif

/* Footer-MicMac-eLiSe-25/06/2007

   Ce logiciel est un programme informatique servant a  la mise en
   correspondances d'images pour la reconstruction du relief.

   Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
   respectant les principes de diffusion des logiciels libres. Vous pouvez
   utiliser, modifier et/ou redistribuer ce programme sous les conditions
   de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
   sur le site "http://www.cecill.info".

   En contrepartie de l'accessibilite au code source et des droits de copie,
   de modification et de redistribution accordes par cette licence, il n'est
   offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
   seule une responsabilite restreinte pese sur l'auteur du programme,  le
   titulaire des droits patrimoniaux et les concedants successifs.

   A cet egard  l'attention de l'utilisateur est attiree sur les risques
   associes au chargement, a l'utilisation, a la modification et/ou au
   developpement et a la reproduction du logiciel par l'utilisateur etant
   donne sa specificite de logiciel libre, qui peut le rendre complexe a
   manipuler et qui le reserve donc a des developpeurs et des professionnels
   avertis possedant  des  connaissances  informatiques approfondies.  Les
   utilisateurs sont donc invites a charger  et  tester  l'adequation  du
   logiciel a leurs besoins dans des conditions permettant d'assurer la
   securite de leurs systemes et ou de leurs donnees et, plus generalement,
   a l'utiliser et l'exploiter dans les memes conditions de securite.

   Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
   pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
   termes.
   Footer-MicMac-eLiSe-25/06/2007/*/
