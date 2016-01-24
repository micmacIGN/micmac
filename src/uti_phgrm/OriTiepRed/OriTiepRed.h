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


#if (1)  // Pour l'instant tout dans un namespace
#define  NS_OriTiePRed_BEGIN namespace OriTiePRed{
#define  NS_OriTiePRed_END };
#define  NS_OriTiePRed_USE using namespace OriTiePRed;

#else
#define  NS_OriTiePRed_BEGIN
#define  NS_OriTiePRed_END
#define  NS_OriTiePRed_USE
#endif




#ifndef _TiepRed_H_
#define _TiepRed_H_



#include "StdAfx.h"

NS_OriTiePRed_BEGIN

class cCameraTiepRed;
class cAppliTiepRed;
class cLnk2ImTiepRed;
class cPMulTiepRed;

/*
    cCameraTiepRed => geometry of camera,


    cPMulTiepRed => one multiple point (topology + ground geometry)

    cLnk2ImTiepRed  => link between two images (contains homologous points in the current box)

#####   cAppliTiepRed => The application ####


When executing 

    mm3d TestOscar Abbey-IMG_0.*jpg OriCalib=Ori-RTL2/
          void DoReduceBox();

    1/ The spliting in boxes is done, for each box K the set of image which footprint intersect the box is compted and
       store in Tmp-ReducTieP/Param_K.xml :

Exemple  of Param_K.xml

<Xml_ParamBoxReducTieP>
     <Box>-6.37202317345678271 -4.53617702320232041 -2.56975959785584473 -0.322177001523337836</Box>
     <Ims>Abbey-IMG_0173.jpg</Ims>
     <Ims>Abbey-IMG_0192.jpg</Ims>
     ....
<Xml_ParamBoxReducTieP>

       This is done in  void cAppliTiepRed::GenerateSplit();


    2/ Then the subprocess are executed
         mm3d TestOscar Abbey-IMG_0.*jpg OriCalib=Ori-RTL2/   KBox=1
         mm3d TestOscar Abbey-IMG_0.*jpg OriCalib=Ori-RTL2/   KBox=2
         ...

      This the same command, the programm "knows" that is called at the second level du the existence of KBox= 


       This is done in  void cAppliTiepRed::DoReduceBox();

###  !!!!!!!!  => One tricky thing 

The code read the tie points that comes from Martini because they are more memory efficient (float value + store on way);
But Martini store tie point corrected from distorsion, focale an principal (ideal sensor with F=1, PP=(0,0)), which means that
if (X,Y) is one point then (X,Y,1) is the direction of the bundle in the camera coordinate system.



*/

extern bool DEBUG_OTR;


typedef cVarSizeMergeTieP<Pt2df>  tMerge;
typedef cStructMergeTieP<tMerge>  tMergeStr;

class cCameraTiepRed
{
    public :
        cCameraTiepRed(cAppliTiepRed & anAppli,const std::string &,CamStenope *);
        const std::string NameIm() const;
 
        //  Intersection of bundles in ground geometry
        Pt3dr BundleIntersection(const Pt2df & aP1,const cCameraTiepRed & aCam2,const Pt2df & aP2,double & Precision) const;

        // return "standard" MicMac stenope camera
        CamStenope  & CS();
        // is the camera to maintains once the tie points are loaded
        bool  SelectOnHom2Im() const;
        const int &   NbPtsHom2Im() const;

        // Load the tie point between this and Cam2
        void LoadHom(cCameraTiepRed & aCam2);

        //  handle numeration of camera (associate a unique integer to each camera), because in topological merging ,
        // images are referenced by numbers
        void SetNum(int aNum);
        const int & Num() const;

        // Transform for "ideal sensor" coordinate to the pixel coordinates
        Pt2dr Hom2Cam(const Pt2df & aP) const;
        void AddCamBox(cCameraTiepRed*,int aKBox);

        void SaveHom();


    private :
        void SaveHom( cCameraTiepRed*,const std::list<int> & aLBox);
        cCameraTiepRed(const cCameraTiepRed &); // Not Implemented


        cAppliTiepRed & mAppli;
        std::string mNameIm;
        CamStenope * mCS;
        int          mNbPtsHom2Im;
        int          mNum;
        std::map<cCameraTiepRed*,std::list<int> > mMapCamBox;
};

class cLnk2ImTiepRed
{
     public :
        cLnk2ImTiepRed(cCameraTiepRed * ,cCameraTiepRed *);
        cCameraTiepRed &     Cam1();
        cCameraTiepRed &     Cam2();
        std::vector<Pt2df>&  VP1();
        std::vector<Pt2df>&  VP2();

        void Add2Merge(tMergeStr *);
     private :
        cCameraTiepRed *    mCam1;
        cCameraTiepRed *    mCam2;
        std::vector<Pt2df>  mVP1;
        std::vector<Pt2df>  mVP2;
};


class cPMulTiepRed
{
     public :
       cPMulTiepRed(tMerge *,cAppliTiepRed &);
       const Pt2dr & Pt() const {return mP;}
       int & HeapIndex() { return mHeapIndex;}
       const int & HeapIndex() const { return mHeapIndex;}
       const double  & Gain() const {return mGain;}
       double  & Gain() {return mGain;}
       const double  & Prec() const {return mPrec;}
       tMerge * Merge() {return mMerge;}
       void InitGain(cAppliTiepRed &);

       bool Removed() const;
       bool Removable() const;
       void Remove();
       void UpdateNewSel(const cPMulTiepRed *,cAppliTiepRed & anAppli);
     private :
       tMerge * mMerge;
       Pt2dr    mP;   // mP + Z => 3D coordinate
       double   mZ;
       double   mPrec;  // Precision of bundle intersection
       double   mGain;  // Gain to select this tie points (takes into account multiplicity and precision)
       int      mHeapIndex; // This memory will be used vy the heap to allow dynamic change of the priority
       bool     mRemoved;
       int      mNbCam0;
       int      mNbCamCur;
       std::vector<U_INT1> mVConserved;
};


typedef cPMulTiepRed * tPMulTiepRedPtr;

// Class to interact with the Quod Tree
class cP2dGroundOfPMul
{
    public :
          Pt2dr operator()(const tPMulTiepRedPtr &  aPM) {return aPM->Pt();}
};
typedef ElQT<cPMulTiepRed*,Pt2dr,cP2dGroundOfPMul>  tTiePRed_QT;

// Classes to interact with the heap
class cParamHeapPMulTiepRed
{
   public :
        static void SetIndex(tPMulTiepRedPtr  &  aPM,int i) { aPM->HeapIndex() = i;}
        static int  Index(const tPMulTiepRedPtr &  aPM) { return aPM->HeapIndex(); }
};

class cCompareHeapPMulTiepRed
{
    public :
        bool operator() (const tPMulTiepRedPtr & aP1,const tPMulTiepRedPtr &  aP2)
        {
             return  aP1->Gain() > aP2->Gain();
        }
};

typedef ElHeap<tPMulTiepRedPtr,cCompareHeapPMulTiepRed,cParamHeapPMulTiepRed>  tTiePRed_Heap;



class cAppliTiepRed 
{
     public :
          cAppliTiepRed(int argc,char **argv); 
          void Exe();
          cVirtInterf_NewO_NameManager & NM();
          const cXml_ParamBoxReducTieP & ParamBox() const;
          const double & ThresoldPrec2Point() const;
          const double & ThresholdPrecMult() const;
          const int    & ThresholdNbPts2Im() const;
          const int    & ThresholdTotalNbPts2Im() const;
          void AddLnk(cLnk2ImTiepRed *);
          cCameraTiepRed * KthCam(int aK);
          const double & StdPrec() const;
          std::vector<int>  & BufICam();
          std::string NameHomol(const std::string &,const std::string &,int aK) const;
          cInterfChantierNameManipulateur* ICNM();

     private :

          void GenerateSplit();
          void DoReduceBox();
          void DoLoadTiePoints();
          void DoFilterCamAnLinks();
          void DoExport();

          cAppliTiepRed(const cAppliTiepRed &); // N.I.

          static const std::string TheNameTmp;

          std::string DirOneImage(const std::string &) const;
          std::string NameParamBox(int aK,bool Bin) const;


          const std::vector<std::string> * mFilesIm;
          double mPrec2Point; // Threshold on precision for a pair of tie P
          double mThresholdPrecMult; // Threshold on precision for multiple points
          int    mThresholdNbPts2Im;
          int    mThresholdTotalNbPts2Im;
          int    mSzTile;    //  Number of pixel / tiles
          double mDistPMul;

          std::string  mDir;
          std::string  mPatImage;
          std::string  mCalib;

          std::map<std::string,cCameraTiepRed *> mMapCam;
          std::vector<cCameraTiepRed *>          mVecCam;
          std::set<std::string>          * mSetFiles;
          cVirtInterf_NewO_NameManager *   mNM ;
          bool                             mCallBack;
          int                              mKBox;
          Box2dr                           mBoxGlob;
          Box2dr                           mBoxLoc;
          double                           mResol;
          cXml_ParamBoxReducTieP           mXmlParBox;
          std::list<cLnk2ImTiepRed *>      mLnk2Im;
          tMergeStr *                      mMergeStruct;
          const std::list<tMerge *> *      mLMerge;
          // std::list<cPMulTiepRed *>        mLPMul;

          cP2dGroundOfPMul                 mPMul2Gr;
          tTiePRed_QT                      *mQT;
          cCompareHeapPMulTiepRed          mPMulCmp;
          tTiePRed_Heap                    *mHeap;
          std::list<tPMulTiepRedPtr>       mListSel; // List of selected multi points
          double                           mStdPrec;
          std::vector<int>                 mBufICam;
          cInterfChantierNameManipulateur* mICNM;
};


inline bool ImTest(const std::string & aName)
{
   return (aName=="Abbey-IMG_0206.jpg") ||  (aName=="Abbey-IMG_0207.jpg");
}

inline bool CpleImTest(const std::string & aName1,const std::string & aName2)
{
   return (aName1!=aName2) && ImTest(aName1) && ImTest(aName2);
}
inline bool CamTest(const cCameraTiepRed & aCam) { return ImTest(aCam.NameIm()); }
inline bool CpleCamTest(const cCameraTiepRed & aCam1,const cCameraTiepRed & aCam2) { return CpleImTest(aCam1.NameIm(),aCam2.NameIm()); }


NS_OriTiePRed_END

#endif // _TiepRed_H_

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
